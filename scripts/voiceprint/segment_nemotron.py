"""
Optional `nemotron3` boundary backend: NVIDIA Nemotron 3 Diarization (OpenMDW-1.1).

    segment(wave, device="cpu") -> LocalSegmentation with n_chunks = 1

The model emits up to 8 speaker slots for the whole recording, ordered by first
arrival, at 10 ms frames. Slots are global (an Arrival-Order Speaker Cache carries
them across the model's internal 27.2 s chunks), so the result is one "chunk" that
spans the file and whose local speakers are the slots. Only the main
`nvidia/Nemotron-3-Diarization` checkpoint is used, never the `-preview` artifact
(evaluation-only licence).

Runtime: Hugging Face Transformers (the Nemotron 3 Diarization model class landed
after the 5.17.0 release; a build that has `transformers.models.nemotron3_diarization`
is required) plus torch. Neither NeMo nor any network access is needed at run time:
weights are fetched once by `fetch_models.py nemotron3` into
$VOICEPRINT_MODEL_DIR/nemotron3-diarization@<rev>/, pinned by sha256.

Memory: the checkpoint's offline configuration (340-frame chunks + 40 look-ahead,
FIFO 40, cache update 300) is run *chunk by chunk* through the streaming API, so the
spectrogram is never computed for the whole file at once. A single offline forward on
a 2 h file peaked at 8.8 GB RSS; the chunked form bounds it by one 30.4 s chunk.

Device: CPU is the default and the reference. MPS is ~2.7× faster but not bit-close
(max |Δp| 0.32, 99.9 % binary frame agreement on an AMI meeting).
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np

from .config import model_dir

HF_REPO = "nvidia/Nemotron-3-Diarization"
HF_REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"   # main @ 2026-09-24
FILES = {   # file → sha256
    "config.json": "dd795ec07ffa09acf18d231aa3fa8642aa5c521760953754e02559a7413b6873",
    "processor_config.json": "91726746788a024a2f0f2f699a6c3667f0d74e6a0e9288fc44fe3018e23af68e",
    "model.safetensors": "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6",
}
NAME = "nemotron3"
SAMPLE_RATE = 16000
FRAME_S = 0.01
OFFLINE_MODE = "offline"            # registered on the processor from the checkpoint's offline sizes


def local_dir() -> Path:
    return model_dir() / f"nemotron3-diarization@{HF_REVISION[:12]}"


def url(file: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/{HF_REVISION}/{file}"


def _has_model_class() -> bool:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
        return False
    try:
        dist = importlib.metadata.distribution("transformers")
    except importlib.metadata.PackageNotFoundError:
        return False
    root = Path(dist.locate_file("transformers"))
    return (root / "models" / "nemotron3_diarization" / "modeling_nemotron3_diarization.py").is_file()


def available() -> bool:
    """transformers build with the model class, torch, and the pinned files present."""
    return _has_model_class() and all((local_dir() / f).is_file() for f in FILES)


_MODELS: dict[str, tuple] = {}


def _load(device: str):
    if device not in _MODELS:
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        d = local_dir()
        missing = [f for f in FILES if not (d / f).is_file()]
        if missing:
            raise FileNotFoundError(f"{d}: missing {missing}; run: python scripts/voiceprint/fetch_models.py nemotron3")
        proc = AutoProcessor.from_pretrained(d, local_files_only=True)
        model = AutoModelForAudioFrameClassification.from_pretrained(d, local_files_only=True).eval().to(device)
        cfg = model.config
        proc.streaming_modes = {**dict(proc.streaming_modes),
                                OFFLINE_MODE: (cfg.chunk_length, cfg.chunk_right_context)}
        proc.set_streaming_mode(OFFLINE_MODE)
        _MODELS[device] = (torch, proc, model)
    return _MODELS[device]


def segment(wave: np.ndarray, device: str = "cpu", num_threads: int | None = None):
    """Whole-file slot activations as a one-chunk LocalSegmentation (10 ms frames, 8 slots)."""
    import os

    from .segment import LocalSegmentation

    torch, proc, model = _load(device)
    torch.set_num_threads(num_threads or int(os.environ.get("VOICEPRINT_THREADS", "4")))
    from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import (
        Nemotron3DiarizationSpeakerCache,
    )

    cfg = model.config
    n = wave.shape[0]
    probs: list[np.ndarray] = []
    cache = Nemotron3DiarizationSpeakerCache(cfg.streaming_config, fifo_length=cfg.fifo_length,
                                            speaker_cache_update_period=cfg.speaker_cache_update_period)
    with torch.inference_mode():
        for audio, first, last in _chunks(proc, wave):
            inputs = proc(audio, sampling_rate=SAMPLE_RATE, is_streaming=True,
                          is_first_audio_chunk=first, is_last_audio_chunk=last).to(device)
            out = model(**inputs, speaker_cache=cache)
            cache = out.speaker_cache
            probs.append(out.logits[0].sigmoid().float().cpu().numpy())
    p = np.concatenate(probs)[: int(np.ceil(n / (SAMPLE_RATE * FRAME_S)))]
    none = np.prod(1.0 - p, axis=1)
    one = sum(p[:, k] * np.prod(np.delete(1.0 - p, k, axis=1), axis=1) for k in range(p.shape[1]))
    meta = {
        "duration": n / SAMPLE_RATE,
        "model": f"{HF_REPO}@{HF_REVISION[:12]}",
        "device": device,
        "speech": (1.0 - none).astype(np.float32),                 # slots treated as independent
        "overlap": np.clip(1.0 - none - one, 0.0, 1.0).astype(np.float32),
    }
    return LocalSegmentation(chunk_starts=np.zeros(1), chunk_dur=n / SAMPLE_RATE, frame_step=FRAME_S,
                             frame_dur=FRAME_S, activations=p[None].astype(np.float32),
                             backend=NAME, meta=meta)


def _chunks(proc, wave: np.ndarray):
    """(audio, is_first, is_last) pieces following the processor's streaming contract."""
    n = wave.shape[0]
    first_n = proc.num_samples_first_audio_chunk
    if n <= first_n:
        yield wave, True, True
        return
    yield wave[:first_n], True, False
    mel = proc.num_mel_frames_per_step
    start = proc.audio_chunk_start(mel)
    while start + proc.num_samples_per_audio_chunk <= n:
        yield wave[start:start + proc.num_samples_per_audio_chunk], False, False
        mel += proc.num_mel_frames_per_step
        start = proc.audio_chunk_start(mel)
    yield wave[start:], False, True
