"""
Neural speech VAD (Silero v5 via sherpa-onnx; no torch) for voiceprint clip gating.

Why: the energy VAD in audio.py passes any loud sound. In phone calls, a
ringtone or hold music that the diarizer attributes to a speaker cluster
scores as near-100 % "speech" by energy but ≈0 by Silero v5, and would
otherwise become that cluster's longest (most trusted) clip. Silero runs ONCE
per decoded span; clips are then kept only if their Silero speech fraction
≥ ClipParams.neural_min_frac.

Cost (measured on an Apple-silicon laptop CPU, 1 thread): ≈10 s per hour of
audio; mask memory 1 byte per 512-sample hop (≈110 KB/h).

Model: silero_vad_v5.onnx from the sherpa-onnx `asr-models` release, pinned
by sha256 below; fetched by fetch_models.py into $VOICEPRINT_MODEL_DIR/sherpa-onnx/.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import model_dir

SAMPLE_RATE = 16000
HOP = 512                      # Silero v5 window at 16 kHz (32 ms)


@dataclass(frozen=True)
class VadSpec:
    name: str
    file: str
    sha256: str

    @property
    def path(self) -> Path:
        return model_dir() / "sherpa-onnx" / self.file

    @property
    def url(self) -> str:
        return f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{self.file}"


SILERO_V5 = VadSpec(name="silero_vad", file="silero_vad_v5.onnx",
                    sha256="6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396")


def available(spec: VadSpec = SILERO_V5) -> bool:
    """sherpa-onnx importable and the model file present (sha256 is checked at fetch time)."""
    return importlib.util.find_spec("sherpa_onnx") is not None and spec.path.is_file()


@dataclass
class SpeechMask:
    """Frame-level Silero speech mask over a decoded wave (frame = HOP samples)."""
    frames: np.ndarray            # bool, one per HOP samples
    hop: int = HOP

    def region(self, a: int, b: int) -> np.ndarray:
        """Sample-level bool mask for wave[a:b]."""
        if b <= a:
            return np.zeros(0, dtype=bool)
        fa, fb = a // self.hop, (b - 1) // self.hop + 1
        m = np.repeat(self.frames[fa:fb], self.hop)
        off = a - fa * self.hop
        out = m[off:off + (b - a)]
        if out.size < b - a:                       # past the last frame → non-speech
            out = np.concatenate([out, np.zeros(b - a - out.size, dtype=bool)])
        return out

    @property
    def speech_fraction(self) -> float:
        return float(self.frames.mean()) if self.frames.size else 0.0


def silero_mask(wave: np.ndarray, sr: int = SAMPLE_RATE, threshold: float = 0.5,
                min_silence_s: float = 0.3, min_speech_s: float = 0.25,
                spec: VadSpec = SILERO_V5, num_threads: int = 1) -> SpeechMask:
    """Run Silero once over `wave` (16 kHz mono float32) and return a frame mask.

    Pauses shorter than `min_silence_s` are bridged (matches the energy VAD's
    0.3 s gap rule). Raises FileNotFoundError / ImportError if unavailable —
    callers should check `available()` first and fall back.

    Fed one HOP window per accept_waveform call: with sherpa-onnx 1.13.8,
    feeding 10 s chunks gave different segments on the same speech (speech
    fraction 0.15 vs 0.55), so per-window feeding is the reference.
    """
    import sherpa_onnx as so

    if not spec.path.is_file():
        raise FileNotFoundError(spec.path)
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = str(spec.path)
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = min_silence_s
    cfg.silero_vad.min_speech_duration = min_speech_s
    cfg.silero_vad.window_size = HOP
    cfg.silero_vad.max_speech_duration = 30.0
    cfg.sample_rate = sr
    cfg.num_threads = num_threads
    vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=120)

    n_frames = (len(wave) + HOP - 1) // HOP
    frames = np.zeros(n_frames, dtype=bool)

    def drain():
        while not vad.empty():
            seg = vad.front
            a = int(seg.start)
            b = a + len(seg.samples)
            frames[a // HOP:(b + HOP - 1) // HOP] = True
            vad.pop()

    w = np.asarray(wave, dtype=np.float32)
    for i in range(0, len(w), HOP):
        vad.accept_waveform(w[i:i + HOP])
        if not vad.empty():
            drain()
    vad.flush()
    drain()
    return SpeechMask(frames)
