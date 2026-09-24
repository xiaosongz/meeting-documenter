"""Real-model checks. Skipped unless torch/sherpa-onnx AND the model files are installed
(point VOICEPRINT_MODEL_DIR at them); CI never downloads models.

The end-to-end check uses two synthetic macOS `say` voices, never real recordings.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import identify_speakers as cli
from voiceprint import audio as au
from voiceprint import embedders as em

SR = au.SAMPLE_RATE
AVAILABLE = em.available_embedders()
needs_model = pytest.mark.skipif(not AVAILABLE, reason="no embedder deps/models installed")


def voice_like(sec=4.0, f0=140.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    phase = 2 * np.pi * np.cumsum(f0 + 15 * np.sin(2 * np.pi * 0.7 * t)) / SR
    x = sum((0.3 / k) * np.sin(k * phase) for k in range(1, 8))
    return np.clip(x + 0.01 * rng.normal(size=t.size), -1, 1).astype(np.float32)


@needs_model
@pytest.mark.parametrize("name", AVAILABLE)
def test_embedder_contract(name):
    emb = em.get_embedder(name, device="cpu")
    assert emb.model_id and emb.dim > 0 and emb.sample_rate == SR
    assert emb.model_id == em.model_id_for(name)                          # readiness checks rely on it
    w = voice_like()
    a, b = emb.embed(w), emb.embed(w)
    assert a.shape == (emb.dim,) and a.dtype == np.float32
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-4
    assert np.allclose(a, b, atol=1e-5)                                   # deterministic on CPU
    batch = emb.embed_batch([w, voice_like(3.0, 220.0, seed=1)])
    assert batch.shape == (2, emb.dim) and np.allclose(batch[0], a, atol=1e-4)
    with pytest.raises(ValueError):
        emb.embed(np.zeros(SR // 4, np.float32))                           # < 0.5 s


@needs_model
def test_load_embedder_auto_device_passes_parity():
    name = AVAILABLE[0]
    emb, why = cli.load_embedder(name, "auto")
    assert why is None and emb.vp_device in ("cpu", "mps")


def tts(tmp_path, voice, text, idx):
    say = shutil.which("say")
    if not say:
        pytest.skip("macOS `say` unavailable")
    out = tmp_path / f"{voice}-{idx}.aiff"
    r = subprocess.run([say, "-v", voice, "-o", str(out), text], capture_output=True, timeout=120)
    if r.returncode != 0 or not out.exists():
        pytest.skip(f"`say` voice {voice} unavailable")
    return au.decode_audio(out)


LINES = [
    "Thanks everyone for joining. Today we will review the release checklist and the open questions.",
    "The build finished overnight, and the integration tests passed on the second attempt.",
    "I will update the documentation this afternoon and share the draft before the next meeting.",
    "Let us make sure the example data stays synthetic, and that nobody commits a real recording.",
]


def tts_meeting(tmp_path, key, voices, seed):
    from scipy.io import wavfile
    rng = np.random.default_rng(seed)
    pieces, utts, t = [], [], 0
    for i, line in enumerate(LINES):
        c = i % len(voices)
        w = tts(tmp_path, voices[c], line, f"{key}-{i}")
        pieces += [w, 0.001 * rng.normal(size=SR).astype(np.float32)]
        utts.append({"speaker": "AB"[c], "start": t, "end": t + int(len(w) / SR * 1000), "text": line})
        t += int(len(w) / SR * 1000) + 1000
    wav = tmp_path / f"{key} Example Meeting.wav"
    wavfile.write(wav, SR, (np.concatenate(pieces) * 32767).astype(np.int16))
    side = tmp_path / f"{key} Example Meeting-Transcript.utterances.json"
    side.write_text(json.dumps({"schema": 1, "source": "assemblyai", "speaker_map": None, "utterances": utts}))
    return wav, side


@needs_model
def test_tts_enroll_identify_end_to_end(tmp_path):
    """Two synthetic voices: enroll from two meetings, identify a third (uncalibrated → low)."""
    model = "redimnet2" if "redimnet2" in AVAILABLE else AVAILABLE[0]
    voices = ["Samantha", "Daniel"]
    store = tmp_path / "vp"
    common = ["--model", model, "--device", "cpu", "--min-net", "1.0"]
    for i, key in enumerate(["2026-01-01-1000", "2026-01-02-1000"]):
        wav, side = tts_meeting(tmp_path, key, voices, seed=i)
        for cluster, name in (("A", "Alice"), ("B", "Bob")):
            assert cli.main(["--store", str(store), "enroll", "--name", name, "--cluster", cluster,
                             "--audio", str(wav), "--utterances", str(side), *common]) == 0
    wav, side = tts_meeting(tmp_path, "2026-01-03-1000", voices[::-1], seed=3)   # swapped order
    out = tmp_path / "r.json"
    assert cli.main(["--store", str(store), "identify", "--audio", str(wav), "--utterances", str(side),
                     "--min-cluster-s", "4", "--json", str(out), *common]) == 0
    r = json.loads(out.read_text())
    assert r["status"] == "ok" and not r["calibrated"]
    best = {d["cluster"]: d["best"] for d in r["decisions"]}
    assert best == {"A": "Bob Example", "B": "Alice Example"}
    assert all(d["confidence"] != "high" for d in r["decisions"])        # uncalibrated cap


def test_bounded_batches_equal_lengths_and_audio_cap():
    sr = em.SAMPLE_RATE
    lengths = [2 * sr] * 30 + [sr] * 3 + [70 * sr]
    batches = em.bounded_batches(lengths)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len({lengths[i] for i in b}) == 1                          # never mixes lengths
        assert len(b) == 1 or sum(lengths[i] for i in b) <= em.MAX_BATCH_S * sr
    assert [lengths[b[0]] for b in batches] == sorted(lengths[b[0]] for b in batches)
    assert [len(b) for b in batches if lengths[b[0]] == 2 * sr] == [24, 6]


def test_footprint_readings():
    from voiceprint import footprint as fp
    cur, peak = fp.phys_footprint(), fp.peak_footprint()
    if cur is None:
        pytest.skip("physical footprint not available on this platform")
    assert 0 < cur <= peak
    fp.FootprintGuard(limit_gb=1e6).check()
    with pytest.raises(fp.FootprintExceeded):
        fp.FootprintGuard(limit_gb=1e-6).check("test")


@pytest.mark.skipif(not os.environ.get("VOICEPRINT_PERF"), reason="set VOICEPRINT_PERF=1 (slow, needs MPS + models)")
@needs_model
def test_mps_embedding_footprint_bounded(tmp_path):
    """Mixed-length input through audio.embed_clips must keep the MPS footprint bounded."""
    tool = Path(__file__).resolve().parents[1] / "tools" / "perf_embed.py"
    r = subprocess.run([sys.executable, str(tool), "--device", "mps", "--minutes", "10",
                        "--max-footprint-gb", "8", "--json", str(tmp_path / "p.json")],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
