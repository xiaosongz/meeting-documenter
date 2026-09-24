"""Device-aware clip cap, --device auto resolution, and the MPS parity fallback (all mocked)."""

import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import identify_speakers as cli
import voiceprint
from voiceprint import audio as au

SR = au.SAMPLE_RATE
SCRIPTS = Path(cli.__file__).parent


# ─── cap selection ──────────────────────────────────────────────────────────

def test_cap_for_device_override_and_ceiling():
    assert au.cap_for("cpu") == 12.0
    assert au.cap_for("mps") == 60.0
    assert au.cap_for("coreml") == 12.0 and au.cap_for(None) == 12.0     # unknown → CPU cap
    assert au.cap_for("cpu", override=20) == 20.0                        # explicit override lifts it
    assert au.cap_for("mps", override=300) == au.SAFETY_CEILING_S        # never above the ceiling


def _monologue(sec=55):
    rng = np.random.default_rng(3)
    wave = (0.001 * rng.normal(size=(sec + 10) * SR)).astype(np.float32)
    t = np.arange(sec * SR) / SR
    wave[2 * SR:(2 + sec) * SR] += (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    u = au.UtteranceSet([au.Utterance("A", 2000, (2 + sec) * 1000)])
    return wave, u


@pytest.mark.parametrize("device,limit", [("cpu", 12.0), ("mps", 60.0)])
def test_longest_run_cap_follows_device(device, limit):
    wave, u = _monologue(55)
    p = au.ClipParams(clip_mode="longest_run", device=device)
    clips = au.select_clips(wave, u, "A", p)
    longest = max(len(c.wave) for c in clips) / SR
    assert longest <= limit
    assert longest > (50 if device == "mps" else 11.9)                 # MPS keeps the ~54.5 s run


def test_window_sweep_params_respect_cap():
    wave, u = _monologue(55)
    for w in (4, 8, 12, 20, 30):
        for dev in ("cpu", "mps"):
            p = au.ClipParams(window_s=w, window_hop_s=w / 2, max_clip_s=w, device=dev)
            clips = au.select_clips(wave, u, "A", replace(p, max_clips=0))
            lens = [len(c.wave) / SR for c in clips]
            assert clips and max(lens) <= min(w, au.cap_for(dev)) + 1e-6


class RecordingEmbedder:
    name, model_id, dim, sample_rate = "rec", "rec@0", 4, SR

    def __init__(self, device):
        self.vp_device = device
        self.max_len = 0

    def embed_batch(self, waves):
        self.max_len = max([self.max_len, *[w.size for w in waves]])
        return np.ones((len(waves), 4), np.float32)


@pytest.mark.parametrize("device,limit", [("cpu", 12.0), ("mps", 60.0)])
def test_embed_meeting_takes_cap_from_embedder_device(device, limit):
    wave, u = _monologue(55)
    emb = RecordingEmbedder(device)
    au.embed_meeting(emb, None, u, au.ClipParams(clip_mode="longest_run", neural_vad="energy"), wave=wave)
    assert 11.9 * SR < emb.max_len <= limit * SR


def test_embed_clips_crop_uses_device_cap():
    mk = lambda: [au.Clip("A", 0, 1, 30.0, np.full(30 * SR, 0.1, np.float32))]
    for dev, limit in (("cpu", 12), ("mps", 30)):                       # 30 s < MPS cap: untouched
        emb = RecordingEmbedder(dev)
        au.embed_clips(emb, mk())
        assert emb.max_len == limit * SR


# ─── device resolution ──────────────────────────────────────────────────────

def _fake_torch(monkeypatch, mps: bool):
    t = types.SimpleNamespace(backends=types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)))
    monkeypatch.setitem(sys.modules, "torch", t)


def test_resolve_device(monkeypatch):
    _fake_torch(monkeypatch, True)
    assert cli.resolve_device("auto", "redimnet2", {"redimnet2": 1}) == "mps"
    assert cli.resolve_device("auto", "eres2netv2", {"redimnet2": 1}) == "cpu"   # sherpa model
    assert cli.resolve_device("cpu", "redimnet2", {"redimnet2": 1}) == "cpu"
    _fake_torch(monkeypatch, False)
    assert cli.resolve_device("auto", "redimnet2", {"redimnet2": 1}) == "cpu"


# ─── MPS parity self-test ───────────────────────────────────────────────────

class ParityEmb:
    dim = 4

    def __init__(self, vec):
        self.vec = np.asarray(vec, np.float32)

    def embed(self, w):
        return self.vec


def _fake_embedders(monkeypatch, mps_vec, calls):
    mod = types.ModuleType("voiceprint.embedders")
    mod.TORCH_SPECS = {"redimnet2": object()}
    mod.available_embedders = lambda: ["redimnet2"]

    def get_embedder(name, device="cpu"):
        calls.append(device)
        return ParityEmb(mps_vec if device == "mps" else [1, 0, 0, 0])
    mod.get_embedder = get_embedder
    monkeypatch.setitem(sys.modules, "voiceprint.embedders", mod)
    monkeypatch.setattr(voiceprint, "embedders", mod, raising=False)
    monkeypatch.setattr(cli, "_PARITY", {})
    _fake_torch(monkeypatch, True)


def test_mps_parity_pass_keeps_mps_and_is_cached(monkeypatch):
    calls = []
    _fake_embedders(monkeypatch, [1, 0, 0, 0], calls)
    emb, why = cli.load_embedder("redimnet2", "auto")
    assert why is None and emb.vp_device == "mps"
    assert cli._PARITY["redimnet2"]["ok"] and cli._PARITY["redimnet2"]["cos"] == pytest.approx(1.0)
    emb2, _ = cli.load_embedder("redimnet2", "auto")
    assert emb2.vp_device == "mps" and calls == ["mps", "cpu", "mps"]      # self-test ran once


def test_mps_parity_fail_falls_back_to_cpu(monkeypatch, capsys):
    calls = []
    _fake_embedders(monkeypatch, [0.95, 0.31, 0, 0], calls)               # cos ≈ 0.95 < 0.999
    emb, why = cli.load_embedder("redimnet2", "mps")
    assert why is None and emb.vp_device == "cpu"
    assert np.allclose(emb.vec, [1, 0, 0, 0])                              # the CPU embedder
    assert "parity self-test failed" in capsys.readouterr().err


def test_mps_parity_error_falls_back_to_cpu(monkeypatch):
    calls = []
    _fake_embedders(monkeypatch, [7, 7, 7, 7], calls)                     # MPS marker vector

    def boom_on_mps(self, w):
        if self.vec.tolist() == [7, 7, 7, 7]:
            raise RuntimeError("mps boom")
        return self.vec
    monkeypatch.setattr(ParityEmb, "embed", boom_on_mps)
    emb, why = cli.load_embedder("redimnet2", "auto")
    assert why is None and emb.vp_device == "cpu"
    assert "mps boom" in cli._PARITY["redimnet2"]["error"]


# ─── no-op path stays torch-free ────────────────────────────────────────────

def test_identify_noop_does_not_import_torch(tmp_path):
    side = tmp_path / "2026-01-01-1000 x.utterances.json"
    side.write_text('{"schema":1,"source":"assemblyai","speaker_map":null,"utterances":[]}')
    code = (f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import identify_speakers as c; "
            f"rc = c.main(['--store', {str(tmp_path / 'empty')!r}, 'identify', '--audio', 'x.m4a', "
            f"'--utterances', {str(side)!r}]); print('RC', rc, 'TORCH', 'torch' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert "RC 0 TORCH False" in out.stdout, out.stdout + out.stderr
