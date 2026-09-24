"""Neural speech gate: ringtone/chirp rejected, speech kept; energy fallback with a warning.

Model tests need sherpa-onnx + silero_vad_v5.onnx (skipped otherwise). The speech
sample is macOS `say` TTS (skipped if unavailable); tests never use real recordings.
"""

import shutil
import subprocess
import warnings

import numpy as np
import pytest

from voiceprint import audio as au
from voiceprint import vad

SR = au.SAMPLE_RATE
needs_silero = pytest.mark.skipif(not vad.available(), reason="silero_vad_v5.onnx / sherpa-onnx missing")


def ringtone(sec=12.0):
    """North-American ring (440+480 Hz, 2 s on / 1 s off) + a fast chirp-train ringtone."""
    t = np.arange(int(sec * SR)) / SR
    ring = 0.25 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t)) * ((t % 3.0) < 2.0)
    f = 800 + 1200 * ((t * 8) % 1.0)                     # 8 up-chirps / s
    chirp = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR) * ((t % 1.0) < 0.6)
    noise = 0.001 * np.random.default_rng(1).normal(size=t.size)   # room floor (energy VAD needs one)
    return (np.where(t < sec / 2, ring, chirp) + noise).astype(np.float32)


def tts_speech(tmp_path, text):
    say = shutil.which("say")
    if not say:
        pytest.skip("macOS `say` unavailable")
    aiff = tmp_path / "s.aiff"
    subprocess.run([say, "-o", str(aiff), text], check=True, timeout=60)
    return au.decode_audio(aiff)


def one_utt(wave):
    return au.UtteranceSet([au.Utterance("A", 0, int(len(wave) / SR * 1000))])


# ─── no model needed ────────────────────────────────────────────────────────

def test_mask_region_mapping():
    m = vad.SpeechMask(np.array([0, 1, 1, 0], bool), hop=4)
    assert m.region(0, 16).tolist() == [False] * 4 + [True] * 8 + [False] * 4
    assert m.region(3, 6).tolist() == [False, True, True]
    assert m.region(14, 20).tolist() == [False] * 6              # past the end → non-speech


def test_gate_uses_mask_fraction():
    rng = np.random.default_rng(0)
    wave = (0.2 * rng.normal(size=10 * SR)).astype(np.float32)   # loud → energy VAD keeps all
    u = one_utt(wave)
    p = au.ClipParams(max_clip_s=12, neural_min_frac=0.6)
    hop = vad.HOP
    n = len(wave) // hop + 1
    for frac, kept in ((1.0, True), (0.7, True), (0.5, False), (0.0, False)):
        frames = np.zeros(n, bool)
        frames[: int(n * frac)] = True
        info = {}
        clips = au.select_clips(wave, u, "A", p, floor_db=-60.0, speech_mask=vad.SpeechMask(frames), info=info)
        assert bool(clips) is kept, frac
        if kept:
            assert clips[0].speech_frac >= 0.6
        else:
            assert info["rejected_non_speech"]["A"] == 1
    assert au.select_clips(wave, u, "A", p, floor_db=-60.0)[0].speech_frac is None   # no mask → not gated


def test_fallback_to_energy_with_warning(monkeypatch):
    monkeypatch.setattr(vad, "available", lambda spec=vad.SILERO_V5: False)
    wave = ringtone()
    info = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        per = au.select_meeting_clips(wave, one_utt(wave), au.ClipParams(), info=info)
    assert any("falling back to energy VAD" in str(x.message) for x in w)
    assert info["vad"] == "energy (fallback)" and per["A"]            # energy VAD lets the ringtone through


def test_energy_mode_skips_silero(monkeypatch):
    monkeypatch.setattr(vad, "silero_mask", lambda *a, **k: pytest.fail("silero called"))
    info = {}
    au.select_meeting_clips(ringtone(), one_utt(ringtone()), au.ClipParams(neural_vad="energy"), info=info)
    assert info["vad"] == "energy"


# ─── Silero model ───────────────────────────────────────────────────────────

@needs_silero
def test_silero_rejects_ringtone_and_chirps():
    wave = ringtone()
    assert vad.silero_mask(wave).speech_fraction < 0.1
    info = {}
    per = au.select_meeting_clips(wave, one_utt(wave), au.ClipParams(), info=info)
    assert per["A"] == [] and info["rejected_non_speech"]["A"] >= 1 and info["vad"] == "silero_v5"
    energy_only = au.select_meeting_clips(wave, one_utt(wave), au.ClipParams(neural_vad="energy"))
    assert energy_only["A"]                                         # the bug being fixed


@needs_silero
def test_silero_keeps_tts_speech(tmp_path):
    speech = tts_speech(tmp_path, "The quarterly report is ready for review. We should compare the new "
                                  "numbers with last month before the investor call on Thursday.")
    assert vad.silero_mask(speech).speech_fraction > 0.6
    info = {}
    per = au.select_meeting_clips(speech, one_utt(speech), au.ClipParams(), info=info)
    assert per["A"] and all(c.speech_frac >= 0.6 for c in per["A"])


@needs_silero
def test_silero_speech_then_ringtone_keeps_only_speech(tmp_path):
    speech = tts_speech(tmp_path, "Hello, can you hear me now? I just joined from my phone, "
                                  "so let me share the dashboard with you in a second.")
    gap = np.zeros(SR, np.float32)
    ring = ringtone(10.0)
    wave = np.concatenate([speech, gap, ring])
    ls = int(len(speech) / SR * 1000)
    u = au.UtteranceSet([au.Utterance("A", 0, ls), au.Utterance("A", ls + 1000, int(len(wave) / SR * 1000))])
    per = au.select_meeting_clips(wave, u, au.ClipParams())
    assert per["A"] and all(c.end_ms <= ls for c in per["A"])       # nothing from the ringtone turn
