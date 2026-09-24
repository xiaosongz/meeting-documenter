"""Boundary layer (segment.py). Pure-logic tests always run; model tests need onnxruntime
+ the seg3 ONNX file (VOICEPRINT_MODEL_DIR) and macOS `say` TTS, and skip otherwise.
Audio is synthetic (TTS or silence), never a real recording."""

import shutil
import subprocess

import numpy as np
import pytest

from voiceprint import segment as sg

SR = sg.SAMPLE_RATE
needs_seg3 = pytest.mark.skipif(not sg.available("seg3"), reason="onnxruntime / seg3 model missing")


def fake_seg(acts, starts=(0.0,), duration=None):
    acts = np.asarray(acts, np.float32)
    return sg.LocalSegmentation(
        chunk_starts=np.asarray(starts, float), chunk_dur=10.0, frame_step=0.1, frame_dur=0.1,
        activations=acts, backend="test", meta={"duration": duration or 1e9})


# ─── pure logic ──────────────────────────────────────────────────────────────

def test_chunk_starts_cover_the_whole_file():
    win = 10 * SR
    assert sg.chunk_starts_for(3 * SR, 2.5).tolist() == [0]                  # short: one padded chunk
    assert sg.chunk_starts_for(win, 2.5).tolist() == [0]
    s = sg.chunk_starts_for(win + 5 * SR, 2.5)
    assert s.tolist() == [0, 2.5 * SR, 5 * SR]                               # exact fit, no extra
    s = sg.chunk_starts_for(win + 6 * SR, 2.5)
    assert s[-1] == 6 * SR and s[-2] == 5 * SR                               # end-aligned tail chunk
    assert np.all(np.diff(s) > 0) and np.all(np.diff(s) <= 2.5 * SR)
    with pytest.raises(ValueError):
        sg.chunk_starts_for(win, 0.0)


def test_runs():
    assert sg._runs(np.array([0, 1, 1, 0, 1], bool)) == [(1, 3), (4, 5)]
    assert sg._runs(np.zeros(4, bool)) == []


def test_local_regions_excludes_overlap_and_short():
    a = np.zeros((1, 40, 3), np.float32)
    a[0, 0:20, 0] = 0.9          # local 0: frames 0-19
    a[0, 15:30, 1] = 0.8         # local 1: frames 15-29 (overlaps 15-19)
    a[0, 35:36, 2] = 0.9         # local 2: one frame → dropped by min_span_s
    seg = fake_seg(a, starts=(5.0,))
    r = {g.local: g for g in sg.local_regions(seg, min_clean_s=0.0, min_span_s=0.15)}
    assert set(r) == {0, 1}
    # frame i centre = 5.0 + 0.1 i + 0.05; interval ± 0.05
    assert r[0].spans == [pytest.approx((5.0, 6.5))]
    assert r[1].spans == [pytest.approx((7.0, 8.0))]
    assert r[0].clean_s == pytest.approx(1.5)
    r2 = {g.local: g for g in sg.local_regions(seg, 0.0, exclude_overlap=False, min_span_s=0.15)}
    assert r2[0].spans == [pytest.approx((5.0, 7.0))] and r2[1].spans == [pytest.approx((6.5, 8.0))]
    assert [g.local for g in sg.local_regions(seg, min_clean_s=1.2)] == [0]


def test_local_regions_clips_to_duration_and_threshold():
    a = np.zeros((1, 40, 3), np.float32)
    a[0, 30:40, 0] = 0.6
    seg = fake_seg(a, duration=3.5)
    (g,) = sg.local_regions(seg, 0.0)
    assert g.spans[0][1] == pytest.approx(3.5)
    assert sg.local_regions(seg, 0.0, threshold=0.7) == []


def test_segment_rejects_bad_input():
    with pytest.raises(ValueError):
        sg.segment(np.zeros(SR, np.float32), sr=8000)
    with pytest.raises(ValueError):
        sg.segment(np.zeros((2, SR), np.float32))
    with pytest.raises(ValueError):
        sg.segment(np.zeros(SR, np.float32), backend="nope")


# ─── model tests (synthetic TTS) ─────────────────────────────────────────────

def tts(tmp_path, voice, text, idx):
    say, ff = shutil.which("say"), shutil.which("ffmpeg")
    if not (say and ff):
        pytest.skip("macOS `say` / ffmpeg unavailable")
    aiff = tmp_path / f"{idx}.aiff"
    r = subprocess.run([say, "-v", voice, "-o", str(aiff), text], capture_output=True, timeout=120)
    if r.returncode != 0 or not aiff.exists():
        pytest.skip(f"`say` voice {voice} unavailable")
    from voiceprint.audio import decode_audio
    return decode_audio(aiff)


def two_voice_dialogue(tmp_path):
    """A (female) / B (male) alternate with 0.7 s pauses; returns wave and reference turns."""
    lines = [("Samantha", "Thanks for joining. Today we review the release checklist and the open items."),
             ("Daniel", "The build finished overnight, and all the integration tests passed."),
             ("Samantha", "Good. I will update the documentation this afternoon."),
             ("Daniel", "Then we can tag the release candidate tomorrow morning.")]
    rng = np.random.default_rng(0)
    pieces, ref, t = [], [], 0.5
    pieces.append(np.zeros(int(0.5 * SR), np.float32))
    for i, (v, text) in enumerate(lines):
        w = tts(tmp_path, v, text, i)
        ref.append((t, t + len(w) / SR, v))
        pieces += [w, np.zeros(int(0.7 * SR), np.float32)]
        t += len(w) / SR + 0.7
    x = np.concatenate(pieces)
    return x + 1e-3 * rng.normal(size=x.size).astype(np.float32), ref


@needs_seg3
def test_seg3_shapes_and_silence():
    x = np.zeros(25 * SR, np.float32)
    seg = sg.segment(x, step=2.5, num_threads=1)
    assert seg.backend == "seg3" and seg.n_frames == sg.N_FRAMES and seg.n_local == 3
    assert seg.chunk_starts.tolist() == [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    assert seg.activations.dtype == np.float32
    assert seg.activations.min() >= 0 and seg.activations.max() <= 1
    assert seg.meta["speech"].max() < 0.5
    assert sg.local_regions(seg, 0.0) == []


@needs_seg3
def test_seg3_separates_two_tts_voices(tmp_path):
    x, ref = two_voice_dialogue(tmp_path)
    seg = sg.segment(x, step=2.5, num_threads=1)
    assert seg.meta["duration"] == pytest.approx(len(x) / SR)
    t = sg.global_frames(seg)
    in_speech = np.zeros(t.size, bool)
    for s, e, _ in ref:
        in_speech |= (t > s + 0.2) & (t < e - 0.2)
    assert seg.meta["speech"][in_speech].mean() > 0.8

    # Every clean region should be (almost) one voice: purity vs the synthetic reference.
    regions = sg.local_regions(seg, min_clean_s=0.5)
    assert regions
    pure, total = 0.0, 0.0
    for g in regions:
        per_voice = {}
        for s, e in g.spans:
            for rs, re_, v in ref:
                per_voice[v] = per_voice.get(v, 0.0) + max(0.0, min(e, re_) - max(s, rs))
        total += g.clean_s
        pure += max(per_voice.values(), default=0.0)
    assert pure / total > 0.9
    # Chunks spanning a turn change see two different local speakers.
    change = ref[0][1] + 0.35
    spanning = [c for c in range(seg.n_chunks)
                if seg.chunk_starts[c] + 2 < change < seg.chunk_starts[c] + 8]
    assert any(len({g.local for g in regions if g.chunk == c}) >= 2 for c in spanning)
