import numpy as np

from voiceprint import audio as au

SR = au.SAMPLE_RATE


def uset(turns, approx=False):
    return au.UtteranceSet([au.Utterance(s, a, b) for s, a, b in turns], approx=approx)


def test_regions_trim_and_overlap_guard():
    u = uset([("A", 0, 10_000), ("B", 6_000, 7_000), ("A", 20_000, 21_500)])
    regs = au.speech_regions(u, "A", au.ClipParams(min_net_s=0.1))
    # A:0–10 s trimmed to 0.25–9.75, minus B±0.3 (5.7–7.3); second A turn 1.5 s → 1.0 s after trim
    assert regs[0] == (0.25, 5.7)
    assert regs[1] == (7.3, 9.75)
    assert regs[2][1] - regs[2][0] == 1.0
    regs2 = au.speech_regions(u, "A", au.ClipParams(min_net_s=2.0))
    assert len(regs2) == 2                               # short pieces dropped


def test_approx_uses_turn_middle_only():
    u = uset([("Bob", 0, 10_000), ("Alice", 10_000, 14_000)], approx=True)
    p = au.ClipParams(approx_window_s=4.0, approx_min_turn_s=8.0)
    assert au.speech_regions(u, "Bob", p) == [(3.0, 7.0)]
    assert au.speech_regions(u, "Alice", p) == []       # 4 s turn < 8 s


def test_utterances_from_dict_gemini_flag():
    d = {"schema": 1, "source": "gemini-approx", "utterances": [
        {"speaker": "X", "start": 5000, "end": 9000}, {"speaker": "Y", "start": 1000, "end": 1000}]}
    us = au.utterances_from_dict(d)
    assert us.approx and len(us.utterances) == 1        # zero-length dropped


def _tone(sec, f=220.0, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_vad_removes_long_silence_keeps_short_pause():
    rng = np.random.default_rng(0)
    noise = lambda s: (0.001 * rng.normal(size=int(s * SR))).astype(np.float32)
    seg = np.concatenate([_tone(2), noise(0.2), _tone(2), noise(1.5), _tone(2)])
    floor = au.noise_floor_db(np.concatenate([seg, noise(10)]))
    kept = au.vad_keep(seg, floor)
    # 1.5 s gap removed, 0.2 s pause kept → ≈ 6.2 s
    assert abs(len(kept) / SR - 6.2) < 0.1


def test_select_clips_windows_and_cap():
    rng = np.random.default_rng(1)
    wave = (0.001 * rng.normal(size=60 * SR)).astype(np.float32)
    wave[1 * SR:31 * SR] += _tone(30)                   # A speaks 1–31 s
    wave[40 * SR:43 * SR] += _tone(3, f=400)            # B speaks 40–43 s
    u = uset([("A", 1000, 31_000), ("B", 40_000, 43_000)])
    clips = au.select_clips(wave, u, "A")
    assert all(6.0 <= c.net_dur <= 10.0 for c in clips)  # 30 s split into 8 s windows
    assert len(clips) >= 6
    capped = au.select_clips(wave, u, "A", au.ClipParams(max_clips=3))
    assert len(capped) == 3
    b = au.select_clips(wave, u, "B")
    assert len(b) == 1 and 2.0 <= b[0].net_dur <= 2.6


def test_meeting_key_from():
    assert au.meeting_key_from("x/2026-01-01-1000 Example Meeting-Transcript.utterances.json") == "2026-01-01-1000"
    assert au.meeting_key_from(None, "foo.m4a") is None


class FakeEmbedder:
    name, model_id, dim, sample_rate = "fake", "fake@0", 8, SR

    def embed_batch(self, waves):
        return np.stack([np.full(8, len(w), np.float32) for w in waves])


def test_embed_clips_normalises_and_frees_waves():
    clips = [au.Clip("A", 0, 1, 2.0, np.zeros(SR * 2, np.float32)) for _ in range(20)]
    E = au.embed_clips(FakeEmbedder(), clips, batch_size=7)
    assert E.shape == (20, 8) and np.allclose(np.linalg.norm(E, axis=1), 1)
    assert all(c.wave is None for c in clips)


def test_no_clip_exceeds_cap_identify_or_enroll():
    rng = np.random.default_rng(2)
    wave = (0.001 * rng.normal(size=80 * SR)).astype(np.float32)
    wave[1 * SR:12 * SR] += _tone(11)                   # 11 s turn (> 10 s default cap)
    wave[20 * SR:75 * SR] += _tone(55)                  # 55 s monologue
    u = uset([("A", 1000, 12_000), ("A", 20_000, 75_000)])
    for params in (au.IDENTIFY_PARAMS, au.ENROLL_PARAMS):
        clips = au.select_clips(wave, u, "A", params)
        assert clips and max(len(c.wave) for c in clips) <= params.max_clip_s * SR
    big = au.select_clips(wave, u, "A", au.ClipParams(max_clip_s=60))   # param can't exceed hard cap
    assert max(len(c.wave) for c in big) <= au.MAX_EMBED_S * SR


class StrictEmbedder(FakeEmbedder):
    """Mimics embedders.py: ValueError on < 0.5 s or NaN; records max length seen."""
    seen = 0

    def embed_batch(self, waves):
        for w in waves:
            if w.size < SR // 2 or not np.isfinite(w).all():
                raise ValueError("bad clip")
            StrictEmbedder.seen = max(StrictEmbedder.seen, w.size)
        return super().embed_batch(waves)


def test_embed_clips_crops_long_and_drops_invalid_keeping_alignment():
    long = au.Clip("A", 0, 1, 30.0, np.ones(30 * SR, np.float32) * 0.1)
    short = au.Clip("A", 0, 1, 0.2, np.ones(SR // 5, np.float32))
    nan = au.Clip("A", 0, 1, 3.0, np.full(3 * SR, np.nan, np.float32))
    ok = au.Clip("A", 5, 6, 3.0, np.ones(3 * SR, np.float32) * 0.1)
    clips = [long, short, nan, ok]
    E = au.embed_clips(StrictEmbedder(), clips, batch_size=4)
    assert clips == [long, ok] and E.shape[0] == 2
    assert StrictEmbedder.seen <= au.MAX_EMBED_S * SR and long.net_dur == au.MAX_EMBED_S


def test_frame_db_chunked_matches_reference():
    rng = np.random.default_rng(5)
    x = (0.1 * rng.normal(size=SR * 7 + 123)).astype(np.float32)
    frame = 320
    got = au._frame_db(x, frame, chunk_frames=37)                     # odd chunk to hit boundaries
    f = x[: (len(x) // frame) * frame].reshape(-1, frame).astype(np.float64)
    ref = 10 * np.log10((f * f).mean(axis=1) + 1e-12)
    assert got.dtype == np.float32 and got.shape == ref.shape and np.allclose(got, ref, atol=1e-3)


def test_decode_span_offsets(tmp_path):
    """Only the utterance span is decoded, and clip times come back in file time."""
    from scipy.io import wavfile
    sec = 120
    t = np.arange(sec * SR) / SR
    x = (0.001 * np.random.default_rng(0).normal(size=t.size)).astype(np.float32)
    x[90 * SR:100 * SR] += (0.3 * np.sin(2 * np.pi * 200 * t[:10 * SR])).astype(np.float32)
    wav = tmp_path / "2026-01-01-0000 x.wav"
    wavfile.write(wav, SR, (x * 32767).astype(np.int16))
    u = uset([("A", 90_000, 100_000)])
    wave, off = au.decode_for(wav, u, pad_s=1.0)
    assert off == 89.0 and abs(len(wave) / SR - 12.0) < 0.05          # 12 s, not 120 s
    full = au.decode_audio(wav)
    assert abs(len(full) / SR - sec) < 0.05
    assert np.allclose(wave[:SR], full[89 * SR:90 * SR], atol=1e-4)    # sample-accurate seek
    per = au.embed_meeting(FakeEmbedder(), wav, u, au.ClipParams(neural_vad="energy"))
    clips, _ = per["A"]
    assert clips and all(90_000 <= c.start_ms and c.end_ms <= 100_000 for c in clips)


def test_quantize_to_half_second_grid_matches_eval_rule():
    rng = np.random.default_rng(4)
    wave = (0.001 * rng.normal(size=40 * SR)).astype(np.float32)
    wave[1 * SR:9 * SR] += _tone(8.0)                      # 7.5 s after trim → already on grid
    wave[12 * SR:18 * SR] += _tone(6.0)                    # 5.5 s
    wave[20 * SR:25 * SR] += _tone(5.0, f=300)             # 4.5 s
    u = uset([("A", 1000, 9_000), ("A", 12_000, 18_130), ("A", 20_000, 25_070)])
    raw = au.select_clips(wave, u, "A", au.ClipParams(quant_s=0))
    assert any(len(c.wave) % (SR // 2) for c in raw)                 # off-grid without quantization
    q = au.select_clips(wave, u, "A", au.ClipParams())               # default quant_s = 0.5
    assert q and all(len(c.wave) % (SR // 2) == 0 for c in q)
    assert all(c.net_dur == len(c.wave) / SR for c in q)
    # centre crop, identical to the offline evaluation's quantization rule
    w = np.arange(int(2.7 * SR), dtype=np.float32)
    c = au.quantize_clips([au.Clip("A", 0, 1, 2.7, w.copy())], au.ClipParams())[0]
    n = int(2.5 * SR)
    assert len(c.wave) == n and c.wave[0] == (len(w) - n) // 2
    assert au.quantize_clips([au.Clip("A", 0, 1, 2.3, w[: int(2.3 * SR)].copy())],
                             au.ClipParams(min_net_s=2.5)) == []  # below min after flooring → dropped
