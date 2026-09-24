"""Tracker core on synthetic activations and embeddings (no audio models needed)."""

import numpy as np
import pytest

from voiceprint import calibrate_track as CT
from voiceprint import track as T
from voiceprint.segment import LocalSegmentation
from voiceprint.timeline import GUEST_PREFIX, UNKNOWN

D = 16
RNG = np.random.default_rng(0)
BASE = T._l2(RNG.normal(size=(5, D)))           # voices 0-4


def voice(i, n=1, noise=0.15, rng=RNG):
    return T._l2(BASE[i] + noise * rng.normal(size=(n, D)))


def slot_seg(pattern, step=0.01, n_slots=3):
    """One-chunk (global-slot) segmentation from [(slot, start_s, end_s), ...]."""
    dur = max(e for _, _, e in pattern)
    n = int(round(dur / step))
    P = np.zeros((n, n_slots), np.float32)
    for k, s, e in pattern:
        P[int(round(s / step)):int(round(e / step)), k] = 0.9
    return LocalSegmentation(np.zeros(1), dur, step, step, P[None], "nemotron3", {"duration": dur})


def test_global_slots_become_tracks_and_units():
    seg = slot_seg([(0, 0, 25), (1, 25, 32), (0, 32, 40), (1, 39, 45)])
    p = T.TrackParams(pad_s=0.0)
    trs = T.tracks_from_segmentation(seg, p)
    assert len(trs.tracks) == 2 and trs.step == 0.01 and abs(trs.t0) < 1e-12
    units = T.build_units(trs, p)
    # slot 0: 25 s of clean speech → 3 units (≤ 10 s each), then 32-39 s (overlap 39-40 excluded)
    u0 = [u for u in units if u.track == 0]
    assert [round(u.clean_s, 2) for u in u0] == [10.0, 10.0, 5.0, 7.0]
    assert all(u.t1 - u.t0 <= p.unit_s + 1e-6 for u in units)
    u1 = [u for u in units if u.track == 1]
    assert round(sum(u.clean_s for u in u1), 2) == 12.0          # 7 s + 5 s (39-40 is overlap)


def test_sparse_speech_pools_until_min_embed():
    pat = [(0, t, t + 0.4) for t in np.arange(0, 20, 4.0)]        # 0.4 s blips every 4 s
    trs = T.tracks_from_segmentation(slot_seg(pat), T.TrackParams())
    units = T.build_units(trs, T.TrackParams(min_embed_s=1.0, unit_s=10.0, pool_s=30.0))
    assert units[0].clean_s >= 1.0 and len(units[0].spans) >= 3


def test_link_chunks_follows_permuted_local_speakers():
    step, fs, n_fr = 2.5, 0.016875, 589
    dur = 40.0
    starts = np.arange(0, dur - 10 + 1e-9, step)
    speaker_at = lambda t: 0 if (t // 5) % 2 == 0 else 1          # noqa: E731  A/B alternate every 5 s
    acts = np.zeros((len(starts), n_fr, 3), np.float32)
    for c, s in enumerate(starts):
        perm = [1, 0, 2] if c % 2 else [0, 1, 2]                 # local order changes between chunks
        t = s + np.arange(n_fr) * fs + 0.03
        for f, tt in enumerate(t):
            acts[c, f, perm[speaker_at(tt)]] = 0.95
    trs = T.link_chunks(acts, starts, fs, 0.0619, dur)
    assert len(trs.tracks) == 2
    A = trs.active(0.5)
    t = trs.time(np.arange(trs.n_frames)) + fs / 2
    truth = np.array([speaker_at(x) for x in t])
    inside = (t > 0.5) & (t < dur - 0.5)
    agree = max((A[inside, 0] == (truth[inside] == 0)).mean(), (A[inside, 0] == (truth[inside] == 1)).mean())
    assert agree > 0.97


def test_hmm_smooths_a_single_outlier_and_fills_gaps():
    E = np.zeros((7, 3))
    E[[0, 1, 2, 4, 5, 6], 0] = 3.0
    E[3, 1] = 2.0                                               # one confusable unit
    post = T.hmm_posteriors(E, switch=6.0)
    assert (post.argmax(1) == 0).all()
    E2 = np.zeros((3, 3))
    E2[0, 1] = E2[2, 1] = 4.0                                   # middle unit has no observation
    assert T.hmm_posteriors(E2, 6.0).argmax(1).tolist() == [1, 1, 1]
    assert np.allclose(T.hmm_posteriors(E, 6.0).sum(1), 1.0, atol=1e-5)


def _world(pattern, voices, roster, rng):
    """Tracks + units + unit embeddings for slot `k` speaking with voice voices[k]."""
    p = T.TrackParams(pad_s=0.0, second_pass=False, guest_min_s=3.0)
    trs = T.tracks_from_segmentation(slot_seg(pattern, n_slots=len(voices)), p)
    units = T.build_units(trs, p)
    rows = [i for i, u in enumerate(units) if u.clean_s >= p.min_embed_s]
    X = (np.concatenate([voice(voices[units[i].track], rng=rng) for i in rows]) if rows
         else np.zeros((0, D), np.float32))
    T.attach_embeddings(units, np.array(rows), p)
    sc = T.Scorer([f"P{i}" for i in roster], BASE[roster], np.zeros((0, D), np.float32))
    cal = T.Calibration(a=20.0, b=0.0, c=0.0, k=-10.0)            # cosine 0.5 ↔ LLR 0
    return p, trs, units, X, sc, cal


def test_decode_names_enrolled_and_abstains_on_unenrolled():
    rng = np.random.default_rng(1)
    pat = [(0, 0, 30), (1, 30, 60), (2, 60, 90), (0, 90, 100)]
    p, trs, units, X, sc, cal = _world(pat, [0, 1, 4], [0, 1, 2], rng)
    fl, info = T.decode_all(trs, units, X, sc, cal, p)
    tl = T.build_timeline(trs, fl, {"x": 1})
    lab = {(round(t.start), round(t.end)): t.speaker for t in tl.turns}
    assert lab[(0, 30)] == "P0" and lab[(30, 60)] == "P1" and lab[(90, 100)] == "P0"
    assert lab[(60, 90)].startswith(GUEST_PREFIX)                 # voice 4 is not enrolled
    assert info["embedded_units"] == len(X)
    ex = tl.exclusive
    assert all(a.end <= b.start + 1e-9 for a, b in zip(ex, ex[1:]))


def test_conflicting_simultaneous_tracks_get_different_names():
    rng = np.random.default_rng(2)
    p = T.TrackParams(pad_s=0.0)
    trs = T.tracks_from_segmentation(slot_seg([(0, 0, 20), (1, 0, 20)], n_slots=2), p)
    units = [T.Unit(k, a, a + 10, [(a, a + 10)], 10.0) for k in (0, 1) for a in (0.0, 10.0)]
    X = np.concatenate([voice(0, rng=rng) for _ in units])       # both tracks sound like P0
    T.attach_embeddings(units, np.arange(len(units)), p)
    sc = T.Scorer(["P0", "P1"], BASE[[0, 1]], np.zeros((0, D), np.float32))
    cal = T.Calibration(a=20.0, b=0.0, c=0.0, k=-10.0)
    dec = T.decode_units(units, trs, T.unit_emissions(units, sc.scores(X), cal, p.theta), sc.names, p)
    for a in (0.0, 10.0):
        at = [dec.label[i] for i, u in enumerate(units) if u.t0 == a]
        assert "P0" in at and len(set(at)) == 2


def test_uncertain_label_when_posterior_is_low():
    E = np.zeros((1, 3), np.float32)
    E[0, :2] = [2.0, 1.9]
    units = [T.Unit(0, 0, 5, [(0, 5)], 5.0, 0, 5.0)]
    trs = T.tracks_from_segmentation(slot_seg([(0, 0, 5)], n_slots=1), T.TrackParams())
    dec = T.decode_units(units, trs, E, ["A", "B"], T.TrackParams(p_uncertain=0.7, theta=-9))
    assert dec.label == ["UNCERTAIN(A|B)"]


def test_second_pass_relabels_a_confident_turn():
    rng = np.random.default_rng(3)
    p, trs, units, X, sc, cal = _world([(0, 0, 12)], [0], [0, 1], rng)
    p = T.TrackParams(pad_s=0.0, turn_min_s=3.0, turn_override_p=0.9)
    fl, _ = T.decode_all(trs, units, X, sc, cal, T.TrackParams(pad_s=0.0, second_pass=False))
    cands = T.long_turns(trs, fl, units, p)
    assert len(cands) == 1 and cands[0].label == "P0"
    S = sc.scores(voice(1, rng=rng))                           # the whole turn sounds like P1
    n = T.apply_second_pass(trs, fl, cands, S, np.array([12.0]), sc.names, cal, p)
    tl = T.build_timeline(trs, fl)
    assert n == 1 and {t.speaker for t in tl.turns} == {"P1"}


def test_calibration_fit_is_duration_aware():
    rng = np.random.default_rng(4)
    n = 4000
    d = rng.choice([0.5, 1.0, 2.0, 8.0], n)
    y = rng.random(n) < 0.3
    sep = 1.0 + np.log2(d / 0.5)                                # longer → better separated
    s = np.where(y, sep, 0.0) + rng.normal(size=n)
    cal, rep = CT.fit_calibration(s, d, y)
    assert rep["all"]["cllr"] < 0.9 and rep["all"]["cllr"] >= rep["all"]["min_cllr"] - 1e-6
    assert cal.b > 0                                             # slope grows with duration
    hi, lo = cal.llr(np.array([6.0]), 8.0)[0], cal.llr(np.array([6.0]), 0.5)[0]
    assert hi > lo
    base = T.Calibration(1.0, 0.0, 0.0, 0.0)
    re = CT.refit_scale(base, s, d, y)
    assert re.b == 0.0 and re.a > 0


def test_cllr_and_eer_sanity():
    y = np.array([True] * 50 + [False] * 50)
    s = np.concatenate([np.full(50, 5.0), np.full(50, -5.0)])
    assert CT.eer(s, y) == 0.0 and CT.cllr(s, y) < 0.01
    assert abs(CT.cllr(np.zeros(100), y) - 1.0) < 1e-9


def test_track_params_roundtrip():
    p = T.TrackParams(theta=1.5)
    assert T.TrackParams(**p.to_dict()) == p
    c = T.Calibration(1, 2, 3, 4, "x")
    assert T.Calibration.from_dict(c.to_dict()) == c
    with pytest.raises(KeyError):
        T.Calibration.from_dict({"a": 1})


def test_label_frames_fill_and_unknown_far_from_units():
    trs = T.tracks_from_segmentation(slot_seg([(0, 0, 3), (0, 40, 41)], n_slots=1), T.TrackParams())
    units = [T.Unit(0, 0.0, 3.0, [(0.0, 3.0)], 3.0, 0, 3.0)]
    fl = T.label_frames(trs, units, ["A"], np.array([0.9], np.float32), T.TrackParams(fill_s=10.0))
    turns = [t for _, t in T.frames_to_turns(trs, fl)]
    assert [(t.speaker, round(t.start), round(t.end)) for t in turns] == [("A", 0, 3), (UNKNOWN, 40, 41)]


def test_stranger_resembling_an_enrolled_track_is_demoted():
    """Track 1 (a stranger) looks weakly like P0 on every unit; track 0 is P0 itself. With
    track-level identities P0 belongs to track 0, so track 1 abstains instead of taking P0."""
    units = [T.Unit(0, a, a + 5, [(a, a + 5)], 5.0, i, 5.0) for i, a in enumerate((0.0, 20.0, 40.0))]
    units += [T.Unit(1, a, a + 5, [(a, a + 5)], 5.0, 3 + i, 5.0) for i, a in enumerate((10.0, 30.0, 50.0))]
    E = np.zeros((6, 3), np.float32)
    E[:3, 0], E[3:, 0] = 5.0, 1.0                                 # names P0, P1; UNKNOWN θ = 0
    E[:, 1] = -5.0
    trs = T.tracks_from_segmentation(slot_seg([(0, 0, 5), (1, 10, 15)], n_slots=2), T.TrackParams())
    assert T.track_identities(units, E, 2) == {0: 0, 1: 2}
    on = T.decode_units(units, trs, E, ["P0", "P1"], T.TrackParams(foreign_p=2.0))
    off = T.decode_units(units, trs, E, ["P0", "P1"], T.TrackParams(foreign_p=1.0))
    assert on.label == ["P0"] * 3 + [UNKNOWN] * 3
    assert off.label == ["P0"] * 6
