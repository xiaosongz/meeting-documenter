"""B0 decoding pieces (numpy only; no models, no audio)."""
import numpy as np

from bench.baselines import window_id as B


def test_viterbi_penalty_smooths_blips():
    s = np.full((9, 2), -1.0)
    s[:, 0] = 3.0
    s[4] = [-1.0, 3.5]                                     # one-window blip to speaker 1
    t = np.arange(9) * 0.5
    assert list(B.viterbi(s, theta=0.0, penalty=0.0, times=t, reset_gap_s=2)) == [0] * 4 + [1] + [0] * 4
    assert list(B.viterbi(s, theta=0.0, penalty=3.0, times=t, reset_gap_s=2)) == [0] * 9


def test_viterbi_unknown_state_and_gap_reset():
    s = np.array([[3.0, 0.0], [3.0, 0.0], [0.2, 0.1], [0.2, 0.1], [0.0, 3.0]])
    path = B.viterbi(s, theta=1.0, penalty=0.5, times=np.arange(5) * 0.5, reset_gap_s=2)
    assert list(path) == [0, 0, 2, 2, 1]                   # index 2 = UNKNOWN
    s = np.array([[3.0, 0.0], [0.0, 1.0]])
    assert list(B.viterbi(s, 0.0, 5.0, np.array([0.0, 0.5]), 2.0)) == [0, 0]
    assert list(B.viterbi(s, 0.0, 5.0, np.array([0.0, 9.0]), 2.0)) == [0, 1]   # long gap: free switch


def test_to_segments_labels_vad_frames_by_nearest_window():
    hop = 512
    vad = np.zeros(200, bool)
    vad[10:60] = True                                       # 0.32–1.92 s
    vad[150:160] = True                                     # 4.8–5.12 s, far from any window
    test = {"vad": vad, "vad_hop": np.array(hop), "starts": np.array([0.0, 0.5])}
    segs = B.to_segments(test, ["A", "B"], np.array([0, 1]), B.WinParams(), fill_s=1.0)
    assert [s.speaker for s in segs] == ["A", "B"]
    assert segs[0].start == 10 * hop / 16000 and segs[-1].end == 60 * hop / 16000
    assert abs(segs[0].end - segs[1].start) < 1e-9


def test_score_windows_asnorm_ranks_target_first():
    rng = np.random.default_rng(0)
    d = 32
    spk = B._l2(rng.standard_normal((3, d)))
    cohort = B._l2(rng.standard_normal((50, d)))
    test = B._l2(spk[[0, 1, 2, 1]] + 0.3 * rng.standard_normal((4, d)))
    names, S = B.score_windows(test, {"a": spk[0], "b": spk[1], "c": spk[2]}, cohort, top_n=20)
    assert names == ["a", "b", "c"] and list(S.argmax(1)) == [0, 1, 2, 1]
    _, raw = B.score_windows(test, {"a": spk[0], "b": spk[1], "c": spk[2]}, cohort[:5], top_n=20)
    assert np.allclose(raw, test @ spk.T)                  # tiny cohort → raw cosine


def test_embed_batches_bounded_and_order_preserved():
    lengths = [48000, 24000, 96000, 24000, 48000, 24000]
    batches = B._batches(lengths, batch=2, max_samples=100000)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(len(b) <= 2 and sum(lengths[i] for i in b) <= 100000 or len(b) == 1 for b in batches)

    class Fake:
        dim = 1

        def embed_batch(self, waves):
            return np.array([[len(w)] for w in waves], np.float32)

    waves = [np.zeros(n, np.float32) for n in lengths]
    assert list(B._embed(Fake(), waves, batch=2)[:, 0]) == lengths
    assert B._embed(Fake(), [], batch=2).shape == (0, 1)


def test_cohort_drops_speakers_matching_an_enrolled_model(tmp_path):
    from bench.protocol import Meeting

    rng = np.random.default_rng(1)
    a, b, other = B._l2(rng.standard_normal((3, 32)))
    mk = lambda mid, series, spk: Meeting(mid, "x", "test", series, spk, 1.0, tmp_path, {})
    pool = [mk("S1a", "S1", ("A",)), mk("S2a", "S2", ("B", "C"))]
    np.savez(tmp_path / "S2a.enroll.npz", speaker=np.array(["B", "B", "C", "C"]),
             emb=np.stack([a, a, other, other]).astype(np.float32))   # B is A under another label
    assert len(B.cohort_vectors(pool[0], pool, tmp_path)) == 4
    kept = B.cohort_vectors(pool[0], pool, tmp_path, {"A": a})
    assert len(kept) == 2 and np.allclose(kept, other)
