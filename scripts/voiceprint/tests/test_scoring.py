"""Scoring tests on synthetic embeddings: random unit 'voices' + per-clip noise."""

import numpy as np
import pytest

from voiceprint import scoring as sc

DIM = 64
# The priors, but marked calibrated (as if `calibrate` had produced them): lets "high" through.
CAL = {k: {f: getattr(t, f) for f in ("tau", "tau_low", "delta", "gap")}
       for k, t in sc.DEFAULT_THRESHOLDS.items()}


def voices(n, seed=0):
    rng = np.random.default_rng(seed)
    return sc.l2norm(rng.normal(size=(n, DIM)))


def clips(center, n, noise=0.35, seed=1):
    rng = np.random.default_rng(seed)
    return sc.l2norm(center[None, :] + noise * rng.normal(size=(n, DIM)) / np.sqrt(DIM) * 4)


def enroll_people(V, names, n=30, seed=10):
    items = []
    for k, (name, v) in enumerate(zip(names, V)):
        for e in clips(v, n, seed=seed + k):
            items.append((name, e, "zoom", 5.0, f"m{k}"))
    return sc.build_enrollment(items)


def cluster(name, center, n=10, seed=99, dur=5.0):
    E = clips(center, n, seed=seed)
    return sc.cluster_embedding(name, E, [dur] * n)


def cohort_for(enr, n_extra=200, seed=500):
    extra = voices(40, seed=seed)
    vecs = np.concatenate([clips(v, n_extra // 40, seed=seed + i) for i, v in enumerate(extra)])
    return sc.build_cohort(enr, vecs, [f"x{i // (n_extra // 40)}" for i in range(len(vecs))])


# ─── basics ─────────────────────────────────────────────────────────────────

def test_mad_prune_drops_outliers():
    V = voices(3)
    E = np.concatenate([clips(V[0], 30, noise=0.2), V[1:3]])   # 2 wrong-voice clips
    keep = sc.mad_prune(E, k=3.0)
    assert keep[:30].sum() >= 27
    assert not keep[30] and not keep[31]


def test_mad_prune_small_n_noop():
    assert sc.mad_prune(voices(3), min_n=5).all()


def test_centroid_is_renormed_mean():
    V = voices(4)
    enr = sc.build_enrollment([("p", v) for v in V], prune_min_n=100)
    exp = V.mean(0) / np.linalg.norm(V.mean(0))
    assert np.allclose(enr["p"].centroid, exp, atol=1e-5)
    assert abs(np.linalg.norm(enr["p"].centroid) - 1) < 1e-5


def test_channel_subcentroid_and_fallback():
    V = voices(2)
    items = [("p", e, "phone") for e in clips(V[0], 5, seed=3)] + \
            [("p", e, "zoom") for e in clips(V[1], 2, seed=4)]
    pm = sc.build_enrollment(items, min_sub=3, prune_min_n=100)["p"]
    assert "phone" in pm.sub and "zoom" not in pm.sub             # zoom has < min_sub
    assert pm.reference("zoom") is pm.centroid                    # fallback
    assert pm.reference("phone") @ V[0] > pm.centroid @ V[0]


def test_cluster_embedding_duration_weighted_and_insufficient():
    V = voices(2)
    E = np.stack([V[0], V[1]])
    cv = sc.cluster_embedding("A", E, [9.0, 1.0], prune_min_n=100)
    assert cv.vec @ V[0] > cv.vec @ V[1]
    assert not cv.insufficient
    short = sc.cluster_embedding("B", E, [3.0, 3.0], min_net_s=8.0, prune_min_n=100)
    assert short.insufficient
    empty = sc.cluster_embedding("C", np.zeros((0, DIM)), [])
    assert empty.vec is None and empty.insufficient


def test_top5_mode():
    V = voices(1)
    pm = sc.build_enrollment([("p", e) for e in clips(V[0], 10)], prune_min_n=100)["p"]
    s = sc.raw_score(V[0], pm, mode="top5")
    top = np.sort(pm.exemplars @ V[0])[::-1][:5].mean()
    assert s == pytest.approx(top, abs=1e-6)


# ─── AS-norm ────────────────────────────────────────────────────────────────

def test_as_norm_math():
    rng = np.random.default_rng(7)
    cohort = sc.l2norm(rng.normal(size=(300, DIM)))
    e, t = voices(2, seed=8)
    s = 0.42
    se, st = np.sort(cohort @ e)[-50:], np.sort(cohort @ t)[-50:]
    exp = 0.5 * ((s - se.mean()) / se.std() + (s - st.mean()) / st.std())
    assert sc.as_norm(s, e, t, cohort, top_n=50) == pytest.approx(exp, rel=1e-5)


def test_asnorm_excludes_candidate_from_cohort_and_falls_back():
    V = voices(3)
    names = ["P0", "P1", "P2"]
    enr = enroll_people(V, names)
    cvs = [cluster("A", V[0])]
    sm = sc.score_matrix(cvs, enr, cohort=cohort_for(enr), min_cohort=50)
    assert sm.kind == "asnorm" and sm.norm is not None
    # target is strongly positive, non-targets not
    j = sm.persons.index("P0")
    assert sm.norm[0, j] > 3 and all(sm.norm[0, k] < sm.norm[0, j] for k in range(3) if k != j)
    tiny = sc.score_matrix(cvs, enr, cohort=cohort_for(enr), min_cohort=10_000)
    assert tiny.kind == "cosine" and tiny.norm is None
    assert any("AS-norm skipped" in w for w in tiny.warnings)


# ─── assignment ─────────────────────────────────────────────────────────────

def test_hungarian_exclusivity():
    V = voices(2)
    enr = enroll_people(V, ["X", "Y"])
    # both clusters sound like X, the second slightly less so
    a = cluster("A", V[0], seed=1)
    b = cluster("B", sc.l2norm(0.8 * V[0] + 0.6 * V[1]), seed=2)
    res = sc.identify([a, b], enr, cohort=cohort_for(enr), use_asnorm=False,
                      thresholds=sc.Thresholds(0.3, 0.2, 0.0, 0.0, "cosine"))
    names = [d["proposed"] for d in res["decisions"] if d["proposed"]]
    assert len(names) == len(set(names))                       # never the same person twice
    assert res["decisions"][0]["proposed"] == "X"


def test_unknown_when_below_tau():
    V = voices(4)
    enr = enroll_people(V[:3], ["P0", "P1", "P2"])
    stranger = cluster("C", V[3])
    res = sc.identify([stranger], enr, cohort=cohort_for(enr))
    d = res["decisions"][0]
    assert d["proposed"] is None and d["confidence"] == "unknown"
    assert not res["calibrated"]
    assert any("UNCALIBRATED" in w for w in res["warnings"])


def test_two_speaker_correct_and_label_inversion_detected():
    V = voices(4)
    enr = enroll_people(V, ["Alice Example", "Bob Example", "P2", "P3"])
    a, b = cluster("A", V[1], seed=5), cluster("B", V[0], seed=6)   # A=Bob, B=Alice
    roster = ["Alice Example", "Bob Example"]
    ok = sc.identify([a, b], enr, roster=roster, cohort=cohort_for(enr), calibration=CAL,
                     labels={"A": "Bob Example", "B": "Alice Example"})
    assert [d["proposed"] for d in ok["decisions"]] == ["Bob Example", "Alice Example"]
    assert all(d["confidence"] == "high" for d in ok["decisions"])
    assert ok["two_speaker_margin"] > ok["thresholds"]["delta"]
    assert ok["label_check"]["status"] == "agree"

    inv = sc.identify([a, b], enr, roster=roster, cohort=cohort_for(enr), calibration=CAL,
                      labels={"A": "Alice Example", "B": "Bob Example"})   # inverted mapping
    assert inv["label_check"]["status"] == "inversion"
    assert inv["label_check"]["pairs"][0]["label_margin"] < 0


def test_two_speaker_margin_below_delta_gives_unknown():
    V = voices(2)
    enr = enroll_people(V, ["X", "Y"])
    mid = sc.l2norm(V[0] + V[1])
    a, b = cluster("A", mid, seed=3, n=10), cluster("B", mid, seed=4, n=10)
    th = sc.Thresholds(tau=0.0, tau_low=-1.0, delta=0.2, gap=0.0, kind="cosine")
    res = sc.identify([a, b], enr, thresholds=th, use_asnorm=False)
    for d in res["decisions"]:
        assert d["proposed"] is None and d["tentative"] in ("X", "Y")
        assert any("swap margin" in r for r in d["reasons"])


def test_closed_roster_uses_tau_low_and_elimination():
    V = voices(3)
    enr = enroll_people(V[:2], ["X", "Y"])
    good = cluster("A", V[0], seed=11)
    short = sc.cluster_embedding("B", clips(V[1], 1, seed=12), [3.0])      # insufficient
    res = sc.identify([good, short], enr, roster=["X", "Y"], cohort=cohort_for(enr), calibration=CAL)
    dA, dB = res["decisions"]
    assert dA["proposed"] == "X" and dA["confidence"] == "high"
    assert dB["insufficient"] and dB["proposed"] == "Y" and dB["confidence"] == "low"
    assert res["elimination"] == {"cluster": "B", "person": "Y"}


def test_roster_member_not_enrolled_reported():
    V = voices(2)
    enr = enroll_people(V[:1], ["X"])
    res = sc.identify([cluster("A", V[0])], enr, roster=["X", "Newcomer"], use_asnorm=False)
    assert res["roster_not_enrolled"] == ["Newcomer"]
    assert res["persons"] == ["X"]


def test_calibration_thresholds_loaded():
    entry = {"asnorm": {"tau": 5.5, "tau_low": 3.0, "delta": 6.0, "gap": 2.0}, "_source": "x.json"}
    th = sc.load_thresholds(entry, "asnorm")
    assert th.calibrated and th.tau == 5.5 and th.source == "x.json"
    th2 = sc.load_thresholds(entry, "cosine")          # kind missing → defaults, uncalibrated
    assert not th2.calibrated and th2.tau == sc.DEFAULT_THRESHOLDS["cosine"].tau


def test_label_check_agree_by_margin_when_clusters_stay_unknown():
    """B3: both clusters below open-set τ (→ Unknown) but labels clearly supported by margin."""
    V = voices(2)
    enr = enroll_people(V, ["X", "Y"])
    a, b = cluster("A", V[0], seed=21), cluster("B", V[1], seed=22)
    th = sc.Thresholds(tau=5.0, tau_low=5.0, delta=0.3, gap=0.0, kind="cosine")   # τ unreachable
    res = sc.identify([a, b], enr, thresholds=th, use_asnorm=False, labels={"A": "X", "B": "Y"})
    assert all(d["proposed"] is None for d in res["decisions"])
    lc = res["label_check"]
    assert lc["status"] == "agree_by_margin" and lc["supported_pairs"] == [["A", "B"]]
    assert lc["pairs"][0]["label_margin"] >= th.delta
    swapped = sc.identify([a, b], enr, thresholds=th, use_asnorm=False, labels={"A": "Y", "B": "X"})
    assert swapped["label_check"]["status"] == "inversion"
    weak = sc.Thresholds(tau=5.0, tau_low=5.0, delta=10.0, gap=0.0, kind="cosine")  # margin < δ
    amb = sc.identify([a, b], enr, thresholds=weak, use_asnorm=False, labels={"A": "X", "B": "Y"})
    assert amb["label_check"]["status"] == "unverifiable" and amb["label_check"]["ambiguous_pairs"]


# ─── roster elimination and label-check guards ──────────────────────────────

def test_elimination_skips_split_duplicate_of_named_speaker():
    """Roster has an absent, non-enrolled attendee; the diarizer split Bob into B and F.
    F must NOT be named after the absent attendee by elimination."""
    V = voices(4)
    names = ["Alice Example", "Bob Example", "Carol Example"]
    enr = enroll_people(V[:3], names)
    cl = [cluster("A", V[0], seed=31), cluster("B", V[1], seed=32), cluster("C", V[2], seed=33),
          cluster("F", V[1], seed=34)]                                     # F = second Bob cluster
    roster = names + ["Dave Example"]                                     # Dave: absent, not enrolled
    res = sc.identify(cl, enr, roster=roster, cohort=cohort_for(enr), calibration=CAL)
    props = {d["cluster"]: d["proposed"] for d in res["decisions"]}
    assert props["A"] == "Alice Example" and props["C"] == "Carol Example"
    assert {props["B"], props["F"]} == {"Bob Example", None}
    assert "Dave Example" not in props.values() and res["elimination"] is None
    sk = res["elimination_skipped"]
    assert sk["person"] == "Dave Example" and sk["looks_like"] == "Bob Example"
    dup = next(d for d in res["decisions"] if d["proposed"] is None)
    assert any("split duplicate" in r for r in dup["reasons"])


def test_elimination_still_names_a_genuinely_unknown_voice():
    V = voices(4)
    enr = enroll_people(V[:2], ["X", "Y"])
    cl = [cluster("A", V[0], seed=41), cluster("B", V[1], seed=42), cluster("C", V[3], seed=43)]
    res = sc.identify(cl, enr, roster=["X", "Y", "Newcomer"], cohort=cohort_for(enr), calibration=CAL)
    dC = res["decisions"][2]
    assert dC["proposed"] == "Newcomer" and dC["by_elimination"] and dC["confidence"] == "low"


def test_label_check_ignores_elimination_names():
    """An elimination name is not voice evidence → no 'agree' from it."""
    V = voices(4)
    enr = enroll_people(V[:2], ["X", "Y"])
    cl = [cluster("A", V[0], seed=51), cluster("B", V[1], seed=52), cluster("C", V[3], seed=53)]
    res = sc.identify(cl, enr, roster=["X", "Y", "Newcomer"], cohort=cohort_for(enr), calibration=CAL,
                      labels={"C": "Newcomer"})
    dC = res["decisions"][2]
    assert dC["by_elimination"] and dC["label_agrees"] is None
    assert res["label_check"]["status"] != "agree" and "C" not in res["label_check"]["agree"]


def test_uncalibrated_caps_confidence_and_disables_elimination():
    """No calibration.json: the same evidence never yields "high", so no roster elimination."""
    V = voices(3)
    enr = enroll_people(V[:2], ["X", "Y"])
    good = cluster("A", V[0], seed=11)
    short = sc.cluster_embedding("B", clips(V[1], 1, seed=12), [3.0])      # insufficient
    cal = sc.identify([good, short], enr, roster=["X", "Y"], cohort=cohort_for(enr), calibration=CAL)
    raw = sc.identify([good, short], enr, roster=["X", "Y"], cohort=cohort_for(enr))
    assert cal["decisions"][0]["confidence"] == "high" and cal["elimination"]
    dA, dB = raw["decisions"]
    assert dA["proposed"] == "X" and dA["confidence"] == "low" and not raw["calibrated"]
    assert any("uncalibrated" in r for r in dA["reasons"])
    assert dB["proposed"] is None and raw["elimination"] is None
    assert any("UNCALIBRATED" in w for w in raw["warnings"])
