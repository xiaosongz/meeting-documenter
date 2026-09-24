"""`calibrate`: leave-one-meeting-out thresholds from a synthetic store (random unit 'voices')."""

import json
import stat

import numpy as np
import pytest

import identify_speakers as cli
from voiceprint import audio as au
from voiceprint import calibrate as cb
from voiceprint import scoring as sc
from voiceprint import store as st

DIM = 64
PEOPLE = ["Alice Example", "Bob Example", "Carol Example", "Dave Example", "Eve Example", "Frank Example"]
MODEL = "fake-model@test"


def unit_vectors(n, seed):
    return sc.l2norm(np.random.default_rng(seed).normal(size=(n, DIM)))


def clip_rows(center, n, seed, noise=0.35):
    rng = np.random.default_rng(seed)
    return sc.l2norm(center[None, :] + noise * rng.normal(size=(n, DIM)) / np.sqrt(DIM) * 4)


def default_sig(device="cpu"):
    return cli.clip_signature(cli.clip_summary(au.ENROLL_PARAMS, device, {}))


def fill_store(root, meetings=10, per_meeting=3, clips=20, sig=None, strangers=True):
    """Each meeting: `per_meeting` of the six people (rotating), 20 clips × 3 s each."""
    V = unit_vectors(len(PEOPLE) + 12, seed=1)
    s = st.VoiceprintStore.open(root)
    sig = sig if sig is not None else default_sig()
    for m in range(meetings):
        mk = f"2026-01-{m + 1:02d}-1000"
        for k in range(per_meeting):
            pi = (m + k) % len(PEOPLE)
            E = clip_rows(V[pi], clips, seed=100 * m + k)
            s.add_exemplars(MODEL, [st.Exemplar(PEOPLE[pi], e, "zoom", meeting_key=mk, cluster="ABC"[k],
                                                net_dur=3.0, clip_sig=sig) for e in E])
        if strangers:
            E = clip_rows(V[len(PEOPLE) + m], 10, seed=900 + m)
            s.add_cohort(MODEL, [st.CohortClip(f"{mk}:D", e, meeting_key=mk, net_dur=3.0, clip_sig=sig)
                                 for e in E])
    s.close()
    return V


def run(store, *argv):
    return cli.main(["--store", str(store), *argv])


# ─── pure threshold rules ───────────────────────────────────────────────────

def test_threshold_rules_and_minimums():
    rng = np.random.default_rng(0)
    pools = cb.Pools(stranger=list(rng.normal(0, 1, 1000)), margins=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                     gaps=list(np.linspace(-0.1, 1.0, 20)), target=[3.0] * 5)
    th, src = cb.thresholds_from(pools, "asnorm")
    assert th.calibrated and th.tau == pytest.approx(np.quantile(pools.stranger, 0.99))
    assert th.tau_low == pytest.approx(np.quantile(pools.stranger, 0.95)) and th.tau_low <= th.tau
    assert th.delta == pytest.approx(np.quantile(pools.margins, 0.05))
    assert th.gap == pytest.approx(max(0.0, np.quantile(pools.gaps, 0.05)))
    few = cb.Pools(stranger=list(rng.normal(size=60)), margins=[1.0], gaps=[0.5])
    th2, src2 = cb.thresholds_from(few, "cosine")
    assert th2.delta == sc.DEFAULT_THRESHOLDS["cosine"].delta and src2["delta"].startswith("prior")
    assert th2.gap == sc.DEFAULT_THRESHOLDS["cosine"].gap and src2["gap"].startswith("prior")
    none, why = cb.thresholds_from(cb.Pools(stranger=[0.1] * 49), "cosine")
    assert none is None and "49 < 50" in why["reason"]


def test_lomo_never_tests_on_its_own_meeting(tmp_path):
    fill_store(tmp_path / "vp")
    s = st.VoiceprintStore.open(tmp_path / "vp", create=False)
    folds = cb.score_folds(s.exemplars(MODEL), s.cohort_clips(MODEL))
    main = [f for f in folds if len(f.truths) > 1]
    assert len(main) == 10 and all(len(f.truths) == 4 for f in main)        # 3 people + 1 cohort voice
    assert all(f.loso[-1] for f in main)                                    # cohort voice = stranger
    loso = [f for f in folds if len(f.truths) == 1]
    assert len(loso) == 30 and all(f.loso == [True] for f in loso)
    for f in loso:                                                          # LOSO: person really removed
        assert f.truths[0] not in f.matrices["cosine"].persons
    pools = cb.fold_pools(main[0], "cosine")
    assert len(pools.target) == 3 and len(pools.margins) == 3 and len(pools.stranger) == 6


# ─── CLI ────────────────────────────────────────────────────────────────────

def test_calibrate_writes_thresholds_clip_config_and_identify_uses_them(tmp_path, capsys):
    store = tmp_path / "vp"
    V = fill_store(store)
    out = tmp_path / "c.json"
    assert run(store, "calibrate", "--json", str(out)) == 0
    text = capsys.readouterr().out
    assert "nested LOMO" in text and "wrote" in text
    cal_path = store / "calibration.json"
    assert stat.S_IMODE(cal_path.stat().st_mode) == 0o600
    entry = json.loads(cal_path.read_text())["models"][MODEL]
    for kind in ("cosine", "asnorm"):
        th = entry[kind]
        assert th["tau_low"] <= th["tau"] and th["delta"] >= 0 and th["gap"] >= 0
        nested = entry["diagnostics"][kind]["nested"]
        assert nested.get("stranger_wrong_high", 0) == 0 and nested.get("enrolled_wrong_high", 0) == 0
        assert entry["diagnostics"][kind]["counts"]["stranger"] >= cb.MIN_STRANGER
    assert entry["clip"]["hard_cap_s"] == au.DEVICE_CAP_S["cpu"]            # explicit, device-independent
    assert entry["clip_sig"] == default_sig() and entry["score_mode"] == "centroid"
    assert json.loads(out.read_text())["status"] == "ok"

    # the recorded clip block reproduces the enrollment signature on any device
    calib = st.VoiceprintStore.open(store, create=False).load_calibration(MODEL)
    for dev in ("cpu", "mps"):
        p, src = cli.resolve_clip_config(cli.argparse.Namespace(), au.ENROLL_PARAMS, calib, identify=False)
        assert cli.clip_signature(cli.clip_summary(p, dev, src)) == default_sig()
        assert cli.clip_config_mismatch(p, calib) == []

    # scoring with the calibrated entry: a known voice is named with high confidence
    s = st.VoiceprintStore.open(store, create=False)
    enr = sc.build_enrollment(s.exemplars(MODEL))
    cv = sc.cluster_embedding("A", clip_rows(V[0], 20, seed=4242), [3.0] * 20)
    res = sc.identify([cv], enr, calibration=calib)
    assert res["calibrated"] and res["decisions"][0]["proposed"] == PEOPLE[0]
    assert res["decisions"][0]["confidence"] == "high"


def test_calibrate_refuses_small_store_and_writes_nothing(tmp_path, capsys):
    store = tmp_path / "vp"
    fill_store(store, meetings=2, per_meeting=2, strangers=False)
    assert run(store, "calibrate") == 5
    assert "not enough data" in capsys.readouterr().err
    assert not (store / "calibration.json").exists()


def test_calibrate_empty_store(tmp_path):
    assert run(tmp_path / "none", "calibrate") == 5
    assert not (tmp_path / "none").exists()


def test_calibrate_refuses_mixed_or_unmatched_clip_configs(tmp_path, capsys):
    store = tmp_path / "vp"
    fill_store(store)
    s = st.VoiceprintStore.open(store)
    s.add_exemplars(MODEL, [st.Exemplar("Alice Example", np.ones(DIM, np.float32), meeting_key="2026-02-01-1000",
                                        net_dur=3.0, clip_sig="0123456789ab")])
    s.close()
    assert run(store, "calibrate") == 4
    assert "different clip configs" in capsys.readouterr().err
    assert not (store / "calibration.json").exists()
    assert run(store, "calibrate", "--force", "--dry-run") == 0
    assert not (store / "calibration.json").exists()

    other = tmp_path / "vp2"
    fill_store(other, sig="fedcba987654")                                   # enrolled with other flags
    assert run(other, "calibrate") == 4
    assert "do not reproduce" in capsys.readouterr().err


def test_calibrate_infers_mps_enrollment_cap(tmp_path):
    store = tmp_path / "vp"
    fill_store(store, sig=default_sig("mps"))
    assert run(store, "calibrate") == 0
    entry = json.loads((store / "calibration.json").read_text())["models"][MODEL]
    assert entry["clip"]["hard_cap_s"] == au.DEVICE_CAP_S["mps"] and entry["clip_sig"] == default_sig("mps")


def test_cosine_only_calibration_disables_asnorm_at_identify():
    ns = cli.argparse.Namespace(no_asnorm=False)
    assert cli.use_asnorm_for(ns, None)
    assert cli.use_asnorm_for(ns, {"asnorm": {"tau": 1}, "cosine": {"tau": 0.5}})
    assert not cli.use_asnorm_for(ns, {"cosine": {"tau": 0.5}})
    assert not cli.use_asnorm_for(cli.argparse.Namespace(no_asnorm=True), None)
