"""bench.protocol and the bench.run CLI on a synthetic two-series dataset (no audio)."""
import json

import pytest

from bench import protocol as P
from bench.datasets.ami import assign_series, series_of
from bench.datasets.aishell4 import group_of
from bench.rttm import Seg, format_rttm

SERIES = {"XX1000": ["A", "B", "C"], "XX2000": ["D", "E"]}


@pytest.fixture
def bench_env(tmp_path, monkeypatch):
    """A fake 'ami' manifest: two series × 3 meetings, C absent from XX1000c."""
    monkeypatch.setenv("SPEAKER_BENCH_DIR", str(tmp_path))
    root = tmp_path / "ami"
    meetings = {}
    for series, spk in SERIES.items():
        for k, letter in enumerate("abc"):
            mid = f"{series}{letter}"
            present = [s for s in spk if not (mid == "XX1000c" and s == "C")]
            segs, t = [], 0.0
            for i, s in enumerate(present * 3):
                d = 0.6 + (i % 4) * 2.5                        # 0.6, 3.1, 5.6, 8.1 s turns
                segs.append(Seg(t, t + d, s))
                t += d - (0.3 if i % 5 == 4 else 0.0)          # occasional overlap
            (root / "rttm").mkdir(parents=True, exist_ok=True)
            (root / "rttm" / f"{mid}.rttm").write_text(format_rttm(mid, segs))
            (root / "rttm" / f"{mid}.uem").write_text(f"{mid} 1 0.000 {t + 1:.3f}\n")
            meetings[mid] = {"split": "test", "series": series, "rttm": f"rttm/{mid}.rttm",
                             "uem": f"rttm/{mid}.uem", "duration_s": t + 1, "speakers": sorted(present),
                             "audio": {"sdm": {"path": f"audio/{mid}.sdm.wav"},
                                       "headset": {"path": f"audio/{mid}.hs.wav"}}}
    (tmp_path / "manifest-ami.json").write_text(json.dumps({"root": "ami", "meetings": meetings}))
    return tmp_path


def test_series_and_group_names():
    assert series_of("ES2004c") == "ES2004" and series_of("IB4001") == "IB4001"
    assert group_of("L_R003S02C02") == "L_R003"
    spk = {"ES2004a": ["A", "B"], "IB4001": ["C", "D"], "IB4002": ["D", "C"], "IB4003": ["C", "E"]}
    assert assign_series(spk) == {"ES2004a": "ES2004", "IB4001": "IB4001+IB4002",
                                  "IB4002": "IB4001+IB4002", "IB4003": "IB4003"}
    with pytest.raises(ValueError):
        group_of("nope")


def test_solo_regions_remove_other_speakers():
    ref = [Seg(0, 10, "A"), Seg(2, 3, "B"), Seg(8, 12, "B")]
    assert P.solo_regions(ref, "A") == [(0, 2), (3, 8)]
    assert P.solo_regions(ref, "A", guard=0.5, min_dur=2.0) == [(3.5, 7.5)]


def test_enrollment_is_cross_session(bench_env):
    pool = P.load_meetings("ami", "test")
    m = next(x for x in pool if x.id == "XX1000a")
    enr = P.enrollment(m, pool)
    assert sorted(enr) == ["A", "B", "C"]
    assert all(src.meeting != "XX1000a" and src.meeting.startswith("XX1000")
               for srcs in enr.values() for src in srcs)
    assert {src.meeting for src in enr["C"]} == {"XX1000b"}            # absent from XX1000c
    assert all(src.audio.name.endswith(".hs.wav") for src in enr["A"])  # headset for enrollment
    assert m.test_audio.name.endswith(".sdm.wav")
    ro = P.rosters(m, list(enr))
    assert ro["closed"] == {"A", "B", "C"} and ro["loo:B"] == {"A", "C"}
    coh = P.cohort_speakers(m, pool)
    assert sorted(coh) == ["XX2000/D", "XX2000/E"]


def test_absent_enrolled_speaker_in_roster(bench_env):
    pool = P.load_meetings("ami", "test")
    m = next(x for x in pool if x.id == "XX1000c")
    ro = P.rosters(m, list(P.enrollment(m, pool)))
    assert "C" in ro["closed"] and "loo:C" not in ro


def _run(args):
    from bench import run
    return run.main(args)


def test_cli_sanity_reference_and_shuffled(bench_env, tmp_path):
    pytest.importorskip("pyannote.metrics")
    for kind in ("reference", "shuffled"):
        out = tmp_path / f"out-{kind}"
        assert _run(["--dataset", "ami", "--split", "test", "--sanity", kind, "--out", str(out)]) == 0
        rep = json.loads((out / "report.json").read_text())
        for ck in ("collar_0", "collar_0.25"):
            ier = rep["pooled"][ck]["ier"]
            assert (ier == 0) if kind == "reference" else (ier > 0.5), (kind, ck, ier)
        assert (out / "report.md").read_text().startswith("# Speaker-tracking benchmark")


def test_cli_scores_system_dir_with_loo(bench_env, tmp_path):
    pytest.importorskip("pyannote.metrics")
    pool = P.load_meetings("ami", "test")
    sysdir = tmp_path / "sys"
    for m in pool:
        ref = m.reference()
        (sysdir / "closed").mkdir(parents=True, exist_ok=True)
        (sysdir / "closed" / f"{m.id}.rttm").write_text(format_rttm(m.id, ref))
        for spk in m.speakers:                         # held-out speaker: half guest, half wrong name
            d = sysdir / "loo" / m.id
            d.mkdir(parents=True, exist_ok=True)
            other = next(s for s in m.speakers if s != spk)
            hyp = []
            for i, s in enumerate(x for x in ref if x.speaker == spk):
                hyp.append(Seg(s.start, s.end, "GUEST_1" if i % 2 else other))
            hyp += [s for s in ref if s.speaker != spk]
            (d / f"{spk}.rttm").write_text(format_rttm(m.id, hyp))
    out = tmp_path / "out"
    assert _run(["--dataset", "ami", "--system", str(sysdir), "--out", str(out), "--collars", "0"]) == 0
    rep = json.loads((out / "report.json").read_text())
    p = rep["pooled"]["collar_0"]
    assert p["ier"] == 0
    assert 0.2 < p["loo"]["false_name_rate"] < 0.8
    assert p["loo"]["variants"] == sum(len(m.speakers) for m in pool)
    assert "false-name" in (out / "report.md").read_text()


def test_cli_missing_system_file_fails(bench_env, tmp_path):
    pytest.importorskip("pyannote.metrics")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        _run(["--dataset", "ami", "--system", str(tmp_path / "empty"), "--out", str(tmp_path / "o")])


def test_enrollment_qc_drops_sources_and_meetings(bench_env):
    qc = {"drop_sources": [["XX1000b", "A"]], "drop_meetings": ["XX2000c"]}
    qc["_drop_sources"] = {tuple(x) for x in qc["drop_sources"]}
    pool = P.load_meetings("ami", "test", qc=qc)
    assert "XX2000c" not in {m.id for m in pool} and len(pool) == 5
    m = next(x for x in pool if x.id == "XX1000a")
    assert [s.meeting for s in P.enrollment(m, pool, qc=qc)["A"]] == ["XX1000c"]
    assert [s.meeting for s in P.enrollment(m, pool)["A"]] == ["XX1000b", "XX1000c"]
    d = next(x for x in pool if x.id == "XX2000a")
    assert "XX1000/A" in P.cohort_speakers(d, pool) and P.source_ok(qc, "XX1000a", "A")
    assert not P.source_ok(qc, "XX1000b", "A") and P.source_ok(None, "XX1000b", "A")


def test_committed_ami_qc_loads():
    qc = P.load_qc("ami")
    assert qc is not None and {"IB4001", "IB4002"} <= set(qc["drop_meetings"])
    assert all(len(x) == 2 for x in qc["drop_sources"]) and qc["_drop_sources"]
    assert P.load_qc("nope") is None
