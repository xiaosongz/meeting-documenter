"""bench.metrics on tiny synthetic RTTMs (no audio, no downloads)."""
import pytest

pytest.importorskip("pyannote.metrics")

from bench import metrics as M
from bench.rttm import Seg, format_rttm, parse_rttm

REF = [Seg(0.0, 4.0, "A"), Seg(4.0, 4.6, "B"), Seg(5.0, 7.0, "A"), Seg(6.5, 8.0, "B"),
       Seg(9.0, 21.0, "C")]
UEM = [(0.0, 22.0)]
ROSTER = {"A", "B", "C"}


def test_reference_vs_reference_is_zero():
    for c in (0.0, 0.25):
        h = M.headline(REF, REF, UEM, ROSTER, c)
        assert h["ier"] == 0 and h["der"] == 0
    assert M.headline(REF, REF, UEM, ROSTER, 0.0)["jer"] == 0
    s = M.summarize(M.breakdown(REF, REF, UEM, ROSTER))
    assert s["enrolled"]["correct"] == 1.0


def test_shuffled_labels_score_worse():
    perm = {"A": "B", "B": "C", "C": "A"}
    hyp = [Seg(s.start, s.end, perm[s.speaker]) for s in REF]
    h = M.headline(REF, hyp, UEM, ROSTER, 0.0)
    assert h["ier"] > 0.9
    assert h["der"] == pytest.approx(0.0, abs=1e-9)      # DER permutes labels, IER does not
    assert M.summarize(M.breakdown(REF, hyp, UEM, ROSTER))["enrolled"]["wrong_name"] > 0.9


def test_frame_breakdown_matches_pyannote_ier_components():
    hyp = [Seg(0.0, 3.0, "A"), Seg(3.0, 4.6, "B"), Seg(5.0, 8.0, "A"), Seg(9.0, 15.0, "C"),
           Seg(15.0, 20.0, "GUEST_1")]
    h = M.headline(REF, hyp, UEM, ROSTER, 0.0)["ier_components"]
    bd = M.breakdown(REF, hyp, UEM, ROSTER)
    correct = bd.enrolled["correct"]
    assert correct == pytest.approx(h["correct"], abs=0.02)
    assert sum(bd.enrolled.values()) == pytest.approx(h["total"], abs=0.02)


def test_wrong_name_vs_abstain_vs_missed():
    ref = [Seg(0, 10, "A")]
    hyp = [Seg(0, 2, "A"), Seg(2, 4, "B"), Seg(4, 6, "UNCERTAIN(A|B)"), Seg(6, 8, "UNKNOWN")]
    s = M.summarize(M.breakdown(ref, hyp, [(0, 10)], {"A", "B"}))["enrolled"]
    assert s["correct"] == pytest.approx(0.2, abs=1e-3)
    assert s["wrong_name"] == pytest.approx(0.2, abs=1e-3)
    assert s["abstain"] == pytest.approx(0.4, abs=1e-3)
    assert s["missed"] == pytest.approx(0.2, abs=1e-3)
    # IER counts UNCERTAIN as an error and UNKNOWN on enrolled speech as confusion
    assert M.headline(ref, hyp, [(0, 10)], {"A", "B"}, 0.0)["ier"] == pytest.approx(0.8, abs=1e-3)


def test_open_set_false_name_rate():
    ref = [Seg(0, 5, "A"), Seg(5, 10, "X")]          # X is not enrolled
    hyp = [Seg(0, 5, "A"), Seg(5, 7, "GUEST_1"), Seg(7, 10, "A")]
    s = M.summarize(M.breakdown(ref, hyp, [(0, 10)], {"A"}))
    assert s["false_name_rate"] == pytest.approx(0.6, abs=1e-3)
    assert s["unenrolled"]["correct"] == pytest.approx(0.4, abs=1e-3)
    assert s["pair_confusion_s"][0] == {"ref": "X", "hyp": "A", "seconds": 3.0}
    # open-set IER: X→UNKNOWN in the reference and GUEST_1→UNKNOWN in the hypothesis
    assert M.headline(ref, hyp, [(0, 10)], {"A"}, 0.0)["ier"] == pytest.approx(0.3, abs=1e-3)


def test_duration_buckets_and_overlap():
    ref = [Seg(0, 0.5, "A"), Seg(1, 3, "B"), Seg(4, 9, "A"), Seg(10, 25, "B"), Seg(20, 22, "A")]
    s = M.summarize(M.breakdown(ref, ref, None, {"A", "B"}))
    secs = {k: v["seconds"] for k, v in s["by_bucket"].items()}
    assert secs == {"<1s": 0.5, "1-3s": 4.0, "3-10s": 5.0, ">=10s": 15.0}
    assert s["by_overlap"]["overlap"]["seconds"] == pytest.approx(4.0)


def test_turn_gap_merges_short_pauses():
    ref = [Seg(0, 0.8, "A"), Seg(1.0, 1.6, "A")]
    b = M.summarize(M.breakdown(ref, ref, None, {"A"}, turn_gap=0.5))["by_bucket"]
    assert b["1-3s"]["seconds"] == pytest.approx(1.4) and b["<1s"]["seconds"] == 0
    b = M.summarize(M.breakdown(ref, ref, None, {"A"}, turn_gap=0.0))["by_bucket"]
    assert b["<1s"]["seconds"] == pytest.approx(1.4)


def test_collar_removes_boundary_frames():
    ref = [Seg(0, 5, "A"), Seg(5, 10, "B")]
    hyp = [Seg(0, 5.2, "A"), Seg(5.2, 10, "B")]
    assert M.headline(ref, hyp, [(0, 10)], {"A", "B"}, 0.0)["ier"] > 0
    assert M.headline(ref, hyp, [(0, 10)], {"A", "B"}, 0.25)["ier"] == 0
    assert M.summarize(M.breakdown(ref, hyp, [(0, 10)], {"A", "B"}, 0.25))["enrolled"]["wrong_name"] == 0


def test_named_false_alarm_and_abstain_fraction():
    ref = [Seg(0, 4, "A")]
    hyp = [Seg(0, 4, "A"), Seg(5, 6, "A"), Seg(6, 8, "UNKNOWN")]
    s = M.summarize(M.breakdown(ref, hyp, [(0, 10)], {"A"}))
    assert s["named_false_alarm_s"] == pytest.approx(1.0)
    assert s["hyp_abstain_fraction"] == pytest.approx(2 / 7, abs=1e-3)


def test_label_kinds_and_wder():
    assert [M.label_kind(x) for x in ("FEE013", "GUEST_2", "UNKNOWN", "UNCERTAIN(A|B)")] == \
        ["name", "guest", "unknown", "uncertain"]
    assert M.wder([("A", "A"), ("A", "B"), ("B", "UNKNOWN"), ("B", "B")]) == 0.5
    assert M.wder([]) is None


def test_rttm_round_trip():
    text = format_rttm("m1", REF)
    assert parse_rttm(text)["m1"] == sorted(REF, key=lambda s: (s.start, s.end, s.speaker))
    with pytest.raises(ValueError):
        parse_rttm("SPKR m1 1 0 1 x\n")
