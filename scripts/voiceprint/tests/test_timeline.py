"""Timeline tests: label conventions, the exclusive-derivation rule, RTTM and JSON round trips."""

import json
import math

import pytest

from voiceprint import timeline as tlm
from voiceprint.timeline import Timeline, Turn


def T(start, end, label, conf=1.0):
    return Turn.from_label(start, end, label, conf)


def spans(turns):
    return [(round(t.start, 6), round(t.end, 6), t.speaker) for t in turns]


def test_parse_label_kinds():
    assert tlm.parse_label("Alice Example") == ("name", ())
    assert tlm.parse_label("GUEST_2") == ("guest", ())
    assert tlm.parse_label("UNKNOWN") == ("unknown", ())
    assert tlm.parse_label("UNCERTAIN(Alice Example|Bob Example)") == (
        "uncertain", ("Alice Example", "Bob Example"))
    assert tlm.parse_label(tlm.overlap_label(["A", "B"])) == ("overlap", ("A", "B"))
    assert tlm.candidates("UNCERTAIN(A|B)") == ("A", "B")
    assert tlm.candidates("A") == ("A",)


def test_turn_validation():
    with pytest.raises(ValueError):
        Turn(2.0, 1.0, "A", "name", 1.0)
    with pytest.raises(ValueError):
        Turn(0.0, 1.0, "A", "person", 1.0)
    with pytest.raises(ValueError):
        Turn(0.0, math.inf, "A", "name", 1.0)
    assert Turn(0, 1, "A", "name", 1.0, ["B"]).alts == ("B",)


def test_non_overlapping_turns_pass_through_and_same_label_merges():
    turns = [T(0, 2, "A"), T(1.5, 3, "A"), T(3, 4, "B"), T(5, 6, "A")]
    ex = tlm.derive_exclusive(turns)
    assert spans(ex) == [(0, 3, "A"), (3, 4, "B"), (5, 6, "A")]


def test_handoff_overlap_splits_at_midpoint():
    ex = tlm.derive_exclusive([T(0, 5.4, "A", 0.9), T(5.0, 9, "B", 0.8)])
    assert spans(ex) == [(0, 5.2, "A"), (5.2, 9, "B")]
    assert [t.kind for t in ex] == ["name", "name"]
    assert ex[0].conf == pytest.approx(0.9) and ex[1].conf == pytest.approx(0.8)


def test_long_handoff_becomes_overlap():
    ex = tlm.derive_exclusive([T(0, 6, "A"), T(4, 9, "B")])
    assert spans(ex) == [(0, 4, "A"), (4, 6, "OVERLAP(A|B)"), (6, 9, "B")]
    assert ex[1].kind == "overlap" and ex[1].alts == ("A", "B")


def test_short_backchannel_goes_to_host_and_records_interjector():
    ex = tlm.derive_exclusive([T(0, 10, "A"), T(4, 4.6, "B")])
    assert spans(ex) == [(0, 4, "A"), (4, 4.6, "A"), (4.6, 10, "A")]
    assert [t.alts for t in ex] == [(), ("B",), ()]


def test_long_interjection_and_three_way_overlap_are_overlap():
    ex = tlm.derive_exclusive([T(0, 10, "A"), T(4, 6, "B")])
    assert ex[1].speaker == "OVERLAP(A|B)"
    ex = tlm.derive_exclusive([T(0, 3, "A"), T(2.8, 5, "B"), T(2.9, 3.5, "C")])
    kinds = {t.speaker: t.kind for t in ex}
    assert kinds.get("OVERLAP(A|B|C)") == "overlap"


def test_overlap_uses_min_conf_and_flattens_compound_labels():
    ex = tlm.derive_exclusive([T(0, 5, "A", 0.9), T(1, 4, "UNCERTAIN(A|B)", 0.4), T(4.5, 6, "C", 0.7)])
    ov = [t for t in ex if t.kind == "overlap"]
    assert ov[0].speaker == "OVERLAP(A|B)" and ov[0].conf == pytest.approx(0.4)


def test_simultaneous_start_is_not_a_handoff():
    ex = tlm.derive_exclusive([T(0, 0.5, "A"), T(0, 3, "B")])
    assert spans(ex) == [(0, 0.5, "OVERLAP(A|B)"), (0.5, 3, "B")]


def test_gaps_stay_gaps_and_zero_length_turns_vanish():
    ex = tlm.derive_exclusive([T(0, 1, "A"), T(1, 1, "B"), T(2, 3, "B")])
    assert spans(ex) == [(0, 1, "A"), (2, 3, "B")]
    assert all(a.end <= b.start for a, b in zip(ex, ex[1:]))


def test_merge_touching_weights_conf_by_duration():
    out = tlm.merge_touching([Turn(0, 1, "A", "name", 1.0), Turn(1, 4, "A", "name", 0.6)])
    assert len(out) == 1 and out[0].conf == pytest.approx(0.7)


def test_exclusive_never_overlaps_on_random_turns():
    import random
    rng = random.Random(7)
    labels = ["A", "B", "C", "GUEST_1", "UNKNOWN", "UNCERTAIN(A|B)"]
    for _ in range(50):
        turns = []
        for _ in range(rng.randint(1, 30)):
            s = round(rng.uniform(0, 60), 2)
            turns.append(T(s, s + round(rng.uniform(0.05, 8), 2), rng.choice(labels), rng.random()))
        ex = tlm.derive_exclusive(turns)
        assert all(a.end <= b.start + 1e-9 for a, b in zip(ex, ex[1:]))
        # exclusive covers exactly the union of the turns
        covered = sum(t.dur for t in ex)
        union, cur_s, cur_e = 0.0, None, None
        for t in sorted(turns, key=lambda t: t.start):
            if cur_e is None or t.start > cur_e:
                union += (cur_e - cur_s) if cur_e is not None else 0
                cur_s, cur_e = t.start, t.end
            else:
                cur_e = max(cur_e, t.end)
        union += cur_e - cur_s
        assert covered == pytest.approx(union)


def test_rttm_round_trip_with_spaces_and_compound_labels():
    tl = Timeline.from_turns([T(0.0, 2.5, "Alice Example", 0.93), T(2.5, 4.0, "UNCERTAIN(Alice Example|Bob Example)", 0.41),
                              T(3.9, 6.25, "GUEST_1", 0.7), T(7.0, 8.0, "UNKNOWN", 1.0), T(8, 9, "Pct%Name", 1.0)])
    text = tlm.to_rttm(tl, "meeting_01")
    assert "Alice%20Example" in text and "Pct%25Name" in text
    for line in text.splitlines():
        assert len(line.split()) == 10
    back = tlm.from_rttm(text)
    assert spans(back.turns) == spans(tl.turns)
    assert [t.kind for t in back.turns] == [t.kind for t in tl.turns]
    assert back.turns[1].alts == ("Alice Example", "Bob Example")
    assert [t.conf for t in back.turns] == pytest.approx([t.conf for t in tl.turns])
    assert spans(back.exclusive) == spans(tl.exclusive)
    assert back.meta["uri"] == "meeting_01"


def test_rttm_exclusive_export_and_multi_uri_selection():
    tl = Timeline.from_turns([T(0, 6, "A"), T(4, 9, "B")])
    ex_text = tlm.to_rttm(tl, "m", exclusive=True)
    assert "OVERLAP(A|B)" in ex_text
    two = tlm.to_rttm(tl, "m1") + tlm.to_rttm(Timeline.from_turns([T(0, 1, "C")]), "m2")
    with pytest.raises(ValueError):
        tlm.from_rttm(two)
    assert tlm.from_rttm(two, uri="m2").speakers() == ["C"]
    with pytest.raises(ValueError):
        tlm.to_rttm(tl, "bad uri")


def test_rttm_reference_style_na_conf_and_comments():
    text = ";; comment\nSPEAKER ES2004a 1 0.50 1.25 <NA> <NA> MEE013 <NA> <NA>\n\n"
    tl = tlm.from_rttm(text)
    assert spans(tl.turns) == [(0.5, 1.75, "MEE013")] and tl.turns[0].conf == 1.0


def test_json_round_trip_is_exact_and_keeps_meta():
    np = pytest.importorskip("numpy")
    turns = [Turn(0.1234567, 2.0, "Alice Example", "name", 0.87654321, ("Bob Example",)),
             Turn(2.0, 3.0, "UNCERTAIN(Alice Example|Bob Example)", "uncertain", float("nan"),
                  ("Alice Example", "Bob Example"))]
    tl = Timeline(turns, list(turns), {"backend": "seg3", "n": np.int64(3), "arr": np.arange(2)})
    text = tlm.to_json(tl)
    json.loads(text)  # strict JSON: no NaN literals
    back = tlm.from_json(text)
    assert back.turns[0] == turns[0]
    assert math.isnan(back.turns[1].conf) and back.turns[1].alts == turns[1].alts
    assert back.meta == {"backend": "seg3", "n": 3, "arr": [0, 1]}
    with pytest.raises(ValueError):
        tlm.from_json(json.dumps({"schema": 99}))
