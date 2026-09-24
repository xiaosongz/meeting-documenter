"""Word attribution tests on synthetic timelines and synthetic AssemblyAI-shaped words."""

import json
import time

import pytest

from voiceprint import words as wm
from voiceprint.timeline import Timeline, Turn
from voiceprint.words import Word


def T(start, end, label, conf=1.0):
    return Turn.from_label(start, end, label, conf)


def TL(*turns):
    return Timeline.from_turns(list(turns))


def W(text, start, end, conf=0.9):
    return Word(text, start, end, conf)


def labels(aws):
    return [w.speaker for w in aws]


def test_words_inside_turns_get_that_speaker():
    tl = TL(T(0, 2, "Alice Example", 0.9), T(2, 4, "Bob Example", 0.8))
    out = wm.assign_words([W("hi", 0.2, 0.5), W("there", 0.6, 1.0), W("yes", 2.5, 2.9)], tl)
    assert labels(out) == ["Alice Example", "Alice Example", "Bob Example"]
    assert [w.kind for w in out] == ["name"] * 3
    assert out[0].margin == 1.0 and out[0].coverage == 1.0 and out[0].flags == ()
    assert out[2].conf == pytest.approx(0.8) and out[0].asr_conf == 0.9


@pytest.mark.parametrize("start,end,winner,straddle", [
    (1.5, 2.5, "A", True),     # 50/50 -> earlier segment wins the tie, flagged
    (1.4, 2.4, "A", True),     # 60/40 -> flagged (boundary inclusive)
    (1.3, 2.3, "A", False),    # 70/30
    (1.75, 2.25, "A", True),   # 50/50 at 0.5 s
    (1.8, 2.8, "B", False),    # 20/80
])
def test_straddling_word_goes_to_larger_intersection(start, end, winner, straddle):
    out = wm.assign_words([W("word", start, end)], TL(T(0, 2, "A"), T(2, 4, "B")))
    assert out[0].speaker == winner and out[0].straddle is straddle
    assert 0.0 <= out[0].margin <= 1.0


def test_straddle_margin_and_conf_tiebreak():
    out = wm.assign_words([W("w", 1.0, 3.0)], TL(T(0, 2, "A", 0.5), T(2, 4, "B", 0.9)))
    assert out[0].speaker == "B" and out[0].margin == 0.0 and out[0].straddle


def test_gap_words_nearest_within_half_second_else_unknown():
    tl = TL(T(0, 2, "A"), T(5, 6, "B"))
    out = wm.assign_words([W("tail", 2.3, 2.6), W("lost", 3.2, 3.6), W("lead", 4.6, 4.9)], tl)
    assert labels(out) == ["A", "UNKNOWN", "B"]
    assert out[0].flags == ("nearest",) and out[0].coverage == 0.0
    assert out[1].flags == ("gap",) and out[1].kind == "unknown" and out[1].conf == 0.0
    # equidistant between two speakers -> the previous one
    tie = wm.assign_words([W("mid", 2.2, 2.3)], TL(T(0, 2, "A"), T(2.5, 3, "B")))
    assert tie[0].speaker == "A"


def test_partially_covered_word_reports_coverage():
    out = wm.assign_words([W("edge", 1.8, 2.2)], TL(T(0, 2, "A")))
    assert out[0].speaker == "A" and out[0].coverage == pytest.approx(0.5) and out[0].margin == 1.0


def test_zero_length_words_use_containing_segment():
    tl = TL(T(0, 2, "A"), T(2, 4, "B"))
    out = wm.assign_words([W("a", 1.0, 1.0), W("b", 2.0, 2.0), W("c", 4.2, 4.2), W("d", 9, 9)], tl)
    assert labels(out) == ["A", "B", "B", "UNKNOWN"]
    assert out[2].flags == ("nearest",)


def test_uncertain_and_overlap_segments_carry_through():
    tl = TL(T(0, 2, "UNCERTAIN(A|B)", 0.3), T(2, 6, "C"), T(4, 5.5, "D"), T(8, 12, "E"), T(9, 9.5, "F"))
    out = wm.assign_words([W("u", 0.5, 0.9), W("o", 4.5, 4.9), W("bc", 9.1, 9.3), W("e", 10, 10.4)], tl)
    assert out[0].speaker == "UNCERTAIN(A|B)" and out[0].kind == "uncertain"
    assert out[0].alts == ("A", "B") and "uncertain" in out[0].flags and out[0].conf == pytest.approx(0.3)
    assert out[1].speaker == "OVERLAP(C|D)" and out[1].kind == "overlap" and "overlap" in out[1].flags
    # resolved backchannel: host keeps the word, interjector recorded, flagged overlap
    assert out[2].speaker == "E" and out[2].alts == ("F",) and "overlap" in out[2].flags
    assert out[3].flags == ()


def test_untimed_words_inherit_neighbour():
    tl = TL(T(0, 2, "A"), T(2, 4, "B"))
    out = wm.assign_words([W("x", None, None), W("a", 0.5, 0.9), W("y", None, None), W("b", 2.5, 3)], tl)
    assert labels(out) == ["A", "A", "A", "B"]
    assert out[0].flags == ("untimed",) and out[2].start is None
    only = wm.assign_words([W("z", None, None)], tl)
    assert only[0].speaker == "UNKNOWN" and only[0].flags == ("untimed",)


def test_never_splits_or_reorders_words_and_empty_timeline():
    ws = [W(f"w{i}", i * 0.3, i * 0.3 + 0.25) for i in range(20)]
    out = wm.assign_words(ws, TL(T(0, 3, "A"), T(2.9, 6.5, "B")))
    assert [w.text for w in out] == [w.text for w in ws] and len(out) == len(ws)
    assert set(labels(wm.assign_words(ws, Timeline([], [], {})))) == {"UNKNOWN"}


def test_turns_only_timeline_derives_exclusive():
    tl = Timeline([T(0, 2, "A"), T(1.8, 4, "B")], [], {})
    out = wm.assign_words([W("a", 0.5, 1.0), W("b", 3.0, 3.5)], tl)
    assert labels(out) == ["A", "B"]


def test_mandarin_shaped_tokens():
    """Shape seen from AssemblyAI zh (universal-2): 1–3 character word tokens, little
    punctuation, contiguous timing (next start == previous end), and long tokens that
    absorb a pause."""
    tl = TL(T(0.0, 1.9, "Alice Example"), T(4.5, 6.5, "Bob Example"))
    ws = [W("我们", 0.10, 0.40), W("今天", 0.40, 0.70), W("讨论", 0.70, 1.00), W("一下", 1.00, 1.30),
          W("Python", 1.30, 1.70), W("方案。", 1.70, 4.50),  # absorbs the pause up to Bob's onset
          W("好的", 4.50, 4.80), W("，", 4.80, 4.80), W("没问题", 4.80, 6.10)]
    out = wm.assign_words(ws, tl)
    assert labels(out)[:6] == ["Alice Example"] * 6 and labels(out)[6:] == ["Bob Example"] * 3
    assert out[5].flags == ("long",) and out[5].coverage == pytest.approx(0.2 / 2.8)
    turns = wm.rebuild_turns(out)
    assert [t.text for t in turns] == ["我们今天讨论一下 Python 方案。", "好的，没问题"]
    # Known limitation: a pause-absorbing token that runs into the next speaker's speech
    # goes to whoever it overlaps more; it is flagged, not corrected.
    crossing = wm.assign_words([W("方案", 1.70, 5.10)], tl)[0]
    assert crossing.speaker == "Bob Example" and "long" in crossing.flags


def test_rebuild_turns_splits_at_every_label_change():
    tl = TL(T(0, 2, "A", 0.9), T(2, 3, "B", 0.5), T(3, 5, "A", 0.7), T(5, 6, "UNCERTAIN(A|B)", 0.2))
    ws = [W("a1", 0.1, 0.5), W("a2", 0.6, 1.9), W("b1", 2.1, 2.9), W("a3", 3.1, 4.0), W("u1", 5.1, 5.5)]
    turns = wm.rebuild_turns(wm.assign_words(ws, tl))
    assert [(t.speaker, t.text) for t in turns] == [("A", "a1 a2"), ("B", "b1"), ("A", "a3"),
                                                     ("UNCERTAIN(A|B)", "u1")]
    assert turns[0].start == 0.1 and turns[0].end == 1.9 and turns[0].conf == pytest.approx(0.9)
    assert turns[3].kind == "uncertain" and turns[3].alts == ("A", "B")


def test_rebuild_turns_pause_and_length_splits():
    aws = wm.unattributed([W("One.", 0, 1), W("two", 1.1, 2), W("three.", 5.0, 6), W("four", 6.1, 7)])
    assert len(wm.rebuild_turns(aws)) == 1
    assert [t.text for t in wm.rebuild_turns(aws, max_pause=2.0)] == ["One. two", "three. four"]
    assert [t.text for t in wm.rebuild_turns(aws, max_turn_s=0.5)] == ["One.", "two three.", "four"]
    assert [t.text for t in wm.rebuild_turns(aws, max_turn_s=0.4)][0] == "One."


def test_rebuild_turns_counts_flags():
    out = wm.assign_words([W("w", 1.5, 2.5), W("x", 2.6, 3)], TL(T(0, 2, "A"), T(2, 4, "B")))
    turns = wm.rebuild_turns(out)
    assert turns[0].flag_counts["straddle"] == 1


def test_join_tokens_spacing():
    assert wm.join_tokens(["Hello,", "world."]) == "Hello, world."
    assert wm.join_tokens(["你好", "世界。", "我们", "用", "Python", "写"]) == "你好世界。我们用 Python 写"
    assert wm.join_tokens(["OK,", "我们", "", " 开始"]) == "OK, 我们开始"


def test_format_ts_and_render_body():
    assert [wm.format_ts(s) for s in (0, 61.9, 599.99, 3600, 3725.2)] == [
        "00:00", "01:01", "09:59", "1:00:00", "1:02:05"]
    tl = TL(T(0, 2, "Alice Example"), T(2, 4, "GUEST_1"))
    turns = wm.rebuild_turns(wm.assign_words([W("Hi.", 0.2, 0.5), W("Hello.", 2.2, 2.6)], tl))
    assert wm.render_body(turns) == "**[00:00] Alice Example:** Hi.\n\n**[00:02] GUEST_1:** Hello."
    assert wm.render_body(turns, labels=False) == "**[00:00]** Hi.\n\n**[00:02]** Hello."


def test_sidecar_round_trip_drops_speakers(tmp_path):
    items = [{"text": "Synthetic", "start": 0, "end": 480, "confidence": 0.98, "speaker": "A"},
             {"text": "words.", "start": 480, "end": 900, "confidence": 0.9, "speaker": "A"},
             {"text": "unaligned", "start": None, "end": None, "confidence": 0.5}]
    meta = {"id": "synthetic-job", "language_code": "en", "audio_duration": 1.5,
            "speech_model_used": "universal-3-5-pro"}
    side = wm.sidecar_from_assemblyai(items, meta)
    text = wm.dump_sidecar(side)
    assert json.loads(text) == side and "speaker" not in text
    assert text.count("\n") >= len(items) + 5  # one word per line
    path = tmp_path / "x.words.json"
    path.write_text(text, encoding="utf-8")
    ws, head = wm.load_words_sidecar(path)
    assert ws == [Word("Synthetic", 0.0, 0.48, 0.98), Word("words.", 0.48, 0.9, 0.9), Word("unaligned", None, None, 0.5)]
    assert head["audio_duration_ms"] == 1500 and head["transcript_id"] == "synthetic-job"
    assert wm.sidecar_from_assemblyai(items[:2])["audio_duration_ms"] == 900
    path.write_text(json.dumps({"schema": 1, "type": "utterances"}), encoding="utf-8")
    with pytest.raises(ValueError):
        wm.load_words_sidecar(path)


def test_attribution_scales_to_long_meetings():
    turns, ws, t = [], [], 0.0
    for i in range(3000):  # ~2 h, 3000 turns, ~24k words
        turns.append(T(t, t + 2.2, f"S{i % 5}"))
        ws.extend(W("w", t + k * 0.3, t + k * 0.3 + 0.28) for k in range(8))
        t += 2.4
    tl = TL(*turns)
    t0 = time.perf_counter()
    out = wm.assign_words(ws, tl)
    wm.rebuild_turns(out)
    assert time.perf_counter() - t0 < 2.0
    assert len(out) == len(ws)


def test_display_label_is_readable():
    assert wm.display_label("Alice Example") == "Alice Example"
    assert wm.display_label("GUEST_2") == "Guest 2"
    assert wm.display_label("UNKNOWN") == "Unknown speaker"
    assert wm.display_label("UNCERTAIN(Alice Example|Bob Example)") == "Alice Example or Bob Example (uncertain)"
    assert wm.display_label("UNCERTAIN(A|B|GUEST_1)") == "A, B or Guest 1 (uncertain)"
    assert wm.display_label("OVERLAP(A|UNKNOWN)") == "A + Unknown speaker (overlap)"
