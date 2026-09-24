"""
Attribute ASR words to speakers from a local Timeline, and rebuild transcript turns.

AssemblyAI runs with speaker_labels=false, so its words carry text and timing
only; speakers come from the local timeline (who speaks when). Standard library
only, so the transcription pipeline can import it without the voiceprint extras.

Rules (WhisperX-style max intersection, against the *exclusive* timeline):
  - A word goes to the label with the largest summed time intersection. Words
    are never split. When the top two labels split the word 40–60 %, the word
    is flagged "straddle" (margin = top share − runner-up share).
  - A word that touches no segment goes to the nearest segment within
    MAX_GAP s (flag "nearest", previous segment on a tie); otherwise it is
    UNKNOWN (flag "gap"). Zero-length words use the segment containing them.
  - The chosen segment's kind carries through: UNCERTAIN(A|B) and
    OVERLAP(A|B) stay as labels (flags "uncertain" / "overlap"); a named
    segment with co-active alts (a resolved backchannel) is flagged "overlap".
  - Words without timestamps (AssemblyAI could not align them) inherit the
    previous word's attribution (flag "untimed").
  - Words longer than LONG_WORD_S are flagged "long": AssemblyAI's Mandarin
    tokens absorb the following pause (contiguous timing, durations up to tens
    of seconds), so their spans overstate the speech they contain.
Turns are rebuilt at every label change, and optionally at long pauses or
after a sentence end once a turn grows long.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .timeline import UNKNOWN, Timeline, Turn, derive_exclusive

MAX_GAP = 0.5               # s; nearest-segment fallback for words in a gap
STRADDLE_MAX_SHARE = 0.6    # top label's share of top-two intersection at or below which a word straddles
LONG_WORD_S = 2.5           # s; see module docstring
SENTENCE_END = tuple(".?!。？！…")
SIDECAR_SCHEMA = 1


@dataclass
class Word:
    text: str
    start: float | None     # seconds; None when the ASR could not align the word
    end: float | None
    conf: float = 1.0       # ASR confidence


@dataclass
class AttributedWord:
    text: str
    start: float | None
    end: float | None
    asr_conf: float
    speaker: str
    kind: str
    conf: float             # timeline confidence of the chosen label, intersection-weighted
    margin: float           # top share − runner-up share of the word's speech intersection
    coverage: float         # fraction of the word's span covered by timeline speech
    alts: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def straddle(self) -> bool:
        return "straddle" in self.flags


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    kind: str
    conf: float
    alts: tuple[str, ...]
    words: list[AttributedWord]

    @property
    def text(self) -> str:
        return join_tokens(w.text for w in self.words)

    @property
    def flag_counts(self) -> Counter:
        return Counter(f for w in self.words for f in w.flags)


# ─── AssemblyAI words ───────────────────────────────────────────────────────

def words_from_assemblyai(items: list[dict]) -> list[Word]:
    """AssemblyAI `words` (ms; start/end may be missing) → Words in seconds."""
    out = []
    for it in items:
        start, end = it.get("start"), it.get("end")
        timed = start is not None and end is not None
        conf = it.get("confidence")
        out.append(Word(text=(it.get("text") or "").strip(),
                        start=start / 1000 if timed else None,
                        end=end / 1000 if timed else None,
                        conf=1.0 if conf is None else float(conf)))
    return out


def sidecar_from_assemblyai(items: list[dict], metadata: dict | None = None) -> dict:
    """Word-level sidecar (schema 1): AssemblyAI text, ms timing and confidence; no speakers."""
    metadata = metadata or {}
    duration_s = metadata.get("audio_duration")
    last_end = next((it["end"] for it in reversed(items) if it.get("end") is not None), 0)
    return {
        "schema": SIDECAR_SCHEMA,
        "type": "words",
        "source": "assemblyai",
        "transcript_id": metadata.get("id"),
        "language_code": metadata.get("language_code"),
        "speech_model_used": metadata.get("speech_model_used"),
        "audio_duration_ms": int(duration_s * 1000) if duration_s is not None else int(last_end),
        "time_unit": "ms",
        "words": [{"text": (it.get("text") or "").strip(),
                   "start": None if it.get("start") is None else int(it["start"]),
                   "end": None if it.get("end") is None else int(it["end"]),
                   "confidence": it.get("confidence")}
                  for it in items],
    }


def dump_sidecar(sidecar: dict) -> str:
    """JSON with one word per line (diffable, still small)."""
    head = {k: v for k, v in sidecar.items() if k != "words"}
    body = ",\n".join(" " + json.dumps(w, ensure_ascii=False) for w in sidecar["words"])
    text = json.dumps(head, ensure_ascii=False, indent=1)
    return text[:-2] + ',\n "words": [\n' + body + "\n ]\n}\n"


def load_words_sidecar(path: str | Path) -> tuple[list[Word], dict]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema") != SIDECAR_SCHEMA or obj.get("type") != "words":
        raise ValueError("not a schema-1 words sidecar")
    return words_from_assemblyai(obj["words"]), {k: v for k, v in obj.items() if k != "words"}


# ─── Attribution ────────────────────────────────────────────────────────────

def _from_turn(w: Word, seg: Turn, conf: float, margin: float, coverage: float,
               flags: tuple[str, ...]) -> AttributedWord:
    if seg.kind == "uncertain":
        flags += ("uncertain",)
    elif seg.kind == "overlap" or seg.alts:
        flags += ("overlap",)
    return AttributedWord(w.text, w.start, w.end, w.conf, seg.speaker, seg.kind,
                          conf, margin, coverage, seg.alts, flags)


def _unknown(w: Word, flags: tuple[str, ...]) -> AttributedWord:
    return AttributedWord(w.text, w.start, w.end, w.conf, UNKNOWN, "unknown", 0.0, 0.0, 0.0, (), flags)


def assign_words(words: list[Word], tl: Timeline, *, max_gap: float = MAX_GAP,
                 straddle_max_share: float = STRADDLE_MAX_SHARE,
                 long_word_s: float = LONG_WORD_S) -> list[AttributedWord]:
    """Attribute each word to one exclusive-timeline label (rules in the module docstring)."""
    segs = sorted(tl.exclusive if tl.exclusive or not tl.turns else derive_exclusive(tl.turns),
                  key=lambda t: t.start)
    starts = [s.start for s in segs]
    ends = [s.end for s in segs]

    def nearest(w: Word, a: float, b: float, flags: tuple[str, ...]) -> AttributedWord:
        k = bisect_right(starts, a)
        prev = segs[k - 1] if k > 0 and segs[k - 1].end <= a else None
        nxt = next((s for s in segs[k:] if s.start >= b), None)
        d_prev = a - prev.end if prev else float("inf")
        d_next = nxt.start - b if nxt else float("inf")
        seg, dist = (prev, d_prev) if d_prev <= d_next else (nxt, d_next)
        if seg is None or dist > max_gap:
            return _unknown(w, flags + ("gap",))
        return _from_turn(w, seg, seg.conf, 1.0, 0.0, flags + ("nearest",))

    out: list[AttributedWord | None] = []
    for w in words:
        if w.start is None or w.end is None:
            out.append(None)
            continue
        a, b = w.start, max(w.start, w.end)
        flags = ("long",) if b - a > long_word_s else ()
        if b - a <= 0:
            k = bisect_right(starts, a) - 1
            if k >= 0 and segs[k].end > a:
                out.append(_from_turn(w, segs[k], segs[k].conf, 1.0, 1.0, flags))
            else:
                out.append(nearest(w, a, b, flags))
            continue
        per: dict[str, list] = {}  # label -> [intersection, conf·intersection, best segment, its intersection]
        for seg in segs[bisect_right(ends, a):]:
            if seg.start >= b:
                break
            inter = min(b, seg.end) - max(a, seg.start)
            if inter <= 0:
                continue
            acc = per.setdefault(seg.speaker, [0.0, 0.0, seg, 0.0])
            acc[0] += inter
            acc[1] += seg.conf * inter
            if inter > acc[3]:
                acc[2], acc[3] = seg, inter
        if not per:
            out.append(nearest(w, a, b, flags))
            continue
        ranked = sorted(per.values(), key=lambda v: (-v[0], -(v[1] / v[0]), v[2].start))
        total = sum(v[0] for v in ranked)
        top = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 0.0
        if second > 0 and top[0] / (top[0] + second) <= straddle_max_share + 1e-9:
            flags += ("straddle",)
        out.append(_from_turn(w, top[2], top[1] / top[0], (top[0] - second) / total,
                              total / (b - a), flags))

    # Untimed words inherit the previous attributed word (or the next one at the start).
    for i, aw in enumerate(out):
        if aw is not None:
            continue
        w = words[i]
        src = next((out[j] for j in range(i - 1, -1, -1) if out[j] is not None), None) or \
            next((out[j] for j in range(i + 1, len(out)) if out[j] is not None), None)
        if src is None:
            out[i] = _unknown(w, ("untimed",))
        else:
            out[i] = AttributedWord(w.text, None, None, w.conf, src.speaker, src.kind, src.conf,
                                    0.0, 0.0, src.alts, ("untimed",))
    return out  # type: ignore[return-value]


def unattributed(words: list[Word]) -> list[AttributedWord]:
    """Words with no timeline at all (transcript before local speaker tracking has run)."""
    return [AttributedWord(w.text, w.start, w.end, w.conf, UNKNOWN, "unknown", 0.0, 0.0, 0.0, (),
                           ("unattributed",)) for w in words]


# ─── Turns ──────────────────────────────────────────────────────────────────

def _utterance(ws: list[AttributedWord], fallback_t: float) -> Utterance:
    timed = [w for w in ws if w.start is not None]
    start = min((w.start for w in timed), default=fallback_t)
    end = max((w.end for w in timed), default=start)
    weights = [max(w.end - w.start, 0.02) for w in timed]
    conf = (sum(w.conf * x for w, x in zip(timed, weights)) / sum(weights)) if timed else ws[0].conf
    alts = tuple(dict.fromkeys(a for w in ws for a in w.alts))
    return Utterance(start, end, ws[0].speaker, ws[0].kind, conf, alts, ws)


def rebuild_turns(attributed: list[AttributedWord], *, max_pause: float | None = None,
                  max_turn_s: float | None = None) -> list[Utterance]:
    """Group words into turns: a new turn at every label change; optionally also at a
    pause > max_pause s, and after a sentence end once a turn exceeds max_turn_s
    (at any word once it exceeds twice that)."""
    turns: list[Utterance] = []
    cur: list[AttributedWord] = []
    cur_start = last_end = None
    for w in attributed:
        split = False
        if cur:
            if w.speaker != cur[-1].speaker:
                split = True
            elif w.start is not None and last_end is not None:
                if max_pause is not None and w.start - last_end > max_pause:
                    split = True
                elif max_turn_s is not None and cur_start is not None:
                    length = last_end - cur_start
                    ended = cur[-1].text.endswith(SENTENCE_END)
                    split = (length >= max_turn_s and ended) or length >= 2 * max_turn_s
        if split:
            turns.append(_utterance(cur, turns[-1].end if turns else 0.0))
            cur, cur_start = [], None
        cur.append(w)
        if w.start is not None:
            cur_start = w.start if cur_start is None else cur_start
            last_end = w.end if last_end is None else max(last_end, w.end)
    if cur:
        turns.append(_utterance(cur, turns[-1].end if turns else 0.0))
    return turns


# ─── Rendering ──────────────────────────────────────────────────────────────

def _cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or 0xF900 <= o <= 0xFAFF
            or 0xFF00 <= o <= 0xFFEF or 0x20000 <= o <= 0x2FA1F)


def join_tokens(tokens) -> str:
    """Join ASR tokens: no space between two CJK tokens (AssemblyAI Mandarin tokens are
    1–3 character words with punctuation attached), a space everywhere else."""
    out = ""
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if out and not (_cjk(out[-1]) and _cjk(tok[0])):
            out += " "
        out += tok
    return out


def format_ts(seconds: float) -> str:
    """Seconds → MM:SS (H:MM:SS past an hour), identical to the AssemblyAI transcript format."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def render_body(utterances: list[Utterance], *, labels: bool = True) -> str:
    """Transcript body lines `**[MM:SS] Label:** text` (or `**[MM:SS]** text` without labels)."""
    lines = []
    for u in utterances:
        text = u.text
        if not text:
            continue
        head = f"**[{format_ts(u.start)}] {u.speaker}:**" if labels else f"**[{format_ts(u.start)}]**"
        lines.append(f"{head} {text}")
    return "\n\n".join(lines)
