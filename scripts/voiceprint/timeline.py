"""
Speaker timeline: who speaks when, as turns plus an exclusive (one speaker at a time) view.

Shared by the boundary/tracking stages (producers), words.py (consumer) and the
public benchmark (scoring). Standard library only, so the transcription pipeline
can import it without the optional voiceprint dependencies.

Times are seconds from the start of the audio. Speaker labels:
    <enrolled name>        kind "name"       e.g. "Alice Example"
    GUEST_<n>              kind "guest"      unenrolled voice, clustered within the meeting
    UNKNOWN                kind "unknown"    speech not attributed to anyone
    UNCERTAIN(A|B)         kind "uncertain"  low margin between candidates; alts = (A, B)
    OVERLAP(A|B)           kind "overlap"    exclusive view only; alts = (A, B)
Names must not contain "|", "(" or ")".

`turns` may overlap (several people at once). `exclusive` never does; it is what
word attribution uses. A producer that has frame-level scores should build
`exclusive` itself; `derive_exclusive` is the rule for when only turns are known
(RTTM import, other systems):

  1. Merge turns of the same label; where exactly one label is active, it wins.
  2. A maximal region where two or more labels are active is an overlap region R:
     a. Hand-off: two labels, one active just before R and the other just after,
        and |R| <= handoff_max -> split R at its midpoint between them.
     b. Backchannel: two labels, the same "host" label active on both sides of R,
        the other only inside it, and |R| <= backchannel_max -> R goes to the host;
        the interjector is recorded in `alts` (name/guest/unknown hosts only).
     c. Anything else -> an "overlap" turn OVERLAP(A|B[|C…]), conf = the lowest
        conf of the labels involved; compound labels contribute their candidates.
  3. Touching pieces with the same label, kind and alts are merged (conf is
     duration-weighted).

RTTM keeps times (ms), labels and conf (column 9, "<NA>" -> 1.0); kind and alts
are re-derived from the label. Whitespace and "%" in labels are percent-encoded,
because RTTM fields are whitespace-separated. JSON keeps everything.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal
from urllib.parse import unquote

Kind = Literal["name", "guest", "unknown", "uncertain", "overlap"]
KINDS: tuple[str, ...] = ("name", "guest", "unknown", "uncertain", "overlap")
UNKNOWN = "UNKNOWN"
GUEST_PREFIX = "GUEST_"
SCHEMA = 1

HANDOFF_MAX = 1.0       # s; longer two-way overlaps stay "overlap" even at a speaker change
BACKCHANNEL_MAX = 1.0   # s; longer interjections inside a host turn stay "overlap"

_COMPOUND = re.compile(r"^(UNCERTAIN|OVERLAP)\((.*)\)$")
_EPS = 1e-9


def uncertain_label(alts: Iterable[str]) -> str:
    return "UNCERTAIN(" + "|".join(alts) + ")"


def overlap_label(alts: Iterable[str]) -> str:
    return "OVERLAP(" + "|".join(alts) + ")"


def parse_label(label: str) -> tuple[str, tuple[str, ...]]:
    """Label -> (kind, alts) under the conventions in the module docstring."""
    m = _COMPOUND.match(label)
    if m:
        kind = "uncertain" if m[1] == "UNCERTAIN" else "overlap"
        return kind, tuple(a for a in m[2].split("|") if a)
    if label == UNKNOWN:
        return "unknown", ()
    if label.startswith(GUEST_PREFIX):
        return "guest", ()
    return "name", ()


def candidates(label: str) -> tuple[str, ...]:
    """The plain speaker labels a label stands for (compound labels expand to their alts)."""
    kind, alts = parse_label(label)
    return alts if kind in ("uncertain", "overlap") else (label,)


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    kind: Kind
    conf: float
    alts: tuple[str, ...] = ()

    def __post_init__(self):
        self.alts = tuple(self.alts)
        if self.kind not in KINDS:
            raise ValueError(f"unknown turn kind {self.kind!r}")
        if not (math.isfinite(self.start) and math.isfinite(self.end)) or self.end < self.start:
            raise ValueError(f"invalid turn span {self.start!r}..{self.end!r}")

    @classmethod
    def from_label(cls, start: float, end: float, label: str, conf: float = 1.0) -> Turn:
        kind, alts = parse_label(label)
        return cls(float(start), float(end), label, kind, float(conf), alts)

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Timeline:
    turns: list[Turn]
    exclusive: list[Turn]
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_turns(cls, turns: Iterable[Turn], meta: dict | None = None, *,
                   handoff_max: float = HANDOFF_MAX,
                   backchannel_max: float = BACKCHANNEL_MAX) -> Timeline:
        ordered = sorted(turns, key=lambda t: (t.start, t.end, t.speaker))
        return cls(ordered, derive_exclusive(ordered, handoff_max=handoff_max,
                                             backchannel_max=backchannel_max),
                   dict(meta or {}))

    def speakers(self) -> list[str]:
        """Distinct turn labels in order of first appearance."""
        return list(dict.fromkeys(t.speaker for t in self.turns))


# ─── Exclusive derivation ───────────────────────────────────────────────────

@dataclass
class _Piece:
    start: float
    end: float
    labels: dict  # label -> (kind, conf, alts); insertion order = first appearance


def _pieces(turns: list[Turn]) -> list[_Piece]:
    """Elementary intervals between all turn boundaries that have at least one active turn."""
    live = [t for t in turns if t.end - t.start > _EPS]
    bounds = sorted({t.start for t in live} | {t.end for t in live})
    order = sorted(live, key=lambda t: (t.start, t.end))
    active: list[Turn] = []
    j = 0
    out: list[_Piece] = []
    for a, b in zip(bounds, bounds[1:]):
        while j < len(order) and order[j].start <= a:
            active.append(order[j])
            j += 1
        active = [t for t in active if t.end > a]  # every boundary is in `bounds`, so these cover [a, b)
        if not active:
            continue
        labels: dict = {}
        for t in sorted(active, key=lambda t: (t.start, t.speaker)):
            if t.speaker in labels:
                kind, conf, alts = labels[t.speaker]
                labels[t.speaker] = (kind, max(conf, t.conf), tuple(dict.fromkeys(alts + t.alts)))
            else:
                labels[t.speaker] = (t.kind, t.conf, t.alts)
        out.append(_Piece(a, b, labels))
    return out


def _single(start: float, end: float, label: str, info: tuple) -> Turn:
    kind, conf, alts = info
    return Turn(start, end, label, kind, conf, alts)


def _resolve_region(region: list[_Piece], before: str | None, after: str | None,
                    handoff_max: float, backchannel_max: float) -> list[Turn]:
    s, e = region[0].start, region[-1].end
    info: dict = {}
    for p in region:
        for label, (kind, conf, alts) in p.labels.items():
            if label in info:
                k0, c0, a0 = info[label]
                info[label] = (k0, max(c0, conf), tuple(dict.fromkeys(a0 + alts)))
            else:
                info[label] = (kind, conf, alts)
    if len(info) == 2:
        if before in info and after in info and before != after and e - s <= handoff_max:
            mid = (s + e) / 2
            return [_single(s, mid, before, info[before]), _single(mid, e, after, info[after])]
        if before is not None and before == after and before in info and e - s <= backchannel_max:
            kind, conf, alts = info[before]
            other = next(label for label in info if label != before)
            if kind in ("name", "guest", "unknown"):
                alts = tuple(dict.fromkeys(alts + (other,)))
            return [Turn(s, e, before, kind, conf, alts)]
    names = tuple(dict.fromkeys(c for label in info for c in candidates(label)))
    conf = min(c for _, c, _ in info.values())
    return [Turn(s, e, overlap_label(names), "overlap", conf, names)]


def derive_exclusive(turns: Iterable[Turn], *, handoff_max: float = HANDOFF_MAX,
                     backchannel_max: float = BACKCHANNEL_MAX) -> list[Turn]:
    """Non-overlapping turns from possibly overlapping ones (rule in the module docstring)."""
    pieces = _pieces(list(turns))
    raw: list[Turn] = []
    i = 0
    while i < len(pieces):
        p = pieces[i]
        if len(p.labels) == 1:
            label, info = next(iter(p.labels.items()))
            raw.append(_single(p.start, p.end, label, info))
            i += 1
            continue
        k = i
        while k + 1 < len(pieces) and len(pieces[k + 1].labels) > 1 and pieces[k + 1].start == pieces[k].end:
            k += 1
        prev = pieces[i - 1] if i > 0 and pieces[i - 1].end == p.start else None
        nxt = pieces[k + 1] if k + 1 < len(pieces) and pieces[k + 1].start == pieces[k].end else None
        before = next(iter(prev.labels)) if prev is not None else None
        after = next(iter(nxt.labels)) if nxt is not None else None
        raw.extend(_resolve_region(pieces[i:k + 1], before, after, handoff_max, backchannel_max))
        i = k + 1
    return merge_touching(raw)


def merge_touching(turns: list[Turn]) -> list[Turn]:
    """Merge sorted, non-overlapping turns that touch and share label, kind and alts."""
    out: list[Turn] = []
    for t in turns:
        if t.end - t.start <= _EPS:
            continue
        last = out[-1] if out else None
        if (last is not None and abs(last.end - t.start) <= _EPS and last.speaker == t.speaker
                and last.kind == t.kind and last.alts == t.alts):
            d0, d1 = last.dur, t.dur
            conf = (last.conf * d0 + t.conf * d1) / (d0 + d1)
            out[-1] = Turn(last.start, t.end, last.speaker, last.kind, conf, last.alts)
        else:
            out.append(Turn(t.start, t.end, t.speaker, t.kind, t.conf, t.alts))
    return out


# ─── RTTM ───────────────────────────────────────────────────────────────────

def _encode(label: str) -> str:
    return "".join("".join(f"%{b:02X}" for b in ch.encode("utf-8")) if ch.isspace() or ch == "%" else ch
                   for ch in label)


def to_rttm(tl: Timeline, uri: str, *, exclusive: bool = False) -> str:
    """RTTM text for `tl.turns` (or `tl.exclusive`); zero-length turns are skipped."""
    if not uri or any(ch.isspace() for ch in uri):
        raise ValueError("RTTM uri must be non-empty and contain no whitespace")
    lines = []
    for t in sorted(tl.exclusive if exclusive else tl.turns, key=lambda t: (t.start, t.end, t.speaker)):
        if t.dur <= 0:
            continue
        conf = f"{t.conf:.3f}" if math.isfinite(t.conf) else "<NA>"
        lines.append(f"SPEAKER {uri} 1 {t.start:.3f} {t.dur:.3f} <NA> <NA> {_encode(t.speaker)} {conf} <NA>")
    return "".join(line + "\n" for line in lines)


def from_rttm(text: str, uri: str | None = None, **rule) -> Timeline:
    """Parse SPEAKER lines of one recording (select it with `uri` if the text has several)."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if not parts or parts[0] != "SPEAKER":
            continue
        if len(parts) < 8:
            raise ValueError("malformed RTTM SPEAKER line")
        rows.append(parts)
    uris = list(dict.fromkeys(p[1] for p in rows))
    if uri is None:
        if len(uris) > 1:
            raise ValueError(f"RTTM holds {len(uris)} recordings; pass uri=")
        uri = uris[0] if uris else None
    turns = []
    for p in rows:
        if p[1] != uri:
            continue
        start, dur = float(p[3]), float(p[4])
        try:
            conf = float(p[8]) if len(p) > 8 else 1.0
        except ValueError:
            conf = 1.0
        turns.append(Turn.from_label(start, start + dur, unquote(p[7]), conf))
    return Timeline.from_turns(turns, {"uri": uri, "source": "rttm"}, **rule)


# ─── JSON ───────────────────────────────────────────────────────────────────

def _turn_obj(t: Turn) -> dict:
    return {"start": t.start, "end": t.end, "speaker": t.speaker, "kind": t.kind,
            "conf": t.conf if math.isfinite(t.conf) else None, "alts": list(t.alts)}


def _obj_turn(o: dict) -> Turn:
    conf = o.get("conf")
    return Turn(float(o["start"]), float(o["end"]), o["speaker"], o["kind"],
                float("nan") if conf is None else float(conf), tuple(o.get("alts") or ()))


def _jsonable(o):
    if hasattr(o, "tolist"):  # numpy scalars and arrays in meta
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def to_json(tl: Timeline, *, indent: int | None = 1) -> str:
    return json.dumps({"schema": SCHEMA, "type": "speaker-timeline", "meta": tl.meta,
                       "turns": [_turn_obj(t) for t in tl.turns],
                       "exclusive": [_turn_obj(t) for t in tl.exclusive]},
                      ensure_ascii=False, indent=indent, allow_nan=False, default=_jsonable)


def from_json(text: str) -> Timeline:
    obj = json.loads(text)
    if obj.get("schema") != SCHEMA or obj.get("type") != "speaker-timeline":
        raise ValueError("not a schema-1 speaker timeline")
    return Timeline([_obj_turn(o) for o in obj["turns"]],
                    [_obj_turn(o) for o in obj["exclusive"]], dict(obj.get("meta") or {}))
