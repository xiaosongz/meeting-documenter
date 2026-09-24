"""Known-speaker tracking metrics: IER (primary), DER, JER, and frame-level breakdowns.

System labels are enrolled speaker IDs (same namespace as the reference) or abstentions:
`GUEST_*`, `UNKNOWN`, `UNCERTAIN(A|B)`. The rule being measured is "unknown is acceptable;
a wrong name is not", so every breakdown separates wrong names from abstentions.

Headline (pyannote.metrics):
    IER  identification error rate, no label permutation. Open-set mapping first: reference
         speakers outside the roster → `UNKNOWN`; hypothesis `GUEST_*` → `UNKNOWN`.
         `UNCERTAIN(...)` stays a label of its own, so it always counts as an error here
         (the abstain rate below reports it separately).
    DER  diarization error rate (optimal mapping; labels as given).
    JER  Jaccard error rate (optimal mapping; no collar in pyannote's JER).

Collar convention: `collar` is NIST-style ±seconds around every reference boundary
(md-eval / dscore "0.25"). pyannote's `collar` is the total width, so it gets 2 × collar.
The frame breakdowns use the same scored region (UEM minus collars; overlap is kept).

Frame breakdown (10 ms frames, over each (frame, reference speaker) pair in the scored region):
    correct     the reference speaker's ID is among the hypothesis labels
    wrong_name  not correct, and the hypothesis names an enrolled ID that is not speaking
    abstain     neither, and the hypothesis has an abstain label
    missed      anything else (no hypothesis, or only other true speakers in overlap)
For reference speakers outside the roster (unenrolled), `correct` means "abstained" and
`wrong_name` is a false name. Tables: by reference-turn duration (<1, 1–3, 3–10, ≥10 s;
a turn is one speaker's speech with internal gaps ≤ `turn_gap`), overlap vs non-overlap,
and wrong-name seconds per (reference, hypothesis) speaker pair.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .rttm import Seg, to_annotation, to_timeline

FRAME = 0.01
BUCKETS = ((0.0, 1.0, "<1s"), (1.0, 3.0, "1-3s"), (3.0, 10.0, "3-10s"), (10.0, math.inf, ">=10s"))
OUTCOMES = ("correct", "wrong_name", "abstain", "missed")
UNKNOWN = "UNKNOWN"
_UNCERTAIN_RE = re.compile(r"^UNCERTAIN(\(.*\))?$")


def label_kind(label: str) -> str:
    """'name' | 'guest' | 'unknown' | 'uncertain'."""
    if label == UNKNOWN:
        return "unknown"
    if label.startswith("GUEST_"):
        return "guest"
    if _UNCERTAIN_RE.match(label):
        return "uncertain"
    return "name"


def is_name(label: str) -> bool:
    return label_kind(label) == "name"


# ─── helpers ────────────────────────────────────────────────────────────────

def _frames(t: float) -> int:
    return int(round(t / FRAME))


def merge_turns(segs: list[Seg], turn_gap: float) -> list[Seg]:
    """Per speaker, merge segments whose gap is ≤ turn_gap (overlapping ones always merge)."""
    out: list[Seg] = []
    by: dict[str, list[Seg]] = {}
    for s in segs:
        by.setdefault(s.speaker, []).append(s)
    for spk, ss in by.items():
        ss = sorted(ss, key=lambda s: s.start)
        cur = ss[0]
        for s in ss[1:]:
            if s.start - cur.end <= turn_gap:
                cur = Seg(cur.start, max(cur.end, s.end), spk)
            else:
                out.append(cur)
                cur = s
        out.append(cur)
    return sorted(out, key=lambda s: (s.start, s.speaker))


def scored_mask(n: int, ref: list[Seg], uem: list[tuple[float, float]] | None,
                collar: float) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in (uem or [(0.0, n * FRAME)]):
        m[max(0, _frames(a)):min(n, _frames(b))] = True
    if collar > 0:
        for s in ref:
            for t in (s.start, s.end):
                m[max(0, _frames(t - collar)):min(n, _frames(t + collar))] = False
    return m


def _activity(segs: list[Seg], labels: list[str], n: int) -> np.ndarray:
    idx = {l: i for i, l in enumerate(labels)}
    act = np.zeros((n, len(labels)), dtype=bool)
    for s in segs:
        act[max(0, _frames(s.start)):min(n, _frames(s.end)), idx[s.speaker]] = True
    return act


# ─── pyannote headline metrics ─────────────────────────────────────────────

def open_set_map(ref: list[Seg], hyp: list[Seg], roster: set[str] | None) -> tuple[list[Seg], list[Seg]]:
    """Reference speakers outside `roster` → UNKNOWN; hypothesis GUEST_* → UNKNOWN."""
    r = [Seg(s.start, s.end, s.speaker if roster is None or s.speaker in roster else UNKNOWN) for s in ref]
    h = [Seg(s.start, s.end, UNKNOWN if label_kind(s.speaker) == "guest" else s.speaker) for s in hyp]
    return r, h


def headline(ref: list[Seg], hyp: list[Seg], uem: list[tuple[float, float]] | None,
             roster: set[str] | None, collar: float, uri: str = "uri") -> dict:
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    from pyannote.metrics.identification import IdentificationErrorRate

    u = to_timeline(uem, uri) if uem else None
    width = 2 * collar
    r_os, h_os = open_set_map(ref, hyp, roster)
    ier = IdentificationErrorRate(collar=width)(to_annotation(r_os, uri), to_annotation(h_os, uri),
                                                uem=u, detailed=True)
    der = DiarizationErrorRate(collar=width)(to_annotation(ref, uri), to_annotation(hyp, uri),
                                             uem=u, detailed=True)
    out = {"ier": ier["identification error rate"], "der": der["diarization error rate"],
           "ier_components": {k: ier[k] for k in ("total", "correct", "confusion",
                                                   "missed detection", "false alarm")},
           "der_components": {k: der[k] for k in ("total", "correct", "confusion",
                                                   "missed detection", "false alarm")}}
    if collar == 0:
        out["jer"] = JaccardErrorRate()(to_annotation(ref, uri), to_annotation(hyp, uri), uem=u)
    return out


# ─── frame breakdowns ──────────────────────────────────────────────────────

@dataclass
class Breakdown:
    """Seconds per outcome, split by bucket / overlap / enrolment."""
    by_bucket: dict[str, dict[str, float]] = field(default_factory=dict)
    by_overlap: dict[str, dict[str, float]] = field(default_factory=dict)
    enrolled: dict[str, float] = field(default_factory=dict)
    unenrolled: dict[str, float] = field(default_factory=dict)
    pairs: dict[tuple[str, str], float] = field(default_factory=dict)
    named_false_alarm_s: float = 0.0
    hyp_speech_s: float = 0.0
    hyp_abstain_s: float = 0.0

    def add(self, other: "Breakdown") -> None:
        for mine, theirs in ((self.by_bucket, other.by_bucket), (self.by_overlap, other.by_overlap)):
            for k, d in theirs.items():
                acc = mine.setdefault(k, dict.fromkeys(OUTCOMES, 0.0))
                for o, v in d.items():
                    acc[o] += v
        for mine, theirs in ((self.enrolled, other.enrolled), (self.unenrolled, other.unenrolled)):
            for o, v in theirs.items():
                mine[o] = mine.get(o, 0.0) + v
        for k, v in other.pairs.items():
            self.pairs[k] = self.pairs.get(k, 0.0) + v
        self.named_false_alarm_s += other.named_false_alarm_s
        self.hyp_speech_s += other.hyp_speech_s
        self.hyp_abstain_s += other.hyp_abstain_s


def breakdown(ref: list[Seg], hyp: list[Seg], uem: list[tuple[float, float]] | None,
              roster: set[str] | None, collar: float = 0.0, turn_gap: float = 0.5) -> Breakdown:
    end = max([s.end for s in ref + hyp] + [b for _, b in (uem or [])] + [0.0])
    n = _frames(end) + 1
    scored = scored_mask(n, ref, uem, collar)
    rl = sorted({s.speaker for s in ref})
    hl = sorted({s.speaker for s in hyp})
    R = _activity(ref, rl, n) & scored[:, None]
    H = _activity(hyp, hl, n) & scored[:, None]
    n_ref = R.sum(1)
    overlap = n_ref >= 2

    kinds = np.array([label_kind(l) for l in hl]) if hl else np.array([], dtype=str)
    name_cols = np.flatnonzero(kinds == "name")
    abstain_any = H[:, np.flatnonzero(kinds != "name")].any(1) if hl else np.zeros(n, bool)
    hcol = {l: i for i, l in enumerate(hl)}
    # a hypothesis name is "wrong" at a frame when that ID is not a reference speaker there
    wrong = np.zeros((n, len(name_cols)), dtype=bool)
    for j, c in enumerate(name_cols):
        lab = hl[c]
        speaking = R[:, rl.index(lab)] if lab in rl else np.zeros(n, bool)
        wrong[:, j] = H[:, c] & ~speaking
    wrong_any = wrong.any(1)

    # turn-duration bucket of every (frame, ref speaker)
    bucket_of = np.full((n, len(rl)), -1, dtype=np.int8)
    for t in merge_turns(ref, turn_gap):
        b = next(i for i, (lo, hi, _) in enumerate(BUCKETS) if lo <= t.dur < hi)
        bucket_of[max(0, _frames(t.start)):min(n, _frames(t.end)), rl.index(t.speaker)] = b

    bd = Breakdown()
    for i, spk in enumerate(rl):
        active = R[:, i]
        if not active.any():
            continue
        enrolled = roster is None or spk in roster
        if enrolled and spk in hcol:
            correct = active & H[:, hcol[spk]]
        elif enrolled:
            correct = np.zeros(n, bool)
        else:
            correct = active & abstain_any & ~wrong_any
        wrong_name = active & ~correct & wrong_any
        abstain = active & ~correct & ~wrong_name & abstain_any
        missed = active & ~correct & ~wrong_name & ~abstain
        masks = dict(zip(OUTCOMES, (correct, wrong_name, abstain, missed)))
        tgt = bd.enrolled if enrolled else bd.unenrolled
        for o, m in masks.items():
            tgt[o] = tgt.get(o, 0.0) + m.sum() * FRAME
        if enrolled:
            for bi, (_, _, name) in enumerate(BUCKETS):
                sel = bucket_of[:, i] == bi
                acc = bd.by_bucket.setdefault(name, dict.fromkeys(OUTCOMES, 0.0))
                for o, m in masks.items():
                    acc[o] += (m & sel).sum() * FRAME
            for key, sel in (("overlap", overlap), ("non_overlap", ~overlap)):
                acc = bd.by_overlap.setdefault(key, dict.fromkeys(OUTCOMES, 0.0))
                for o, m in masks.items():
                    acc[o] += (m & sel).sum() * FRAME
        for j, c in enumerate(name_cols):
            s = (wrong_name & wrong[:, j]).sum() * FRAME
            if s:
                key = (spk, hl[c])
                bd.pairs[key] = bd.pairs.get(key, 0.0) + s
    nonspeech = scored & (n_ref == 0)
    bd.named_false_alarm_s = float((nonspeech & H[:, name_cols].any(1)).sum() * FRAME) if len(name_cols) else 0.0
    hyp_any = H.any(1)
    bd.hyp_speech_s = float(hyp_any.sum() * FRAME)
    bd.hyp_abstain_s = float((hyp_any & ~H[:, name_cols].any(1)).sum() * FRAME) if hl else 0.0
    return bd


def rates(d: dict[str, float]) -> dict[str, float]:
    tot = sum(d.get(o, 0.0) for o in OUTCOMES)
    out = {"seconds": round(tot, 2)}
    for o in OUTCOMES:
        out[o] = round(d.get(o, 0.0) / tot, 4) if tot else None
    return out


def summarize(bd: Breakdown, top_pairs: int = 10) -> dict:
    unen = bd.unenrolled
    unen_tot = sum(unen.values())
    return {
        "enrolled": rates(bd.enrolled),
        "unenrolled": rates(unen) if unen_tot else None,
        "false_name_rate": round(unen.get("wrong_name", 0.0) / unen_tot, 4) if unen_tot else None,
        "abstain_rate": rates(bd.enrolled)["abstain"],
        "hyp_abstain_fraction": round(bd.hyp_abstain_s / bd.hyp_speech_s, 4) if bd.hyp_speech_s else None,
        "named_false_alarm_s": round(bd.named_false_alarm_s, 2),
        "by_bucket": {name: rates(bd.by_bucket.get(name, {})) for _, _, name in BUCKETS},
        "by_overlap": {k: rates(bd.by_overlap.get(k, {})) for k in ("non_overlap", "overlap")},
        "pair_confusion_s": [
            {"ref": r, "hyp": h, "seconds": round(s, 2)}
            for (r, h), s in sorted(bd.pairs.items(), key=lambda kv: -kv[1])[:top_pairs]],
    }


def frame_ier(ref: list[Seg], hyp: list[Seg], uem: list[tuple[float, float]] | None,
              roster: set[str] | None, collar: float = 0.0) -> tuple[float, float]:
    """Fast frame-level IER with the same open-set mapping as `headline` (for parameter
    sweeps). Returns (errors_s, total_s); IER = errors / total. Per frame with r reference
    and h hypothesis labels and c shared: error = r − c + max(0, h − r)."""
    r_os, h_os = open_set_map(ref, hyp, roster)
    end = max([s.end for s in ref + hyp] + [b for _, b in (uem or [])] + [0.0])
    n = _frames(end) + 1
    scored = scored_mask(n, ref, uem, collar)
    labels = sorted({s.speaker for s in r_os + h_os})
    R = _activity(r_os, labels, n) & scored[:, None]
    H = _activity(h_os, labels, n) & scored[:, None]
    r, h, c = R.sum(1), H.sum(1), (R & H).sum(1)
    return float((r - c + np.maximum(0, h - r)).sum() * FRAME), float(r.sum() * FRAME)


def wder(pairs: list[tuple[str, str]]) -> float | None:
    """Word diarization error rate over (reference speaker, hypothesis speaker) per word,
    labels already in one namespace (no permutation). Abstentions count as errors."""
    if not pairs:
        return None
    return sum(r != h for r, h in pairs) / len(pairs)
