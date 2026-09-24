"""Cross-session enrollment protocol over a prepared dataset manifest.

For a test meeting M in series S (AMI scenario series ES2004a–d; AISHELL-4 session group
L_R003), each participant of S is enrolled only from the *other* meetings of S in which
they speak, using their reference-labelled non-overlapped regions from the enrollment
stream (AMI: headset mix; AISHELL-4: the same array channel 0). M itself is never used.

Rosters:
    closed      every participant of S with enrollment material (may include someone
                absent from M, e.g. AMI EN2002c)
    loo:<spk>   closed minus one speaker present in M; that speaker's speech is
                unenrolled and measures the open-set false-name rate
Cohort (AS-norm): speakers of the other series of the same split, never anyone in S.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import bench_dir, manifest_path
from .rttm import Seg, read_rttm, read_uem

# test / enrollment stream per dataset
STREAMS = {"ami": ("sdm", "headset"), "aishell4": ("ch0", "ch0")}


@dataclass(frozen=True)
class Meeting:
    id: str
    dataset: str
    split: str
    series: str
    speakers: tuple[str, ...]
    duration_s: float
    root: Path
    entry: dict

    def audio(self, stream: str) -> Path:
        return self.root / self.entry["audio"][stream]["path"]

    @property
    def test_audio(self) -> Path:
        return self.audio(STREAMS[self.dataset][0])

    @property
    def enroll_audio(self) -> Path:
        return self.audio(STREAMS[self.dataset][1])

    def reference(self) -> list[Seg]:
        return read_rttm(self.root / self.entry["rttm"])[self.id]

    def uem(self) -> list[tuple[float, float]] | None:
        if not self.entry.get("uem"):
            return None
        return read_uem(self.root / self.entry["uem"]).get(self.id)


def load_manifest(dataset: str) -> dict:
    p = manifest_path(dataset)
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing; run: python -m bench.datasets.{dataset}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_meetings(dataset: str, split: str | None = None, manifest: dict | None = None) -> list[Meeting]:
    man = manifest or load_manifest(dataset)
    root = bench_dir() / man["root"]
    out = [Meeting(mid, dataset, e["split"], e["series"], tuple(e["speakers"]), e["duration_s"], root, e)
           for mid, e in man["meetings"].items() if split is None or e["split"] == split]
    return sorted(out, key=lambda m: m.id)


def solo_regions(ref: list[Seg], speaker: str, guard: float = 0.0,
                 min_dur: float = 0.0) -> list[tuple[float, float]]:
    """The speaker's reference speech with every other speaker's speech (±guard) removed."""
    own = sorted((s.start, s.end) for s in ref if s.speaker == speaker)
    cuts = sorted((s.start - guard, s.end + guard) for s in ref if s.speaker != speaker)
    out = []
    for a, b in own:
        pieces = [(a, b)]
        for ca, cb in cuts:
            if cb <= a or ca >= b:
                continue
            nxt = []
            for pa, pb in pieces:
                if cb <= pa or ca >= pb:
                    nxt.append((pa, pb))
                    continue
                if ca > pa:
                    nxt.append((pa, ca))
                if cb < pb:
                    nxt.append((cb, pb))
            pieces = nxt
        out += [(pa, pb) for pa, pb in pieces if pb - pa >= min_dur]
    return out


@dataclass(frozen=True)
class EnrollSource:
    speaker: str
    meeting: str
    audio: Path
    spans: tuple[tuple[float, float], ...]

    @property
    def seconds(self) -> float:
        return sum(b - a for a, b in self.spans)


def enrollment(meeting: Meeting, pool: list[Meeting], guard: float = 0.0,
               min_dur: float = 0.0) -> dict[str, list[EnrollSource]]:
    """Enrollment sources for every participant of `meeting`'s series, excluding `meeting`."""
    series = [m for m in pool if m.series == meeting.series and m.id != meeting.id]
    participants = sorted({s for m in pool if m.series == meeting.series for s in m.speakers})
    out: dict[str, list[EnrollSource]] = {}
    for spk in participants:
        srcs = []
        for m in series:
            if spk in m.speakers:
                spans = solo_regions(m.reference(), spk, guard, min_dur)
                if spans:
                    srcs.append(EnrollSource(spk, m.id, m.enroll_audio, tuple(spans)))
        if srcs:
            out[spk] = srcs
    return out


def rosters(meeting: Meeting, enrolled: list[str]) -> dict[str, set[str]]:
    closed = set(enrolled)
    out = {"closed": closed}
    for spk in meeting.speakers:
        if spk in closed:
            out[f"loo:{spk}"] = closed - {spk}
    return out


def cohort_speakers(meeting: Meeting, pool: list[Meeting]) -> dict[str, list[Meeting]]:
    """Speakers from other series of the pool (never anyone in this meeting's series)."""
    in_series = {s for m in pool if m.series == meeting.series for s in m.speakers}
    out: dict[str, list[Meeting]] = {}
    for m in pool:
        if m.series == meeting.series:
            continue
        for s in m.speakers:
            if s not in in_series:
                out.setdefault(f"{m.series}/{s}", []).append(m)
    return out
