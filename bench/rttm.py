"""Minimal RTTM / UEM I/O for the benchmark. Stdlib only (pyannote conversion is lazy).

RTTM line: SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <speaker> <NA> <NA>
UEM line:  <uri> <chan> <start> <end>
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Seg:
    start: float
    end: float
    speaker: str

    @property
    def dur(self) -> float:
        return self.end - self.start


def parse_rttm(text: str) -> dict[str, list[Seg]]:
    out: dict[str, list[Seg]] = defaultdict(list)
    for n, line in enumerate(text.splitlines(), 1):
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        if f[0] != "SPEAKER" or len(f) < 8:
            raise ValueError(f"RTTM line {n}: expected 'SPEAKER uri chan start dur ... speaker'")
        start, dur = float(f[3]), float(f[4])
        if dur < 0:
            raise ValueError(f"RTTM line {n}: negative duration")
        if dur > 0:
            out[f[1]].append(Seg(start, start + dur, f[7]))
    for segs in out.values():
        segs.sort(key=lambda s: (s.start, s.end, s.speaker))
    return dict(out)


def read_rttm(path: str | Path) -> dict[str, list[Seg]]:
    return parse_rttm(Path(path).read_text(encoding="utf-8"))


def format_rttm(uri: str, segs: list[Seg]) -> str:
    return "".join(f"SPEAKER {uri} 1 {s.start:.3f} {s.dur:.3f} <NA> <NA> {s.speaker} <NA> <NA>\n"
                   for s in sorted(segs, key=lambda s: (s.start, s.speaker)) if s.dur > 0)


def write_rttm(path: str | Path, uri: str, segs: list[Seg]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(format_rttm(uri, segs), encoding="utf-8")


def read_uem(path: str | Path) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= 4:
            out[f[0]].append((float(f[2]), float(f[3])))
    return dict(out)


def to_annotation(segs: list[Seg], uri: str = "uri"):
    from pyannote.core import Annotation, Segment

    ann = Annotation(uri=uri)
    for i, s in enumerate(segs):
        ann[Segment(s.start, s.end), i] = s.speaker
    return ann


def to_timeline(spans: list[tuple[float, float]], uri: str = "uri"):
    from pyannote.core import Segment, Timeline

    return Timeline([Segment(a, b) for a, b in spans], uri=uri)
