"""AMI meeting corpus: BUT diarization setup (only_words RTTMs) + official-mirror audio.

    python -m bench.datasets.ami [--splits test dev] [--workers 4]

Layout under $SPEAKER_BENCH_DIR/ami/:
    setup-<commit12>/            BUT AMI-diarization-setup at SETUP_COMMIT (lists, RTTMs, UEMs)
    audio/<meeting>.<stream>.wav Array1-01 (SDM, far-field; test input) and Mix-Headset
                                 (close-talk mix; enrollment source)
and $SPEAKER_BENCH_DIR/manifest-ami.json (files, sizes, sha256, speakers, recurrence check).

AMI participant IDs (e.g. FEE013) are corpus-global, so a scenario series such as
ES2004a–d shares its four participants; `verify_recurrence` checks this from the RTTMs.
Licences: AMI audio and annotations CC BY 4.0; BUT setup Apache-2.0.
"""

from __future__ import annotations

import argparse
import io
import re
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from ..paths import dataset_dir, manifest_path
from ..rttm import read_rttm, read_uem
from .download import (USER_AGENT, disk_usage, fetch_many, load_checksums, stderr_log,
                       write_manifest)

SETUP_REPO = "BUTSpeechFIT/AMI-diarization-setup"
SETUP_COMMIT = "2509d8933721023fab4def2618aabd5c28eb82e9"
AUDIO_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
STREAMS = {"sdm": "Array1-01", "headset": "Mix-Headset"}
SPLITS = ("train", "dev", "test")
CHECKSUMS = Path(__file__).with_name("checksums") / "ami.sha256"
LICENCES = {
    "audio+annotations": "CC BY 4.0 (AMI Meeting Corpus, https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)",
    "diarization setup": f"Apache-2.0 ({SETUP_REPO}@{SETUP_COMMIT[:12]})",
}


def root() -> Path:
    return dataset_dir("ami")


def setup_dir() -> Path:
    return root() / f"setup-{SETUP_COMMIT[:12]}"


def audio_path(meeting: str, stream: str) -> Path:
    return root() / "audio" / f"{meeting}.{STREAMS[stream]}.wav"


def audio_url(meeting: str, stream: str) -> str:
    return f"{AUDIO_MIRROR}/{meeting}/audio/{meeting}.{STREAMS[stream]}.wav"


def series_of(meeting: str) -> str:
    """ES2004a → ES2004. Meetings without a session letter (IB4001) are their own series."""
    m = re.fullmatch(r"([A-Z]{2}\d{4})[a-d]", meeting)
    return m.group(1) if m else meeting


def assign_series(speakers: dict[str, list[str]]) -> dict[str, str]:
    """Series per meeting. Lettered meetings use `series_of`. Unlettered ones (the dev
    split's IB40xx) are grouped with the other unlettered meetings that have exactly the
    same participants (IB4001 + IB4002 → "IB4001+IB4002"); a unique set stays alone."""
    groups: dict[frozenset, list[str]] = defaultdict(list)
    out = {}
    for m in sorted(speakers):
        s = series_of(m)
        if s != m:
            out[m] = s
        else:
            groups[frozenset(speakers[m])].append(m)
    for ms in groups.values():
        out.update({m: "+".join(ms) for m in ms})
    return out


def fetch_setup(log=stderr_log) -> Path:
    d = setup_dir()
    if (d / "lists" / "test.meetings.txt").is_file():
        return d
    url = f"https://github.com/{SETUP_REPO}/archive/{SETUP_COMMIT}.tar.gz"
    log(f"fetching BUT setup {SETUP_COMMIT[:12]}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    tmp = d.with_name(d.name + ".tmp")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(tmp, filter="data")
    (next(tmp.iterdir())).rename(d)
    tmp.rmdir()
    return d


def meetings(split: str) -> list[str]:
    return (setup_dir() / "lists" / f"{split}.meetings.txt").read_text().split()


def rttm_path(meeting: str, split: str) -> Path:
    return setup_dir() / "only_words" / "rttms" / split / f"{meeting}.rttm"


def uem_path(meeting: str, split: str) -> Path:
    return setup_dir() / "uems" / split / f"{meeting}.uem"


def verify_recurrence(split_meetings: dict[str, list[str]]) -> dict:
    """Per split: speakers per meeting, and whether every speaker of each meeting also
    speaks in at least one other meeting of the split (needed for cross-session enrollment)."""
    report = {}
    for split, ms in split_meetings.items():
        spk = {m: sorted({s.speaker for s in read_rttm(rttm_path(m, split))[m]}) for m in ms}
        where: dict[str, list[str]] = defaultdict(list)
        for m, ss in spk.items():
            for s in ss:
                where[s].append(m)
        missing = {m: [s for s in ss if len(where[s]) < 2] for m, ss in spk.items()}
        report[split] = {
            "speakers_per_meeting": spk,
            "meetings_per_speaker": {s: sorted(v) for s, v in sorted(where.items())},
            "speakers_without_other_session": {m: v for m, v in missing.items() if v},
            "all_recur": not any(missing.values()),
        }
    return report


def prepare(splits=("test", "dev"), streams=("sdm", "headset"), workers: int = 4,
            log=stderr_log) -> dict:
    t0 = time.time()
    fetch_setup(log)
    pinned = load_checksums(CHECKSUMS)
    split_meetings = {s: meetings(s) for s in splits}
    jobs = []
    for split, ms in split_meetings.items():
        for m in ms:
            for st in streams:
                p = audio_path(m, st)
                jobs.append((audio_url(m, st), p, pinned.get(f"audio/{p.name}")))
    log(f"AMI: {len(jobs)} audio files for splits {list(splits)}")
    fetched = {r.path.name: r for r in fetch_many(jobs, workers=workers, log=log)}

    entries = {}
    for split, ms in split_meetings.items():
        speakers = {m: sorted({s.speaker for s in read_rttm(rttm_path(m, split))[m]}) for m in ms}
        series = assign_series(speakers)
        for m in ms:
            uem = read_uem(uem_path(m, split)).get(m, [])
            entries[m] = {
                "split": split,
                "series": series[m],
                "rttm": str(rttm_path(m, split).relative_to(root())),
                "uem": str(uem_path(m, split).relative_to(root())),
                "duration_s": round(sum(b - a for a, b in uem), 3),
                "speakers": speakers[m],
                "audio": {st: {"path": str(audio_path(m, st).relative_to(root())),
                               "size": fetched[audio_path(m, st).name].size,
                               "sha256": fetched[audio_path(m, st).name].sha256}
                          for st in streams},
            }
    manifest = {
        "dataset": "ami",
        "root": "ami",
        "setup": {"repo": SETUP_REPO, "commit": SETUP_COMMIT, "references": "only_words"},
        "audio_mirror": AUDIO_MIRROR,
        "streams": {k: STREAMS[k] for k in streams},
        "licences": LICENCES,
        "meetings": entries,
        "recurrence": verify_recurrence(split_meetings),
        "disk_bytes": disk_usage(root()),
        "prepared_s": round(time.time() - t0, 1),
    }
    write_manifest(manifest_path("ami"), manifest)
    unpinned = [k for k in (f"audio/{j[1].name}" for j in jobs) if k not in pinned]
    if unpinned:
        log(f"note: {len(unpinned)} files have no pinned sha256 in {CHECKSUMS.name}; "
            "run with --write-checksums to pin them")
    return manifest


def write_checksums(manifest: dict) -> None:
    lines = sorted(f"{a['sha256']}  {a['path']}"
                   for e in manifest["meetings"].values() for a in e["audio"].values())
    merged = load_checksums(CHECKSUMS)
    merged.update({ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in lines})
    CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUMS.write_text("# sha256 of AMI mirror audio (sha256sum format), relative to $SPEAKER_BENCH_DIR/ami\n"
                         + "".join(f"{v}  {k}\n" for k, v in sorted(merged.items())))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", nargs="+", default=["test", "dev"], choices=SPLITS)
    ap.add_argument("--streams", nargs="+", default=["sdm", "headset"], choices=list(STREAMS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write-checksums", action="store_true",
                    help="pin the downloaded files' sha256 into bench/datasets/checksums/ami.sha256")
    a = ap.parse_args(argv)
    man = prepare(a.splits, a.streams, a.workers)
    if a.write_checksums:
        write_checksums(man)
    rec = {s: r["all_recur"] for s, r in man["recurrence"].items()}
    stderr_log(f"AMI ready: {len(man['meetings'])} meetings, {man['disk_bytes'] / 1e9:.2f} GB, "
               f"speaker recurrence {rec} → {manifest_path('ami')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
