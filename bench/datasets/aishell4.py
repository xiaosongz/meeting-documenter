"""AISHELL-4 evaluation set (OpenSLR 111, `test.tar.gz`): channel 0 of the 8-mic array as mono.

    python -m bench.datasets.aishell4 [--workers 4] [--keep-multichannel]

Layout under $SPEAKER_BENCH_DIR/aishell4/:
    download/test.tar.gz          the OpenSLR archive (resumable; sha256 pinned in checksums/)
    test/TextGrid/<session>.rttm  official references (kept from the archive)
    ch0/<session>.wav             channel 0, 16 kHz 16-bit mono (the benchmark input)
and $SPEAKER_BENCH_DIR/manifest-aishell4.json.

Speaker IDs (FINDING, see bench/README.md): labels such as `002-F` are local to a
*group* of sessions, not global. Sessions are named <room>_R<group>S<session>C<n>;
the four sessions of one group (e.g. L_R003S01..S04) carry the same label set, while
the same label in another group is a different person (it can even change gender).
So the cross-session protocol enrolls from the other sessions of the same group
(`group_of`), and `bench.datasets.verify_ids` checks acoustically that a label is one
voice across its group's sessions.
Licence: CC BY-SA 4.0 (https://www.openslr.org/111/).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..paths import dataset_dir, manifest_path
from ..rttm import read_rttm
from .download import disk_usage, fetch, load_checksums, sha256_file, stderr_log, write_manifest

URL = "https://www.openslr.org/resources/111/test.tar.gz"
MIRRORS = ("https://openslr.elda.org/resources/111/test.tar.gz",
           "https://openslr.trmal.net/resources/111/test.tar.gz")
CHECKSUMS = Path(__file__).with_name("checksums") / "aishell4.sha256"
LICENCE = "CC BY-SA 4.0 (AISHELL-4, OpenSLR 111, https://www.openslr.org/111/)"
SESSION_RE = re.compile(r"([LMS]_R\d{3})S\d{2}C\d{2}")


def root() -> Path:
    return dataset_dir("aishell4")


def archive_path() -> Path:
    return root() / "download" / "test.tar.gz"


def wav_path(session: str) -> Path:
    return root() / "ch0" / f"{session}.wav"


def rttm_path(session: str) -> Path:
    return root() / "test" / "TextGrid" / f"{session}.rttm"


def group_of(session: str) -> str:
    """L_R003S02C02 → L_R003 (room + participant group)."""
    m = SESSION_RE.fullmatch(session)
    if not m:
        raise ValueError(f"unexpected AISHELL-4 session name {session!r}")
    return m.group(1)


def sessions() -> list[str]:
    return sorted(p.stem for p in (root() / "test" / "TextGrid").glob("*.rttm"))


def _extract(log) -> None:
    if (root() / "test" / "TextGrid").is_dir() and len(list((root() / "test" / "wav").glob("*.flac"))) == 20:
        return
    log("extracting test.tar.gz")
    with tarfile.open(archive_path(), "r:gz") as tf:
        tf.extractall(root(), filter="data")


def _ffmpeg_ch0(src: Path, dst: Path) -> None:
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-af", "pan=mono|c0=c0", "-ar", "16000", "-sample_fmt", "s16", str(tmp)], check=True)
    tmp.rename(dst)


def _duration_s(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return info.frames / info.samplerate


def prepare(workers: int = 4, keep_multichannel: bool = True, log=stderr_log) -> dict:
    t0 = time.time()
    pinned = load_checksums(CHECKSUMS).get("download/test.tar.gz")
    got = fetch(URL, archive_path(), pinned, log=log)
    _extract(log)
    ss = sessions()
    flacs = {s: root() / "test" / "wav" / f"{s}.flac" for s in ss}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda s: _ffmpeg_ch0(flacs[s], wav_path(s)), ss))
    if not keep_multichannel:
        for f in flacs.values():
            f.unlink(missing_ok=True)

    entries = {}
    for s in ss:
        segs = read_rttm(rttm_path(s))
        if list(segs) != [s]:
            raise ValueError(f"{rttm_path(s).name}: expected uri {s}, got {list(segs)}")
        entries[s] = {
            "split": "test",
            "series": group_of(s),
            "rttm": str(rttm_path(s).relative_to(root())),
            "uem": None,  # whole recording
            "duration_s": round(_duration_s(wav_path(s)), 3),
            "speakers": sorted({x.speaker for x in segs[s]}),
            "audio": {"ch0": {"path": str(wav_path(s).relative_to(root())),
                              "size": wav_path(s).stat().st_size,
                              "sha256": sha256_file(wav_path(s))}},
        }
    groups: dict[str, list[str]] = {}
    for s in ss:
        groups.setdefault(group_of(s), []).append(s)
    label_sets = {g: sorted({tuple(entries[s]["speakers"]) for s in v}) for g, v in groups.items()}
    manifest = {
        "dataset": "aishell4",
        "root": "aishell4",
        "archive": {"url": URL, "mirrors": list(MIRRORS), "size": got.size, "sha256": got.sha256},
        "licences": {"audio+annotations": LICENCE},
        "channel": "array channel 0, mono 16 kHz",
        "meetings": entries,
        "speaker_ids": {
            "scope": "group",
            "groups": groups,
            "same_label_set_within_group": {g: len(v) == 1 for g, v in label_sets.items()},
            "note": "labels are local to a session group (<room>_R<group>); see verify_ids for the "
                    "acoustic consistency check",
        },
        "disk_bytes": disk_usage(root()),
        "prepared_s": round(time.time() - t0, 1),
    }
    write_manifest(manifest_path("aishell4"), manifest)
    if not pinned:
        log(f"note: archive sha256 {got.sha256[:12]} not pinned; run with --write-checksums")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--drop-multichannel", action="store_true",
                    help="delete the extracted 8-channel FLACs after deriving channel 0")
    ap.add_argument("--write-checksums", action="store_true")
    a = ap.parse_args(argv)
    man = prepare(a.workers, keep_multichannel=not a.drop_multichannel)
    if a.write_checksums:
        CHECKSUMS.parent.mkdir(parents=True, exist_ok=True)
        CHECKSUMS.write_text("# sha256 of the OpenSLR 111 archive, relative to $SPEAKER_BENCH_DIR/aishell4\n"
                             f"{man['archive']['sha256']}  download/test.tar.gz\n")
    stderr_log(f"AISHELL-4 ready: {len(man['meetings'])} sessions, {man['disk_bytes'] / 1e9:.2f} GB "
               f"→ {manifest_path('aishell4')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
