"""Resumable HTTP downloads, checksums and dataset manifests. Stdlib only.

A download writes `<dest>.part`, resumes it with an HTTP Range request, and is
renamed to `<dest>` only once the byte count matches Content-Length (and the
pinned sha256, when one is given). Re-running a prepare step is therefore
cheap: finished files are skipped, partial ones continue where they stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20
USER_AGENT = "meeting-documenter-bench/1 (+https://github.com)"


@dataclass
class Fetched:
    url: str
    path: Path
    size: int
    sha256: str
    skipped: bool  # already complete before this call


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_size(url: str, timeout: float) -> int | None:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        n = r.headers.get("Content-Length")
        return int(n) if n else None


def fetch(url: str, dest: Path, sha256: str | None = None, retries: int = 5,
          timeout: float = 60.0, log=None) -> Fetched:
    """Download `url` to `dest`, resuming `dest.part`. Verifies size and, if given, sha256."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        digest = sha256_file(dest)
        if sha256 and digest != sha256:
            raise RuntimeError(f"{dest.name}: sha256 {digest[:12]} != pinned {sha256[:12]}")
        return Fetched(url, dest, dest.stat().st_size, digest, True)

    part = dest.with_name(dest.name + ".part")
    total = _remote_size(url, timeout)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total is not None and have >= total:
            break
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if have and r.status != 206:      # server ignored Range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f, CHUNK)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if log:
                log(f"{dest.name}: attempt {attempt}/{retries} failed ({e}); resuming")
            time.sleep(min(30, 2 ** attempt))
    else:
        raise RuntimeError(f"{dest.name}: download failed after {retries} attempts")

    size = part.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(f"{dest.name}: got {size} bytes, expected {total}")
    digest = sha256_file(part)
    if sha256 and digest != sha256:
        part.unlink()
        raise RuntimeError(f"{dest.name}: sha256 {digest[:12]} != pinned {sha256[:12]}; removed")
    os.replace(part, dest)
    return Fetched(url, dest, size, digest, False)


def fetch_many(jobs: list[tuple[str, Path, str | None]], workers: int = 4,
               log=None) -> list[Fetched]:
    """Parallel `fetch` over (url, dest, sha256|None) jobs; raises after all finish if any failed."""
    out: list[Fetched] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, u, d, s, log=log): d for u, d, s in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
                out.append(r)
                if log:
                    log(f"[{i}/{len(jobs)}] {r.path.name} {'ok (cached)' if r.skipped else 'downloaded'}"
                        f" {r.size / 1e6:.1f} MB")
            except Exception as e:  # noqa: BLE001 - report every failure, then raise
                errors.append(f"{futs[fut].name}: {e}")
    if errors:
        raise RuntimeError("downloads failed:\n  " + "\n  ".join(errors))
    return sorted(out, key=lambda r: str(r.path))


def disk_usage(root: Path) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_checksums(path: Path) -> dict[str, str]:
    """`sha256  relative/path` lines (sha256sum format) → {relative path: sha256}."""
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, name = line.split(None, 1)
            out[name.strip()] = digest
    return out


def stderr_log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
