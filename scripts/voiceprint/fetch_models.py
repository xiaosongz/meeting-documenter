#!/usr/bin/env python3
"""Idempotently vendor voiceprint model code + weights into VOICEPRINT_MODEL_DIR.

    python scripts/voiceprint/fetch_models.py            # all backends
    python scripts/voiceprint/fetch_models.py redimnet2 eres2netv2

Pins (commit SHA, sha256) live in embedders.py. Existing files whose sha256
matches are skipped; downloads go to *.part and are renamed only after the
hash checks out. Stdlib only (no torch needed to fetch).
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voiceprint.embedders import (  # noqa: E402
    ALL_NAMES, ONNX_SPECS, REDIMNET2_REPO, REDIMNET2_SHA, TORCH_SPECS, model_dir, sha256_file,
)
from voiceprint.vad import SILERO_V5  # noqa: E402  (speech VAD for clip gating; pin lives in vad.py)

FETCH_NAMES = ALL_NAMES + [SILERO_V5.name]

# Inference-only subset of the upstream repo (training code + images skipped).
_VENDOR_KEEP = ("redimnet2/", "LICENSE", "README.md", "hubconf.py", "requirements.txt")


def _download(url: str, dest: Path, sha256: str) -> None:
    if dest.is_file() and sha256_file(dest) == sha256:
        print(f"ok      {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"fetch   {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = sha256_file(part)
    if got != sha256:
        part.unlink()
        raise RuntimeError(f"sha256 mismatch for {url}: got {got}, pinned {sha256}")
    part.replace(dest)
    print(f"saved   {dest}")


def _vendor_redimnet2_code(code_dir: Path) -> None:
    marker = code_dir / "VENDOR.json"
    if (code_dir / "redimnet2" / "__init__.py").is_file() and marker.is_file():
        print(f"ok      redimnet2 code @ {REDIMNET2_SHA[:12]}")
        return
    url = f"https://codeload.github.com/{REDIMNET2_REPO}/tar.gz/{REDIMNET2_SHA}"
    print(f"fetch   {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    code_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for m in tf.getmembers():
            rel = m.name.split("/", 1)[1] if "/" in m.name else ""
            if not rel or not rel.startswith(_VENDOR_KEEP) or not (m.isfile() or m.isdir()):
                continue
            m.name = rel
            tf.extract(m, code_dir, filter="data")
    marker.write_text(json.dumps(
        {"repo": REDIMNET2_REPO, "commit": REDIMNET2_SHA, "source": url,
         "kept": list(_VENDOR_KEEP)}, indent=2) + "\n")
    print(f"saved   {code_dir}")


def fetch(names: list[str]) -> None:
    for n in names:
        if n in TORCH_SPECS:
            s = TORCH_SPECS[n]
            _vendor_redimnet2_code(s.code_dir)
            _download(s.url, s.weight_path, s.sha256)
        elif n in ONNX_SPECS:
            s = ONNX_SPECS[n]
            _download(s.url, s.path, s.sha256)
        elif n == SILERO_V5.name:
            _download(SILERO_V5.url, SILERO_V5.path, SILERO_V5.sha256)
        else:
            raise SystemExit(f"unknown model {n!r}; known: {FETCH_NAMES}")


if __name__ == "__main__":
    targets = sys.argv[1:] or FETCH_NAMES
    print(f"VOICEPRINT_MODEL_DIR={model_dir()}")
    fetch(targets)
