"""
Voiceprint paths and store-location guard. Stdlib only, so the CLI can parse
arguments and take its graceful no-op path without numpy/torch installed.

Environment (all optional; see .env.example):
    VOICEPRINT_DIR              enrollment store (biometric data)
                                default ${XDG_DATA_HOME:-~/.local/share}/meeting-documenter/voiceprints
    VOICEPRINT_MODEL_DIR        vendored model code, weights and caches
                                default ${XDG_CACHE_HOME:-~/.cache}/meeting-documenter/models
    VOICEPRINT_FORBIDDEN_ROOTS  extra directories the store must never be inside
                                (os.pathsep-separated, e.g. "/srv/notes:/mnt/shared")
    VAULT_PATH, MEETING_NOTES_DIR
                                notes locations from the existing .env; always forbidden

The store is also refused inside this clone and inside any directory that
contains `.obsidian/` (an Obsidian vault), so biometric vectors cannot land in
synced notes or be committed by accident.
"""

from __future__ import annotations

import os
from pathlib import Path

APP = "meeting-documenter"
SOURCES = ("self_check", "user_ear", "confirmed_meeting")
CHANNELS = ("phone", "zoom", "room", "unknown")
PERSON_CAP = 200          # max exemplars per (person, model)
# Hard cap (s) on any wave handed to an embedder, by device, and the ceiling no override may exceed.
DEVICE_CAP_S = {"cpu": 12.0, "mps": 60.0}
SAFETY_CEILING_S = 60.0
NOTES_ENV_VARS = ("VAULT_PATH", "MEETING_NOTES_DIR")
REPO_ROOT = Path(__file__).resolve().parents[2]


class VaultPathError(ValueError):
    """Raised when a biometric store path resolves inside notes or this clone."""


def _xdg(var: str, fallback: str) -> Path:
    base = os.environ.get(var)
    return Path(base).expanduser() if base else Path.home() / fallback


def default_store_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / APP / "voiceprints"


def default_model_dir() -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache") / APP / "models"


def model_dir() -> Path:
    raw = os.environ.get("VOICEPRINT_MODEL_DIR")
    return Path(raw).expanduser() if raw else default_model_dir()


def forbidden_roots() -> list[Path]:
    """This clone, notes roots from the environment, and VOICEPRINT_FORBIDDEN_ROOTS."""
    raw = [os.environ.get(v) for v in NOTES_ENV_VARS]
    raw += (os.environ.get("VOICEPRINT_FORBIDDEN_ROOTS") or "").split(os.pathsep)
    return [REPO_ROOT, *(Path(r).expanduser() for r in raw if r and r.strip())]


def _resolve(p: Path) -> Path:
    try:
        return p.expanduser().resolve()
    except OSError:
        return p.expanduser().absolute()


def assert_outside_vault(p: str | Path) -> None:
    """Refuse a store path inside this clone, a configured notes root, or an Obsidian vault."""
    p = _resolve(Path(p))
    for root in forbidden_roots():
        root_r = _resolve(root)
        if p == root_r or root_r in p.parents:
            raise VaultPathError(f"refusing biometric store inside a forbidden root ({root_r}): {p}")
    for anc in (p, *p.parents):
        if (anc / ".obsidian").is_dir():
            raise VaultPathError(f"refusing biometric store inside an Obsidian vault ({anc}): {p}")


def resolve_store_dir(path: str | Path | None = None) -> Path:
    """Store directory (argument > VOICEPRINT_DIR > XDG default), resolved and guarded."""
    raw = path or os.environ.get("VOICEPRINT_DIR") or default_store_dir()
    p = _resolve(Path(raw))
    assert_outside_vault(p)
    return p
