#!/usr/bin/env python3
"""
PII / private-data guard for this public repository (CI and local pre-commit hook).

Rules (always on):
    path       absolute home or mounted-volume paths (macOS user folders, /Volumes,
               Linux home directories). Use ~, $HOME or /path/to/... in docs.
    email      e-mail addresses, except reserved example domains (example.com/.org/.net,
               *.example, *.invalid, *.test, localhost).
    file       forbidden files: voiceprint stores (*.sqlite*), calibration.json,
               *.utterances.json sidecars, embeddings (*.npy/*.npz), model weights
               (*.pt/*.onnx), audio, .env files and the private registries.
Rule (local only, when a registry is available):
    roster     any canonical_name / alias / email from the private
               references/KNOWN_SPEAKERS.yaml (or --roster PATH, or
               $MEETING_DOCUMENTER_KNOWN_SPEAKERS). Case-insensitive, whole-word for
               Latin-script terms; terms shorter than --min-term-len, and bare
               placeholder first names (PLACEHOLDER_NAMES), are skipped.

Modes:
    --tracked          every tracked file (default)
    --staged           staged additions and file names (pre-commit hook)
    --log RANGE        commit messages, added lines and file names of `git log RANGE`
    PATH ...           the given files

Findings are printed with the match masked, so CI logs never repeat the value.
Exit status: 0 clean, 1 findings, 2 usage/git error.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROSTER = ROOT / "references" / "KNOWN_SPEAKERS.yaml"

PATH_RE = re.compile(r"/(?:Users|Volumes|home)/[A-Za-z0-9._-]+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})")
ALLOWED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "localhost")
ALLOWED_EMAIL_TLDS = (".example", ".invalid", ".test", ".localhost")

FORBIDDEN_FILES = (
    "*.sqlite", "*.sqlite3", "*.sqlite-journal", "*.sqlite-wal", "calibration.json", "*.utterances.json",
    "*.npy", "*.npz", "*.pt", "*.pth", "*.onnx", "*.ckpt", "*.safetensors",
    "*.m4a", "*.mp3", "*.wav", "*.ogg", "*.oga", "*.opus", "*.flac", "*.aac", "*.aiff", "*.aif",
    "*.webm", "*.wma", "*.caf",
    ".env", ".env.*", "references/KNOWN_SPEAKERS.yaml", "references/PROJECT_KEYWORDS.yaml",
)
ALLOWED_FILES = (".env.example",)
# The copyright notice names the copyright holder by design; the roster rule skips it.
ROSTER_EXEMPT_PATHS = ("LICENSE",)
# Fictional placeholder first names used in docs/tests. A roster alias that is exactly one of
# these is not a leak on its own; multi-word roster names containing them are still checked.
PLACEHOLDER_NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank")


# ─── roster ─────────────────────────────────────────────────────────────────

def _roster_terms_fallback(text: str) -> list[str]:
    """Minimal parser for canonical_name / aliases / email lines when PyYAML is absent."""
    terms: list[str] = []
    for m in re.finditer(r"^\s*-?\s*(?:canonical_name|email)\s*:\s*(.+?)\s*$", text, re.M):
        terms.append(m.group(1).strip("\"'"))
    for m in re.finditer(r"^\s*aliases\s*:\s*\[(.*)\]\s*$", text, re.M):
        terms += [a.strip().strip("\"'") for a in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", m.group(1))]
    return terms


def load_roster_terms(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    terms: list[str] = []
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(text) or {}
        for sp in data.get("speakers") or []:
            for key in ("canonical_name", "email"):
                if sp.get(key):
                    terms.append(str(sp[key]))
            terms += [str(a) for a in sp.get("aliases") or []]
    except ImportError:
        terms = _roster_terms_fallback(text)
    terms += [m.group(0) for m in EMAIL_RE.finditer(text)]        # e.g. addresses inside notes
    seen, out = set(), []
    for t in (t.strip() for t in terms):
        if t and "REPLACE_ME" not in t and t.casefold() not in seen:
            seen.add(t.casefold())
            out.append(t)
    return out


def roster_patterns(terms: list[str], min_len: int) -> tuple[list[re.Pattern], int]:
    pats, skipped = [], 0
    placeholders = {n.casefold() for n in PLACEHOLDER_NAMES}
    for t in terms:
        latin = all(ord(c) < 0x250 for c in t)
        if len(t) < (min_len if latin else 2) or t.casefold() in placeholders:
            skipped += 1
            continue
        body = re.escape(t).replace(r"\ ", r"\s+")
        pats.append(re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])" if latin else body, re.I))
    return pats, skipped


# ─── checks ─────────────────────────────────────────────────────────────────

def mask(s: str) -> str:
    return s[0] + "*" * min(len(s) - 1, 12) if s else s


def email_allowed(domain: str) -> bool:
    d = domain.lower()
    return d in ALLOWED_EMAIL_DOMAINS or any(d.endswith("." + a) for a in ALLOWED_EMAIL_DOMAINS) \
        or d.endswith(ALLOWED_EMAIL_TLDS)


def check_name(path: str) -> list[str]:
    p = path.replace(os.sep, "/")
    base = p.rsplit("/", 1)[-1]
    if base in ALLOWED_FILES:
        return []
    for pat in FORBIDDEN_FILES:
        if fnmatch.fnmatch(base, pat) or fnmatch.fnmatch(p, pat):
            return [f"{path}: file: forbidden file type ({pat})"]
    return []


def check_line(where: str, line: str, roster: list[re.Pattern], roster_ok: bool) -> list[str]:
    out = []
    for m in PATH_RE.finditer(line):
        out.append(f"{where}: path: {mask(m.group(0))}")
    for m in EMAIL_RE.finditer(line):
        if not email_allowed(m.group(1)):
            out.append(f"{where}: email: {mask(m.group(0))}")
    if roster_ok:
        for pat in roster:
            for m in pat.finditer(line):
                out.append(f"{where}: roster: {mask(m.group(0))}")
    return out


def roster_applies(path: str) -> bool:
    return path.replace(os.sep, "/") not in ROSTER_EXEMPT_PATHS


def scan_text(label: str, text: str, roster, roster_ok=True) -> list[str]:
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        out += check_line(f"{label}:{i}", line, roster, roster_ok)
    return out


def scan_file(path: str, roster) -> list[str]:
    out = check_name(path)
    full = ROOT / path
    try:
        data = full.read_bytes()
    except OSError:
        return out
    if b"\0" in data[:8192]:
        return out
    return out + scan_text(path, data.decode("utf-8", errors="replace"), roster, roster_applies(path))


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="replace").stdout


def scan_patch(patch: str, roster, label: str) -> list[str]:
    """Commit messages, added lines and file names of `git log -p` / `git diff` output."""
    out, current, n = [], label, 0
    in_msg = True
    for line in patch.split("\n"):                              # not splitlines(): it splits on \x1e
        if line.startswith("\x1ecommit "):                        # our --format separator
            in_msg, current, n = True, f"{label} message {line.split()[1][:12]}", 0
            continue
        if line.startswith("diff --git "):
            in_msg, current, n = False, line.split(" b/", 1)[-1], 0
            out += check_name(current)
            continue
        if in_msg:
            n += 1
            out += check_line(f"{current}:{n}", line, roster, True)
            continue
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            n = int(m.group(1)) - 1 if m else 0
            continue
        if line.startswith(("+++ ", "--- ", "index ", "new file mode", "deleted file mode", "similarity",
                            "rename ", "old mode", "new mode", "Binary files")):
            continue
        if line.startswith("+"):
            n += 1
            out += check_line(f"{current}:{n}", line[1:], roster, roster_applies(current))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--tracked", action="store_true", help="scan every tracked file (default)")
    mode.add_argument("--staged", action="store_true", help="scan staged additions (pre-commit)")
    mode.add_argument("--log", metavar="RANGE", help="scan messages + added lines of a commit range")
    ap.add_argument("paths", nargs="*", help="scan these files instead")
    ap.add_argument("--roster", type=Path, help="private KNOWN_SPEAKERS.yaml to derive the denylist from")
    ap.add_argument("--no-roster", action="store_true", help="skip the roster denylist")
    ap.add_argument("--min-term-len", type=int, default=3, help="skip shorter Latin-script roster terms")
    a = ap.parse_args(argv)

    roster: list[re.Pattern] = []
    if not a.no_roster:
        env = os.environ.get("MEETING_DOCUMENTER_KNOWN_SPEAKERS")
        rpath = a.roster or (Path(env) if env else DEFAULT_ROSTER)
        if rpath.is_file():
            roster, skipped = roster_patterns(load_roster_terms(rpath), a.min_term_len)
            print(f"check_pii: roster denylist active ({len(roster)} terms"
                  + (f"; {skipped} short/placeholder terms skipped" if skipped else "") + ")", file=sys.stderr)
        elif a.roster:
            print(f"check_pii: roster file not found: {a.roster}", file=sys.stderr)
            return 2

    try:
        if a.paths:
            findings = [f for p in a.paths for f in scan_file(p, roster)]
        elif a.staged:
            names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").split("\0")
            findings = [f for n in names if n for f in check_name(n)]
            findings += scan_patch(git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff"), roster, "staged")
        elif a.log:
            patch = git("log", "-p", "-U0", "--no-color", "--no-ext-diff", "--format=%x1ecommit %H%n%B", a.log)
            findings = scan_patch(patch, roster, "log")
        else:
            names = git("ls-files", "-z").split("\0")
            findings = [f for n in names if n for f in scan_file(n, roster)]
    except subprocess.CalledProcessError as e:
        print(f"check_pii: git failed: {e.stderr.strip()}", file=sys.stderr)
        return 2

    for f in findings:
        print(f)
    if findings:
        print(f"check_pii: {len(findings)} finding(s). Replace with placeholders (Alice, Bob, "
              "~/..., name@example.com) or remove the file.", file=sys.stderr)
        return 1
    print("check_pii: clean", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
