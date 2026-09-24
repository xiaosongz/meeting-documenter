"""
Voiceprint enrollment store — biometric data, kept OUTSIDE notes and this clone.

Layout (VOICEPRINT_DIR, see config.py; chmod 700):
    voiceprints.sqlite   (chmod 600) metadata + float32 embedding BLOBs
    calibration.json     (chmod 600) thresholds + clip config per model, written by
                         `identify_speakers.py calibrate`

Why one SQLite file with BLOBs instead of SQLite + .npy per model:
  * `forget --name` is one atomic transaction — no orphaned vector files;
  * one file to chmod / back up / shred;
  * size is trivial (≈200 clips × ~20 people × 256 dims × 4 B ≈ 4 MB per model);
  * rows are namespaced by `model_id` (embeddings from different models or
    model versions are not comparable), so several models coexist in one file.

Tables:
    exemplars  enrolled clips of named people (source: self_check|user_ear|confirmed_meeting)
    cohort     clips of other / non-enrolled voices, used only for AS-norm
    people     optional per-person consent / notes
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .config import (  # noqa: F401  (re-exported for callers)
    CHANNELS, PERSON_CAP, SOURCES, VaultPathError, assert_outside_vault, resolve_store_dir,
)

DB_NAME = "voiceprints.sqlite"
CALIBRATION_NAME = "calibration.json"
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS exemplars (
    id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL,
    person TEXT NOT NULL,
    channel TEXT NOT NULL DEFAULT 'unknown',
    language TEXT,
    meeting_key TEXT,
    cluster TEXT,
    start_ms INTEGER, end_ms INTEGER,
    net_dur REAL,
    source TEXT NOT NULL,
    enrolled_date TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ex_model_person ON exemplars(model_id, person);
CREATE TABLE IF NOT EXISTS cohort (
    id INTEGER PRIMARY KEY,
    model_id TEXT NOT NULL,
    label TEXT NOT NULL,              -- canonical name if known, else "<meeting_key>:<cluster>"
    person TEXT,                      -- canonical name if known (so forget() finds it)
    meeting_key TEXT,
    start_ms INTEGER, end_ms INTEGER,
    net_dur REAL,
    added_date TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_co_model ON cohort(model_id);
CREATE TABLE IF NOT EXISTS people (
    name TEXT PRIMARY KEY,
    consent TEXT,
    notes TEXT,
    created TEXT
);
"""


# ─── Records ────────────────────────────────────────────────────────────────

@dataclass
class Exemplar:
    person: str
    embedding: np.ndarray = field(repr=False)
    channel: str = "unknown"
    language: str | None = None
    meeting_key: str | None = None
    cluster: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    net_dur: float = 0.0
    source: str = "confirmed_meeting"
    enrolled_date: str = field(default_factory=lambda: date.today().isoformat())
    id: int | None = None
    clip_sig: str | None = None      # identify_speakers.clip_signature of the clip config used


@dataclass
class CohortClip:
    label: str
    embedding: np.ndarray = field(repr=False)
    person: str | None = None
    meeting_key: str | None = None
    start_ms: int | None = None
    end_ms: int | None = None
    net_dur: float = 0.0
    added_date: str = field(default_factory=lambda: date.today().isoformat())
    clip_sig: str | None = None


def _blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).ravel().tobytes()


def _vec(b: bytes, dim: int) -> np.ndarray:
    v = np.frombuffer(b, dtype=np.float32)
    if v.size != dim:
        raise ValueError(f"corrupt embedding blob: {v.size} != dim {dim}")
    return v.copy()


# ─── Store ──────────────────────────────────────────────────────────────────

class VoiceprintStore:
    """SQLite-backed store. Use `VoiceprintStore.open(create=False)` for read paths
    so a missing store is reported as empty instead of being created."""

    def __init__(self, root: Path, conn: sqlite3.Connection | None):
        self.root = root
        self.conn = conn

    # -- lifecycle --
    @classmethod
    def open(cls, path: str | Path | None = None, create: bool = True) -> "VoiceprintStore":
        root = resolve_store_dir(path)
        db = root / DB_NAME
        if not create and not db.exists():
            return cls(root, None)
        old = os.umask(0o077)
        try:
            root.mkdir(parents=True, exist_ok=True)
            os.chmod(root, stat.S_IRWXU)                      # 700
            conn = sqlite3.connect(str(db))
            conn.executescript(_SCHEMA)
            for table in ("exemplars", "cohort"):          # migration: clip config signature
                cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if "clip_sig" not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN clip_sig TEXT")
            conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (str(SCHEMA_VERSION),))
            conn.commit()
            os.chmod(db, stat.S_IRUSR | stat.S_IWUSR)         # 600
        finally:
            os.umask(old)
        return cls(root, conn)

    @property
    def exists(self) -> bool:
        return self.conn is not None

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def _tx(self):
        if self.conn is None:
            raise RuntimeError("store not initialised (opened with create=False and no DB)")
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- exemplars --
    def add_exemplars(self, model_id: str, rows: list[Exemplar],
                      replace_meeting: bool = True, cap: int = PERSON_CAP) -> dict:
        """Insert exemplars. With replace_meeting, existing rows for the same
        (model, person, meeting_key) are replaced first, so re-enrolling a
        meeting is idempotent. Enforces `cap` per (person, model) by trimming
        the shortest clips of the most-represented meeting (keeps diversity)."""
        added, replaced, trimmed = 0, 0, {}
        with self._tx() as c:
            if replace_meeting:
                for person, mk in {(r.person, r.meeting_key) for r in rows if r.meeting_key}:
                    cur = c.execute("DELETE FROM exemplars WHERE model_id=? AND person=? AND meeting_key=?",
                                    (model_id, person, mk))
                    replaced += cur.rowcount
            for r in rows:
                if r.source not in SOURCES:
                    raise ValueError(f"source must be one of {SOURCES}, got {r.source!r}")
                emb = np.asarray(r.embedding, dtype=np.float32).ravel()
                c.execute(
                    "INSERT INTO exemplars (model_id, person, channel, language, meeting_key, cluster,"
                    " start_ms, end_ms, net_dur, source, enrolled_date, dim, embedding, clip_sig)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (model_id, r.person, r.channel or "unknown", r.language, r.meeting_key, r.cluster,
                     r.start_ms, r.end_ms, float(r.net_dur), r.source, r.enrolled_date,
                     int(emb.size), _blob(emb), r.clip_sig))
                added += 1
            for person in {r.person for r in rows}:
                n = self._trim_to_cap(c, model_id, person, cap)
                if n:
                    trimmed[person] = n
        return {"added": added, "replaced": replaced, "trimmed": trimmed}

    @staticmethod
    def _trim_to_cap(c: sqlite3.Connection, model_id: str, person: str, cap: int) -> int:
        n_removed = 0
        while True:
            (n,) = c.execute("SELECT COUNT(*) FROM exemplars WHERE model_id=? AND person=?",
                             (model_id, person)).fetchone()
            if n <= cap:
                return n_removed
            (mk,) = c.execute(
                "SELECT meeting_key FROM exemplars WHERE model_id=? AND person=?"
                " GROUP BY meeting_key ORDER BY COUNT(*) DESC LIMIT 1", (model_id, person)).fetchone()
            excess = n - cap
            cur = c.execute(
                "DELETE FROM exemplars WHERE id IN (SELECT id FROM exemplars WHERE model_id=? AND person=?"
                " AND meeting_key IS ? ORDER BY net_dur ASC LIMIT ?)",
                (model_id, person, mk, min(excess, 10)))
            n_removed += cur.rowcount

    def exemplars(self, model_id: str, people: list[str] | None = None,
                  exclude_meetings: list[str] | None = None) -> list[Exemplar]:
        if self.conn is None:
            return []
        q = ("SELECT id, person, channel, language, meeting_key, cluster, start_ms, end_ms, net_dur,"
             " source, enrolled_date, dim, embedding, clip_sig FROM exemplars WHERE model_id=?")
        args: list = [model_id]
        if people:
            q += f" AND person IN ({','.join('?' * len(people))})"
            args += list(people)
        out = []
        excl = set(exclude_meetings or [])
        for row in self.conn.execute(q + " ORDER BY id", args):
            (i, person, ch, lang, mk, cl, s, e, nd, src, ed, dim, blob, sig) = row
            if mk in excl:
                continue
            out.append(Exemplar(person, _vec(blob, dim), ch, lang, mk, cl, s, e, nd, src, ed, i, sig))
        return out

    def backfill_clip_sig(self, model_id: str, sig: str, only_null: bool = True) -> dict:
        """Tag rows written without a clip signature (e.g. by library callers) with `sig`.
        Use only when those rows are KNOWN to come from the config that `sig` describes."""
        cond = " AND clip_sig IS NULL" if only_null else ""
        with self._tx() as c:
            ex = c.execute(f"UPDATE exemplars SET clip_sig=? WHERE model_id=?{cond}", (sig, model_id)).rowcount
            co = c.execute(f"UPDATE cohort SET clip_sig=? WHERE model_id=?{cond}", (sig, model_id)).rowcount
        return {"exemplars": ex, "cohort": co}

    # -- cohort --
    def add_cohort(self, model_id: str, rows: list[CohortClip], replace_meeting: bool = True) -> int:
        with self._tx() as c:
            if replace_meeting:
                for label, mk in {(r.label, r.meeting_key) for r in rows if r.meeting_key}:
                    c.execute("DELETE FROM cohort WHERE model_id=? AND label=? AND meeting_key=?",
                              (model_id, label, mk))
            for r in rows:
                emb = np.asarray(r.embedding, dtype=np.float32).ravel()
                c.execute(
                    "INSERT INTO cohort (model_id, label, person, meeting_key, start_ms, end_ms, net_dur,"
                    " added_date, dim, embedding, clip_sig) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (model_id, r.label, r.person, r.meeting_key, r.start_ms, r.end_ms,
                     float(r.net_dur), r.added_date, int(emb.size), _blob(emb), r.clip_sig))
        return len(rows)

    def cohort(self, model_id: str, exclude_meetings: list[str] | None = None
               ) -> tuple[np.ndarray, list[str]]:
        """(N, dim) cohort-table embeddings and their labels (does NOT include exemplars)."""
        if self.conn is None:
            return np.zeros((0, 0), np.float32), []
        excl = set(exclude_meetings or [])
        vecs, labels = [], []
        for label, person, mk, dim, blob in self.conn.execute(
                "SELECT label, person, meeting_key, dim, embedding FROM cohort WHERE model_id=? ORDER BY id",
                (model_id,)):
            if mk in excl:
                continue
            vecs.append(_vec(blob, dim))
            labels.append(person or label)
        if not vecs:
            return np.zeros((0, 0), np.float32), []
        return np.stack(vecs), labels

    def cohort_clips(self, model_id: str) -> list[CohortClip]:
        """All cohort rows for `model_id` as records (used by calibration)."""
        if self.conn is None:
            return []
        return [CohortClip(label, _vec(blob, dim), person, mk, s, e, nd or 0.0, ad, sig)
                for label, person, mk, s, e, nd, ad, dim, blob, sig in self.conn.execute(
                    "SELECT label, person, meeting_key, start_ms, end_ms, net_dur, added_date, dim,"
                    " embedding, clip_sig FROM cohort WHERE model_id=? ORDER BY id", (model_id,))]

    # -- admin --
    def forget(self, name: str) -> dict:
        """Delete every exemplar, cohort row and people row for `name` (all models)."""
        if self.conn is None:
            return {"exemplars": 0, "cohort": 0, "people": 0}
        with self._tx() as c:
            ex = c.execute("DELETE FROM exemplars WHERE person=?", (name,)).rowcount
            co = c.execute("DELETE FROM cohort WHERE person=? OR label=?", (name, name)).rowcount
            pe = c.execute("DELETE FROM people WHERE name=?", (name,)).rowcount
        # reclaim pages so deleted vectors don't linger in the file
        self.conn.execute("VACUUM")
        return {"exemplars": ex, "cohort": co, "people": pe}

    def set_person_note(self, name: str, consent: str | None = None, notes: str | None = None) -> None:
        with self._tx() as c:
            c.execute("INSERT INTO people (name, consent, notes, created) VALUES (?,?,?,?)"
                      " ON CONFLICT(name) DO UPDATE SET consent=COALESCE(excluded.consent, consent),"
                      " notes=COALESCE(excluded.notes, notes)",
                      (name, consent, notes, date.today().isoformat()))

    def model_ids(self) -> list[str]:
        if self.conn is None:
            return []
        rows = self.conn.execute("SELECT DISTINCT model_id FROM exemplars UNION "
                                 "SELECT DISTINCT model_id FROM cohort ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    def count(self, model_id: str | None = None) -> int:
        if self.conn is None:
            return 0
        if model_id is None:
            return self.conn.execute("SELECT COUNT(*) FROM exemplars").fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM exemplars WHERE model_id=?",
                                 (model_id,)).fetchone()[0]

    def summary(self) -> list[dict]:
        """Per (model, person): counts, channels, meetings, sources, net seconds. No vectors."""
        if self.conn is None:
            return []
        rows = self.conn.execute(
            "SELECT model_id, person, COUNT(*), GROUP_CONCAT(DISTINCT channel),"
            " COUNT(DISTINCT meeting_key), GROUP_CONCAT(DISTINCT source), SUM(net_dur),"
            " MIN(enrolled_date), MAX(enrolled_date)"
            " FROM exemplars GROUP BY model_id, person ORDER BY person, model_id").fetchall()
        out = []
        for m, p, n, chs, nm, srcs, dur, d0, d1 in rows:
            out.append({"model_id": m, "person": p, "samples": n,
                        "channels": sorted((chs or "").split(",")), "meetings": nm,
                        "sources": sorted((srcs or "").split(",")), "net_speech_s": round(dur or 0, 1),
                        "first_enrolled": d0, "last_enrolled": d1})
        return out

    def cohort_summary(self) -> list[dict]:
        if self.conn is None:
            return []
        rows = self.conn.execute("SELECT model_id, COUNT(*), COUNT(DISTINCT label) FROM cohort"
                                 " GROUP BY model_id").fetchall()
        return [{"model_id": m, "clips": n, "speakers": s} for m, n, s in rows]

    # -- calibration --
    def calibration_path(self) -> Path:
        return self.root / CALIBRATION_NAME

    def load_calibration(self, model_id: str, model_name: str | None = None) -> dict | None:
        """calibration.json entry for this model: exact model_id first, then short name."""
        p = self.calibration_path()
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        models = data.get("models", data)
        entry = models.get(model_id) or (models.get(model_name) if model_name else None)
        if entry is not None:
            entry = dict(entry)
            entry.setdefault("_source", str(p))
        return entry

    def save_calibration(self, model_id: str, entry: dict) -> Path:
        """Write/replace this model's entry, keeping other models; atomic, chmod 600."""
        p = self.calibration_path()
        data: dict = {"models": {}}
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                raise RuntimeError(f"existing {p.name} is unreadable; move it aside first ({e})") from e
            if "models" not in data:
                data = {"models": data}
        data["models"][model_id] = {k: v for k, v in entry.items() if not k.startswith("_")}
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".calibration-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return p
