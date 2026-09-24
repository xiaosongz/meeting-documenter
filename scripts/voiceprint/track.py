"""
Known-speaker tracking (architecture A): who speaks when, from enrolled voiceprints.

Contract (see the speaker-tracking plan):
    track(audio_path, roster, store, embedder, *, backend="auto", params=TrackParams()) -> Timeline

Pipeline
    decode 16 kHz → boundary layer (segment.py; nemotron3 when available, else seg3)
    → speaker *tracks*: nemotron3's global slots as they are; seg3's chunk-local speakers
      linked across overlapping chunks into chains (`link_chunks`)
    → speech = union of track activity (the boundary model, not Silero: Silero recalled only
      77.5 % of AMI far-field speech), padded by `pad_s` into silence only
    → *units*: per track, its clean (no other track active) spans grouped into ≤ `unit_s`
      of wall time; sparse speech keeps growing a unit (up to `pool_s`) until it holds
      `min_embed_s` of clean audio → one ReDimNet2 embedding per unit
    → AS-norm score vs every roster model → duration-aware calibrated LLR (Calibration)
    → per track, an HMM over its units (states = roster ∪ UNKNOWN; UNKNOWN has constant
      log-odds θ; a state change costs λ) → forward-backward posteriors; units without an
      embedding take their state from their neighbours on the same track
    → exclusive assignment: two tracks speaking at the same time cannot carry one name
    → labels: MAP state; a name whose posterior is below `p_uncertain` → UNCERTAIN(A|B)
    → UNKNOWN units clustered across tracks → GUEST_n (≥ `guest_min_s` of speech)
    → second pass: every contiguous turn ≥ `turn_min_s` is re-embedded whole and relabelled
      when its own posterior is confident
    → Timeline: `turns` (per track, may overlap) and `exclusive` (the most active track)

Everything after the embeddings is numpy only, so the benchmark can re-decode cached
embeddings with other parameters at no model cost. B1 (bench/baselines/ahc_id.py) reuses
the tracks, units and scoring and only swaps the decoder.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .timeline import GUEST_PREFIX, UNKNOWN, Timeline, Turn, merge_touching, uncertain_label

SAMPLE_RATE = 16000


# ─── Parameters ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrackParams:
    # tracks and speech
    threshold: float = 0.5          # track activity threshold for units (clean audio to embed)
    speech_threshold: float | None = None   # activity threshold for labelling speech (None: threshold)
    pad_s: float = 0.25             # extend each track's speech by this much into silence
    min_span_s: float = 0.1         # clean spans shorter than this are ignored
    link_min: float = 0.4           # seg3 chunk linking: minimum soft-Dice agreement
    # units and embeddings
    unit_s: float = 10.0            # a unit spans at most this much wall time …
    pool_s: float = 30.0            # … unless it still has < min_embed_s clean audio (then up to this)
    min_embed_s: float = 1.0        # clean audio needed for a unit embedding
    max_embed_s: float = 10.0       # clean audio used per unit embedding (longest spans first)
    quant_s: float = 0.5            # embedding input lengths are floored to a multiple of this
    # scoring
    top_n: int = 100                # AS-norm cohort top-N
    # decoding
    theta: float = 0.0              # UNKNOWN log-odds, in calibrated LLR units
    switch: float = 10.0            # λ: cost of a state change between consecutive units of a track
    fill_s: float = 10.0            # speech further than this from any unit of its track → UNKNOWN
    p_uncertain: float = 0.0        # MAP posterior below this turns a name into UNCERTAIN(A|B)
    conflict_s: float = 0.5         # simultaneous speech that makes two tracks' equal names a conflict
    foreign_p: float = 2.0          # a name that is another track's primary identity needs this posterior
                                    # (1.0 = off, > 1 = never allowed); see demote_foreign
    # guests
    guest_cos: float = 0.55         # AHC (average linkage) cosine threshold between UNKNOWN groups
    guest_min_s: float = 5.0        # guest clusters with less speech stay UNKNOWN
    # second pass
    turn_min_s: float = 3.0         # re-score contiguous turns at least this long
    turn_gap_s: float = 1.0         # a pause longer than this on the track ends a turn
    turn_max_embed_s: float = 30.0  # clean audio per turn embedding (capped by the device cap)
    turn_override_p: float = 0.95   # posterior needed to relabel a turn
    second_pass: bool = False       # neutral on AMI dev (IER 17.73 → 17.69 at best) for ~20x the
                                    # embedding time, so off by default

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Calibration: AS-norm score + duration → LLR ────────────────────────────

@dataclass(frozen=True)
class Calibration:
    """llr = a·s + b·s·ln(d) + c·ln(d) + k, with s the AS-norm score and d the seconds of
    clean audio behind the embedding (a quality measure function on log-duration). Fitted
    by prior-weighted logistic regression, so the output is a log-likelihood ratio. d is
    clamped to [d_min, d_max], the durations the fit saw: the duration terms must not be
    extrapolated (a 30 s turn is scored as a d_max one)."""
    a: float
    b: float
    c: float
    k: float
    source: str = "default"
    d_min: float = 0.5              # durations are clamped to the range the fit has seen
    d_max: float = 10.0

    def llr(self, s: np.ndarray, d: np.ndarray | float) -> np.ndarray:
        ld = np.log(np.clip(np.asarray(d, np.float64), self.d_min, self.d_max))
        s = np.asarray(s, np.float64)
        ld = ld if ld.ndim == 0 or ld.ndim == s.ndim else ld.reshape(ld.shape + (1,) * (s.ndim - ld.ndim))
        return (self.a * s + self.b * s * ld + self.c * ld + self.k).astype(np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(float(d["a"]), float(d["b"]), float(d["c"]), float(d["k"]), str(d.get("source", "?")),
                   float(d.get("d_min", 0.5)), float(d.get("d_max", 10.0)))


# Fitted on the public AMI dev split (ReDimNet2, far-field test vs headset enrollment,
# enrollment-QC protocol); see calibrate_track.py. The private store refits a and k.
DEFAULT_CALIBRATION = Calibration(a=0.5826, b=0.0799, c=-3.0205, k=-2.2843, source="ami-dev-2026-09-24",
                                  d_min=1.0, d_max=9.0)


# ─── Tracks ─────────────────────────────────────────────────────────────────

@dataclass
class Track:
    start: int                      # first global frame
    prob: np.ndarray                # (n,) activity probability from `start`

    @property
    def end(self) -> int:
        return self.start + len(self.prob)


@dataclass
class Tracks:
    """Speaker tracks on one global frame grid: frame f covers [t0 + f·step, t0 + (f+1)·step)."""
    step: float
    t0: float
    n_frames: int
    tracks: list[Track]
    backend: str
    duration: float
    meta: dict = field(default_factory=dict)

    def time(self, f) -> np.ndarray:
        return self.t0 + np.asarray(f) * self.step

    def frame(self, t) -> np.ndarray:
        return np.floor((np.asarray(t) - self.t0) / self.step).astype(np.int64)

    def active(self, threshold: float) -> np.ndarray:
        """(n_frames, n_tracks) bool, dense. Only for callers that need all tracks at once."""
        A = np.zeros((self.n_frames, len(self.tracks)), bool)
        for k, tr in enumerate(self.tracks):
            A[tr.start:tr.end, k] = tr.prob >= threshold
        return A


def tracks_from_segmentation(seg, params: TrackParams = TrackParams()) -> Tracks:
    """segment.LocalSegmentation → Tracks (global slots as-is; chunk-local speakers linked)."""
    dur = float(seg.meta.get("duration", seg.chunk_starts[-1] + seg.chunk_dur))
    if seg.n_chunks == 1:
        P = seg.activations[0]
        t0 = seg.frame_dur / 2 - seg.frame_step / 2
        tr = [Track(0, P[:, k].astype(np.float32)) for k in range(P.shape[1])
              if (P[:, k] >= params.threshold).any()]
        return Tracks(seg.frame_step, t0, P.shape[0], tr, seg.backend, dur)
    return link_chunks(seg.activations, seg.chunk_starts, seg.frame_step, seg.frame_dur, dur,
                       params, backend=seg.backend)


def link_chunks(acts: np.ndarray, chunk_starts: np.ndarray, step: float, frame_dur: float,
                duration: float, params: TrackParams = TrackParams(), backend: str = "seg3") -> Tracks:
    """Chain chunk-local speakers across overlapping chunks.

    Chunks are visited in time order. Each active local speaker of chunk c is compared, on
    the frames it shares with earlier chunks, with the running mean activity of every open
    chain (soft Dice); a Hungarian match ≥ `link_min` extends that chain, otherwise it opens
    a new one. A chain's activity at a frame is its summed probability over the number of
    chunks covering the frame, as in pyannote's aggregation. Chains are stored sparsely
    (their own frame span), so memory stays O(audio), not O(audio × chains).
    """
    from scipy.optimize import linear_sum_assignment

    n_chunks, n_fr, n_loc = acts.shape
    offs = np.rint(np.asarray(chunk_starts) / step).astype(np.int64)
    n_global = int(offs[-1] + n_fr)
    cover = np.zeros(n_global, np.float32)
    for o in offs:
        cover[o:o + n_fr] += 1
    chains: list[dict] = []                               # {"start", "sum": list[np.ndarray] …}
    open_ids: list[int] = []
    for c in range(n_chunks):
        o = int(offs[c])
        A = acts[c].astype(np.float32)
        live = [k for k in range(n_loc) if (A[:, k] >= params.threshold).any()]
        open_ids = [i for i in open_ids if chains[i]["end"] > o]          # still overlapping chunk c
        match: dict[int, int] = {}
        if live and open_ids:
            S = np.zeros((len(live), len(open_ids)), np.float32)
            for j, i in enumerate(open_ids):
                ch = chains[i]
                a, b = max(o, ch["start"]), min(o + n_fr, ch["end"])
                if b <= a:
                    continue
                mean = ch["sum"][a - ch["start"]:b - ch["start"]] / cover[a:b]
                for r, k in enumerate(live):
                    x = A[a - o:b - o, k]
                    den = float(x.sum() + mean.sum())
                    S[r, j] = 2 * float(np.minimum(x, mean).sum()) / den if den > 0 else 0.0
            rows, cols = linear_sum_assignment(-S)
            match = {live[r]: open_ids[j] for r, j in zip(rows, cols) if S[r, j] >= params.link_min}
        for k in live:
            if k in match:
                ch = chains[match[k]]
                if o + n_fr > ch["end"]:                                   # grow the chain's span
                    ch["sum"] = np.concatenate([ch["sum"], np.zeros(o + n_fr - ch["end"], np.float32)])
                    ch["end"] = o + n_fr
                ch["sum"][o - ch["start"]:o - ch["start"] + n_fr] += A[:, k]
            else:
                chains.append({"start": o, "end": o + n_fr, "sum": A[:, k].copy()})
                open_ids.append(len(chains) - 1)
    tracks = [Track(ch["start"], (ch["sum"] / cover[ch["start"]:ch["end"]]).astype(np.float32))
              for ch in chains]
    tracks = [t for t in tracks if (t.prob >= params.threshold).any()]
    return Tracks(step, frame_dur / 2 - step / 2, n_global, tracks, backend, duration,
                  {"chains": len(tracks)})


# ─── Units ──────────────────────────────────────────────────────────────────

@dataclass
class Unit:
    track: int
    t0: float                       # wall range covered by its spans
    t1: float
    spans: list[tuple[float, float]]
    clean_s: float
    emb_row: int = -1               # row in the unit embedding matrix (−1: no embedding)
    emb_s: float = 0.0              # seconds of audio behind the embedding


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def n_active(trs: Tracks, threshold: float) -> np.ndarray:
    n = np.zeros(trs.n_frames, np.int16)
    for tr in trs.tracks:
        n[tr.start:tr.end] += tr.prob >= threshold
    return n


def build_units(trs: Tracks, params: TrackParams = TrackParams()) -> list[Unit]:
    """Per track, clean spans grouped into units (see the module docstring)."""
    n_act = n_active(trs, params.threshold)
    units: list[Unit] = []
    for k, tr in enumerate(trs.tracks):
        act = tr.prob >= params.threshold
        clean = act & (n_act[tr.start:tr.end] == 1)
        spans = []
        for a, b in _runs(clean):
            s, e = float(trs.time(tr.start + a)), float(trs.time(tr.start + b))
            e = min(e, trs.duration)
            if e - s < params.min_span_s:
                continue
            while e - s > params.unit_s:                          # very long clean runs: split
                spans.append((s, s + params.unit_s))
                s += params.unit_s
            spans.append((s, e))
        cur: list[tuple[float, float]] = []
        for s, e in spans:
            if cur:
                wall = e - cur[0][0]
                have = sum(b - a for a, b in cur)
                if wall > params.pool_s or (wall > params.unit_s and have >= params.min_embed_s):
                    units.append(_unit(k, cur))
                    cur = []
            cur.append((s, e))
        if cur:
            units.append(_unit(k, cur))
    return units


def _unit(k: int, spans: list[tuple[float, float]]) -> Unit:
    return Unit(k, spans[0][0], spans[-1][1], list(spans), float(sum(e - s for s, e in spans)))


def gather(wave: np.ndarray, spans: list[tuple[float, float]], max_s: float, quant_s: float,
           sr: int = SAMPLE_RATE) -> np.ndarray | None:
    """Concatenate the longest spans (in time order) up to `max_s`, floored to `quant_s`."""
    order = sorted(range(len(spans)), key=lambda i: spans[i][0] - spans[i][1])
    keep, tot = [], 0.0
    for i in order:
        if tot >= max_s:
            break
        s, e = spans[i]
        d = min(e - s, max_s - tot)
        keep.append((s, s + d))
        tot += d
    keep.sort()
    x = np.concatenate([wave[int(s * sr):int(e * sr)] for s, e in keep]) if keep else np.zeros(0, np.float32)
    q = int(round(quant_s * sr)) if quant_s else 1
    n = (len(x) // q) * q
    if n < sr // 2:
        return None
    return np.ascontiguousarray(x[:n], dtype=np.float32)


def embed_spans(embedder, wave: np.ndarray, groups: list[list[tuple[float, float]]], max_s: float,
                quant_s: float, guard=None) -> tuple[np.ndarray, np.ndarray]:
    """Embed each span group (audio.embed_clips: bounded MPS batches). Returns
    (embeddings (n_ok, D), index of the group each row came from); groups with < 0.5 s drop out."""
    from . import audio as au

    clips, idx = [], []
    for i, g in enumerate(groups):
        w = gather(wave, g, max_s, quant_s)
        if w is not None:
            clips.append(au.Clip("u", 0, 0, len(w) / SAMPLE_RATE, w))
            idx.append(i)
    if not clips:
        return np.zeros((0, int(getattr(embedder, "dim", 0))), np.float32), np.zeros(0, int)
    E, rows, order = [], [], np.argsort([len(c.wave) for c in clips], kind="stable")
    step = 256
    for a in range(0, len(order), step):                      # bounded chunks; guard between them
        part = [clips[i] for i in order[a:a + step]]
        keep_ids = [idx[i] for i in order[a:a + step]]
        tag = {id(c): g for c, g in zip(part, keep_ids)}
        e = au.embed_clips(embedder, part, batch_size=64)      # drops rejected clips in place
        E.append(e)
        rows += [tag[id(c)] for c in part]
        if guard is not None:
            guard.check("unit embeddings")
    return np.concatenate(E), np.asarray(rows, int)


def attach_embeddings(units: list[Unit], rows: np.ndarray, params: TrackParams) -> None:
    """Mark which units have an embedding (row i of the matrix belongs to units[rows[i]])."""
    for u in units:
        u.emb_row, u.emb_s = -1, 0.0
    for r, i in enumerate(rows):
        u = units[int(i)]
        u.emb_row = r
        q = params.quant_s or 1e-9
        u.emb_s = math.floor(min(u.clean_s, params.max_embed_s) / q + 1e-9) * q


def embed_units(embedder, wave: np.ndarray, units: list[Unit], params: TrackParams,
                guard=None) -> np.ndarray:
    groups = [u.spans if u.clean_s >= params.min_embed_s else [] for u in units]
    E, rows = embed_spans(embedder, wave, groups, params.max_embed_s, params.quant_s, guard)
    attach_embeddings(units, rows, params)
    return E


# ─── Scoring ────────────────────────────────────────────────────────────────

def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _top_stats(S: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean / std of the top-N values along axis 0 of S (cohort × items)."""
    n = min(top_n, S.shape[0])
    top = np.partition(S, S.shape[0] - n, axis=0)[S.shape[0] - n:]
    return top.mean(0), np.maximum(top.std(0), 1e-4)


@dataclass
class Scorer:
    """Roster models (one reference vector per name, channel-matched by the caller) and an
    AS-norm cohort. `owners` (one label per cohort row, optional) marks rows that belong to
    a roster name; those rows are left out of that name's normalisation on both sides, so
    enrolled people can serve as each other's cohort. A name with fewer than `min_cohort`
    usable rows falls back to raw cosine (and so does everything without a cohort)."""
    names: list[str]
    models: np.ndarray              # (K, D) L2-normed
    cohort: np.ndarray              # (N, D) L2-normed
    top_n: int = 100
    owners: np.ndarray | None = None
    min_cohort: int = 10
    _masks: list = field(default_factory=list, repr=False)
    _enroll_stats: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        dim = self.models.shape[1] if np.ndim(self.models) == 2 and len(self.models) else (
            self.cohort.shape[1] if self.cohort.ndim == 2 else 1)
        self.models = _l2(self.models) if len(self.names) else np.zeros((0, dim), np.float32)
        self.cohort = _l2(self.cohort) if self.cohort.size else np.zeros((0, dim), np.float32)
        own = np.asarray(self.owners if self.owners is not None else [None] * len(self.cohort), dtype=object)
        self._masks, self._enroll_stats = [], []
        for j, n in enumerate(self.names):
            m = own != n
            if m.sum() < self.min_cohort:
                self._masks.append(None)
                self._enroll_stats.append(None)
                continue
            self._masks.append(m)
            mu, sd = _top_stats((self.cohort[m] @ self.models[j])[:, None], self.top_n)
            self._enroll_stats.append((float(mu[0]), float(sd[0])))

    @property
    def use_asnorm(self) -> bool:
        return any(m is not None for m in self._masks)

    def scores(self, X: np.ndarray) -> np.ndarray:
        """(n, K) AS-norm scores (raw cosine for names without a usable cohort)."""
        if not len(self.names) or not len(X):
            return np.zeros((len(X), len(self.names)), np.float32)
        X = _l2(X)
        out = X @ self.models.T
        if not self.use_asnorm:
            return out.astype(np.float32)
        C = self.cohort @ X.T                                       # (N, n), computed once
        shared = None
        for j, m in enumerate(self._masks):
            if m is None:
                continue
            if m.all():
                if shared is None:
                    shared = _top_stats(C, self.top_n)
                mu_t, sd_t = shared
            else:
                mu_t, sd_t = _top_stats(C[m], self.top_n)
            mu_e, sd_e = self._enroll_stats[j]
            out[:, j] = 0.5 * ((out[:, j] - mu_e) / sd_e + (out[:, j] - mu_t) / sd_t)
        return out.astype(np.float32)


# ─── Decoding ───────────────────────────────────────────────────────────────

def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def hmm_posteriors(E: np.ndarray, switch: float) -> np.ndarray:
    """Forward-backward over a chain of units. E: (n, S) log-likelihood ratios (a row of
    zeros = no observation). Transition: stay ∝ 1, move to any other state ∝ e^−λ."""
    n, S = E.shape
    if n == 0:
        return np.zeros((0, S), np.float32)
    lt = np.full((S, S), -switch, np.float64)
    np.fill_diagonal(lt, 0.0)
    lt -= _logsumexp(lt, axis=1)[:, None]
    a = np.empty((n, S))
    a[0] = E[0] - math.log(S)
    for t in range(1, n):
        a[t] = _logsumexp(a[t - 1][:, None] + lt, axis=0) + E[t]
    b = np.zeros((n, S))
    for t in range(n - 2, -1, -1):
        b[t] = _logsumexp(lt + (E[t + 1] + b[t + 1])[None, :], axis=1)
    g = a + b
    g -= _logsumexp(g, axis=1)[:, None]
    return np.exp(g).astype(np.float32)


@dataclass
class Decoded:
    """Per-unit decision. `state` indexes names + [UNKNOWN]; `post` the posteriors."""
    names: list[str]
    post: np.ndarray                # (n_units, K+1)
    state: np.ndarray               # (n_units,)
    label: list[str]
    conf: np.ndarray                # (n_units,) posterior of the chosen state


def unit_emissions(units: list[Unit], S: np.ndarray, cal: Calibration, theta: float) -> np.ndarray:
    """(n_units, K+1) emissions: calibrated LLR per name, θ for UNKNOWN; zeros where a unit
    has no embedding. S holds one AS-norm row per embedding row."""
    K = S.shape[1]
    E = np.zeros((len(units), K + 1), np.float32)
    for i, u in enumerate(units):
        if u.emb_row >= 0:
            E[i, :K] = cal.llr(S[u.emb_row], u.emb_s)
            E[i, K] = theta
    return E


def decode_units(units: list[Unit], trs: Tracks, E: np.ndarray, names: list[str],
                 params: TrackParams) -> Decoded:
    """Per-track HMM posteriors, then exclusive assignment across simultaneous tracks."""
    K = len(names)
    post = np.zeros((len(units), K + 1), np.float32)
    by_track: dict[int, list[int]] = {}
    for i, u in enumerate(units):
        by_track.setdefault(u.track, []).append(i)
    for idx in by_track.values():
        idx.sort(key=lambda i: units[i].t0)
        post[idx] = hmm_posteriors(E[idx].astype(np.float64), params.switch)
    state = post.argmax(1)
    if params.foreign_p != 1.0 and K:
        state = demote_foreign(units, post, state, K, params, E)
    state = resolve_conflicts(units, trs, post, state, K, params)
    labels, conf = [], np.zeros(len(units), np.float32)
    for i, s in enumerate(state):
        conf[i] = post[i, s]
        if s == K:
            labels.append(UNKNOWN)
            continue
        if params.p_uncertain and post[i, s] < params.p_uncertain:
            alt = [j for j in np.argsort(-post[i]) if j != s][0]
            labels.append(uncertain_label([names[s]] + ([names[alt]] if alt < K else [])))
        else:
            labels.append(names[s])
    return Decoded(names, post, state, labels, conf)


def track_identities(units: list[Unit], E: np.ndarray, K: int) -> dict[int, int]:
    """Primary identity per track: Hungarian over tracks × (names ∪ one UNKNOWN per track) on
    each track's mean emission (clean-duration-weighted calibrated LLR; θ for UNKNOWN). The
    mean, not the sum or the HMM posterior: a talkative stranger who resembles X on every
    unit must not outweigh X's own track, whose evidence for X is far stronger per unit."""
    from scipy.optimize import linear_sum_assignment

    tracks = sorted({u.track for u in units})
    row = {k: r for r, k in enumerate(tracks)}
    acc = np.zeros((len(tracks), K + 1))
    w = np.zeros(len(tracks))
    for i, u in enumerate(units):
        if u.emb_row >= 0:
            acc[row[u.track]] += E[i] * u.emb_s
            w[row[u.track]] += u.emb_s
    mean = acc / np.maximum(w, 1e-9)[:, None]
    n = len(tracks)
    gain = np.full((n, K + n), -1e9)
    gain[:, :K] = np.where(w[:, None] > 0, mean[:, :K], -1e9)
    gain[np.arange(n), K + np.arange(n)] = np.where(w > 0, mean[:, K], 0.0)
    r, c = linear_sum_assignment(-gain)
    return {tracks[a]: (int(b) if b < K else K) for a, b in zip(r, c)}


def demote_foreign(units: list[Unit], post: np.ndarray, state: np.ndarray, K: int,
                   params: TrackParams, E: np.ndarray) -> np.ndarray:
    """With global slots, a speaker normally lives on one track. A unit labelled with a name
    that is *another* track's primary identity keeps it only with posterior ≥ foreign_p;
    otherwise it falls back to UNKNOWN (an abstention, never a different guess)."""
    prim = track_identities(units, E, K)
    owner = {s: k for k, s in prim.items() if s < K}
    state = state.copy()
    for i, u in enumerate(units):
        s = int(state[i])
        if s < K and owner.get(s, u.track) != u.track and post[i, s] < params.foreign_p:
            state[i] = K
    return state


def resolve_conflicts(units: list[Unit], trs: Tracks, post: np.ndarray, state: np.ndarray,
                      K: int, params: TrackParams) -> np.ndarray:
    """Two tracks active together for ≥ conflict_s cannot both be the same enrolled person:
    the weaker unit (lower posterior for that name) moves to its best free state."""
    state = state.copy()
    if len(units) < 2:
        return state
    act = {k: (tr.start, tr.prob >= params.threshold) for k, tr in enumerate(trs.tracks)}

    def together(u: Unit, v: Unit) -> float:
        a, b = max(u.t0, v.t0), min(u.t1, v.t1)
        if b <= a:
            return 0.0
        fa, fb = int(trs.frame(a)), int(trs.frame(b)) + 1
        (su, au_), (sv, av) = act[u.track], act[v.track]
        lo, hi = max(fa, su, sv), min(fb, su + len(au_), sv + len(av))
        if hi <= lo:
            return 0.0
        return float((au_[lo - su:hi - su] & av[lo - sv:hi - sv]).sum() * trs.step)

    order = sorted(range(len(units)), key=lambda i: units[i].t0)
    starts = np.array([units[i].t0 for i in order])
    for _ in range(3):                                         # a few sweeps settle chains of moves
        changed = False
        for p, i in enumerate(order):
            if state[i] == K:
                continue
            u = units[i]
            hi = np.searchsorted(starts, u.t1, side="left")
            for j in order[p + 1:hi]:
                v = units[j]
                if v.track == u.track or state[j] != state[i] or together(u, v) < params.conflict_s:
                    continue
                lose = i if post[i, state[i]] < post[j, state[j]] else j
                taken = {state[x] for x in (i, j) if x != lose}
                for s in np.argsort(-post[lose]):
                    if s == K or s not in taken:
                        state[lose] = s
                        break
                changed = True
                if lose == i:
                    break
        if not changed:
            break
    return state


# ─── Guests ─────────────────────────────────────────────────────────────────

def cluster_guests(units: list[Unit], labels: list[str], X: np.ndarray, params: TrackParams) -> list[str]:
    """UNKNOWN units → GUEST_n. Consecutive UNKNOWN units of one track form a group; groups
    are merged by average-linkage AHC on their mean embedding (cosine ≥ guest_cos); clusters
    with ≥ guest_min_s of clean speech get a guest label, numbered by first appearance."""
    out = list(labels)
    groups: list[list[int]] = []
    by_track: dict[int, list[int]] = {}
    for i, u in enumerate(units):
        by_track.setdefault(u.track, []).append(i)
    for idx in by_track.values():
        idx.sort(key=lambda i: units[i].t0)
        cur: list[int] = []
        for i in idx:
            if labels[i] == UNKNOWN:
                cur.append(i)
            elif cur:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
    vecs, keep = [], []
    for g in groups:
        rows = [units[i].emb_row for i in g if units[i].emb_row >= 0]
        if rows:
            w = np.array([units[i].emb_s for i in g if units[i].emb_row >= 0], np.float32)
            vecs.append(_l2((X[rows] * w[:, None]).sum(0)))
            keep.append(g)
    if not keep:
        return out
    cl = ahc(np.stack(vecs), params.guest_cos)
    clean = {}
    first = {}
    for g, c in zip(keep, cl):
        clean[c] = clean.get(c, 0.0) + sum(units[i].clean_s for i in g)
        first[c] = min(first.get(c, math.inf), min(units[i].t0 for i in g))
    ok = sorted((c for c in clean if clean[c] >= params.guest_min_s), key=lambda c: first[c])
    name = {c: f"{GUEST_PREFIX}{n + 1}" for n, c in enumerate(ok)}
    for g, c in zip(keep, cl):
        if c in name:
            for i in g:
                out[i] = name[c]
    return out


def ahc(X: np.ndarray, min_cos: float) -> np.ndarray:
    """Average-linkage agglomerative clustering on cosine similarity; stop below `min_cos`."""
    n = len(X)
    if n == 1:
        return np.zeros(1, int)
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(_l2(X), method="average", metric="cosine")
    return fcluster(Z, t=1.0 - min_cos, criterion="distance") - 1


# ─── Frames → turns ─────────────────────────────────────────────────────────

@dataclass
class FrameLabels:
    """Per-track frame assignment: label index per active frame (−1 = silent)."""
    labels: list[str]                              # label vocabulary
    per_track: list[tuple[int, np.ndarray, np.ndarray]]   # (start frame, label idx, conf)


def label_frames(trs: Tracks, units: list[Unit], unit_labels: list[str], unit_conf: np.ndarray,
                 params: TrackParams) -> FrameLabels:
    """Each track's speech frames take the label of the unit of that track nearest in time
    (within fill_s; else UNKNOWN). Speech is the track's activity, padded by pad_s into
    frames where no track is active."""
    vocab: dict[str, int] = {}
    thr = params.threshold if params.speech_threshold is None else params.speech_threshold
    n_act = n_active(trs, thr)
    pad = int(round(params.pad_s / trs.step))
    by_track: dict[int, list[int]] = {}
    for i, u in enumerate(units):
        by_track.setdefault(u.track, []).append(i)
    per = []
    unk = vocab.setdefault(UNKNOWN, len(vocab))
    owner = _pad_owner(trs, thr, pad, n_act) if pad else None
    for k, tr in enumerate(trs.tracks):
        act = tr.prob >= thr
        if owner is not None:
            act = act | (owner[tr.start:tr.end] == k)
        lab = np.full(len(act), -1, np.int32)
        conf = np.zeros(len(act), np.float32)
        f = np.flatnonzero(act)
        idx = sorted(by_track.get(k, []), key=lambda i: units[i].t0)
        if len(f) and idx:
            t = trs.time(tr.start + f) + trs.step / 2
            t0 = np.array([units[i].t0 for i in idx])
            t1 = np.array([units[i].t1 for i in idx])
            j = np.clip(np.searchsorted(t0, t, side="right") - 1, 0, len(idx) - 1)
            d_prev = np.maximum(0.0, t - t1[j])                  # 0 inside unit j
            jn = np.minimum(j + 1, len(idx) - 1)
            d_next = np.maximum(0.0, t0[jn] - t)
            use_next = (d_next < d_prev) & (jn != j)
            jj = np.where(use_next, jn, j)
            dist = np.where(use_next, d_next, d_prev)
            ids = np.array([vocab.setdefault(unit_labels[i], len(vocab)) for i in idx])
            cf = np.array([unit_conf[i] for i in idx], np.float32)
            near = dist <= params.fill_s
            lab[f] = np.where(near, ids[jj], unk)
            conf[f] = np.where(near, cf[jj], 0.0)
        elif len(f):
            lab[f] = unk
        per.append((tr.start, lab, conf))
    return FrameLabels(list(vocab), per)


def _pad_owner(trs: Tracks, threshold: float, pad: int, n_act: np.ndarray) -> np.ndarray:
    """Silent frames within `pad` frames of a track's speech edge → that track (the nearest
    edge wins, so a short gap between two speakers is split between them, not doubled)."""
    best = np.full(trs.n_frames, pad + 1, np.int32)
    owner = np.full(trs.n_frames, -1, np.int32)
    for k, tr in enumerate(trs.tracks):
        act = tr.prob >= threshold
        for a, b in _runs(act):
            for lo, hi, dist in ((max(0, a - pad), a, lambda f, a=a: a - f),
                                 (b, min(len(act), b + pad), lambda f, b=b: f - b + 1)):
                if hi <= lo:
                    continue
                f = np.arange(lo, hi)
                g = tr.start + f
                d = dist(f)
                take = (n_act[g] == 0) & (d < best[g])
                best[g[take]] = d[take]
                owner[g[take]] = k
    return owner


def frames_to_turns(trs: Tracks, fl: FrameLabels) -> list[tuple[int, Turn]]:
    """(track, Turn) for every run of one label on one track."""
    out = []
    for k, (start, lab, conf) in enumerate(fl.per_track):
        if not len(lab):
            continue
        brk = np.flatnonzero(np.diff(lab) != 0) + 1
        for a, b in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(lab)]])):
            if lab[a] < 0:
                continue
            s = float(trs.time(start + a))
            e = min(float(trs.time(start + b)), trs.duration)
            if e > s:
                out.append((k, Turn.from_label(s, e, fl.labels[lab[a]], float(conf[a:b].mean()))))
    return out


def exclusive_turns(trs: Tracks, fl: FrameLabels) -> list[Turn]:
    """One label per frame: the active track with the highest probability wins."""
    best_p = np.full(trs.n_frames, -1.0, np.float32)
    best_l = np.full(trs.n_frames, -1, np.int32)
    best_c = np.zeros(trs.n_frames, np.float32)
    for tr, (start, lab, conf) in zip(trs.tracks, fl.per_track):
        sl = slice(start, start + len(lab))
        p = np.where(lab >= 0, tr.prob[:len(lab)], -1.0)
        win = p > best_p[sl]
        best_p[sl] = np.where(win, p, best_p[sl])
        best_l[sl] = np.where(win, lab, best_l[sl])
        best_c[sl] = np.where(win, conf, best_c[sl])
    out = []
    brk = np.flatnonzero(np.diff(best_l) != 0) + 1
    for a, b in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(best_l)]])):
        if best_l[a] < 0:
            continue
        s, e = float(trs.time(a)), min(float(trs.time(b)), trs.duration)
        if e > s:
            out.append(Turn.from_label(s, e, fl.labels[best_l[a]], float(best_c[a:b].mean())))
    return merge_touching(out)


# ─── Second pass ────────────────────────────────────────────────────────────

@dataclass
class TurnCandidate:
    track: int
    f0: int                         # global frames [f0, f1)
    f1: int
    label: str
    spans: list[tuple[float, float]]
    clean_s: float


def long_turns(trs: Tracks, fl: FrameLabels, units: list[Unit], params: TrackParams) -> list[TurnCandidate]:
    """Contiguous same-label runs of one track (pauses ≤ turn_gap_s allowed) lasting
    ≥ turn_min_s, with the track's clean spans inside them."""
    out = []
    by_track: dict[int, list[Unit]] = {}
    for u in units:
        by_track.setdefault(u.track, []).append(u)
    for k, (start, lab, _) in enumerate(fl.per_track):
        f = np.flatnonzero(lab >= 0)
        if not len(f):
            continue
        gap = int(round(params.turn_gap_s / trs.step))
        brk = np.flatnonzero((np.diff(lab[f]) != 0) | (np.diff(f) > gap)) + 1
        for a, b in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(f)]])):
            f0, f1 = start + int(f[a]), start + int(f[b - 1]) + 1
            t0, t1 = float(trs.time(f0)), float(trs.time(f1))
            label = fl.labels[lab[f[a]]]
            if t1 - t0 < params.turn_min_s or label.startswith(GUEST_PREFIX):
                continue
            spans = [(max(s, t0), min(e, t1)) for u in by_track.get(k, []) for s, e in u.spans
                     if e > t0 and s < t1]
            spans = [(s, e) for s, e in spans if e > s]
            cs = sum(e - s for s, e in spans)
            if cs >= params.min_embed_s:
                out.append(TurnCandidate(k, f0, f1, label, spans, cs))
    return out


def apply_second_pass(trs: Tracks, fl: FrameLabels, cands: list[TurnCandidate], S: np.ndarray,
                      emb_s: np.ndarray, names: list[str], cal: Calibration, params: TrackParams) -> int:
    """Relabel candidate turns whose own posterior (calibrated LLR + θ prior, no HMM) is
    ≥ turn_override_p for a different state, unless another track speaking at the same
    time already carries that name. Returns the number of relabelled turns."""
    K = len(names)
    changed = 0
    for c, s_row, d in zip(cands, S, emb_s):
        e = np.concatenate([cal.llr(s_row, d), [params.theta]]).astype(np.float64)
        p = np.exp(e - _logsumexp(e, 0))
        best = int(p.argmax())
        new = names[best] if best < K else UNKNOWN
        if new == c.label or p[best] < params.turn_override_p:
            continue
        if best < K and _name_elsewhere(fl, c, new):
            continue
        start, lab, conf = fl.per_track[c.track]
        ids = {l: i for i, l in enumerate(fl.labels)}
        if new not in ids:
            fl.labels.append(new)
            ids[new] = len(fl.labels) - 1
        seg = slice(c.f0 - start, c.f1 - start)
        m = lab[seg] >= 0
        lab[seg][m] = ids[new]
        conf[seg][m] = p[best]
        changed += 1
    return changed


def _name_elsewhere(fl: FrameLabels, c: TurnCandidate, name: str) -> bool:
    if name not in fl.labels:
        return False
    li = fl.labels.index(name)
    for k, (start, lab, _) in enumerate(fl.per_track):
        if k == c.track:
            continue
        a, b = max(c.f0, start), min(c.f1, start + len(lab))
        if b > a and (lab[a - start:b - start] == li).any():
            return True
    return False


# ─── Assembly ───────────────────────────────────────────────────────────────

def build_timeline(trs: Tracks, fl: FrameLabels, meta: dict | None = None) -> Timeline:
    turns = [t for _, t in frames_to_turns(trs, fl)]
    turns.sort(key=lambda t: (t.start, t.end, t.speaker))
    return Timeline(turns, exclusive_turns(trs, fl), dict(meta or {}))


def decode_all(trs: Tracks, units: list[Unit], X: np.ndarray, scorer: Scorer, cal: Calibration,
               params: TrackParams, turn_scores=None) -> tuple[FrameLabels, dict]:
    """Everything after the unit embeddings. `turn_scores(cands) -> (S, emb_s)` supplies the
    second-pass scores (it embeds the candidate turns); None skips the second pass."""
    S = scorer.scores(X)
    E = unit_emissions(units, S, cal, params.theta)
    dec = decode_units(units, trs, E, scorer.names, params)
    labels = cluster_guests(units, dec.label, X, params) if len(X) else dec.label
    fl = label_frames(trs, units, labels, dec.conf, params)
    info = {"units": len(units), "embedded_units": int(sum(u.emb_row >= 0 for u in units)),
            "tracks": len(trs.tracks)}
    if params.second_pass and turn_scores is not None:
        cands = long_turns(trs, fl, units, params)
        if cands:
            St, dt = turn_scores(cands)
            info["second_pass"] = {"candidates": len(cands),
                                   "relabelled": apply_second_pass(trs, fl, cands, St, dt, scorer.names, cal, params)}
    return fl, info


# ─── Store / roster plumbing and the public entry point ────────────────────

def scorer_from_store(store, model_id: str, roster: list[str] | None, channel: str | None = None,
                      exclude_meetings: list[str] | None = None, top_n: int = 100) -> Scorer:
    """Roster models (channel sub-centroid when the store has one) and the AS-norm cohort:
    every enrolled person's exemplars plus the cohort table, each row tagged with its owner
    so a candidate is never normalised against their own clips."""
    from . import scoring as sc

    ex = store.exemplars(model_id, exclude_meetings=exclude_meetings)
    enr = sc.build_enrollment(ex)
    names = [n for n in (roster or sorted(enr)) if n in enr]
    models = np.stack([enr[n].reference(channel) for n in names]) if names else np.zeros((0, 1), np.float32)
    vecs = [pm.exemplars for pm in enr.values()]
    owners = [n for n, pm in enr.items() for _ in range(len(pm.exemplars))]
    extra, labels = store.cohort(model_id, exclude_meetings=exclude_meetings)
    if extra.size:
        vecs.append(extra)
        owners += list(labels)
    cohort = np.concatenate(vecs) if vecs else np.zeros((0, models.shape[1]), np.float32)
    return Scorer(names, models, cohort, top_n, owners=np.array(owners, dtype=object))


def load_calibration(store, model_id: str) -> Calibration:
    entry = store.load_calibration(model_id) if store is not None else None
    tr = (entry or {}).get("track")
    return Calibration.from_dict(tr) if tr else DEFAULT_CALIBRATION


def resolve_backend(backend: str) -> str:
    from . import segment as sg
    if backend == "auto":
        return "nemotron3" if sg.available("nemotron3") else "seg3"
    return backend


def track(audio_path, roster: list[str] | None, store, embedder, *, backend: str = "auto",
          params: TrackParams = TrackParams(), channel: str | None = None,
          calibration: Calibration | None = None, exclude_meetings: list[str] | None = None,
          wave: np.ndarray | None = None, info: dict | None = None) -> Timeline:
    """Who speaks when in `audio_path`, as a Timeline over `roster` (None: everyone enrolled).

    `embedder` follows the embedders.Embedder protocol; its `model_id` selects the store's
    exemplars and calibration. `backend`: "auto" (nemotron3 if installed, else seg3),
    "nemotron3" or "seg3". `exclude_meetings` leaves those meetings' exemplars out of
    enrollment and cohort (leave-one-meeting-out evaluation).
    """
    from . import audio as au
    from . import segment as sg
    from .footprint import FootprintGuard, gb, peak_footprint

    info = info if info is not None else {}
    guard = FootprintGuard()
    t_all = time.time()
    if wave is None:
        wave = au.decode_audio(audio_path)
    dur = len(wave) / SAMPLE_RATE
    backend = resolve_backend(backend)
    t = time.time()
    seg = sg.segment(wave, backend=backend)
    info["segment_s"] = round(time.time() - t, 2)
    trs = tracks_from_segmentation(seg, params)
    del seg
    units = build_units(trs, params)
    t = time.time()
    X = embed_units(embedder, wave, units, params, guard)
    info["embed_s"] = round(time.time() - t, 2)
    scorer = scorer_from_store(store, embedder.model_id, roster, channel, exclude_meetings, params.top_n)
    cal = calibration or load_calibration(store, embedder.model_id)
    cap = min(params.turn_max_embed_s, au.cap_for(au.embedder_device(embedder)))

    def turn_scores(cands):
        E, rows = embed_spans(embedder, wave, [c.spans for c in cands], cap, params.quant_s, guard)
        S = np.zeros((len(cands), len(scorer.names)), np.float32)
        d = np.array([min(c.clean_s, cap) for c in cands], np.float32)
        if len(rows):
            S[rows] = scorer.scores(E)
        missing = np.setdiff1d(np.arange(len(cands)), rows)
        d[missing] = 0.25                                   # no audio: an uninformative LLR
        S[missing] = 0.0
        return S, d

    t = time.time()
    fl, dinfo = decode_all(trs, units, X, scorer, cal, params, turn_scores)
    info["decode_s"] = round(time.time() - t, 2)
    info.update(dinfo)
    meta = {"backend": backend, "params": params.to_dict(), "calibration": cal.to_dict(),
            "roster": scorer.names, "model_id": embedder.model_id, "duration": round(dur, 3)}
    tl = build_timeline(trs, fl, meta)
    info.update(total_s=round(time.time() - t_all, 2), xrt=round(dur / max(time.time() - t_all, 1e-6), 1),
                peak_footprint_gb=gb(peak_footprint()))
    tl.meta["info"] = dict(info)
    return tl
