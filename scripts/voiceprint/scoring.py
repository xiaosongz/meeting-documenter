"""
Voiceprint scoring: enrollment models, cluster embeddings, AS-norm, Hungarian+Unknown.

Everything here is in-memory numpy — no store, no embedder, no audio — so the
LOMO evaluation can rebuild enrollment from arbitrary exemplar sets.

Defaults (standard speaker-verification practice):
  * centroid = mean of L2-normed exemplars, re-normed; per-(person, channel)
    sub-centroids when a channel has ≥ min_sub exemplars (else fall back);
  * MAD pruning: drop items whose cosine to their centroid < median − k·MAD;
  * cluster embedding = duration-weighted mean after pruning; < 8 s → insufficient;
  * AS-norm s' = ½[(s−μe)/σe + (s−μt)/σt] over the top-N cohort scores; the
    cohort for candidate p excludes p's own clips on both sides;
  * Hungarian over clusters × (candidates ∪ one Unknown dummy per cluster at τ);
  * two-speaker permutation margin Δ and per-cluster best-vs-second gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = 1e-12


# ─── Basics ─────────────────────────────────────────────────────────────────

def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + EPS)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + EPS)


def mean_direction(embs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) mean of L2-normed rows, re-normed."""
    e = l2norm(embs)
    w = np.ones(len(e), np.float32) if weights is None else np.asarray(weights, np.float32)
    return l2norm((e * w[:, None]).sum(axis=0) / (w.sum() + EPS))


def mad_prune(embs: np.ndarray, k: float = 3.0, weights: np.ndarray | None = None,
              min_n: int = 5) -> np.ndarray:
    """Boolean keep-mask: cosine to (weighted) centroid ≥ median − k·MAD (raw MAD).

    Skipped (all True) when n < min_n or MAD == 0.
    """
    n = len(embs)
    keep = np.ones(n, dtype=bool)
    if n < min_n:
        return keep
    cos = l2norm(embs) @ mean_direction(embs, weights)
    med = float(np.median(cos))
    mad = float(np.median(np.abs(cos - med)))
    if mad <= 0:
        return keep
    keep = cos >= med - k * mad
    return keep


# ─── Enrollment ─────────────────────────────────────────────────────────────

@dataclass
class PersonModel:
    name: str
    centroid: np.ndarray = field(repr=False)
    exemplars: np.ndarray = field(repr=False)        # kept (post-prune) L2-normed rows
    channels: list[str] = field(default_factory=list)  # per kept exemplar
    sub: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    n_total: int = 0
    n_pruned: int = 0
    net_dur: float = 0.0
    meetings: list[str] = field(default_factory=list)

    def reference(self, channel: str | None = None) -> np.ndarray:
        if channel and channel in self.sub:
            return self.sub[channel]
        return self.centroid

    def info(self) -> dict:
        return {"name": self.name, "n_total": self.n_total, "n_pruned": self.n_pruned,
                "net_dur": round(self.net_dur, 1), "sub_channels": sorted(self.sub),
                "meetings": len(self.meetings)}


def build_enrollment(items: Iterable, prune_k: float = 3.0, min_sub: int = 3,
                     prune_min_n: int = 5) -> dict[str, PersonModel]:
    """Build PersonModels from exemplar-like objects.

    Each item needs `.person` and `.embedding`; optional `.channel`,
    `.net_dur`, `.meeting_key` (store.Exemplar fits). Tuples
    (person, embedding[, channel[, net_dur[, meeting_key]]]) also work.
    """
    groups: dict[str, list[tuple]] = {}
    for it in items:
        if isinstance(it, tuple):
            person, emb, *rest = it
            ch = rest[0] if len(rest) > 0 else "unknown"
            dur = rest[1] if len(rest) > 1 else 0.0
            mk = rest[2] if len(rest) > 2 else None
        else:
            person, emb = it.person, it.embedding
            ch = getattr(it, "channel", None) or "unknown"
            dur = getattr(it, "net_dur", 0.0) or 0.0
            mk = getattr(it, "meeting_key", None)
        groups.setdefault(person, []).append((np.asarray(emb, np.float32), ch, float(dur), mk))

    models: dict[str, PersonModel] = {}
    for person, rows in groups.items():
        E = l2norm(np.stack([r[0] for r in rows]))
        chans = [r[1] for r in rows]
        durs = np.array([r[2] for r in rows], np.float32)
        keep = mad_prune(E, prune_k, min_n=prune_min_n)
        Ek = E[keep]
        ck = [c for c, k in zip(chans, keep) if k]
        sub = {}
        for ch in sorted(set(ck)):
            idx = [i for i, c in enumerate(ck) if c == ch]
            if ch != "unknown" and len(idx) >= min_sub:
                sub[ch] = mean_direction(Ek[idx])
        models[person] = PersonModel(
            name=person, centroid=mean_direction(Ek), exemplars=Ek, channels=ck, sub=sub,
            n_total=len(rows), n_pruned=int((~keep).sum()), net_dur=float(durs[keep].sum()),
            meetings=sorted({r[3] for r in rows if r[3]}))
    return models


# ─── Cluster embedding ──────────────────────────────────────────────────────

@dataclass
class ClusterVec:
    name: str
    vec: np.ndarray | None = field(repr=False)
    net_dur: float
    n_clips: int
    n_pruned: int
    insufficient: bool


def cluster_embedding(name: str, embs: np.ndarray, durs: Iterable[float],
                      prune_k: float = 3.0, min_net_s: float = 8.0,
                      prune_min_n: int = 5) -> ClusterVec:
    durs = np.asarray(list(durs), np.float32)
    if len(embs) == 0:
        return ClusterVec(name, None, 0.0, 0, 0, True)
    keep = mad_prune(embs, prune_k, weights=durs, min_n=prune_min_n)
    vec = mean_direction(embs[keep], durs[keep])
    net = float(durs[keep].sum())
    return ClusterVec(name, vec, round(net, 2), int(len(embs)), int((~keep).sum()), net < min_net_s)


# ─── Scoring ────────────────────────────────────────────────────────────────

def raw_score(vec: np.ndarray, pm: PersonModel, channel: str | None = None,
              mode: str = "centroid", top_k: int = 5) -> float:
    if mode == "top5":
        pool = pm.exemplars
        if channel:
            idx = [i for i, c in enumerate(pm.channels) if c == channel]
            if len(idx) >= top_k:
                pool = pm.exemplars[idx]
        sims = np.sort(pool @ vec)[::-1]
        return float(sims[:top_k].mean())
    return float(pm.reference(channel) @ vec)


def build_cohort(enrollment: dict[str, PersonModel], extra_vecs: np.ndarray | None = None,
                 extra_labels: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Cohort = every enrolled person's kept exemplars + extra (non-enrolled) clips.

    Returns (vecs (N, D), labels (N,)); per-candidate exclusion happens at score time.
    """
    vecs, labels = [], []
    for name, pm in enrollment.items():
        vecs.append(pm.exemplars)
        labels += [name] * len(pm.exemplars)
    if extra_vecs is not None and len(extra_vecs):
        vecs.append(l2norm(extra_vecs))
        labels += list(extra_labels or ["cohort"] * len(extra_vecs))
    if not vecs:
        return np.zeros((0, 0), np.float32), np.array([], dtype=object)
    return np.concatenate(vecs, axis=0), np.array(labels, dtype=object)


def _top_stats(sims: np.ndarray, top_n: int) -> tuple[float, float]:
    n = min(top_n, sims.size)
    top = np.partition(sims, sims.size - n)[sims.size - n:]
    return float(top.mean()), float(max(top.std(), 1e-4))


def as_norm(s: float, enroll_vec: np.ndarray, test_vec: np.ndarray,
            cohort: np.ndarray, top_n: int = 100) -> float:
    """Adaptive symmetric normalisation: ½[(s−μe)/σe + (s−μt)/σt] over top-N cohort scores."""
    mu_e, sd_e = _top_stats(cohort @ enroll_vec, top_n)
    mu_t, sd_t = _top_stats(cohort @ test_vec, top_n)
    return 0.5 * ((s - mu_e) / sd_e + (s - mu_t) / sd_t)


@dataclass
class ScoreMatrix:
    clusters: list[str]
    persons: list[str]
    raw: np.ndarray                   # (C, P) cosine
    norm: np.ndarray | None           # (C, P) AS-norm, None if cohort too small
    kind: str                         # "asnorm" | "cosine" — what `used` holds
    cohort_size: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def used(self) -> np.ndarray:
        return self.norm if self.norm is not None else self.raw


def score_matrix(clusters: list[ClusterVec], enrollment: dict[str, PersonModel],
                 candidates: list[str] | None = None,
                 cohort: tuple[np.ndarray, np.ndarray] | None = None,
                 channel: str | None = None, mode: str = "centroid",
                 top_n: int = 100, min_cohort: int = 50,
                 use_asnorm: bool = True) -> ScoreMatrix:
    persons = [p for p in (candidates or sorted(enrollment)) if p in enrollment]
    C, P = len(clusters), len(persons)
    raw = np.full((C, P), np.nan, np.float32)
    for i, cv in enumerate(clusters):
        if cv.vec is None:
            continue
        for j, p in enumerate(persons):
            raw[i, j] = raw_score(cv.vec, enrollment[p], channel, mode)

    warnings: list[str] = []
    if cohort is None:
        cohort = build_cohort(enrollment)
    cvecs, clabels = cohort
    norm = None
    csize = int(len(cvecs))
    if use_asnorm and P:
        sizes = [int((clabels != p).sum()) for p in persons] if csize else [0]
        if min(sizes) < min_cohort:
            warnings.append(f"AS-norm skipped: cohort {min(sizes)} clips < min {min_cohort}; "
                            "scores are raw cosine (model-specific, less portable thresholds)")
        else:
            norm = np.full((C, P), np.nan, np.float32)
            for j, p in enumerate(persons):
                co = cvecs[clabels != p]
                e_vec = enrollment[p].reference(channel)
                for i, cv in enumerate(clusters):
                    if cv.vec is None:
                        continue
                    norm[i, j] = as_norm(float(raw[i, j]), e_vec, cv.vec, co, top_n)
            if csize < 500:
                warnings.append(f"AS-norm cohort is small ({csize} clips; research target ≥500 from ≥30 speakers)")
    kind = "asnorm" if norm is not None else "cosine"
    return ScoreMatrix([c.name for c in clusters], persons, raw, norm, kind, csize, warnings)


# ─── Thresholds ─────────────────────────────────────────────────────────────

@dataclass
class Thresholds:
    tau: float          # open-set accept threshold (Unknown dummy cost)
    tau_low: float      # relaxed threshold when the roster is closed
    delta: float        # min two-speaker permutation margin
    gap: float          # min best-vs-second gap for "high"
    kind: str           # "asnorm" | "cosine"
    calibrated: bool = False
    source: str | None = None


# Conservative, UNCALIBRATED priors (more Unknowns rather than wrong names). They are not
# tuned on any dataset; without calibration, assign() never reports "high" confidence.
DEFAULT_THRESHOLDS = {
    "asnorm": Thresholds(tau=4.0, tau_low=2.5, delta=4.0, gap=1.5, kind="asnorm"),
    "cosine": Thresholds(tau=0.60, tau_low=0.45, delta=0.30, gap=0.10, kind="cosine"),
}


def load_thresholds(entry: dict | None, kind: str) -> Thresholds:
    """Pick thresholds for `kind` from a calibration.json model entry, else defaults.

    Entry shape: {"asnorm": {"tau":…, "tau_low":…, "delta":…, "gap":…}, "cosine": {…}, …}
    """
    base = DEFAULT_THRESHOLDS[kind]
    if entry and isinstance(entry.get(kind), dict):
        e = entry[kind]
        return Thresholds(tau=float(e.get("tau", base.tau)), tau_low=float(e.get("tau_low", base.tau_low)),
                          delta=float(e.get("delta", base.delta)), gap=float(e.get("gap", base.gap)),
                          kind=kind, calibrated=True, source=entry.get("_source"))
    return Thresholds(**{**asdict(base), "calibrated": False})


# ─── Assignment ─────────────────────────────────────────────────────────────

@dataclass
class Decision:
    cluster: str
    proposed: str | None            # None → keep "Speaker X"
    confidence: str                 # "high" | "low" | "unknown"
    score: float | None = None
    second: float | None = None
    second_name: str | None = None
    gap: float | None = None
    best: str | None = None         # best-scoring candidate regardless of decision
    tentative: str | None = None    # name withheld because of margin/threshold
    reasons: list[str] = field(default_factory=list)
    insufficient: bool = False
    net_dur: float = 0.0
    existing_label: str | None = None
    label_agrees: bool | None = None
    by_elimination: bool = False    # name inferred from the roster, NOT from the voice


def assign(sm: ScoreMatrix, th: Thresholds, cluster_info: list[ClusterVec] | None = None,
           roster: list[str] | None = None) -> tuple[list[Decision], dict]:
    """Hungarian over clusters × (candidates ∪ Unknown dummies). Returns (decisions, extras).

    `roster` given ⇒ closed set: Unknown cost is τ_low and leftover elimination applies.
    Uncalibrated thresholds (`th.calibrated` False) cap every proposal at "low", which also
    disables roster elimination: voice alone never names anyone until `calibrate` has run.
    """
    S = sm.used
    C, P = S.shape
    closed = bool(roster)
    floor = th.tau_low if closed else th.tau
    info = cluster_info or [ClusterVec(c, None, 0.0, 0, 0, False) for c in sm.clusters]
    insuff = np.array([ci.insufficient or np.isnan(S[i]).all() if P else True
                       for i, ci in enumerate(info)], dtype=bool)

    decisions = [Decision(c, None, "unknown", insufficient=bool(insuff[i]), net_dur=info[i].net_dur)
                 for i, c in enumerate(sm.clusters)]
    extras: dict = {"closed_roster": closed, "floor": floor, "pair_margins": []}

    # best / second for every cluster (diagnostic, independent of the decision)
    for i, d in enumerate(decisions):
        if P == 0 or np.isnan(S[i]).all():
            continue
        order = np.argsort(-np.nan_to_num(S[i], nan=-np.inf))
        d.best = sm.persons[order[0]]
        d.score = float(S[i, order[0]])        # best score (overwritten by the assigned score if named)
        if P > 1:
            d.second_name = sm.persons[order[1]]
            d.second = float(S[i, order[1]])

    valid = np.flatnonzero(~insuff)
    if P and len(valid):
        M = np.nan_to_num(S[valid], nan=-1e6)
        big = np.hstack([M, np.full((len(valid), len(valid)), floor, np.float64)])
        rows, cols = linear_sum_assignment(big, maximize=True)
        for r, c in zip(rows, cols):
            i = int(valid[r])
            d = decisions[i]
            if c < P:
                d.proposed = sm.persons[c]
                d.score = float(S[i, c])
                others = np.delete(np.nan_to_num(S[i], nan=-np.inf), c)
                if others.size:
                    d.second = float(others.max())
                    d.second_name = sm.persons[int(np.flatnonzero(np.arange(P) != c)[int(others.argmax())])]
                    d.gap = d.score - d.second
                if d.score >= th.tau and (d.gap is None or d.gap >= th.gap) and th.calibrated:
                    d.confidence = "high"
                elif d.score >= th.tau and (d.gap is None or d.gap >= th.gap):
                    d.confidence = "low"
                    d.reasons.append("uncalibrated thresholds: capped at low (run `identify_speakers.py calibrate`)")
                else:
                    d.confidence = "low"
                    if d.score < th.tau:
                        d.reasons.append(f"score {d.score:.2f} < τ {th.tau:.2f} (accepted via closed roster τ_low)")
                    if d.gap is not None and d.gap < th.gap:
                        d.reasons.append(f"gap to {d.second_name} {d.gap:.2f} < {th.gap:.2f}")
                if d.best and d.best != d.proposed:
                    d.reasons.append(f"best candidate {d.best} taken by another cluster (exclusivity)")
            else:
                if not np.isnan(S[i]).all():
                    bs = float(np.nanmax(S[i]))
                    d.score = bs
                    if bs >= floor:
                        d.reasons.append(f"best {d.best} {bs:.2f} ≥ {'τ_low' if closed else 'τ'} but {d.best} "
                                         "is assigned to another cluster (exclusivity) — possible split duplicate")
                    else:
                        d.reasons.append(f"best {d.best} {bs:.2f} < {'τ_low' if closed else 'τ'} {floor:.2f}")

    # pairwise swap margins between named clusters
    named = [i for i, d in enumerate(decisions) if d.proposed]
    pidx = {p: j for j, p in enumerate(sm.persons)}
    for a in range(len(named)):
        for b in range(a + 1, len(named)):
            i1, i2 = named[a], named[b]
            p1, p2 = pidx[decisions[i1].proposed], pidx[decisions[i2].proposed]
            margin = float(S[i1, p1] + S[i2, p2] - S[i1, p2] - S[i2, p1])
            extras["pair_margins"].append({"clusters": [sm.clusters[i1], sm.clusters[i2]],
                                           "margin": round(margin, 3)})
            if margin < th.delta:
                for i in (i1, i2):
                    d = decisions[i]
                    d.reasons.append(f"swap margin vs {sm.clusters[i2 if i == i1 else i1]} "
                                     f"{margin:.2f} < δ {th.delta:.2f}")
                    if len(valid) == 2:          # two-speaker rule: ambiguous swap → Unknown
                        d.tentative, d.proposed, d.confidence = d.proposed, None, "unknown"
                    elif d.confidence == "high":
                        d.confidence = "low"
    if len(valid) == 2 and len(named) == 2:
        extras["two_speaker_margin"] = extras["pair_margins"][0]["margin"]

    # roster elimination: one leftover cluster + one leftover roster person
    if closed:
        taken = {d.proposed for d in decisions if d.proposed}
        left_people = [p for p in roster if p not in taken]
        left_clusters = [d for d in decisions if not d.proposed and not d.tentative]
        confident = all(d.confidence == "high" for d in decisions if d.proposed)
        if len(left_people) == 1 and len(left_clusters) == 1 and confident and taken:
            d = left_clusters[0]
            person = left_people[0]
            # Guard: a leftover cluster that sounds like an already-named speaker (≥ τ_low) is
            # most likely a diarization split-duplicate of that speaker, not the absent attendee.
            dup = (d.best is not None and d.best != person and d.score is not None
                   and d.score >= th.tau_low)
            if dup:
                d.reasons.append(f"not eliminated to {person}: sounds like {d.best} ({d.score:.2f} ≥ "
                                 f"τ_low {th.tau_low:.2f}) — likely a split duplicate of {d.best}")
                extras["elimination_skipped"] = {"cluster": d.cluster, "person": person,
                                                 "looks_like": d.best, "score": round(d.score, 3)}
            else:
                d.proposed, d.confidence, d.by_elimination = person, "low", True
                d.reasons.append("by roster elimination (not a voice match) — verify with content")
                extras["elimination"] = {"cluster": d.cluster, "person": person}
    return decisions, extras


# ─── Label self-check ───────────────────────────────────────────────────────

def label_check(sm: ScoreMatrix, decisions: list[Decision], labels: dict[str, str] | None,
                th: Thresholds) -> dict:
    """Compare an existing cluster→name map (e.g. sidecar speaker_map) with the voice evidence.

    Two independent signals, reported separately:
      * label margins — for each labelled cluster pair, s(c1,L1)+s(c2,L2)−s(c1,L2)−s(c2,L1);
        works even when both clusters stay Unknown (open-set τ not met);
      * decisions — whether the proposed/tentative names match the labels.
    status (first match wins):
      inversion          some pair margin ≤ −δ
      conflict           a high-confidence proposal contradicts its label
      weak_disagreement  only low/unknown-confidence proposals contradict
      agree              proposals match labels (and no pair margin ≤ −δ)
      agree_by_margin    no proposal to compare, but every scored pair margin ≥ +δ
      unverifiable       otherwise (e.g. |margin| < δ and no named proposals)
      no_labels
    """
    if not labels:
        return {"status": "no_labels"}
    pidx = {p: j for j, p in enumerate(sm.persons)}
    cidx = {c: i for i, c in enumerate(sm.clusters)}
    S = sm.used
    out: dict = {"pairs": []}
    disagree_hi, disagree_lo, agree = [], [], []
    for d in decisions:
        lab = labels.get(d.cluster)
        d.existing_label = lab
        voice = None if d.by_elimination else (d.proposed or d.tentative)   # voice evidence only
        if not lab or voice is None:
            continue
        d.label_agrees = (voice == lab)
        if d.label_agrees:
            agree.append(d.cluster)
        elif d.confidence == "high":
            disagree_hi.append(d.cluster)
        else:
            disagree_lo.append(d.cluster)
    # label-permutation margins: negative ⇒ the swapped labels fit the voices better
    labelled = [(c, p) for c, p in labels.items() if c in cidx and p in pidx]
    inversions, supported, ambiguous = [], [], []
    for a in range(len(labelled)):
        for b in range(a + 1, len(labelled)):
            (c1, p1), (c2, p2) = labelled[a], labelled[b]
            i1, i2, j1, j2 = cidx[c1], cidx[c2], pidx[p1], pidx[p2]
            m = float(S[i1, j1] + S[i2, j2] - S[i1, j2] - S[i2, j1])
            if np.isnan(m):
                continue
            out["pairs"].append({"clusters": [c1, c2], "labels": [p1, p2], "label_margin": round(m, 3)})
            if m <= -th.delta:
                inversions.append([c1, c2])
            elif m >= th.delta:
                supported.append([c1, c2])
            else:
                ambiguous.append([c1, c2])
    out["unenrolled_labels"] = sorted({p for p in labels.values() if p not in pidx})
    if inversions:
        out["status"] = "inversion"
        out["inverted_pairs"] = inversions
    elif disagree_hi:
        out["status"] = "conflict"
    elif disagree_lo:
        out["status"] = "weak_disagreement"
    elif agree:
        out["status"] = "agree"
    elif supported and not ambiguous:
        out["status"] = "agree_by_margin"
    else:
        out["status"] = "unverifiable"
    out.update({"agree": agree, "disagree_high": disagree_hi, "disagree_low": disagree_lo,
                "supported_pairs": supported, "ambiguous_pairs": ambiguous})
    return out


# ─── End-to-end (in-memory) ─────────────────────────────────────────────────

def identify(cluster_vecs: list[ClusterVec], enrollment: dict[str, PersonModel],
             thresholds: Thresholds | None = None, calibration: dict | None = None,
             roster: list[str] | None = None, labels: dict[str, str] | None = None,
             cohort: tuple[np.ndarray, np.ndarray] | None = None,
             channel: str | None = None, mode: str = "centroid",
             top_n: int = 100, min_cohort: int = 50, use_asnorm: bool = True) -> dict:
    """Score + assign + label-check. Returns a JSON-serialisable dict.

    Candidates = roster ∩ enrolled when a roster is given, else all enrolled.
    """
    candidates = [p for p in roster if p in enrollment] if roster else None
    sm = score_matrix(cluster_vecs, enrollment, candidates, cohort, channel, mode,
                      top_n, min_cohort, use_asnorm)
    th = thresholds or load_thresholds(calibration, sm.kind)
    if th.kind != sm.kind:
        th = load_thresholds(calibration, sm.kind)
    decisions, extras = assign(sm, th, cluster_vecs, roster)
    lc = label_check(sm, decisions, labels, th)

    def _mat(a):
        return None if a is None else [[None if np.isnan(x) else round(float(x), 4) for x in row] for row in a]

    return {
        "status": "ok",
        "score_kind": sm.kind,
        "score_mode": mode,
        "calibrated": th.calibrated,
        "thresholds": asdict(th),
        "clusters": sm.clusters,
        "persons": sm.persons,
        "raw_cosine": _mat(sm.raw),
        "asnorm": _mat(sm.norm),
        "cohort_size": sm.cohort_size,
        "cluster_info": [{"cluster": c.name, "net_dur": c.net_dur, "n_clips": c.n_clips,
                          "n_pruned": c.n_pruned, "insufficient": c.insufficient} for c in cluster_vecs],
        "decisions": [asdict(d) for d in decisions],
        "two_speaker_margin": extras.get("two_speaker_margin"),
        "pair_margins": extras["pair_margins"],
        "elimination": extras.get("elimination"),
        "elimination_skipped": extras.get("elimination_skipped"),
        "closed_roster": extras["closed_roster"],
        "roster_not_enrolled": sorted(set(roster or []) - set(enrollment)),
        "label_check": lc,
        "warnings": sm.warnings + ([] if th.calibrated else
                                   [f"UNCALIBRATED thresholds ({sm.kind} priors) — confidence capped at low; "
                                    "run `identify_speakers.py calibrate`"]),
    }
