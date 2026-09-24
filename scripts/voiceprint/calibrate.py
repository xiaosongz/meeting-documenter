"""
Threshold calibration from the user's own enrolled voiceprints (no audio, no labels file).

Leave-one-meeting-out (LOMO) over the store: every meeting that contributed
exemplars is a fold. For fold M, enrollment is rebuilt from all other meetings,
and each person's exemplars from M form one test "cluster" (the longest
`test_clips` clips, like `identify`). This yields three score pools per kind
(raw cosine, and AS-norm when the cohort is large enough):

    target    test unit vs its own person (the person is enrolled from other meetings)
    stranger  test unit vs everyone, when its person is NOT enrolled: a person whose
              only meeting is M, a leave-one-speaker-out (LOSO) pass that removes the
              person from enrollment and cohort, and cohort-table voices from M
    margins   true-permutation margins s(c1,P1)+s(c2,P2)−s(c1,P2)−s(c2,P1) for
              every pair of enrolled people who both spoke in M
    gaps      target minus best non-target

Thresholds (per kind), from all folds:
    tau      = 99th percentile of stranger scores (≈1 % false-accept of unknown voices)
    tau_low  = 95th percentile of stranger scores, ≤ tau (closed-roster floor)
    delta    = 5th percentile of true pair margins (≥ 0)
    gap      = max(0, 5th percentile of gaps)
Minimum pool sizes: stranger ≥ 50 (else that kind is not calibrated), margins ≥ 5
and gaps ≥ 10 (else the conservative prior is kept and marked as such).

Nested estimate: each fold is also decided with thresholds calibrated from the
OTHER folds only, so the reported error counts are not measured on the data that
set the thresholds.

Pure numpy over exemplar-like objects: the CLI (`identify_speakers.py calibrate`)
reads the store and writes calibration.json.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from . import scoring as sc

KINDS = ("cosine", "asnorm")
MIN_STRANGER, MIN_MARGINS, MIN_GAPS = 50, 5, 10
CHANNEL_REFS = ("phone", "zoom", "room")


@dataclass
class Pools:
    target: list[float] = field(default_factory=list)
    stranger: list[float] = field(default_factory=list)
    insider: list[float] = field(default_factory=list)
    margins: list[float] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)

    def extend(self, other: "Pools", keys: Iterable[str] | None = None) -> None:
        for k in keys or ("target", "stranger", "insider", "margins", "gaps"):
            getattr(self, k).extend(getattr(other, k))

    def counts(self) -> dict:
        return {k: len(v) for k, v in asdict(self).items()}


@dataclass
class FoldScores:
    """One scored fold: matrices per kind plus which test row belongs to whom."""
    meeting: str
    clusters: list[sc.ClusterVec]
    truths: list[str]
    loso: list[bool]                       # row scored with its person removed (stranger trial)
    matrices: dict[str, sc.ScoreMatrix]    # kind → matrix restricted to that kind in `.used`


def _unit_key(e) -> tuple[str, str | None]:
    return e.person, e.meeting_key


def _test_cluster(rows: list, name: str, test_clips: int, min_net_s: float) -> sc.ClusterVec:
    rows = sorted(rows, key=lambda e: -(e.net_dur or 0.0))[:test_clips]
    E = np.stack([np.asarray(e.embedding, np.float32) for e in rows])
    return sc.cluster_embedding(name, E, [e.net_dur or 0.0 for e in rows], min_net_s=min_net_s)


def _fold_channel(rows: list) -> str | None:
    ch = Counter(e.channel for e in rows).most_common(1)
    return ch[0][0] if ch and ch[0][0] in CHANNEL_REFS else None


def _kind_view(sm: sc.ScoreMatrix, kind: str) -> sc.ScoreMatrix | None:
    if kind == "cosine":
        return sc.ScoreMatrix(sm.clusters, sm.persons, sm.raw, None, "cosine", sm.cohort_size)
    if sm.norm is None:
        return None
    return sm


def _cohort_arrays(rows: list) -> tuple[np.ndarray | None, list[str]]:
    if not rows:
        return None, []
    return (np.stack([np.asarray(r.embedding, np.float32) for r in rows]),
            [r.person or r.label for r in rows])


def _view_all(sm: sc.ScoreMatrix) -> dict[str, sc.ScoreMatrix]:
    return {k: v for k in KINDS if (v := _kind_view(sm, k)) is not None}


def score_folds(exemplars: list, cohort: list | None = None, mode: str = "centroid",
                test_clips: int = 20, min_net_s: float = 8.0, top_n: int = 100,
                min_cohort: int = 50) -> list[FoldScores]:
    """Score every fold (main pass + LOSO) against enrollment rebuilt without that meeting.

    `exemplars`: store.Exemplar-like (person, embedding, channel, net_dur, meeting_key).
    `cohort`: store.CohortClip-like (label, person, embedding, net_dur, meeting_key); rows
    from the fold's meeting are excluded from its cohort and scored as stranger voices.
    """
    cohort = cohort or []
    by_meeting: dict[str, list] = defaultdict(list)
    for e in exemplars:
        if e.meeting_key:
            by_meeting[e.meeting_key].append(e)
    out = []
    for mk in sorted(by_meeting):
        rest = [e for e in exemplars if e.meeting_key != mk]
        if not rest:
            continue
        enr = sc.build_enrollment(rest)
        co_rows = [r for r in cohort if r.meeting_key != mk]
        co_vecs, co_labels = _cohort_arrays(co_rows)
        channel = _fold_channel(by_meeting[mk])

        clusters, truths, loso = [], [], []
        units: dict[str, list] = defaultdict(list)
        for e in by_meeting[mk]:
            units[e.person].append(e)
        for person, rows in sorted(units.items()):
            clusters.append(_test_cluster(rows, person, test_clips, min_net_s))
            truths.append(person)
            loso.append(person not in enr)
        # non-enrolled voices of this meeting from the cohort table → stranger rows
        strangers: dict[str, list] = defaultdict(list)
        for r in cohort:
            label = r.person or r.label
            if r.meeting_key == mk and label not in enr and label not in units:
                strangers[label].append(r)
        for label, rows in sorted(strangers.items()):
            clusters.append(_test_cluster(rows, label, test_clips, min_net_s))
            truths.append(label)
            loso.append(True)

        sm = sc.score_matrix(clusters, enr, None, sc.build_cohort(enr, co_vecs, co_labels),
                             channel, mode, top_n, min_cohort)
        out.append(FoldScores(mk, clusters, truths, loso, _view_all(sm)))

        # LOSO: each enrolled person of M removed from enrollment and cohort → a stranger trial
        for i, person in enumerate(truths):
            if loso[i] or clusters[i].insufficient:
                continue
            rest_p = [e for e in rest if e.person != person]
            if not rest_p:
                continue
            enr_p = sc.build_enrollment(rest_p)
            cv, cl = _cohort_arrays([r for r in co_rows if (r.person or r.label) != person])
            sm_p = sc.score_matrix([clusters[i]], enr_p, None, sc.build_cohort(enr_p, cv, cl),
                                   channel, mode, top_n, min_cohort)
            out.append(FoldScores(mk, [clusters[i]], [person], [True], _view_all(sm_p)))
    return out


def fold_pools(fs: FoldScores, kind: str) -> Pools | None:
    sm = fs.matrices.get(kind)
    if sm is None:
        return None
    S = sm.used
    pidx = {p: j for j, p in enumerate(sm.persons)}
    pools = Pools()
    named = []
    for i, (cv, truth, is_loso) in enumerate(zip(fs.clusters, fs.truths, fs.loso)):
        if cv.insufficient or np.isnan(S[i]).all():
            continue
        if is_loso or truth not in pidx:
            pools.stranger += [float(x) for x in S[i] if np.isfinite(x)]
            continue
        j = pidx[truth]
        t = float(S[i, j])
        others = [float(S[i, k]) for k in range(len(sm.persons)) if k != j and np.isfinite(S[i, k])]
        pools.target.append(t)
        pools.insider += others
        if others:
            pools.gaps.append(t - max(others))
        named.append((i, j))
    for a in range(len(named)):
        for b in range(a + 1, len(named)):
            (i1, j1), (i2, j2) = named[a], named[b]
            m = float(S[i1, j1] + S[i2, j2] - S[i1, j2] - S[i2, j1])
            if np.isfinite(m):
                pools.margins.append(m)
    return pools


def thresholds_from(pools: Pools, kind: str) -> tuple[sc.Thresholds | None, dict]:
    """(Thresholds or None when the stranger pool is too small, provenance per field)."""
    base = sc.DEFAULT_THRESHOLDS[kind]
    src = {}
    st = np.asarray(pools.stranger, np.float64)
    if st.size < MIN_STRANGER:
        return None, {"reason": f"stranger trials {st.size} < {MIN_STRANGER}"}
    tau = float(np.quantile(st, 0.99))
    tau_low = min(float(np.quantile(st, 0.95)), tau)
    src.update(tau="p99 stranger", tau_low="p95 stranger")
    if len(pools.margins) >= MIN_MARGINS:
        delta = max(0.0, float(np.quantile(pools.margins, 0.05)))
        src["delta"] = "p05 pair margins"
    else:
        delta = base.delta
        src["delta"] = f"prior (pair margins {len(pools.margins)} < {MIN_MARGINS})"
    if len(pools.gaps) >= MIN_GAPS:
        gap = max(0.0, float(np.quantile(pools.gaps, 0.05)))
        src["gap"] = "p05 target-minus-best-other"
    else:
        gap = base.gap
        src["gap"] = f"prior (gaps {len(pools.gaps)} < {MIN_GAPS})"
    return sc.Thresholds(tau=tau, tau_low=tau_low, delta=delta, gap=gap, kind=kind, calibrated=True), src


def eer(target: list[float], nontarget: list[float]) -> float | None:
    t, n = np.asarray(target), np.asarray(nontarget)
    if not t.size or not n.size:
        return None
    thr = np.unique(np.concatenate([t, n]))
    pmiss = np.array([(t < x).mean() for x in thr])
    pfa = np.array([(n >= x).mean() for x in thr])
    i = int(np.argmin(np.abs(pmiss - pfa)))
    return round(float((pmiss[i] + pfa[i]) / 2), 4)


def nested_estimate(folds: list[FoldScores], kind: str) -> dict:
    """Decide each fold with thresholds calibrated on the other meetings only (open set)."""
    per_meeting: dict[str, Pools] = defaultdict(Pools)
    for fs in folds:
        p = fold_pools(fs, kind)
        if p is not None:
            per_meeting[fs.meeting].extend(p)
    res = Counter()
    for fs in folds:
        sm = fs.matrices.get(kind)
        if sm is None:
            continue
        others = Pools()
        for mk, p in per_meeting.items():
            if mk != fs.meeting:
                others.extend(p)
        th, _ = thresholds_from(others, kind)
        if th is None:
            res["folds_skipped"] += 1
            continue
        decisions, _ = sc.assign(sm, th, fs.clusters, None)
        for d, truth, is_loso in zip(decisions, fs.truths, fs.loso):
            if d.insufficient:
                res["insufficient"] += 1
                continue
            enrolled = (not is_loso) and truth in sm.persons
            key = "enrolled" if enrolled else "stranger"
            res[f"{key}_units"] += 1
            if d.proposed is None:
                res[f"{key}_unknown"] += 1
            elif d.proposed == truth:
                res[f"{key}_correct_{d.confidence}"] += 1
            else:
                res[f"{key}_wrong_{d.confidence}"] += 1
    return dict(sorted(res.items()))


def calibrate(exemplars: list, cohort: list | None = None, mode: str = "centroid",
              test_clips: int = 20, min_net_s: float = 8.0, top_n: int = 100,
              min_cohort: int = 50) -> dict:
    """Full calibration result: {kind: {thresholds, sources, counts, eer, nested}}, plus data summary."""
    folds = score_folds(exemplars, cohort, mode, test_clips, min_net_s, top_n, min_cohort)
    people = Counter()
    for (person, mk) in {_unit_key(e) for e in exemplars if e.meeting_key}:
        people[person] += 1
    out: dict = {"folds": len({f.meeting for f in folds}), "people": len(people),
                 "people_multi_meeting": sum(1 for n in people.values() if n >= 2), "kinds": {}}
    for kind in KINDS:
        pools = Pools()
        any_fold = False
        for fs in folds:
            p = fold_pools(fs, kind)
            if p is not None:
                any_fold = True
                pools.extend(p)
        if not any_fold:
            out["kinds"][kind] = {"calibrated": False, "reason": "no folds scored with this kind"}
            continue
        th, src = thresholds_from(pools, kind)
        entry = {"calibrated": th is not None, "counts": pools.counts(), "sources": src,
                 "eer_all": eer(pools.target, pools.insider + pools.stranger),
                 "eer_stranger": eer(pools.target, pools.stranger)}
        if th is not None:
            entry["thresholds"] = {k: round(v, 4) for k, v in asdict(th).items()
                                   if k in ("tau", "tau_low", "delta", "gap")}
            entry["target_accept_at_tau"] = (round(float(np.mean(np.asarray(pools.target) >= th.tau)), 4)
                                             if pools.target else None)
            entry["nested"] = nested_estimate(folds, kind)
        else:
            entry["reason"] = src.get("reason")
        out["kinds"][kind] = entry
    return out
