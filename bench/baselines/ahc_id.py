"""Baseline B1: diarize, then identify clusters ("community-1 style").

Same boundary layer, tracks, units, unit embeddings, enrollment, cohort and calibration as
the tracker (bench/systems/tracker.py drives both); only the decision differs:

    embedded units → average-linkage AHC on cosine (stop below `cos`) → clusters
    → cluster vector = duration-weighted mean → AS-norm vs roster → calibrated LLR at the
      cluster's total embedded duration (capped at `max_dur_s`)
    → Hungarian over clusters × (roster ∪ one UNKNOWN per cluster at log-odds θ)
    → every unit takes its cluster's label; units without an embedding take the label of
      the nearest embedded unit on the same track (UNKNOWN if the track has none).

    python -m bench.systems.tracker run --system b1 --b1 b1.json ...
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class B1Params:
    cos: float = 0.5               # AHC stopping similarity
    theta: float = 0.0             # UNKNOWN log-odds (calibrated LLR units)
    max_dur_s: float = 60.0        # duration fed to the calibration for a cluster


def cluster_identify(units, X: np.ndarray, scorer, cal, b1: B1Params):
    """→ (label per unit, confidence per unit, info)."""
    from scipy.optimize import linear_sum_assignment

    from voiceprint.timeline import UNKNOWN
    from voiceprint.track import _l2, ahc

    emb = [i for i, u in enumerate(units) if u.emb_row >= 0]
    labels = [UNKNOWN] * len(units)
    conf = np.zeros(len(units), np.float32)
    if not emb:
        return labels, conf, {"clusters": 0}
    V = X[[units[i].emb_row for i in emb]]
    w = np.array([units[i].emb_s for i in emb], np.float32)
    cl = ahc(V, b1.cos)
    C = int(cl.max()) + 1
    vec = np.stack([_l2((V[cl == c] * w[cl == c, None]).sum(0)) for c in range(C)])
    dur = np.array([min(w[cl == c].sum(), b1.max_dur_s) for c in range(C)], np.float32)
    K = len(scorer.names)
    L = cal.llr(scorer.scores(vec), dur) if K else np.zeros((C, 0), np.float32)
    cost = np.full((C, K + C), 1e6)
    cost[:, :K] = -L
    cost[np.arange(C), K + np.arange(C)] = -b1.theta
    rows, cols = linear_sum_assignment(cost)
    lab_c = [UNKNOWN] * C
    conf_c = np.zeros(C, np.float32)
    for r, c in zip(rows, cols):
        e = np.concatenate([L[r], [b1.theta]]).astype(np.float64)
        p = np.exp(e - e.max())
        p /= p.sum()
        if c < K:
            lab_c[r], conf_c[r] = scorer.names[c], p[c]
        else:
            conf_c[r] = p[-1]
    for i, c in zip(emb, cl):
        labels[i], conf[i] = lab_c[c], conf_c[c]
    # units without an embedding: nearest embedded unit on the same track
    by_track: dict[int, list[int]] = {}
    for i in emb:
        by_track.setdefault(units[i].track, []).append(i)
    for i, u in enumerate(units):
        if u.emb_row >= 0:
            continue
        cand = by_track.get(u.track)
        if not cand:
            continue
        mid = (u.t0 + u.t1) / 2
        j = min(cand, key=lambda j: abs((units[j].t0 + units[j].t1) / 2 - mid))
        labels[i], conf[i] = labels[j], conf[j]
    return labels, conf, {"clusters": C, "named_clusters": sum(l != UNKNOWN for l in lab_c)}
