"""
Duration-aware score calibration for the tracker (track.Calibration).

    fit_calibration(s, d, y) -> (Calibration, report)
        s AS-norm scores, d seconds of audio behind the test embedding, y target?
        Prior-weighted (π = 0.5) logistic regression on [s, s·ln d, ln d, 1], so the output
        is a log-likelihood ratio; ridge 1e-3; Newton/IRLS. The report gives Cllr, min Cllr
        (PAV) and EER overall and per duration bucket, before and after calibration.
    refit_scale(base, s, d, y) -> Calibration
        Keeps the duration shape of `base` and refits only an affine map of its LLR
        (llr' = α·llr + β). Used on the private store, whose exemplars are all ≥ 2 s, so the
        short-duration behaviour must come from the public fit.
    store_trials(store, model_id, top_n) -> (s, d, y, meeting)
        Leave-one-meeting-out trials from enrolled exemplars: every exemplar of meeting m is
        scored against models and cohort built without m.
    calibrate_store(store, model_id, base) -> (Calibration, report)

Numpy only (the store trials import scoring.py).
"""

from __future__ import annotations

import math

import numpy as np

from .track import Calibration, Scorer

DUR_BUCKETS = ((0.0, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 1e9))


def _features(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    ld = np.log(np.asarray(d, np.float64))
    s = np.asarray(s, np.float64)
    return np.stack([s, s * ld, ld, np.ones_like(s)], axis=1)


def _logistic(F: np.ndarray, y: np.ndarray, prior: float = 0.5, ridge: float = 1e-3,
              iters: int = 50) -> np.ndarray:
    y = y.astype(np.float64)
    nt, nn = y.sum(), (1 - y).sum()
    if nt == 0 or nn == 0:
        raise ValueError("calibration needs both target and non-target trials")
    w = np.where(y > 0, prior / nt, (1 - prior) / nn)
    off = math.log(prior / (1 - prior))
    beta = np.zeros(F.shape[1])
    reg = ridge * np.eye(F.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        z = F @ beta + off
        p = 1 / (1 + np.exp(-z))
        g = F.T @ (w * (p - y)) + reg @ beta
        H = (F * (w * p * (1 - p))[:, None]).T @ F + reg
        step = np.linalg.solve(H, g)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def cllr(llr: np.ndarray, y: np.ndarray) -> float:
    """Cost of log-likelihood ratio (bits); 1.0 = uninformative, 0 = perfect."""
    t, n = llr[y], llr[~y]
    if not len(t) or not len(n):
        return float("nan")
    ct = np.mean(np.logaddexp(0, -t)) / math.log(2)
    cn = np.mean(np.logaddexp(0, n)) / math.log(2)
    return float(0.5 * (ct + cn))


def pav_llr(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Optimal monotone LLRs (pool-adjacent-violators) for min Cllr."""
    order = np.argsort(s, kind="stable")
    ys = y[order].astype(np.float64)
    vals, wts = [], []
    for v in ys:
        vals.append(v)
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w
            vals.pop()
            wts.pop()
    post = np.repeat(vals, np.array(wts, int))
    eps = 1e-6
    post = np.clip(post, eps, 1 - eps)
    nt, nn = y.sum(), (~y).sum()
    llr = np.log(post / (1 - post)) - math.log(nt / nn)
    out = np.empty_like(llr)
    out[order] = llr
    return out


def eer(s: np.ndarray, y: np.ndarray) -> float:
    t, n = np.sort(s[y]), np.sort(s[~y])
    if not len(t) or not len(n):
        return float("nan")
    thr = np.unique(np.concatenate([t, n]))
    frr = np.searchsorted(t, thr, side="left") / len(t)
    far = 1 - np.searchsorted(n, thr, side="left") / len(n)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


def _summary(s, llr, y) -> dict:
    """EER of the scores; Cllr of the LLRs and its floor over monotone maps of them (PAV)."""
    return {"n_target": int(y.sum()), "n_nontarget": int((~y).sum()), "eer": round(eer(s, y), 4),
            "cllr": round(cllr(llr, y), 4), "min_cllr": round(cllr(pav_llr(llr, y), y), 4)}


def report(cal: Calibration, s: np.ndarray, d: np.ndarray, y: np.ndarray) -> dict:
    llr = cal.llr(s, d).astype(np.float64)
    rep = {"all": _summary(s, llr, y), "by_duration": {}}
    for lo, hi in DUR_BUCKETS:
        m = (d >= lo) & (d < hi)
        if m.sum() and y[m].any() and (~y[m]).any():
            rep["by_duration"][f"{lo:g}-{hi:g}s" if hi < 1e8 else f">={lo:g}s"] = _summary(s[m], llr[m], y[m])
    return rep


def fit_calibration(s, d, y, source: str = "fit") -> tuple[Calibration, dict]:
    s, d, y = np.asarray(s, np.float64), np.asarray(d, np.float64), np.asarray(y, bool)
    lo, hi = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    a, b, c, k = _logistic(_features(np.asarray(s), np.clip(d, lo, hi)), y)
    cal = Calibration(float(a), float(b), float(c), float(k), source, round(lo, 2), round(hi, 2))
    return cal, report(cal, s, d, y)


def refit_scale(base: Calibration, s, d, y, source: str = "store") -> Calibration:
    s, d, y = np.asarray(s, np.float64), np.asarray(d, np.float64), np.asarray(y, bool)
    z = base.llr(s, d).astype(np.float64)
    al, be = _logistic(np.stack([z, np.ones_like(z)], axis=1), y)
    return Calibration(base.a * al, base.b * al, base.c * al, base.k * al + be, source, base.d_min, base.d_max)


def store_trials(store, model_id: str, top_n: int = 100):
    """LOMO trials from the store: (s, d, y, meeting) arrays. For every meeting m, each of its
    exemplars is scored against every person enrolled from the *other* meetings, with the
    same models and cohort that track(roster=None, exclude_meetings=[m]) would use."""
    from .track import scorer_from_store

    ex = store.exemplars(model_id)
    meetings = sorted({e.meeting_key for e in ex if e.meeting_key})
    S, D, Y, MK = [], [], [], []
    for m in meetings:
        test = [e for e in ex if e.meeting_key == m]
        by_ch: dict = {}
        for e in test:
            ch = e.channel if e.channel in ("phone", "zoom", "room") else None
            if ch not in by_ch:
                by_ch[ch] = scorer_from_store(store, model_id, None, ch, [m], top_n)
            scorer = by_ch[ch]
            if len(scorer.names) < 2:
                continue
            row = scorer.scores(np.asarray(e.embedding, np.float32)[None])[0]
            S += row.tolist()
            D += [float(e.net_dur or 0.0)] * len(scorer.names)
            Y += [n == e.person for n in scorer.names]
            MK += [m] * len(scorer.names)
    return np.array(S), np.array(D), np.array(Y, bool), np.array(MK)


def calibrate_store(store, model_id: str, base: Calibration, top_n: int = 100) -> tuple[Calibration, dict]:
    s, d, y, _ = store_trials(store, model_id, top_n)
    if not y.any() or y.all():
        raise ValueError("the store needs targets and non-targets from ≥ 2 meetings to calibrate")
    cal = refit_scale(base, s, d, y, source=f"store-lomo(base={base.source})")
    return cal, {"base": report(base, s, d, y), "refit": report(cal, s, d, y)}
