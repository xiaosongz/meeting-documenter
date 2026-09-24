"""Acoustic check that a speaker label is one voice across the sessions of its series.

    python -m bench.datasets.verify_ids --dataset aishell4 [--out verify.json]

Uses B0's cached enrollment-clip embeddings (run `python -m bench.baselines.window_id
--embed-only` first). For every series, each (session, label) gets a centroid; then
    within series: for each ordered session pair (a, b), is label x in a closest to the
                   same label x in b? (label match rate; same- vs different-label cosine)
    across series: the most similar centroid pairs between different series, flagged
                   when above the 5th percentile of within-series same-label cosine
                   (a hint that a label in another series may be the same person).
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

from ..baselines import window_id as B
from ..protocol import load_meetings


def centroids(dataset: str, split: str | None, cdir: Path) -> dict[tuple[str, str, str], np.ndarray]:
    out = {}
    for m in load_meetings(dataset, split):
        z = B._load(cdir, m.id, "enroll")
        for spk in sorted(set(z["speaker"].tolist())):
            E = B._l2(z["emb"][z["speaker"] == spk])
            if len(E) >= 3:
                out[(m.series, m.id, spk)] = B._l2(E.mean(0))
    return out


def verify(cents: dict[tuple[str, str, str], np.ndarray], top_cross: int = 10) -> dict:
    series = sorted({k[0] for k in cents})
    same, diff, hits, trials, per_series = [], [], 0, 0, {}
    for s in series:
        sess = sorted({k[1] for k in cents if k[0] == s})
        sh = st = 0
        for a, b in permutations(sess, 2):
            la = [k for k in cents if k[1] == a]
            lb = [k for k in cents if k[1] == b]
            if not lb:
                continue
            Cb = np.stack([cents[k] for k in lb])
            for ka in la:
                sims = Cb @ cents[ka]
                for kb, v in zip(lb, sims):
                    (same if kb[2] == ka[2] else diff).append(float(v))
                if any(kb[2] == ka[2] for kb in lb):
                    st += 1
                    sh += lb[int(sims.argmax())][2] == ka[2]
        per_series[s] = {"sessions": len(sess), "label_match": f"{sh}/{st}"}
        hits, trials = hits + sh, trials + st
    thr = float(np.percentile(same, 5)) if same else None
    pooled = {}
    for s in series:
        for spk in sorted({k[2] for k in cents if k[0] == s}):
            vs = [v for k, v in cents.items() if k[0] == s and k[2] == spk]
            pooled[(s, spk)] = B._l2(np.mean(vs, 0))
    keys = sorted(pooled)
    cross = []
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            if ka[0] != kb[0]:
                cross.append((float(pooled[ka] @ pooled[kb]), ka, kb))
    cross.sort(reverse=True)
    return {
        "label_match_rate": hits / trials if trials else None,
        "label_match": f"{hits}/{trials}",
        "per_series": per_series,
        "same_label_cos": {"mean": float(np.mean(same)), "p05": thr} if same else None,
        "diff_label_cos": {"mean": float(np.mean(diff)), "p95": float(np.percentile(diff, 95))} if diff else None,
        "cross_series_top": [{"cos": round(c, 3), "a": "/".join(a), "b": "/".join(b),
                              "above_same_label_p05": thr is not None and c >= thr}
                             for c, a, b in cross[:top_cross]],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=["ami", "aishell4"])
    ap.add_argument("--split")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    _, embedders, _, _ = B._vp()
    cdir = B.cache_dir(a.dataset, embedders.get_embedder("redimnet2", device="cpu").model_id, B.WinParams())
    rep = verify(centroids(a.dataset, a.split, cdir))
    text = json.dumps(rep, indent=2)
    if a.out:
        a.out.write_text(text + "\n")
    print(text, file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
