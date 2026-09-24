"""Enrollment-source quality check for AMI (the source of the dev vs test IER gap).

    python -m bench.datasets.enroll_qc --dataset ami [--out bench/datasets/qc/ami.json]

Needs B0's embedding cache (`python -m bench.baselines.window_id --embed-only`). Two checks:

1. Headset vs far field, same meeting. For each (meeting, speaker), the centroid of the
   enrollment clips (headset-mix stream) is compared with the centroid of the SDM test
   windows that lie inside the speaker's solo reference regions. Same person, same time,
   different microphone: good pairs sit at cosine 0.78-0.98. Pairs below `--min-cos`
   (0.75) mean the headset mix does not carry that speaker (a dead or missing headset
   channel, so the "clips" are other people's crosstalk). These sources are dropped.
2. Label consistency across a series, far field only. The SDM centroid of each label in
   one meeting should be closest to the same label in the other meetings of its series.
   A series where labels map to other labels (a permutation) cannot be used for
   cross-session enrollment and is dropped whole.
3. A series is also dropped when, after (1), some meeting in it has fewer than two
   participants left with enrollment material (it would be almost entirely open set).

The result is a small JSON of public-corpus IDs that `bench.protocol` applies when a
system or the scorer is run with `--enroll-qc`.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np

from ..protocol import load_meetings, solo_regions

QC_DIR = Path(__file__).resolve().parent / "qc"


def _l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


def run(dataset: str, cdir: Path, min_cos: float = 0.75, win_s: float = 1.5) -> dict:
    meetings = load_meetings(dataset)
    far: dict[tuple[str, str], np.ndarray] = {}
    sources, drop = [], []
    for m in meetings:
        with np.load(cdir / f"{m.id}.test.npz") as t, np.load(cdir / f"{m.id}.enroll.npz") as e:
            st, temb, espk, eemb = t["starts"], t["emb"], e["speaker"], e["emb"]
        ref = m.reference()
        for s in m.speakers:
            ok = np.zeros(len(st), bool)
            for a, b in solo_regions(ref, s, 0.1, win_s):
                ok |= (st >= a) & (st + win_s <= b)
            if ok.sum() >= 5:
                far[(m.id, s)] = _l2(temb[ok].mean(0))
            sel = espk == s
            if sel.sum() >= 3 and (m.id, s) in far:
                c = float(_l2(_l2(eemb[sel]).mean(0)) @ far[(m.id, s)])
                sources.append({"meeting": m.id, "speaker": s, "cos": round(c, 3)})
                if c < min_cos:
                    drop.append([m.id, s])
    bad_series = {}
    for series in sorted({m.series for m in meetings}):
        ids = [m.id for m in meetings if m.series == series]
        wrong = []
        for a, b in permutations(ids, 2):
            la = [k for k in far if k[0] == a]
            lb = [k for k in far if k[0] == b]
            for ka in la:
                if not any(kb[1] == ka[1] for kb in lb):
                    continue
                best = max(lb, key=lambda kb: float(far[ka] @ far[kb]))
                if best[1] != ka[1]:
                    wrong.append(f"{a}/{ka[1]} -> {b}/{best[1]}")
        if wrong:
            bad_series[series] = wrong
            continue
        sm = [m for m in meetings if m.series == series]
        dropped = {tuple(x) for x in drop}
        for m in sm:
            n_ok = sum(any(s in o.speakers and (o.id, s) not in dropped for o in sm if o.id != m.id)
                       for s in m.speakers)
            if n_ok < 2:
                bad_series[series] = [f"{m.id}: {n_ok} enrollable participant(s) after dropping bad sources"]
                break
    return {"dataset": dataset, "min_cos": min_cos,
            "drop_sources": drop,
            "drop_meetings": sorted(m.id for m in meetings if m.series in bad_series),
            "inconsistent_series": bad_series,
            "sources": sources}


def main(argv=None) -> int:
    from ..baselines import window_id as B
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="ami", choices=["ami"])
    ap.add_argument("--min-cos", type=float, default=0.75)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    _, embedders, _, _ = B._vp()
    model_id = embedders.get_embedder("redimnet2", device="cpu").model_id
    res = run(a.dataset, B.cache_dir(a.dataset, model_id, B.WinParams()), a.min_cos)
    out = a.out or QC_DIR / f"{a.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"{len(res['drop_sources'])} sources dropped, meetings dropped {res['drop_meetings']} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
