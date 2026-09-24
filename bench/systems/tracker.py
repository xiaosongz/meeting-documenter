"""Benchmark driver for the known-speaker tracker (scripts/voiceprint/track.py) and B1.

    python -m bench.systems.tracker prepare   --dataset ami --split dev [--backend nemotron3]
    python -m bench.systems.tracker calibrate --dataset ami --split dev --out cal.json
    python -m bench.systems.tracker tune      --dataset ami --split dev --cal cal.json --out tune.json
    python -m bench.systems.tracker run       --dataset ami --split test --cal cal.json \
                                              --params params.json --out <dir> [--system track|b1]
    python -m bench.systems.tracker logo      --dataset aishell4 --grid grid.json --out <dir> \
                                              [--objective bal|strict|safe] [--system track|b1]
        leave-one-group-out for a split without a dev set: for each series, calibrate and
        pick the grid point on the other series, then decode the held-out one

Stages, all cached under $SPEAKER_BENCH_DIR/cache/track/ so later stages cost no model time:
    boundary   activations from `--boundary-cache` roots (the st-boundary eval caches) or
               computed with segment() and written to the first root
    units      track.build_units on the test stream + ReDimNet2 unit embeddings (MPS, bounded)
    turns      second-pass turn embeddings, filled on demand (only `run` with an embedder)
Enrollment and AS-norm cohort are B0's (bench/baselines/window_id.py: headset-stream clips
from the other meetings of the series, cohort from the other series), so B0, B1 and the
tracker see exactly the same enrollment. `--enroll-qc` applies bench/datasets/qc/<ds>.json.

Rosters: closed, plus one leave-one-out roster per present speaker (open-set false names),
written in the layout bench.run expects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .. import metrics as M
from ..baselines import window_id as B
from ..paths import bench_dir
from ..protocol import Meeting, load_meetings, load_qc, rosters
from ..rttm import Seg, write_rttm

REPO = Path(__file__).resolve().parents[2]
SR = 16000
BOUNDARY_TAG = {"nemotron3": "nemotron3-cpu", "seg3": "seg3-step2.5"}


def _tk():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    from voiceprint import track as T
    return T


# ─── boundary layer ────────────────────────────────────────────────────────

def boundary_roots(extra: list[Path] | None) -> list[Path]:
    return [*(extra or []), bench_dir() / "cache" / "boundary"]


def load_segmentation(m: Meeting, backend: str, roots: list[Path], wave=None):
    """LocalSegmentation from a cached .npz (eval_boundary format) or computed and cached."""
    _tk()
    from voiceprint.segment import LocalSegmentation
    tag = BOUNDARY_TAG[backend]
    for r in roots:
        p = r / m.dataset / tag / f"{m.id}.npz"
        if p.is_file():
            with np.load(p) as z:
                d = {k: z[k] for k in z.files}
            break
    else:
        from voiceprint import segment as sg
        from voiceprint.audio import decode_audio
        wave = decode_audio(m.test_audio) if wave is None else wave
        seg = sg.segment(wave, backend=backend)
        d = ({"probs": seg.activations[0].astype(np.float16)} if backend == "nemotron3" else
             {"acts": seg.activations.astype(np.float16), "window_starts": seg.chunk_starts,
              "frame_step": seg.frame_step, "frame_dur": seg.frame_dur})
        d["duration"] = len(wave) / SR
        p = roots[0] / m.dataset / tag / f"{m.id}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p.with_suffix(".tmp.npz"), **d)
        os.replace(p.with_suffix(".tmp.npz"), p)
    dur = float(d["duration"])
    if backend == "nemotron3":
        return LocalSegmentation(np.zeros(1), dur, 0.01, 0.01, d["probs"].astype(np.float32)[None],
                                 backend, {"duration": dur})
    return LocalSegmentation(d["window_starts"], 10.0, float(d["frame_step"]), float(d["frame_dur"]),
                             d["acts"].astype(np.float32), backend, {"duration": dur})


# ─── units + embeddings cache ──────────────────────────────────────────────

UNIT_KEYS = ("threshold", "min_span_s", "link_min", "unit_s", "pool_s", "min_embed_s", "max_embed_s", "quant_s")


def unit_key(model_id: str, backend: str, params) -> str:
    d = {k: getattr(params, k) for k in UNIT_KEYS}
    d.update(model=model_id, backend=backend)
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def track_cache(dataset: str, backend: str, key: str) -> Path:
    return bench_dir() / "cache" / "track" / dataset / f"{backend}-{key}"


def _save_units(p: Path, units, X: np.ndarray, info: dict) -> None:
    sp = [(i, s, e) for i, u in enumerate(units) for s, e in u.spans]
    np.savez(p.with_suffix(".tmp.npz"),
             track=np.array([u.track for u in units], np.int32),
             t0=np.array([u.t0 for u in units]), t1=np.array([u.t1 for u in units]),
             clean_s=np.array([u.clean_s for u in units]),
             emb_row=np.array([u.emb_row for u in units], np.int32),
             emb_s=np.array([u.emb_s for u in units], np.float32),
             span=np.array(sp, np.float64).reshape(-1, 3), emb=X.astype(np.float32),
             info=np.array(json.dumps(info)))
    os.replace(p.with_suffix(".tmp.npz"), p)


def _load_units(p: Path):
    T = _tk()
    with np.load(p) as z:
        d = {k: z[k] for k in z.files}
    spans: dict[int, list] = {}
    for i, s, e in d["span"]:
        spans.setdefault(int(i), []).append((float(s), float(e)))
    units = [T.Unit(int(k), float(a), float(b), spans.get(i, []), float(c), int(r), float(es))
             for i, (k, a, b, c, r, es) in enumerate(zip(d["track"], d["t0"], d["t1"], d["clean_s"],
                                                         d["emb_row"], d["emb_s"]))]
    return units, d["emb"], json.loads(str(d["info"]))


_EMB = None


def embedder(device: str):
    global _EMB
    if _EMB is None:
        _, embedders, _, _ = B._vp()
        _EMB = embedders.get_embedder("redimnet2", device=device)
    return _EMB


@dataclass
class Prepared:
    meeting: Meeting
    trs: object                                # track.Tracks
    units: list
    X: np.ndarray
    models: dict[str, np.ndarray]
    cohort: np.ndarray
    turn_cache: Path
    turns: dict = field(default_factory=dict)  # (track, f0, f1) → (emb, dur)
    info: dict = field(default_factory=dict)


def prepare_meeting(m: Meeting, pool: list[Meeting], params, backend: str, roots: list[Path],
                    model_id: str, device: str, qc: dict | None, log=print) -> Prepared:
    T = _tk()
    from voiceprint.footprint import FootprintGuard, gb, peak_footprint
    seg = load_segmentation(m, backend, roots)
    trs = T.tracks_from_segmentation(seg, params)
    del seg
    cdir = track_cache(m.dataset, backend, unit_key(model_id, backend, params))
    cdir.mkdir(parents=True, exist_ok=True)
    up = cdir / f"{m.id}.units.npz"
    if up.is_file():
        units, X, info = _load_units(up)
    else:
        from voiceprint.audio import decode_audio
        t0 = time.time()
        wave = decode_audio(m.test_audio)
        t1 = time.time()
        units = T.build_units(trs, params)
        X = T.embed_units(embedder(device), wave, units, params, FootprintGuard())
        t2 = time.time()
        dur = len(wave) / SR
        info = {"meeting": m.id, "audio_s": round(dur, 1), "units": len(units),
                "embedded": int(sum(u.emb_row >= 0 for u in units)), "tracks": len(trs.tracks),
                "embedded_audio_s": round(float(sum(u.emb_s for u in units if u.emb_row >= 0)), 1),
                "decode_s": round(t1 - t0, 1), "embed_s": round(t2 - t1, 1),
                "embed_xrt": round(dur / max(t2 - t1, 1e-6), 1),
                "peak_footprint_gb": gb(peak_footprint())}
        _save_units(up, units, X, info)
        log(json.dumps(info))
    b0dir = B.cache_dir(m.dataset, model_id, B.WinParams())
    models = B.enrollment_models(m, pool, b0dir, qc)
    cohort = B.cohort_vectors(m, pool, b0dir, models, qc)
    p = Prepared(m, trs, units, X, models, cohort, cdir / f"{m.id}.turns.npz", info=info)
    load_turns(p)
    return p


def load_turns(p: Prepared) -> None:
    if p.turn_cache.is_file():
        with np.load(p.turn_cache) as z:
            for k, e, d in zip(z["key"], z["emb"], z["dur"]):
                p.turns[tuple(int(x) for x in k)] = (e, float(d))


def save_turns(p: Prepared) -> None:
    if not p.turns:
        return
    keys = list(p.turns)
    np.savez(p.turn_cache.with_suffix(".tmp.npz"), key=np.array(keys, np.int64),
             emb=np.stack([p.turns[k][0] for k in keys]).astype(np.float32),
             dur=np.array([p.turns[k][1] for k in keys], np.float32))
    os.replace(p.turn_cache.with_suffix(".tmp.npz"), p.turn_cache)


# ─── decoding ──────────────────────────────────────────────────────────────

def scorer_for(p: Prepared, roster: set[str], top_n: int):
    T = _tk()
    names = sorted(n for n in roster if n in p.models)
    models = np.stack([p.models[n] for n in names]) if names else np.zeros((0, p.X.shape[1]), np.float32)
    return T.Scorer(names, models, p.cohort, top_n)


def make_turn_scores(p: Prepared, scorer, params, device: str | None):
    """Second-pass scorer over the turn cache; with `device`, misses are embedded (and cached),
    otherwise a miss gets an uninformative score (the turn keeps its label)."""
    T = _tk()
    cap = params.turn_max_embed_s

    def turn_scores(cands):
        keys = [(c.track, c.f0, c.f1) for c in cands]
        miss = [i for i, k in enumerate(keys) if k not in p.turns]
        if miss and device:
            from voiceprint.audio import decode_audio
            wave = decode_audio(p.meeting.test_audio)
            E, rows = T.embed_spans(embedder(device), wave, [cands[i].spans for i in miss], cap, params.quant_s)
            for r, j in zip(E, rows):
                i = miss[int(j)]
                p.turns[keys[i]] = (r, float(min(cands[i].clean_s, cap)))
            save_turns(p)
        S = np.zeros((len(cands), len(scorer.names)), np.float32)
        d = np.full(len(cands), 0.25, np.float32)
        have = [i for i, k in enumerate(keys) if k in p.turns]
        if have:
            S[have] = scorer.scores(np.stack([p.turns[keys[i]][0] for i in have]))
            d[have] = [p.turns[keys[i]][1] for i in have]
        return S, d
    return turn_scores


def to_segs(tl) -> list[Seg]:
    return [Seg(t.start, t.end, t.speaker) for t in tl.turns]


def decode_track(p: Prepared, roster: set[str], params, cal, device: str | None = None) -> tuple[list[Seg], dict]:
    T = _tk()
    sc = scorer_for(p, roster, params.top_n)
    ts = make_turn_scores(p, sc, params, device) if params.second_pass else None
    fl, info = T.decode_all(p.trs, p.units, p.X, sc, cal, params, ts)
    return to_segs(T.build_timeline(p.trs, fl)), info


def decode_b1(p: Prepared, roster: set[str], params, cal, b1) -> tuple[list[Seg], dict]:
    from ..baselines import ahc_id
    T = _tk()
    sc = scorer_for(p, roster, params.top_n)
    labels, conf, info = ahc_id.cluster_identify(p.units, p.X, sc, cal, b1)
    fl = T.label_frames(p.trs, p.units, labels, conf, params)
    return to_segs(T.build_timeline(p.trs, fl)), info


# ─── evaluation helpers ────────────────────────────────────────────────────

def quick_eval(preps: list[Prepared], decode, collar: float = 0.0) -> dict:
    """Closed IER (frame level) + leave-one-out false-name rate and IER, pooled."""
    ce = ct = le = lt = 0.0
    bd_loo, bd_closed = M.Breakdown(), M.Breakdown()
    for p in preps:
        ref, uem = p.meeting.reference(), p.meeting.uem()
        rs = rosters(p.meeting, sorted(p.models))
        segs, _ = decode(p, rs["closed"])
        e, t = M.frame_ier(ref, segs, uem, rs["closed"], collar)
        ce, ct = ce + e, ct + t
        bd_closed.add(M.breakdown(ref, segs, uem, rs["closed"], collar))
        for k, r in rs.items():
            if k.startswith("loo:"):
                segs, _ = decode(p, r)
                e, t = M.frame_ier(ref, segs, uem, r, collar)
                le, lt = le + e, lt + t
                bd_loo.add(M.breakdown(ref, segs, uem, r, collar))
    sl, sc = M.summarize(bd_loo), M.summarize(bd_closed)
    return {"ier": ce / ct if ct else None, "loo_ier": le / lt if lt else None,
            "false_name": sl["false_name_rate"], "abstain": sc["abstain_rate"],
            "wrong_name": sc["enrolled"]["wrong_name"], "missed": sc["enrolled"]["missed"],
            "ge3s_correct": _ge3(sc)}


def _ge3(s: dict) -> float | None:
    b = s["by_bucket"]
    secs = b["3-10s"]["seconds"] + b[">=10s"]["seconds"]
    if not secs:
        return None
    return (b["3-10s"]["correct"] * b["3-10s"]["seconds"] + b[">=10s"]["correct"] * b[">=10s"]["seconds"]) / secs


def calibration_trials(preps: list[Prepared], params, min_purity: float = 0.8) -> dict:
    """(AS-norm score, duration, target?) for every embedded unit whose clean audio is
    ≥ min_purity one reference speaker, against every model of the meeting's closed roster."""
    s_all, d_all, y_all, mid = [], [], [], []
    for p in preps:
        ref = p.meeting.reference()
        n = int(max(s.end for s in ref) / M.FRAME) + 2
        labels = sorted({s.speaker for s in ref})
        R = M._activity(ref, labels, n)
        solo = R & (R.sum(1) == 1)[:, None]
        sc = scorer_for(p, set(p.models), params.top_n)
        S = sc.scores(p.X)
        for u in p.units:
            if u.emb_row < 0:
                continue
            tot = np.zeros(len(labels))
            for a, b in u.spans:
                fa, fb = int(a / M.FRAME), min(n, int(b / M.FRAME))
                tot += solo[fa:fb].sum(0)
            span = sum(b - a for a, b in u.spans) / M.FRAME
            if not span or tot.max() / span < min_purity:
                continue
            true = labels[int(tot.argmax())]
            for j, name in enumerate(sc.names):
                s_all.append(S[u.emb_row, j])
                d_all.append(u.emb_s)
                y_all.append(name == true)
                mid.append(p.meeting.id)
    return {"s": np.array(s_all, np.float32), "d": np.array(d_all, np.float32),
            "y": np.array(y_all, bool), "meeting": np.array(mid)}


# ─── worker plumbing (tuning across processes, CPU only) ───────────────────

_PREPS: list[Prepared] = []


def _init_worker(preps):
    global _PREPS
    _PREPS = preps
    os.environ["OMP_NUM_THREADS"] = "1"


def _eval_cfg(args):
    kind, params, cal_d, b1 = args
    T = _tk()
    cal = T.Calibration.from_dict(cal_d)
    if kind == "b1":
        dec = lambda p, r: decode_b1(p, r, params, cal, b1)  # noqa: E731
    else:
        dec = lambda p, r: decode_track(p, r, params, cal, None)  # noqa: E731
    return quick_eval(_PREPS, dec)


def eval_grid(preps, configs: list[tuple], workers: int) -> list[dict]:
    if workers <= 1:
        _init_worker(preps)
        return [_eval_cfg(c) for c in configs]
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker, initargs=(preps,)) as ex:
        return list(ex.map(_eval_cfg, configs))


# ─── CLI ───────────────────────────────────────────────────────────────────

def _log_to(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        f.write(line + "\n")
        f.flush()
    return log


def load_params(path: Path | None, overrides: dict | None = None):
    T = _tk()
    d = json.loads(path.read_text())["params"] if path else {}
    d.update(overrides or {})
    return T.TrackParams(**{k: v for k, v in d.items() if k in T.TrackParams.__dataclass_fields__})


def load_cal(path: Path | None):
    T = _tk()
    if not path:
        return T.DEFAULT_CALIBRATION
    return T.Calibration.from_dict(json.loads(path.read_text())["calibration"])


def prepare_split(a, params, log) -> list[Prepared]:
    qc = load_qc(a.dataset) if a.enroll_qc else None
    pool = load_meetings(a.dataset, a.split, qc=qc)
    if a.meetings:
        todo = [m for m in pool if m.id in a.meetings]
    else:
        todo = pool
    _, embedders, _, _ = B._vp()
    model_id = embedders.get_embedder("redimnet2", device="cpu").model_id
    roots = boundary_roots(a.boundary_cache)
    return [prepare_meeting(m, pool, params, a.backend, roots, model_id, a.device, qc, log) for m in todo]


def pick(rows: list[dict], objective: str, max_fn: float = 0.01, ier_slack: float = 0.01) -> dict:
    if objective == "safe":
        # fewest confident errors (wrong names on enrolled speech + false names on unenrolled
        # speech) among the points within ier_slack of the grid's best closed IER
        best = min(r["ier"] for r in rows)
        ok = [r for r in rows if r["ier"] <= best + ier_slack]
        return min(ok, key=lambda r: (r["wrong_name"] + (r["false_name"] or 0.0), r["ier"]))
    if objective == "strict":
        ok = [r for r in rows if r["false_name"] is not None and r["false_name"] <= max_fn]
        if ok:
            return min(ok, key=lambda r: r["ier"])
        return min(rows, key=lambda r: (r["false_name"], r["ier"]))
    return min(rows, key=lambda r: 0.5 * (r["ier"] + (r["loo_ier"] if r["loo_ier"] is not None else r["ier"])))


def logo(a, preps: list[Prepared], params, log) -> int:
    """Leave-one-series-out calibration + grid choice, then decode each held-out series."""
    sys.path.insert(0, str(REPO / "scripts"))
    from voiceprint import calibrate_track as CT
    from ..baselines import ahc_id
    grid = json.loads(a.grid.read_text()) if a.grid else [{}]
    base_b1 = ahc_id.B1Params(**(json.loads(a.b1.read_text())["b1"] if a.b1 else {}))
    series = sorted({p.meeting.series for p in preps})
    sysdir, chosen, stats = a.out / "system", {}, []
    for g in series:
        train = [p for p in preps if p.meeting.series != g]
        test = [p for p in preps if p.meeting.series == g]
        tr = calibration_trials(train, params)
        cal, rep = CT.fit_calibration(tr["s"], tr["d"], tr["y"], source=f"{a.dataset}/logo-{g}")
        if a.system == "b1":
            cfgs = [("b1", params, cal.to_dict(), replace(base_b1, **o)) for o in grid]
        else:
            cfgs = [("track", replace(params, second_pass=False, **o), cal.to_dict(), None) for o in grid]
        rows = [{"override": o, **r} for o, r in zip(grid, eval_grid(train, cfgs, a.workers))]
        for r in rows:
            log(f"logo {g} grid " + json.dumps(r))
        best = pick(rows, a.objective, ier_slack=a.ier_slack)
        chosen[g] = {"override": best["override"], "train": {k: v for k, v in best.items() if k != "override"},
                     "calibration": cal.to_dict(), "cal_report": rep["all"]}
        log(f"logo {g}: {json.dumps(chosen[g]['override'])} train IER {100 * best['ier']:.2f} "
            f"FN {100 * (best['false_name'] or 0):.2f}")
        prm = replace(params, **best["override"]) if a.system != "b1" else params
        b1 = replace(base_b1, **best["override"]) if a.system == "b1" else None
        for p in test:
            for k, r in rosters(p.meeting, sorted(p.models)).items():
                t = time.time()
                if a.system == "b1":
                    segs, info = decode_b1(p, r, prm, cal, b1)
                else:
                    segs, info = decode_track(p, r, prm, cal, a.device)
                path = sysdir / "closed" / f"{p.meeting.id}.rttm" if k == "closed" else \
                    sysdir / "loo" / p.meeting.id / f"{k[4:]}.rttm"
                write_rttm(path, p.meeting.id, segs)
                if k == "closed":
                    stats.append({**p.info, "decode_s": round(time.time() - t, 2)})
    (a.out / "run-params.json").write_text(json.dumps(
        {"system": a.system, "protocol": "leave-one-series-out", "objective": a.objective,
         "backend": a.backend, "params": asdict(params), "chosen": chosen, "stats": stats}, indent=1) + "\n")
    from .. import run as bench_run
    bench_run.main(["--dataset", a.dataset, "--split", a.split, *(["--enroll-qc"] if a.enroll_qc else []),
                    "--system", str(sysdir), "--out", str(a.out)])
    log("done")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["prepare", "calibrate", "tune", "run", "logo"])
    ap.add_argument("--dataset", required=True, choices=["ami", "aishell4"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="nemotron3", choices=["nemotron3", "seg3"])
    ap.add_argument("--boundary-cache", type=Path, action="append")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--enroll-qc", action="store_true")
    ap.add_argument("--meetings", nargs="*")
    ap.add_argument("--params", type=Path, help="JSON with a 'params' object (TrackParams fields)")
    ap.add_argument("--set", nargs="*", default=[], help="TrackParams overrides k=v")
    ap.add_argument("--cal", type=Path, help="JSON with a 'calibration' object")
    ap.add_argument("--system", default="track", choices=["track", "b1"])
    ap.add_argument("--b1", type=Path, help="B1 params JSON (ahc_id.B1Params fields under 'b1')")
    ap.add_argument("--grid", type=Path, help="tune: JSON list of override dicts")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--objective", default="bal", choices=["bal", "strict", "safe"],
                    help="grid choice: min mean(closed IER, leave-one-out IER) as B0's tuning (bal), "
                         "min closed IER with false-name ≤ 1 %% (strict), or min wrong-name + "
                         "false-name rate within --ier-slack of the best closed IER (safe)")
    ap.add_argument("--ier-slack", type=float, default=0.0075,
                    help="safe objective: IER slack (fraction; 0.75 points leaves headroom under a 1-point budget)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    log = _log_to(a.out.with_suffix(".log") if a.out.suffix else a.out / "tracker.log")
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = json.loads(v)
    params = load_params(a.params, over)
    t0 = time.time()
    preps = prepare_split(a, params, log)
    log(f"prepared {len(preps)} meetings in {time.time() - t0:.0f} s")
    if a.cmd == "prepare":
        return 0
    if a.cmd == "calibrate":
        sys.path.insert(0, str(REPO / "scripts"))
        from voiceprint import calibrate_track as CT
        tr = calibration_trials(preps, params)
        cal, rep = CT.fit_calibration(tr["s"], tr["d"], tr["y"], source=f"{a.dataset}/{a.split}")
        a.out.write_text(json.dumps({"calibration": cal.to_dict(), "report": rep}, indent=1) + "\n")
        log(f"calibration {cal.to_dict()} {json.dumps(rep)}")
        return 0
    if a.cmd == "logo":
        return logo(a, preps, params, log)
    cal = load_cal(a.cal)
    b1 = None
    if a.system == "b1":
        from ..baselines import ahc_id
        b1 = ahc_id.B1Params(**(json.loads(a.b1.read_text())["b1"] if a.b1 else {}))
    if a.cmd == "tune":
        grid = json.loads(a.grid.read_text()) if a.grid else [{}]
        cfgs = []
        for g in grid:
            if a.system == "b1":
                from ..baselines import ahc_id
                cfgs.append(("b1", params, cal.to_dict(), replace(b1, **g)))
            else:
                cfgs.append(("track", replace(params, **g), cal.to_dict(), None))
        t = time.time()
        res = eval_grid(preps, cfgs, a.workers)
        rows = [{"override": g, **r} for g, r in zip(grid, res)]
        for r in rows:
            log(json.dumps(r))
        a.out.write_text(json.dumps({"params": asdict(params), "b1": asdict(b1) if b1 else None,
                                     "calibration": cal.to_dict(), "rows": rows}, indent=1) + "\n")
        log(f"tuned {len(grid)} configs in {time.time() - t:.0f} s")
        return 0
    # run: write RTTMs (closed + leave-one-out), then score with bench.run
    sysdir = a.out / "system"
    stats = []
    for p in preps:
        rs = rosters(p.meeting, sorted(p.models))
        for k, r in rs.items():
            t = time.time()
            if a.system == "b1":
                segs, info = decode_b1(p, r, params, cal, b1)
            else:
                segs, info = decode_track(p, r, params, cal, a.device)
            dt = time.time() - t
            path = sysdir / "closed" / f"{p.meeting.id}.rttm" if k == "closed" else \
                sysdir / "loo" / p.meeting.id / f"{k[4:]}.rttm"
            write_rttm(path, p.meeting.id, segs)
            if k == "closed":
                stats.append({**p.info, "decode_s": round(dt, 2), **{f"d_{x}": y for x, y in info.items()}})
    (a.out / "run-params.json").write_text(json.dumps(
        {"system": a.system, "backend": a.backend, "params": asdict(params), "calibration": cal.to_dict(),
         "b1": asdict(b1) if b1 else None, "enroll_qc": a.enroll_qc, "stats": stats}, indent=1) + "\n")
    from .. import run as bench_run
    bench_run.main(["--dataset", a.dataset, "--split", a.split, *(["--enroll-qc"] if a.enroll_qc else []),
                    *(["--meetings", *a.meetings] if a.meetings else []),
                    "--system", str(sysdir), "--out", str(a.out)])
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
