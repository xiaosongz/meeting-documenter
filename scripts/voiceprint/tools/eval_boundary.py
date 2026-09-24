#!/usr/bin/env python3
"""Boundary-layer quality on public corpora (AMI, AISHELL-4); not run in CI.

    python scripts/voiceprint/tools/eval_boundary.py --dataset ami --split test \
        --backend seg3 [--step 2.5] [--device cpu] [--cache DIR] [--json out.json]

Reads the bench manifest ($SPEAKER_BENCH_DIR/manifest-<dataset>.json; built by the
bench harness), runs segment() on every meeting of the split (activations cached as
float16 .npz under --cache, so a rerun only scores), and scores against the reference
RTTM. Everything is computed on a 10 ms grid inside the UEM (whole file when the
dataset has none). A hypothesis frame is active when its activation is ≥ 0.5.

Speaker-change points (same rule for reference and hypothesis). Per-label activity is
turned into turns after filling same-label gaps < 0.25 s (hypothesis turns < 0.1 s are
dropped). A turn start is a change point when a different label was active within the
previous 1.0 s: at the start itself when that label is still active (overlap onset),
otherwise at the midpoint of the gap. A turn end is a change point when another label
continues past it (overlap offset). Pure speech/non-speech boundaries are not changes.
Hypothesis and reference points are matched one-to-one within ±tolerance (0.25 s,
0.5 s); precision, recall and F1 are pooled over meetings.

  seg3      labels are chunk-local, so change points are found inside each 10 s chunk
            (with that chunk's context) and each chunk keeps only the points in the
            span it owns (half-way to the neighbouring chunks' centres).
  nemotron3 slots are global: change points come from the whole-file slot turns.

Local-speaker purity / coverage, per 10 s window (seg3's chunk grid, used for both
backends so they are comparable; each window is one tracker input):
  purity    for each (window, local speaker) region of *clean* frames (that speaker
            alone, as local_regions() feeds the embedder): fraction of its frames where
            its dominant reference speaker speaks alone. The rest splits into another
            reference speaker alone (confusion), reference overlap, and reference
            non-speech (false alarm). Duration-weighted; also the share of regions with
            purity ≥ 0.9, for all regions and for regions with ≥ 1 s clean.
  coverage  for each (window, reference speaker) with speech: largest fraction of that
            speaker's frames captured by a single local speaker (overlap frames
            included); duration-weighted. `missed` = speaker frames with no local
            speaker active at all.
nemotron3 also gets whole-file (global slot) purity/coverage.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GRID = 0.01                 # seconds per scoring frame
THRESHOLD = 0.5
FILL_GAP_S = 0.25
MIN_HYP_TURN_S = 0.1
CHANGE_GAP_S = 1.0
TOLERANCES = (0.25, 0.5)
WINDOW_S = 10.0
AUDIO_KEY = {"ami": "sdm", "aishell4": "ch0"}


# ─── pure metric helpers (unit-tested) ───────────────────────────────────────


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of the True runs of a 1-D bool array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def turns_from_activity(act: np.ndarray, t0: float = 0.0, fill_gap_s: float = FILL_GAP_S,
                        min_turn_s: float = 0.0) -> list[tuple[float, float, int]]:
    """(start, end, label) turns from a (n_frames, n_labels) bool grid at GRID seconds."""
    fill = int(round(fill_gap_s / GRID))
    out = []
    for k in range(act.shape[1]):
        rs = runs(act[:, k])
        merged: list[list[int]] = []
        for a, b in rs:
            if merged and a - merged[-1][1] < fill:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        out += [(t0 + a * GRID, t0 + b * GRID, k) for a, b in merged if (b - a) * GRID >= min_turn_s]
    out.sort()
    return out


def change_points(turns: list[tuple[float, float, int]], gap_s: float = CHANGE_GAP_S) -> np.ndarray:
    """Speaker-change instants of a turn list (see the module docstring for the rule)."""
    pts: list[float] = []
    last_end: dict[int, float] = {}
    for s, e, lab in sorted(turns):
        prev = max((v for k, v in last_end.items() if k != lab), default=-np.inf)
        if prev >= s - gap_s:
            pts.append(s if prev >= s else (prev + s) / 2)
        last_end[lab] = max(last_end.get(lab, -np.inf), e)
    by_label: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lab in {t[2] for t in turns}:
        ts = sorted((s, e) for s, e, k in turns if k == lab)       # one label's turns never overlap
        by_label[lab] = (np.array([t[0] for t in ts]), np.array([t[1] for t in ts]))
    for s, e, lab in turns:
        for k, (ss, ee) in by_label.items():
            if k == lab:
                continue
            i = np.searchsorted(ss, e, side="left") - 1           # last turn of k starting before e
            if i >= 0 and ee[i] > e:
                pts.append(e)
                break
    return np.unique(np.round(np.asarray(pts, float), 3))


def match_points(ref: np.ndarray, hyp: np.ndarray, tol: float) -> int:
    """Maximum one-to-one matching of sorted 1-D points within ±tol (greedy is optimal here)."""
    ref, hyp = np.sort(ref), np.sort(hyp)
    i = j = hits = 0
    while i < len(ref) and j < len(hyp):
        if hyp[j] < ref[i] - tol:
            j += 1
        elif hyp[j] > ref[i] + tol:
            i += 1
        else:
            hits += 1
            i += 1
            j += 1
    return hits


def prf(hits: int, n_ref: int, n_hyp: int) -> dict:
    p = hits / n_hyp if n_hyp else 0.0
    r = hits / n_ref if n_ref else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "hits": hits, "n_ref": n_ref, "n_hyp": n_hyp}


def window_stats(ref: np.ndarray, hyp: np.ndarray, min_clean_frames: int) -> dict:
    """Purity / coverage sums for one window. ref (n, S) and hyp (n, K) bool, same grid."""
    single = ref.sum(1) == 1
    speech = ref.any(1)
    clean = hyp & (hyp.sum(1, keepdims=True) == 1)
    st = defaultdict(float)
    for k in range(hyp.shape[1]):
        m = clean[:, k]
        n = int(m.sum())
        if n == 0:
            continue
        alone = (ref[m] & single[m, None]).sum(0)          # frames where ref speaker s is alone
        pur = alone.max() / n if alone.size else 0.0
        big = n >= min_clean_frames
        st["frames"] += n
        st["pure"] += pur * n
        st["confused"] += alone.sum() - (alone.max() if alone.size else 0)
        st["overlap"] += int((m & ~single & speech).sum())
        st["false_alarm"] += int((m & ~speech).sum())
        st["regions"] += 1
        st["regions_pure90"] += pur >= 0.9
        if big:
            st["frames_1s"] += n
            st["pure_1s"] += pur * n
            st["regions_1s"] += 1
            st["regions_1s_pure90"] += pur >= 0.9
    anyh = hyp.any(1)
    for s in range(ref.shape[1]):
        r = ref[:, s]
        n = int(r.sum())
        if n == 0:
            continue
        cov = (hyp[r].sum(0).max() / n) if hyp.shape[1] else 0.0
        st["ref_frames"] += n
        st["covered"] += cov * n
        st["missed"] += int((r & ~anyh).sum())
    return st


def summarize_windows(st: dict) -> dict:
    f = max(st.get("frames", 0), 1)
    return {
        "purity": st.get("pure", 0) / f,
        "confusion": st.get("confused", 0) / f,
        "ref_overlap": st.get("overlap", 0) / f,
        "false_alarm": st.get("false_alarm", 0) / f,
        "regions": int(st.get("regions", 0)),
        "regions_purity_ge_0.9": st.get("regions_pure90", 0) / max(st.get("regions", 0), 1),
        "purity_clean_ge_1s": st.get("pure_1s", 0) / max(st.get("frames_1s", 0), 1),
        "regions_clean_ge_1s": int(st.get("regions_1s", 0)),
        "regions_clean_ge_1s_purity_ge_0.9": st.get("regions_1s_pure90", 0) / max(st.get("regions_1s", 0), 1),
        "coverage": st.get("covered", 0) / max(st.get("ref_frames", 0), 1),
        "missed": st.get("missed", 0) / max(st.get("ref_frames", 0), 1),
    }


# ─── reference / hypothesis on the grid ─────────────────────────────────────


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split()
        if len(f) >= 8 and f[0] == "SPEAKER" and float(f[4]) > 0:
            out.append((float(f[3]), float(f[3]) + float(f[4]), f[7]))
    return out


def ref_grid(segs: list[tuple[float, float, str]], n: int) -> tuple[np.ndarray, list[str]]:
    spk = sorted({s for _, _, s in segs})
    g = np.zeros((n, len(spk)), bool)
    for a, b, s in segs:
        g[int(round(a / GRID)):int(round(b / GRID)), spk.index(s)] = True
    return g, spk


def uem_mask(path: Path | None, n: int) -> np.ndarray:
    if path is None or not path.is_file():
        return np.ones(n, bool)
    m = np.zeros(n, bool)
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) >= 4:
            m[int(round(float(f[2]) / GRID)):int(round(float(f[3]) / GRID))] = True
    return m


def seg3_window(acts: np.ndarray, start: float, frame_step: float, frame_dur: float, n: int) -> np.ndarray:
    """One seg3 chunk's (n_frames, 3) activations resampled to `n` grid frames from `start`."""
    t = (np.arange(n) + 0.5) * GRID
    i = np.clip(np.floor((t - frame_dur / 2) / frame_step + 0.5).astype(int), 0, acts.shape[0] - 1)
    return acts[i] >= THRESHOLD


def nemo_grid(p: np.ndarray, n: int) -> np.ndarray:
    g = np.zeros((n, p.shape[1]), bool)
    m = min(n, p.shape[0])
    g[:m] = p[:m] >= THRESHOLD
    return g


# ─── per-meeting scoring ─────────────────────────────────────────────────────


def score_meeting(cache: dict, ref: np.ndarray, uem: np.ndarray, backend: str) -> dict:
    n = ref.shape[0]
    ref = ref & uem[:, None]
    wn = int(round(WINDOW_S / GRID))
    starts = cache["window_starts"]
    # change points: reference
    r_pts = change_points(turns_from_activity(ref))
    if backend == "seg3":
        acts, fs, fd = cache["acts"], float(cache["frame_step"]), float(cache["frame_dur"])
        centres = starts + WINDOW_S / 2
        bounds = np.concatenate([[-np.inf], (centres[1:] + centres[:-1]) / 2, [np.inf]])
        h_pts = []
    else:
        g = nemo_grid(cache["probs"].astype(np.float32), n)
        h_pts = change_points(turns_from_activity(g, min_turn_s=MIN_HYP_TURN_S))
    tot = defaultdict(float)
    for c, s in enumerate(starts):
        a = int(round(s / GRID))
        if backend == "seg3":
            h = seg3_window(acts[c].astype(np.float32), s, fs, fd, wn)[: n - a]
            p = change_points(turns_from_activity(h, t0=s, min_turn_s=MIN_HYP_TURN_S))
            h_pts.append(p[(p >= bounds[c]) & (p < bounds[c + 1])])
        else:
            h = g[a:a + wn]
        st = window_stats(ref[a:a + h.shape[0]], h & uem[a:a + h.shape[0], None], int(round(1.0 / GRID)))
        for k, v in st.items():
            tot[k] += v
    if backend == "seg3":
        h_pts = np.concatenate(h_pts) if h_pts else np.zeros(0)
    in_uem = lambda p: p[uem[np.clip((p / GRID).astype(int), 0, n - 1)]]
    r_pts, h_pts = in_uem(r_pts), in_uem(h_pts)
    out = {"n_ref": len(r_pts), "n_hyp": len(h_pts),
           "hits": {str(t): match_points(r_pts, h_pts, t) for t in TOLERANCES}}
    out["windows"] = dict(tot)
    if backend == "nemotron3":
        out["global"] = dict(window_stats(ref, g & uem[:, None], int(round(1.0 / GRID))))
    return out


# ─── segmentation with a cache ───────────────────────────────────────────────


def segment_cached(audio: Path, cache_path: Path, backend: str, step: float, device: str,
                   threads: int) -> dict:
    if cache_path.is_file():
        return dict(np.load(cache_path))
    from voiceprint import segment as sg
    from voiceprint.audio import decode_audio

    wave = decode_audio(audio)
    dur = len(wave) / sg.SAMPLE_RATE
    t = time.perf_counter()
    if backend == "seg3":
        seg = sg.segment(wave, backend="seg3", step=step, num_threads=threads)
        el = time.perf_counter() - t
        d = {"acts": seg.activations.astype(np.float16), "window_starts": seg.chunk_starts,
             "frame_step": seg.frame_step, "frame_dur": seg.frame_dur}
    else:
        seg = sg.segment(wave, backend="nemotron3", device=device, num_threads=threads)
        el = time.perf_counter() - t
        d = {"probs": seg.activations[0].astype(np.float16),
             "window_starts": sg.chunk_starts_for(len(wave), step) / sg.SAMPLE_RATE}
    d.update(duration=dur, segment_s=el)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez(tmp, **d)
    os.replace(tmp, cache_path)
    return d


def bench_dir() -> Path:
    raw = os.environ.get("SPEAKER_BENCH_DIR")
    if raw:
        return Path(raw).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base).expanduser() / "meeting-documenter" / "bench-data"


def main(argv: list[str] | None = None) -> int:
    global THRESHOLD
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(AUDIO_KEY))
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="seg3", choices=("seg3", "nemotron3"))
    ap.add_argument("--step", type=float, default=2.5, help="seg3 chunk step and scoring-window step (s)")
    ap.add_argument("--device", default="cpu", help="nemotron3: cpu | mps")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cache", type=Path, help="activation cache dir (default <bench>/cache/boundary)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="activation threshold")
    ap.add_argument("--meetings", nargs="*", help="subset of meeting IDs")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    THRESHOLD = a.threshold

    root = bench_dir()
    man = json.loads((root / f"manifest-{a.dataset}.json").read_text())
    ds = root / man["root"]
    tag = f"{a.backend}-step{a.step:g}" if a.backend == "seg3" else f"{a.backend}-{a.device}"
    cache_dir = (a.cache or root / "cache" / "boundary") / a.dataset / tag
    meetings = {k: v for k, v in sorted(man["meetings"].items())
                if v["split"] == a.split and (not a.meetings or k in a.meetings)}
    hits = defaultdict(int)
    n_ref = n_hyp = 0
    win, glob = defaultdict(float), defaultdict(float)
    per, audio_s, seg_s = {}, 0.0, 0.0
    for uri, m in meetings.items():
        c = segment_cached(ds / m["audio"][AUDIO_KEY[a.dataset]]["path"], cache_dir / f"{uri}.npz",
                           a.backend, a.step, a.device, a.threads)
        dur = float(c["duration"])
        n = int(np.ceil(dur / GRID))
        ref, _ = ref_grid(read_rttm(ds / m["rttm"]), n)
        uem = uem_mask(ds / m["uem"] if m.get("uem") else None, n)
        r = score_meeting(c, ref, uem, a.backend)
        n_ref += r["n_ref"]
        n_hyp += r["n_hyp"]
        for t, h in r["hits"].items():
            hits[t] += h
        for k, v in r["windows"].items():
            win[k] += v
        for k, v in r.get("global", {}).items():
            glob[k] += v
        audio_s += dur
        seg_s += float(c["segment_s"])
        per[uri] = {"duration_s": round(dur, 1), "x_realtime": round(dur / float(c["segment_s"]), 1),
                    **{f"f1@{t}": round(prf(h, r["n_ref"], r["n_hyp"])["f1"], 4) for t, h in r["hits"].items()},
                    "purity": round(summarize_windows(r["windows"])["purity"], 4),
                    "coverage": round(summarize_windows(r["windows"])["coverage"], 4)}
        print(uri, json.dumps(per[uri]), flush=True)
    res = {
        "dataset": a.dataset, "split": a.split, "backend": a.backend, "tag": tag, "threshold": THRESHOLD,
        "meetings": len(meetings), "audio_h": round(audio_s / 3600, 2),
        "x_realtime_segment": round(audio_s / seg_s, 1) if seg_s else None,
        "change": {f"tol_{t}": prf(h, n_ref, n_hyp) for t, h in hits.items()},
        "windows": summarize_windows(win),
        **({"global": summarize_windows(glob)} if glob else {}),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             / (2**30 if sys.platform == "darwin" else 2**20), 2),
        "per_meeting": per,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "per_meeting"}, indent=2))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
