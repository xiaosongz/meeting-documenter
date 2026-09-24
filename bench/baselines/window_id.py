"""Baseline B0: per-window identification + Viterbi smoothing. The bar the tracker must beat.

    python -m bench.baselines.window_id --dataset ami --split test --out <dir> \
        [--tune-split dev | --theta T --penalty P] [--device mps] [--workers 2]

Pipeline per test meeting:
    decode 16 kHz → Silero VAD (scripts/voiceprint/vad.py)
    → 1.5 s windows at 0.5 s hop, kept when ≥ 50 % Silero speech
    → ReDimNet2 embeddings (scripts/voiceprint/embedders.py)
    → cosine vs enrollment centroids (cross-session, bench/protocol.py) + AS-norm with a
      cohort of the other series' speakers
    → per-window scores; UNKNOWN is an extra state with constant score θ (the absolute
      threshold); Viterbi with switch penalty λ between states (λ = 0 is plain argmax)
    → every VAD speech frame takes the label of the nearest kept window (≤ 1 s away)
    → RTTM (closed roster, plus one leave-one-out roster per present speaker)

Embeddings are cached under $SPEAKER_BENCH_DIR/cache/b0/, keyed by embedder model id and
window parameters, so re-scoring with other θ / λ / rosters costs no model time.
Enrollment clips: the speaker's non-overlapped reference regions (±0.25 s guard) in the
other meetings of the series, from the enrollment stream (AMI headset mix), cut into
≤ 6 s pieces (0.5 s quantized), ≥ 50 % Silero speech, the 30 longest per (speaker, meeting).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..paths import bench_dir
from ..protocol import Meeting, cohort_speakers, enrollment, load_meetings, rosters, solo_regions
from ..rttm import Seg, write_rttm

REPO = Path(__file__).resolve().parents[2]
SR = 16000
UNK = "UNKNOWN"


@dataclass(frozen=True)
class WinParams:
    win_s: float = 1.5
    hop_s: float = 0.5
    min_speech_frac: float = 0.5
    enroll_guard_s: float = 0.25
    enroll_min_s: float = 1.5
    enroll_max_s: float = 6.0
    enroll_quant_s: float = 0.5
    enroll_per_meeting: int = 30
    batch: int = 64

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]


@dataclass(frozen=True)
class DecodeParams:
    theta: float = 2.0          # UNKNOWN state score (AS-norm units)
    penalty: float = 4.0        # switch penalty λ
    reset_gap_s: float = 2.0    # no penalty across a speech gap longer than this
    top_n: int = 100            # AS-norm cohort top-N
    fill_s: float = 1.0         # VAD frames further than this from any window stay unlabelled


# ─── voiceprint imports (scripts/ is not a package root) ───────────────────

def _vp():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    from voiceprint import audio, embedders, scoring, vad  # noqa: F401
    return audio, embedders, scoring, vad


def cache_dir(dataset: str, model_id: str, wp: WinParams) -> Path:
    tag = hashlib.sha1(model_id.encode()).hexdigest()[:10]
    return bench_dir() / "cache" / "b0" / dataset / f"{tag}-{wp.key()}"


# ─── embedding stage (one process per worker) ──────────────────────────────

_EMB = None


def _embedder(device: str):
    global _EMB
    if _EMB is None:
        _, embedders, _, _ = _vp()
        _EMB = embedders.get_embedder("redimnet2", device=device)
    return _EMB


MAX_BATCH_S = 48.0  # audio seconds per forward pass; bounds transient MPS memory


def _batches(lengths: list[int], batch: int, max_samples: int) -> list[list[int]]:
    """Index batches of ≤ `batch` clips and ≤ `max_samples` total samples, length-sorted so
    equal-length clips share a forward pass (embed_batch only stacks identical lengths)."""
    out, cur, tot = [], [], 0
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        if cur and (len(cur) == batch or tot + lengths[i] > max_samples):
            out.append(cur)
            cur, tot = [], 0
        cur.append(i)
        tot += lengths[i]
    return out + [cur] if cur else out


def _embed(emb, waves: list[np.ndarray], batch: int) -> np.ndarray:
    """Embed in bounded batches. On MPS the caching allocator keeps buffers per input shape
    (measured: 83 GB footprint for mixed enrollment lengths), so free it after every batch."""
    out = np.zeros((len(waves), emb.dim), np.float32)
    mps = str(getattr(emb, "_device", "")) == "mps"
    for idx in _batches([len(w) for w in waves], batch, int(MAX_BATCH_S * SR)):
        out[idx] = emb.embed_batch([waves[i] for i in idx])
        if mps:
            import torch
            torch.mps.empty_cache()
    return out


def embed_meeting(meeting: Meeting, cdir: Path, wp: WinParams, device: str) -> dict:
    """Cache test windows (test stream) and enrollment clips (enrollment stream) for one meeting."""
    audio, _, _, vad = _vp()
    emb = _embedder(device)
    cdir.mkdir(parents=True, exist_ok=True)
    stats = {"meeting": meeting.id}
    t_path, e_path = cdir / f"{meeting.id}.test.npz", cdir / f"{meeting.id}.enroll.npz"

    if not t_path.is_file():
        t0 = time.time()
        wave = audio.decode_audio(meeting.test_audio)
        t1 = time.time()
        mask = vad.silero_mask(wave)
        t2 = time.time()
        W, H = int(wp.win_s * SR), int(wp.hop_s * SR)
        starts = np.arange(0, max(1, len(wave) - W + 1), H)
        fr = np.repeat(mask.frames, vad.HOP)[:len(wave)]
        csum = np.concatenate([[0], np.cumsum(fr, dtype=np.int64)])
        frac = (csum[np.minimum(starts + W, len(wave))] - csum[starts]) / W
        keep = starts[frac >= wp.min_speech_frac]
        E = _embed(emb, [wave[s:s + W] for s in keep], wp.batch)
        t3 = time.time()
        np.savez(t_path, starts=keep / SR, emb=E.astype(np.float32), vad=mask.frames,
                 vad_hop=np.array(vad.HOP), duration=np.array(len(wave) / SR))
        dur = len(wave) / SR
        stats.update(test_audio_s=round(dur, 1), windows=int(len(keep)), decode_s=round(t1 - t0, 1),
                     vad_s=round(t2 - t1, 1), embed_s=round(t3 - t2, 1),
                     embed_xrt=round(dur / max(t3 - t2, 1e-6), 1), total_xrt=round(dur / (t3 - t0), 1))
        del wave

    if not e_path.is_file():
        t0 = time.time()
        wave = audio.decode_audio(meeting.enroll_audio)
        mask = vad.silero_mask(wave)
        ref = meeting.reference()
        spk, spans, waves = [], [], []
        q = wp.enroll_quant_s
        for s in meeting.speakers:
            pieces = []
            for a, b in solo_regions(ref, s, wp.enroll_guard_s, wp.enroll_min_s):
                t = a
                while b - t >= wp.enroll_min_s:
                    d = min(wp.enroll_max_s, b - t)
                    d = np.floor(d / q) * q
                    pieces.append((t, t + d))
                    t += d
            ok = []
            for a, b in pieces:
                ia, ib = int(a * SR), min(int(b * SR), len(wave))
                if ib - ia >= int(wp.enroll_min_s * SR) and mask.region(ia, ib).mean() >= wp.min_speech_frac:
                    ok.append((a, b))
            ok = sorted(ok, key=lambda p: p[0] - p[1])[:wp.enroll_per_meeting]
            for a, b in sorted(ok):
                spk.append(s)
                spans.append((a, b))
                waves.append(wave[int(a * SR):int(a * SR) + int(round((b - a) * SR))])
        E = _embed(emb, waves, wp.batch)
        np.savez(e_path, speaker=np.array(spk), spans=np.array(spans, dtype=np.float64).reshape(-1, 2),
                 emb=E.astype(np.float32))
        stats.update(enroll_clips=len(waves), enroll_s=round(time.time() - t0, 1))
    stats["maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)
    return stats


def _worker_init(env: dict) -> None:
    os.environ.update(env)


def embed_all(meetings: list[Meeting], cdir: Path, wp: WinParams, device: str, workers: int,
              log) -> list[dict]:
    todo = [m for m in meetings
            if not ((cdir / f"{m.id}.test.npz").is_file() and (cdir / f"{m.id}.enroll.npz").is_file())]
    log(f"embedding {len(todo)}/{len(meetings)} meetings on {device} with {workers} worker(s) → {cdir}")
    if not todo:
        return []
    out = []
    if workers <= 1:
        for m in todo:
            out.append(embed_meeting(m, cdir, wp, device))
            log(json.dumps(out[-1]))
        return out
    env = {k: v for k, v in os.environ.items() if k.startswith(("VOICEPRINT", "SPEAKER_BENCH", "PYTORCH"))}
    with ProcessPoolExecutor(workers, initializer=_worker_init, initargs=(env,)) as ex:
        for st in ex.map(embed_meeting, todo, [cdir] * len(todo), [wp] * len(todo), [device] * len(todo)):
            out.append(st)
            log(json.dumps(st))
    return out


# ─── scoring and decoding (numpy only) ─────────────────────────────────────

def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def _load(cdir: Path, mid: str, kind: str) -> dict:
    with np.load(cdir / f"{mid}.{kind}.npz") as z:
        return {k: z[k] for k in z.files}


def enrollment_models(meeting: Meeting, pool: list[Meeting], cdir: Path) -> dict[str, np.ndarray]:
    _, _, scoring, _ = _vp()
    by: dict[str, list[np.ndarray]] = {}
    for spk, srcs in enrollment(meeting, pool).items():
        for src in srcs:
            z = _load(cdir, src.meeting, "enroll")
            sel = z["speaker"] == spk
            if sel.any():
                by.setdefault(spk, []).append(z["emb"][sel])
    out = {}
    for spk, parts in by.items():
        E = _l2(np.concatenate(parts))
        out[spk] = scoring.mean_direction(E[scoring.mad_prune(E)])
    return out


def cohort_vectors(meeting: Meeting, pool: list[Meeting], cdir: Path) -> np.ndarray:
    vecs = []
    for key, ms in cohort_speakers(meeting, pool).items():
        spk = key.split("/", 1)[1]
        for m in ms:
            z = _load(cdir, m.id, "enroll")
            vecs.append(z["emb"][z["speaker"] == spk])
    return _l2(np.concatenate(vecs)) if vecs else np.zeros((0, 1), np.float32)


def _top_stats(S: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean / std of the top-N values along axis 0 of S (cohort × items)."""
    n = min(top_n, S.shape[0])
    top = np.partition(S, S.shape[0] - n, axis=0)[S.shape[0] - n:]
    return top.mean(0), np.maximum(top.std(0), 1e-4)


def score_windows(test_emb: np.ndarray, models: dict[str, np.ndarray], cohort: np.ndarray,
                  top_n: int) -> tuple[list[str], np.ndarray]:
    names = sorted(models)
    C = np.stack([models[n] for n in names])            # (K, D)
    X = _l2(test_emb)
    raw = X @ C.T                                        # (T, K)
    if cohort.shape[0] < 10:
        return names, raw
    mu_e, sd_e = _top_stats(cohort @ C.T, top_n)         # (K,)
    mu_t, sd_t = _top_stats(cohort @ X.T, top_n)         # (T,)
    return names, 0.5 * ((raw - mu_e) / sd_e + (raw - mu_t[:, None]) / sd_t[:, None])


def viterbi(scores: np.ndarray, theta: float, penalty: float, times: np.ndarray,
            reset_gap_s: float) -> np.ndarray:
    """Best state path over K speaker states + UNKNOWN (index K, constant score θ);
    each change of state costs `penalty`, except across gaps > reset_gap_s."""
    T, K = scores.shape
    E = np.concatenate([scores, np.full((T, 1), theta)], axis=1)
    if T == 0:
        return np.zeros(0, int)
    back = np.zeros((T, K + 1), dtype=np.int32)
    v = E[0].copy()
    for t in range(1, T):
        pen = 0.0 if times[t] - times[t - 1] > reset_gap_s else penalty
        best = int(v.argmax())
        stay = v
        switch = v[best] - pen
        take_switch = switch > stay
        back[t] = np.where(take_switch, best, np.arange(K + 1))
        v = np.where(take_switch, switch, stay) + E[t]
    path = np.empty(T, dtype=np.int32)
    path[-1] = int(v.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def to_segments(test: dict, labels: list[str], path: np.ndarray, wp: WinParams,
                fill_s: float) -> list[Seg]:
    """Label each VAD speech frame with the nearest kept window's state; run-length encode."""
    hop = int(test["vad_hop"]) / SR
    frames = test["vad"].astype(bool)
    centers = test["starts"] + wp.win_s / 2
    idx = np.flatnonzero(frames)
    if not len(idx) or not len(centers):
        return []
    t = (idx + 0.5) * hop
    j = np.clip(np.searchsorted(centers, t), 0, len(centers) - 1)
    jl = np.maximum(j - 1, 0)
    j = np.where(np.abs(t - centers[jl]) <= np.abs(t - centers[j]), jl, j)
    near = np.abs(t - centers[j]) <= fill_s
    f, st = idx[near], path[j][near]
    if not len(f):
        return []
    # a new run starts where the frame index jumps or the state changes
    brk = np.flatnonzero((np.diff(f) != 1) | (np.diff(st) != 0)) + 1
    starts = np.concatenate([[0], brk])
    ends = np.concatenate([brk, [len(f)]]) - 1
    names = labels + [UNK]
    return [Seg(f[a] * hop, (f[b] + 1) * hop, names[st[a]]) for a, b in zip(starts, ends)]


@dataclass
class Prepared:
    meeting: Meeting
    test: dict
    names: list[str]
    scores: np.ndarray           # AS-norm (T, K) over the closed roster
    roster_sets: dict[str, set[str]]


def prepare_scores(meetings: list[Meeting], pool: list[Meeting], cdir: Path, top_n: int) -> list[Prepared]:
    out = []
    for m in meetings:
        models = enrollment_models(m, pool, cdir)
        names, S = score_windows(_load(cdir, m.id, "test")["emb"], models, cohort_vectors(m, pool, cdir), top_n)
        out.append(Prepared(m, _load(cdir, m.id, "test"), names, S, rosters(m, names)))
    return out


def decode(p: Prepared, roster: set[str], dp: DecodeParams, wp: WinParams) -> list[Seg]:
    cols = [i for i, n in enumerate(p.names) if n in roster]
    names = [p.names[i] for i in cols]
    path = viterbi(p.scores[:, cols], dp.theta, dp.penalty, p.test["starts"], dp.reset_gap_s)
    return to_segments(p.test, names, path, wp, dp.fill_s)


def tune(prepared: list[Prepared], wp: WinParams, thetas, penalties, log) -> tuple[DecodeParams, list]:
    """Grid search minimizing mean(closed IER, leave-one-out IER) at collar 0 (frame IER)."""
    from ..metrics import frame_ier

    refs = {p.meeting.id: (p.meeting.reference(), p.meeting.uem()) for p in prepared}
    grid = []
    for th in thetas:
        for pen in penalties:
            dp = DecodeParams(theta=th, penalty=pen)
            ce = ct = le = lt = 0.0
            for p in prepared:
                ref, uem = refs[p.meeting.id]
                e, t = frame_ier(ref, decode(p, p.roster_sets["closed"], dp, wp), uem, p.roster_sets["closed"])
                ce, ct = ce + e, ct + t
                for k, r in p.roster_sets.items():
                    if k.startswith("loo:"):
                        e, t = frame_ier(ref, decode(p, r, dp, wp), uem, r)
                        le, lt = le + e, lt + t
            obj = 0.5 * (ce / ct + (le / lt if lt else ce / ct))
            grid.append({"theta": th, "penalty": pen, "ier_closed": ce / ct,
                         "ier_loo": le / lt if lt else None, "objective": obj})
            log(f"tune θ={th:g} λ={pen:g}: closed {100 * ce / ct:.2f} loo {100 * le / max(lt, 1e-9):.2f}")
    best = min(grid, key=lambda g: g["objective"])
    return DecodeParams(theta=best["theta"], penalty=best["penalty"]), grid


def write_system(prepared: list[Prepared], dp: DecodeParams, wp: WinParams, out: Path) -> None:
    for p in prepared:
        mid = p.meeting.id
        write_rttm(out / "closed" / f"{mid}.rttm", mid, decode(p, p.roster_sets["closed"], dp, wp))
        for k, r in p.roster_sets.items():
            if k.startswith("loo:"):
                write_rttm(out / "loo" / mid / f"{k[4:]}.rttm", mid, decode(p, r, dp, wp))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=["ami", "aishell4"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tune-split", help="tune θ/λ on this split of the same dataset")
    ap.add_argument("--tune-dataset", help="dataset of --tune-split (default: --dataset)")
    ap.add_argument("--params", type=Path, help="decode params JSON (e.g. a previous run's b0-params.json)")
    ap.add_argument("--theta", type=float)
    ap.add_argument("--penalty", type=float)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0, 0.5, 1, 1.5, 2, 2.5, 3, 4])
    ap.add_argument("--penalties", type=float, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--embed-only", action="store_true")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "b0.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, file=sys.stderr, flush=True)
        logf.write(line + "\n")
        logf.flush()

    wp = WinParams()
    _, embedders, _, _ = _vp()
    model_id = embedders.get_embedder("redimnet2", device="cpu").model_id  # cheap: weights verify only
    timing = []
    splits = [(a.dataset, a.split)]
    if a.tune_split:
        splits.insert(0, (a.tune_dataset or a.dataset, a.tune_split))
    pools = {}
    for ds, sp in splits:
        pool = load_meetings(ds, sp)
        pools[(ds, sp)] = pool
        cdir = cache_dir(ds, model_id, wp)
        t0 = time.time()
        timing += embed_all(pool, cdir, wp, a.device, a.workers, log)
        log(f"{ds}/{sp}: embedding stage {time.time() - t0:.0f} s")
    if a.embed_only:
        return 0

    dp = DecodeParams()
    grid = None
    if a.params:
        dp = DecodeParams(**json.loads(a.params.read_text())["decode"])
    elif a.tune_split:
        ds, sp = splits[0]
        prep = prepare_scores(pools[(ds, sp)], pools[(ds, sp)], cache_dir(ds, model_id, wp), dp.top_n)
        dp, grid = tune(prep, wp, a.thetas, a.penalties, log)
        log(f"tuned on {ds}/{sp}: θ={dp.theta:g} λ={dp.penalty:g}")
    if a.theta is not None or a.penalty is not None:
        dp = DecodeParams(theta=dp.theta if a.theta is None else a.theta,
                          penalty=dp.penalty if a.penalty is None else a.penalty)

    pool = pools[(a.dataset, a.split)]
    prep = prepare_scores(pool, pool, cache_dir(a.dataset, model_id, wp), dp.top_n)
    for name, d in (("b0", dp), ("b0-argmax", DecodeParams(theta=dp.theta, penalty=0.0))):
        write_system(prep, d, wp, a.out / name / "system")
    (a.out / "b0-params.json").write_text(json.dumps(
        {"window": asdict(wp), "decode": asdict(dp), "model_id": model_id,
         "tuned_on": f"{splits[0][0]}/{splits[0][1]}" if grid else None, "grid": grid,
         "timing": timing}, indent=2) + "\n")
    from .. import run as bench_run
    for name in ("b0", "b0-argmax"):
        bench_run.main(["--dataset", a.dataset, "--split", a.split,
                        "--system", str(a.out / name / "system"), "--out", str(a.out / name)])
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
