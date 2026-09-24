"""
Boundary layer: chunk-local speaker activity for known-speaker tracking.

Contract (see the speaker-tracking plan):
    segment(wave, sr=16000, backend="seg3", **kw) -> LocalSegmentation
    local_regions(seg, min_clean_s, exclude_overlap=True) -> list[LocalRegion]

Backends
    seg3       pyannote segmentation-3.0 (MIT) run as the sherpa-onnx ONNX export on
               onnxruntime CPU. Sliding 10 s chunks; each chunk has up to 3 *local*
               speakers whose indices mean nothing across chunks (linking them is the
               tracker's job). Powerset output (7 classes: none, 3 singles, 3 pairs) is
               decoded to per-local-speaker marginals, P(k) = sum of classes holding k.
    nemotron3  NVIDIA Nemotron 3 Diarization (OpenMDW-1.1); see segment_nemotron.py.
               Global arrival-ordered slots, so n_chunks = 1.

Only the segmentation *model* is used, never pyannote's diarization pipeline or its
reconstruction step: chunks are streamed through the model in small batches straight
into one preallocated activation array, and the global speech/overlap tracks are
overlap-added on the fly. Peak memory ≈ wave + activations (≈ 25 MB/h at the default
step) + one batch.

Frame timing (seg3, from the model's receptive field): frame i of a chunk starting at
s covers samples [s + 270 i, s + 270 i + 991) at 16 kHz, i.e. frame_step = 16.875 ms,
frame_dur = 61.9 ms. A frame stands for the frame_step-wide interval around its centre
s + i·frame_step + frame_dur/2; spans are built from those intervals.

Parity: the ONNX export matches pyannote.audio's torch model to max |Δp| = 6e-6 on
synthetic speech (argmax agreement 100 %); the int8 export does not (max |Δp| 0.79),
so only the float model is pinned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import model_dir

SAMPLE_RATE = 16000

# ─── seg3 model pin (fetch_models.py reads this) ──────────────────────────────


@dataclass(frozen=True)
class Seg3Spec:
    name: str = "seg3"
    archive: str = "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    archive_sha256: str = "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488"
    member: str = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    sha256: str = "220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079"
    extra_members: tuple[str, ...] = ("sherpa-onnx-pyannote-segmentation-3-0/LICENSE",)

    @property
    def dir(self) -> Path:
        return model_dir() / "sherpa-onnx" / "pyannote-segmentation-3-0"

    @property
    def path(self) -> Path:
        return self.dir / "model.onnx"

    @property
    def url(self) -> str:
        return ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                f"speaker-segmentation-models/{self.archive}")


SEG3 = Seg3Spec()

CHUNK_S = 10.0                      # seg-3.0 training chunk length
FRAME_STEP_SAMPLES = 270
FRAME_DUR_SAMPLES = 991
N_FRAMES = 589                      # frames per 10 s chunk
# Powerset class → local speakers (pyannote order: ∅, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}).
POWERSET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                     [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
DEFAULT_STEP_S = 2.5
BACKENDS = ("seg3", "nemotron3")


def available(backend: str = "seg3") -> bool:
    """Deps importable and model present. Cheap: no heavy imports."""
    import importlib.util
    if backend == "seg3":
        return importlib.util.find_spec("onnxruntime") is not None and SEG3.path.is_file()
    if backend == "nemotron3":
        from . import segment_nemotron
        return segment_nemotron.available()
    return False


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class LocalSegmentation:
    chunk_starts: np.ndarray        # (n_chunks,) seconds; not necessarily evenly spaced (last chunk is end-aligned)
    chunk_dur: float                # seconds covered by every chunk
    frame_step: float               # seconds between frames
    frame_dur: float                # receptive field of one frame (seconds)
    activations: np.ndarray         # (n_chunks, n_frames, n_local) float32 in [0, 1]
    backend: str
    meta: dict = field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        return int(self.activations.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.activations.shape[1])

    @property
    def n_local(self) -> int:
        return int(self.activations.shape[2])

    def frame_centers(self, chunk: int) -> np.ndarray:
        """Absolute centre time (s) of every frame of `chunk`."""
        return (self.chunk_starts[chunk] + np.arange(self.n_frames) * self.frame_step
                + self.frame_dur / 2)


@dataclass
class LocalRegion:
    chunk: int
    local: int
    spans: list[tuple[float, float]]   # absolute seconds
    clean_s: float                     # total duration of `spans`


# ─── Public API ──────────────────────────────────────────────────────────────


def segment(wave: np.ndarray, sr: int = SAMPLE_RATE, backend: str = "seg3", **kw) -> LocalSegmentation:
    """Chunk-local speaker activations for a 16 kHz mono float32 wave.

    seg3 keywords: step (s, default 2.5), batch_size (32), num_threads
    (VOICEPRINT_THREADS or 4). nemotron3 keywords: see segment_nemotron.segment.
    """
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}; resample first")
    w = np.asarray(wave)
    if w.ndim != 1:
        raise ValueError(f"expected mono 1-D wave, got shape {w.shape}")
    if w.dtype != np.float32:
        w = w.astype(np.float32)
    if backend == "seg3":
        return _segment_seg3(w, **kw)
    if backend == "nemotron3":
        from . import segment_nemotron
        return segment_nemotron.segment(w, **kw)
    raise ValueError(f"unknown backend {backend!r}; known: {BACKENDS}")


def local_regions(seg: LocalSegmentation, min_clean_s: float, exclude_overlap: bool = True,
                  *, threshold: float = 0.5, min_span_s: float = 0.1) -> list[LocalRegion]:
    """Each (chunk, local speaker)'s spans where that speaker is active.

    A frame is active for local k when its marginal ≥ `threshold`; with
    `exclude_overlap` it counts only when no other local speaker of the chunk is
    active. Spans shorter than `min_span_s` are dropped (single-frame flicker), and
    only (chunk, local) pairs with ≥ `min_clean_s` of spans in total are returned.
    Spans are clipped to the audio duration.
    """
    out: list[LocalRegion] = []
    dur = float(seg.meta.get("duration", np.inf))
    half = seg.frame_step / 2
    for c in range(seg.n_chunks):
        active = seg.activations[c] >= threshold                  # (n_frames, n_local)
        if not active.any():
            continue
        if exclude_overlap:
            active = active & (active.sum(axis=1, keepdims=True) == 1)
        centers = seg.frame_centers(c)
        for k in range(seg.n_local):
            spans = []
            for a, b in _runs(active[:, k]):
                s = max(0.0, centers[a] - half)
                e = min(dur, centers[b - 1] + half)
                if e - s >= min_span_s:
                    spans.append((float(s), float(e)))
            total = sum(e - s for s, e in spans)
            if spans and total >= min_clean_s:
                out.append(LocalRegion(c, k, spans, float(total)))
    return out


def global_frames(seg: LocalSegmentation) -> np.ndarray:
    """Centre times (s) of the global frame grid used by meta['speech'] / meta['overlap']."""
    n = int(seg.meta["speech"].shape[0])
    return np.arange(n) * seg.frame_step + seg.frame_dur / 2


# ─── seg3 backend ────────────────────────────────────────────────────────────

_SESSIONS: dict[tuple[str, int], object] = {}


def _session(path: Path, num_threads: int):
    key = (str(path), num_threads)
    if key not in _SESSIONS:
        import onnxruntime as ort  # lazy
        from .embedders import _verify
        _verify(path, SEG3.sha256)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        _SESSIONS[key] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def chunk_starts_for(n_samples: int, step_s: float) -> np.ndarray:
    """Chunk start samples: every `step_s`, plus one end-aligned chunk covering the tail."""
    win = int(CHUNK_S * SAMPLE_RATE)
    step = int(round(step_s * SAMPLE_RATE))
    if step <= 0:
        raise ValueError("step must be > 0")
    if n_samples <= win:
        return np.zeros(1, dtype=np.int64)
    starts = np.arange(0, n_samples - win + 1, step, dtype=np.int64)
    if starts[-1] != n_samples - win:
        starts = np.append(starts, n_samples - win)
    return starts


def _segment_seg3(wave: np.ndarray, step: float = DEFAULT_STEP_S, batch_size: int = 32,
                  num_threads: int | None = None, model_path: str | Path | None = None) -> LocalSegmentation:
    if step > CHUNK_S:
        raise ValueError(f"step {step} s > chunk {CHUNK_S} s would leave gaps")
    num_threads = num_threads or int(os.environ.get("VOICEPRINT_THREADS", "4"))
    sess = _session(Path(model_path) if model_path else SEG3.path, num_threads)
    win = int(CHUNK_S * SAMPLE_RATE)
    n = wave.shape[0]
    padded = wave
    if n < win:                                   # short file: one zero-padded chunk
        padded = np.zeros(win, dtype=np.float32)
        padded[:n] = wave
    starts = chunk_starts_for(n, step)
    view = np.lib.stride_tricks.sliding_window_view(padded, win)   # no copy

    fs = FRAME_STEP_SAMPLES / SAMPLE_RATE
    n_global = int(round(max(n, win) / FRAME_STEP_SAMPLES)) + 1
    speech_sum = np.zeros(n_global, np.float32)
    overlap_sum = np.zeros(n_global, np.float32)
    cover = np.zeros(n_global, np.float32)
    acts = np.empty((len(starts), N_FRAMES, 3), np.float32)
    offsets = np.rint(starts / FRAME_STEP_SAMPLES).astype(np.int64)

    for b in range(0, len(starts), batch_size):
        idx = starts[b:b + batch_size]
        x = np.ascontiguousarray(view[idx])[:, None, :]            # (B, 1, win)
        logp = sess.run(None, {"x": x})[0]                          # (B, 589, 7) log-softmax
        p = np.exp(logp, dtype=np.float32)
        acts[b:b + len(idx)] = p @ POWERSET
        speech = 1.0 - p[..., 0]
        overlap = p[..., 4:].sum(axis=-1)
        for j, off in enumerate(offsets[b:b + len(idx)]):
            e = min(off + N_FRAMES, n_global)
            m = e - off
            speech_sum[off:e] += speech[j, :m]
            overlap_sum[off:e] += overlap[j, :m]
            cover[off:e] += 1
    cover = np.maximum(cover, 1)
    meta = {
        "duration": n / SAMPLE_RATE,
        "step": float(step),
        "model": f"pyannote-segmentation-3.0-onnx@sha256:{SEG3.sha256[:12]}",
        "num_threads": num_threads,
        "speech": speech_sum / cover,       # (n_global,) mean P(any speech) over covering chunks
        "overlap": overlap_sum / cover,     # (n_global,) mean P(≥ 2 speakers)
    }
    return LocalSegmentation(chunk_starts=starts / SAMPLE_RATE, chunk_dur=CHUNK_S, frame_step=fs,
                             frame_dur=FRAME_DUR_SAMPLES / SAMPLE_RATE, activations=acts,
                             backend="seg3", meta=meta)


# ─── helpers ─────────────────────────────────────────────────────────────────


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of the True runs in a 1-D bool array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
