"""Speaker-embedding backends for voiceprint identification.

Contract:
    get_embedder(name="redimnet2", device="cpu") -> Embedder
    available_embedders() -> list[str]
    Embedder.embed(wave) -> (dim,) float32, L2-normalized
    Embedder.embed_batch(waves) -> (n, dim) float32, L2-normalized

Input is always float32 mono 16 kHz in [-1, 1]; callers resample (ffmpeg) before
calling. Nothing here resamples. torch / sherpa_onnx are imported lazily inside
get_embedder so light CLI paths never pay for them.

Model files live outside notes and this clone under VOICEPRINT_MODEL_DIR
(default ~/.cache/meeting-documenter/models, see config.py); populate with
`python scripts/voiceprint/fetch_models.py`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import model_dir

SAMPLE_RATE = 16000
MIN_SAMPLES = SAMPLE_RATE // 2  # 0.5 s; shorter clips carry too little speaker info
# Audio per forward pass. The MPS caching allocator keeps buffers for every input shape it
# has seen, and that memory is invisible to RSS: ReDimNet2 reached a 74-85 GB physical
# footprint on mixed clip lengths while RSS showed 3-8 GB. Batches are therefore capped at
# this much audio and the MPS cache is released after each one (9 GB peak, no slowdown).
MAX_BATCH_S = 48.0
# PYTORCH_MPS_{HIGH,LOW}_WATERMARK_RATIO defaults, set before the first MPS allocation. The
# high mark turns runaway growth into an error; the low mark must not exceed it (torch's
# default low mark on unified memory is 1.4, which fails validation against 0.15).
MPS_WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.15", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.1"}


# ---------------------------------------------------------------------------
# Pinned model specs (single source of truth; fetch_models.py reads these)
# ---------------------------------------------------------------------------

REDIMNET2_REPO = "PalabraAI/redimnet2"
REDIMNET2_SHA = "c5bbe0b76e37df698c403f8844e41304ceab6307"  # main @ 2026-08-28
REDIMNET2_RELEASE = "v1.0.0"


@dataclass(frozen=True)
class TorchSpec:
    name: str
    model_size: str
    dataset: str
    train_type: str
    weight_file: str
    sha256: str

    @property
    def code_dir(self) -> Path:
        return model_dir() / f"redimnet2@{REDIMNET2_SHA}"

    @property
    def weight_path(self) -> Path:
        return self.code_dir / "weights" / self.weight_file

    @property
    def url(self) -> str:
        return (f"https://github.com/{REDIMNET2_REPO}/releases/download/"
                f"{REDIMNET2_RELEASE}/{self.weight_file}")


@dataclass(frozen=True)
class OnnxSpec:
    name: str
    tag: str  # human-readable model tag used in model_id
    file: str
    sha256: str

    @property
    def path(self) -> Path:
        return model_dir() / "sherpa-onnx" / self.file

    @property
    def url(self) -> str:
        return ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                f"speaker-recongition-models/{self.file}")  # sic: upstream tag typo


TORCH_SPECS: dict[str, TorchSpec] = {
    "redimnet2": TorchSpec(
        name="redimnet2", model_size="b6", dataset="vb2+vox2+cnc2_v0", train_type="lm",
        weight_file="b6-vb2+vox2+cnc2_v0-lm.pt",
        sha256="287365f6f485b19e65e5176554f8f7123bfa8d85185f3d2c040eab51acec9868",
    ),
}

ONNX_SPECS: dict[str, OnnxSpec] = {
    "eres2netv2": OnnxSpec(
        name="eres2netv2", tag="3dspeaker-eres2netv2-zh-cn-200k",
        file="3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx",
        sha256="bf1a75b9930474cf3389ef415e6e5d38ca96fea4a3a00f7e301d080a58ee2239",
    ),
    "campplus_zh_en": OnnxSpec(
        name="campplus_zh_en", tag="3dspeaker-campplus-zh-en-advanced",
        file="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
    ),
    "wespeaker_r34": OnnxSpec(
        name="wespeaker_r34", tag="wespeaker-voxceleb-resnet34-lm",
        file="wespeaker_en_voxceleb_resnet34_LM.onnx",
        sha256="e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
    ),
}

ALL_NAMES = list(TORCH_SPECS) + list(ONNX_SPECS)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    name: str
    model_id: str
    dim: int
    sample_rate: int

    def embed(self, wave: np.ndarray) -> np.ndarray: ...
    def embed_batch(self, waves: list[np.ndarray]) -> np.ndarray: ...


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def available_embedders() -> list[str]:
    """Names whose deps and model files are present. Cheap: no heavy imports."""
    out: list[str] = []
    if _has("torch") and _has("torchaudio") and _has("scipy"):
        for n, s in TORCH_SPECS.items():
            if s.weight_path.is_file() and (s.code_dir / "redimnet2" / "__init__.py").is_file():
                out.append(n)
    if _has("sherpa_onnx"):
        out.extend(n for n, s in ONNX_SPECS.items() if s.path.is_file())
    return out


def model_id_for(name: str) -> str:
    """The `model_id` an embedder built from the pinned files reports, without loading it:
    weights are sha256-verified against the pin at load, so the id follows from the spec.
    Lets cheap readiness checks find the store's exemplars and calibration."""
    if name in TORCH_SPECS:
        s = TORCH_SPECS[name]
        return f"redimnet2-{s.model_size}-{s.dataset}-{s.train_type}@{REDIMNET2_SHA[:12]}+w{s.sha256[:12]}"
    if name in ONNX_SPECS:
        s = ONNX_SPECS[name]
        return f"{s.tag}@sha256:{s.sha256[:12]}"
    raise ValueError(f"unknown embedder {name!r}; known: {ALL_NAMES}")


def get_embedder(name: str = "redimnet2", device: str = "cpu") -> Embedder:
    """Build an embedder. device: 'cpu' (default, deterministic) | 'mps' (redimnet2)
    | 'coreml' (sherpa backends). Raises FileNotFoundError if models are missing."""
    _set_cache_env()
    if name in TORCH_SPECS:
        return _ReDimNet2Embedder(TORCH_SPECS[name], device)
    if name in ONNX_SPECS:
        return _SherpaEmbedder(ONNX_SPECS[name], device)
    raise ValueError(f"unknown embedder {name!r}; known: {ALL_NAMES}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_cache_env() -> None:
    root = model_dir()
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    os.environ.setdefault("HF_HOME", str(root / "hf"))


def _num_threads() -> int:
    # Fixed thread count keeps float reductions reproducible run-to-run.
    return int(os.environ.get("VOICEPRINT_THREADS", "4"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} missing; run: python scripts/voiceprint/fetch_models.py")
    got = sha256_file(path)
    if got != expected:
        raise RuntimeError(f"sha256 mismatch for {path}: got {got[:12]}, pinned {expected[:12]}")
    return got


def _check_wave(wave: np.ndarray) -> np.ndarray:
    w = np.asarray(wave, dtype=np.float32)
    if w.ndim != 1:
        raise ValueError(f"expected mono 1-D wave, got shape {w.shape}")
    if w.shape[0] < MIN_SAMPLES:
        raise ValueError(f"clip too short: {w.shape[0]} samples < {MIN_SAMPLES} (0.5 s @16k)")
    if not np.isfinite(w).all():
        raise ValueError("wave contains NaN/inf")
    return np.ascontiguousarray(w)


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


# ---------------------------------------------------------------------------
# ReDimNet2 (vendored torch code, local weights)
# ---------------------------------------------------------------------------

class _ReDimNet2Embedder:
    sample_rate = SAMPLE_RATE

    def __init__(self, spec: TorchSpec, device: str):
        if device == "mps" and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
            os.environ.update(MPS_WATERMARKS)
        import torch  # lazy

        if device not in ("cpu", "mps"):
            raise ValueError(f"redimnet2 supports device cpu|mps, got {device!r}")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        if not (spec.code_dir / "redimnet2" / "__init__.py").is_file():
            raise FileNotFoundError(
                f"vendored code missing at {spec.code_dir}; run fetch_models.py")
        _verify(spec.weight_path, spec.sha256)

        torch.set_num_threads(_num_threads())
        code = str(spec.code_dir)
        if code not in sys.path:
            sys.path.insert(0, code)
        from redimnet2 import ReDimNet2Wrap  # vendored at REDIMNET2_SHA

        ckpt = torch.load(spec.weight_path, map_location="cpu", weights_only=True)
        model = ReDimNet2Wrap(**ckpt["model_config"])
        res = model.load_state_dict(ckpt["state_dict"])
        if res.missing_keys or res.unexpected_keys:
            raise RuntimeError(f"state_dict mismatch: {res}")
        model.eval().to(device)

        self._torch = torch
        self._model = model
        self._device = device
        self._since_release = 0
        self.name = spec.name
        self.dim = int(ckpt["model_config"]["embed_dim"])
        self.model_id = model_id_for(spec.name)      # _verify pinned the weights above

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        t = self._torch.from_numpy(batch).to(self._device)
        with self._torch.inference_mode():
            e = self._model(t)
        out = e.float().cpu().numpy()
        if self._device == "mps":
            # Release the cache once per MAX_BATCH_S of audio, not per call: per-call release
            # of one-clip forwards cost ~25 % speed for no footprint gain.
            del t, e
            self._since_release += batch.size
            if self._since_release >= MAX_BATCH_S * SAMPLE_RATE:
                self._torch.mps.empty_cache()
                self._since_release = 0
        return out

    def embed(self, wave: np.ndarray) -> np.ndarray:
        return _l2(self._forward(_check_wave(wave)[None, :]))[0]

    def embed_batch(self, waves: list[np.ndarray]) -> np.ndarray:
        # The model z-scores each clip over its own samples and mean-normalizes the
        # log-mel over its own frames, so zero-padding would change embeddings.
        # Batch only clips of identical length, at most MAX_BATCH_S of audio per pass.
        ws = [_check_wave(w) for w in waves]
        out = np.empty((len(ws), self.dim), dtype=np.float32)
        for idx in bounded_batches([w.shape[0] for w in ws]):
            out[idx] = self._forward(np.stack([ws[i] for i in idx]))
        return _l2(out)


def bounded_batches(lengths: list[int], max_samples: int | None = None) -> list[list[int]]:
    """Index batches of equal-length clips holding ≤ `max_samples` samples in total (default
    MAX_BATCH_S of audio; a single longer clip gets a batch of its own), in length order."""
    cap = max_samples or int(MAX_BATCH_S * SAMPLE_RATE)
    out: list[list[int]] = []
    cur: list[int] = []
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        if cur and (lengths[i] != lengths[cur[0]] or (len(cur) + 1) * lengths[i] > cap):
            out.append(cur)
            cur = []
        cur.append(i)
    return out + [cur] if cur else out


# ---------------------------------------------------------------------------
# sherpa-onnx backends (3D-Speaker / WeSpeaker ONNX exports)
# ---------------------------------------------------------------------------

class _SherpaEmbedder:
    sample_rate = SAMPLE_RATE

    def __init__(self, spec: OnnxSpec, device: str):
        import sherpa_onnx  # lazy

        provider = {"cpu": "cpu", "coreml": "coreml"}.get(device)
        if provider is None:
            raise ValueError(f"{spec.name} supports device cpu|coreml, got {device!r}")
        _verify(spec.path, spec.sha256)
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(spec.path), num_threads=_num_threads(), debug=False, provider=provider)
        if not cfg.validate():
            raise RuntimeError(f"invalid sherpa-onnx config for {spec.path}")
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.name = spec.name
        self.dim = int(self._ex.dim)
        self.model_id = model_id_for(spec.name)      # _verify pinned the weights above

    def embed(self, wave: np.ndarray) -> np.ndarray:
        w = _check_wave(wave)
        s = self._ex.create_stream()
        s.accept_waveform(sample_rate=SAMPLE_RATE, waveform=w)
        s.input_finished()
        if not self._ex.is_ready(s):
            raise ValueError("sherpa-onnx: not enough audio for an embedding")
        return _l2(np.asarray(self._ex.compute(s), dtype=np.float32))

    def embed_batch(self, waves: list[np.ndarray]) -> np.ndarray:
        if not waves:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.stack([self.embed(w) for w in waves])
