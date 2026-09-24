"""
Voiceprint audio helpers: decode once, then pick clean single-speaker clips.

Pipeline:
    ffmpeg → 16 kHz mono float32  (8 kHz phone audio is upsampled; no bandwidth extension)
    per utterance: trim ends → drop ±guard around other speakers → energy VAD
                   → keep ≥ min_net_s net speech → split long speech into overlapping windows
    per cluster:   keep the longest `max_clips`

Gemini-approx sidecars (turn start only; `end` = next turn start) use only the
middle `approx_window_s` of turns ≥ `approx_min_turn_s`, because their
boundaries drift.

No torch/embedder imports at module level — `embed_clips` only duck-types the
Embedder protocol from embedders.py.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass, field, asdict, replace
from pathlib import Path

import numpy as np

# DEVICE_CAP_S: hard cap on any wave handed to an embedder, by device. ReDimNet2 memory grows
# with clip length (measured on Apple silicon): CPU peak RSS 1.7 GB @ 8 s, 2.9 GB @ 15 s,
# 5.3 GB @ 30 s, 6.7 GB @ 60 s; MPS driver 2.3 GB @ 8 s, 5.8 GB @ 30 s, 12.8 GB @ 60 s
# (MPS≡CPU, cos 1.000). No override may exceed SAFETY_CEILING_S.
from .config import DEVICE_CAP_S, SAFETY_CEILING_S

SAMPLE_RATE = 16000
MAX_EMBED_S = DEVICE_CAP_S["cpu"]   # back-compat alias: the CPU cap
MIN_EMBED_S = 0.5     # embedders raise ValueError below this
MEETING_KEY_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{4})")


# ─── Parameters ─────────────────────────────────────────────────────────────

@dataclass
class ClipParams:
    trim_s: float = 0.25            # AAI boundaries bleed; trim each end
    overlap_guard_s: float = 0.3    # drop audio within ±guard of another speaker's turn
    min_net_s: float = 2.0          # min net speech per clip (after VAD)
    max_clip_s: float = 10.0        # speech ≤ this stays one clip; longer is windowed
    window_s: float = 8.0           # window length (sweepable: 4/8/12/20/30 s)
    window_hop_s: float = 4.0       # 50 % overlap
    clip_mode: str = "window"       # "window" | "longest_run" (each contiguous VAD run, up to the cap)
    device: str = "cpu"             # selects the device hard cap (embed_meeting sets it from the embedder)
    hard_cap_s: float | None = None # override the device cap (still ≤ SAFETY_CEILING_S)
    max_clips: int = 20             # longest N per cluster (identify); enroll uses more
    quant_s: float = 0.5            # centre-crop each clip to a multiple of this (0 = off). Bounds the
                                    # set of input shapes (MPS compiles once per length, ~300 ms each)
                                    # and matches the offline evaluation's quantization rule.
    # energy VAD
    vad_frame_s: float = 0.02
    vad_margin_db: float = 8.0      # speech = frame dB ≥ noise floor + margin
    vad_abs_floor_db: float = -55.0 # never call anything quieter than this speech
    vad_max_gap_s: float = 0.3      # non-speech runs longer than this are removed
    vad_min_speech_s: float = 0.1   # isolated blips shorter than this are dropped
    vad_floor_pct: float = 3.0      # noise floor = this percentile of non-silent frame energy
    # neural speech gate (Silero v5, sherpa-onnx): energy VAD passes ringtones / hold music
    neural_vad: str = "silero"      # "silero" | "energy" (energy-only; also the fallback)
    neural_min_frac: float = 0.6    # keep a clip only if ≥ this fraction of its samples is Silero speech
    # gemini-approx
    approx_min_turn_s: float = 8.0
    approx_window_s: float = 4.0    # middle 3–5 s of the turn


def cap_for(device: str | None, override: float | None = None) -> float:
    """Hard clip cap (s): override if given, else the device default (unknown → CPU),
    never above SAFETY_CEILING_S."""
    base = override if override else DEVICE_CAP_S.get(device or "cpu", DEVICE_CAP_S["cpu"])
    return float(min(base, SAFETY_CEILING_S))


def clip_cap_s(params: "ClipParams") -> float:
    return cap_for(params.device, params.hard_cap_s)


IDENTIFY_PARAMS = ClipParams()
ENROLL_PARAMS = ClipParams(max_clips=60)


@dataclass
class Utterance:
    speaker: str
    start_ms: int
    end_ms: int
    text: str = ""


@dataclass
class UtteranceSet:
    utterances: list[Utterance]
    approx: bool = False
    source: str = "unknown"
    speaker_map: dict | None = None
    transcript_id: str | None = None
    audio_duration_ms: int | None = None
    path: str | None = None

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for u in self.utterances:
            seen.setdefault(u.speaker, None)
        return list(seen)


@dataclass
class Clip:
    speaker: str
    start_ms: int                   # span in the meeting timeline (pre-VAD)
    end_ms: int
    net_dur: float                  # seconds of speech actually kept
    wave: np.ndarray | None = field(default=None, repr=False)
    speech_frac: float | None = None   # Silero speech fraction of the kept samples (None = not gated)

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("wave", None)
        return d


# ─── Decode / load ──────────────────────────────────────────────────────────

def _ffmpeg_bin() -> str:
    for cand in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if cand and Path(cand).exists():
            return cand
    raise FileNotFoundError("ffmpeg not found (expected /opt/homebrew/bin/ffmpeg)")


def _probe_duration_s(path: Path) -> float | None:
    ffprobe = Path(_ffmpeg_bin()).with_name("ffprobe")
    try:
        out = subprocess.run([str(ffprobe), "-v", "error", "-show_entries", "format=duration",
                              "-of", "default=nw=1:nk=1", str(path)],
                             capture_output=True, text=True, timeout=60, check=False)
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def decode_audio(path: str | Path, sr: int = SAMPLE_RATE, start_s: float = 0.0,
                 dur_s: float | None = None) -> np.ndarray:
    """Decode [start_s, start_s + dur_s) of any ffmpeg-readable file to mono float32 at `sr` Hz.

    Streams raw f32le from ffmpeg straight into one preallocated array (no temp
    WAV, no second bytes copy): peak ≈ 4 B × sr × decoded seconds (~230 MB/h).
    Input-side -ss/-t, so only the requested span is decoded. Times in the
    returned wave are relative to `start_s`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    start_s = max(0.0, float(start_s or 0.0))
    cmd = [_ffmpeg_bin(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    if dur_s:
        cmd += ["-t", f"{dur_s:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr),
            "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    if not dur_s:
        total = _probe_duration_s(path)
        dur_s = max(0.0, total - start_s) if total else None
    cap = int((dur_s + 1.0) * sr) if dur_s else sr * 600          # grows if the guess is short
    buf = np.empty(cap, dtype=np.float32)
    n = 0
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                if n == buf.size:
                    buf = np.resize(buf, buf.size * 2)                 # rare: only if probe was wrong
                got = proc.stdout.readinto(memoryview(buf[n:]).cast("B"))
                if not got:
                    break
                if got % 4:                                            # partial sample: finish it
                    rest = proc.stdout.read(4 - got % 4)
                    memoryview(buf[n:]).cast("B")[got:got + len(rest)] = rest
                    got += len(rest)
                n += got // 4
        finally:
            proc.stdout.close()
            rc = proc.wait()
        if rc != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg decode failed for {path.name}: "
                               f"{err.read().decode(errors='replace').strip()[:400]}")
    return buf[:n]


def utterance_span_s(uset: "UtteranceSet", speakers: list[str] | None = None,
                     pad_s: float = 1.0) -> tuple[float, float] | None:
    """(start, end) seconds covering the given speakers' utterances (all if None), padded."""
    us = [u for u in uset.utterances if speakers is None or u.speaker in speakers]
    if not us:
        return None
    return (max(0.0, min(u.start_ms for u in us) / 1000 - pad_s),
            max(u.end_ms for u in us) / 1000 + pad_s)


def shift_utterances(uset: "UtteranceSet", offset_ms: int) -> "UtteranceSet":
    """Copy with every utterance moved by offset_ms (negative = earlier)."""
    return replace(uset, utterances=[Utterance(u.speaker, u.start_ms + offset_ms, u.end_ms + offset_ms, u.text)
                                     for u in uset.utterances])


def decode_for(path: str | Path, uset: "UtteranceSet", speakers: list[str] | None = None,
               pad_s: float = 1.0, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Decode only the span the (selected) speakers' utterances cover. Returns (wave, offset_s)."""
    span = utterance_span_s(uset, speakers, pad_s)
    if span is None:
        return np.zeros(0, np.float32), 0.0
    return decode_audio(path, sr, start_s=span[0], dur_s=span[1] - span[0]), span[0]


def load_utterances(path: str | Path) -> UtteranceSet:
    """Read a `<stem>.utterances.json` sidecar (schema 1; see references/BACKENDS.md)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return utterances_from_dict(data, path=str(path))


def utterances_from_dict(data: dict, path: str | None = None) -> UtteranceSet:
    raw = data.get("utterances") or []
    utts = []
    for u in raw:
        start, end = int(u["start"]), int(u.get("end") or u["start"])
        if end <= start:
            continue
        utts.append(Utterance(str(u["speaker"]), start, end, u.get("text") or ""))
    utts.sort(key=lambda u: u.start_ms)
    source = data.get("source") or "unknown"
    approx = bool(data.get("approx")) or source.startswith("gemini")
    return UtteranceSet(utts, approx=approx, source=source,
                        speaker_map=data.get("speaker_map"),
                        transcript_id=data.get("transcript_id"),
                        audio_duration_ms=data.get("audio_duration_ms"), path=path)


def meeting_key_from(*paths: str | Path | None) -> str | None:
    """First `YYYY-MM-DD-HHMM` found in the given file names."""
    for p in paths:
        if p:
            m = MEETING_KEY_RE.search(Path(p).name)
            if m:
                return m.group(1)
    return None


# ─── Interval helpers (seconds) ─────────────────────────────────────────────

def _subtract(span: tuple[float, float], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """span minus the union of cuts; cuts need not be sorted/merged."""
    pieces = [span]
    for c0, c1 in sorted(cuts):
        nxt = []
        for a, b in pieces:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        pieces = nxt
    return [(a, b) for a, b in pieces if b > a]


def speech_regions(uset: UtteranceSet, speaker: str,
                   params: ClipParams = IDENTIFY_PARAMS) -> list[tuple[float, float]]:
    """Candidate single-speaker regions (seconds) for `speaker`, before VAD."""
    others = [(u.start_ms / 1000 - params.overlap_guard_s, u.end_ms / 1000 + params.overlap_guard_s)
              for u in uset.utterances if u.speaker != speaker]
    regions: list[tuple[float, float]] = []
    for u in uset.utterances:
        if u.speaker != speaker:
            continue
        s, e = u.start_ms / 1000, u.end_ms / 1000
        if uset.approx:
            if e - s < params.approx_min_turn_s:
                continue
            mid = (s + e) / 2
            half = params.approx_window_s / 2
            span = (mid - half, mid + half)
        else:
            span = (s + params.trim_s, e - params.trim_s)
        if span[1] - span[0] <= 0:
            continue
        # cheap pre-filter: only cuts that can touch this span
        local = [c for c in others if c[1] > span[0] and c[0] < span[1]]
        regions.extend(_subtract(span, local))
    return [r for r in regions if r[1] - r[0] >= params.min_net_s]


# ─── Energy VAD ─────────────────────────────────────────────────────────────

def _frame_db(x: np.ndarray, frame: int, chunk_frames: int = 8192) -> np.ndarray:
    """Per-frame RMS in dBFS, float32 out. Chunked so no full-length float64 copy
    is ever made (the old version cost ~16 B/sample ≈ 0.9 GB per hour)."""
    n = len(x) // frame
    out = np.empty(n, dtype=np.float32)
    for a in range(0, n, chunk_frames):
        b = min(n, a + chunk_frames)
        f = x[a * frame:b * frame].reshape(b - a, frame)
        ms = np.einsum("ij,ij->i", f, f, dtype=np.float64) / frame     # accumulate in f64, per chunk
        out[a:b] = 10 * np.log10(ms + 1e-12)
    return out


def noise_floor_db(wave: np.ndarray, params: ClipParams = IDENTIFY_PARAMS,
                   sr: int = SAMPLE_RATE) -> float:
    """Meeting-level noise floor: low percentile of frame energy (dBFS).

    Low (3rd) percentile because meetings can be >90 % speech; digital
    silence (< −100 dBFS, e.g. padding) is ignored so it can't drag the floor down.
    """
    db = _frame_db(wave, max(1, int(params.vad_frame_s * sr)))
    db = db[db > -100]
    if db.size == 0:
        return params.vad_abs_floor_db
    return float(np.percentile(db, params.vad_floor_pct))


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """Run-length encode a boolean array → [(start, end, value)]."""
    if mask.size == 0:
        return []
    change = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate(([0], change, [mask.size]))
    return [(int(a), int(b), bool(mask[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def vad_runs(seg: np.ndarray, floor_db: float, params: ClipParams = IDENTIFY_PARAMS,
             sr: int = SAMPLE_RATE) -> list[tuple[int, int]]:
    """Contiguous speech runs (sample ranges); pauses ≤ vad_max_gap_s are bridged."""
    frame = max(1, int(params.vad_frame_s * sr))
    db = _frame_db(seg, frame)
    if db.size == 0:
        return []
    thr = max(floor_db + params.vad_margin_db, params.vad_abs_floor_db)
    mask = db >= thr
    max_gap = int(round(params.vad_max_gap_s / params.vad_frame_s))
    min_sp = int(round(params.vad_min_speech_s / params.vad_frame_s))
    # drop tiny speech blips, then fill short pauses between speech
    for a, b, v in _runs(mask):
        if v and (b - a) < min_sp:
            mask[a:b] = False
    runs = _runs(mask)
    for i, (a, b, v) in enumerate(runs):
        if not v and 0 < i < len(runs) - 1 and (b - a) <= max_gap:
            mask[a:b] = True
    return [(a * frame, b * frame) for a, b, v in _runs(mask) if v]


def vad_keep(seg: np.ndarray, floor_db: float, params: ClipParams = IDENTIFY_PARAMS,
             sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return `seg` with non-speech runs > vad_max_gap_s removed (short pauses kept)."""
    runs = vad_runs(seg, floor_db, params, sr)
    if not runs:
        return seg[:0]
    return np.concatenate([seg[a:b] for a, b in runs])


# ─── Clip selection ─────────────────────────────────────────────────────────

def _windows(n: int, params: ClipParams, sr: int) -> list[tuple[int, int]]:
    cap = clip_cap_s(params)
    w = int(min(params.window_s, cap) * sr)
    if n <= int(min(params.max_clip_s, cap) * sr) or n <= w:
        return [(0, n)]
    h = max(1, int(params.window_hop_s * sr))
    out = [(a, a + w) for a in range(0, n - w + 1, h)]
    if out and out[-1][1] < n - h // 2:        # tail not covered well → add last window
        out.append((n - w, n))
    return out


def select_clips(wave: np.ndarray, uset: UtteranceSet, speaker: str,
                 params: ClipParams = IDENTIFY_PARAMS, floor_db: float | None = None,
                 sr: int = SAMPLE_RATE, keep_wave: bool = True,
                 speech_mask=None, info: dict | None = None) -> list[Clip]:
    """Clean clips for one speaker/cluster, longest first, capped at params.max_clips.

    `speech_mask` (vad.SpeechMask over `wave`): when given, a clip is kept only
    if its Silero speech fraction ≥ params.neural_min_frac (energy VAD is the
    pre-filter). `info` (optional dict) accumulates gate counts.
    """
    if floor_db is None:
        floor_db = noise_floor_db(wave, params, sr)
    clips: list[Clip] = []
    rejected = 0
    total = len(wave)
    for s, e in speech_regions(uset, speaker, params):
        a, b = max(0, int(s * sr)), min(total, int(e * sr))
        if b - a < int(params.min_net_s * sr):
            continue
        seg = wave[a:b]
        runs = vad_runs(seg, floor_db, params, sr)
        if not runs:
            continue
        pieces: list[tuple[np.ndarray, np.ndarray | None]] = []   # (samples, silero mask of samples)
        if params.clip_mode == "longest_run":
            cap = int(clip_cap_s(params) * sr)
            for ra, rb in runs:
                if rb - ra > cap:                             # centre-crop to the cap
                    ra += (rb - ra - cap) // 2
                    rb = ra + cap
                pieces.append((seg[ra:rb], speech_mask.region(a + ra, a + rb) if speech_mask else None))
        elif params.clip_mode == "window":
            speech = np.concatenate([seg[ra:rb] for ra, rb in runs])
            sil = (np.concatenate([speech_mask.region(a + ra, a + rb) for ra, rb in runs])
                   if speech_mask else None)
            pieces = [(speech[wa:wb], sil[wa:wb] if sil is not None else None)
                      for wa, wb in _windows(len(speech), params, sr)]
        else:
            raise ValueError(f"clip_mode must be 'window' or 'longest_run', got {params.clip_mode!r}")
        for piece, sil in pieces:
            net = len(piece) / sr
            if net < params.min_net_s:
                continue
            frac = None
            if sil is not None:
                frac = float(sil.mean()) if sil.size else 0.0
                if frac < params.neural_min_frac:
                    rejected += 1
                    continue
            clips.append(Clip(speaker, int(s * 1000), int(e * 1000), round(net, 3),
                              piece.copy() if keep_wave else None,
                              None if frac is None else round(frac, 3)))
    if info is not None:
        info.setdefault("rejected_non_speech", {})[speaker] = rejected
    clips.sort(key=lambda c: c.net_dur, reverse=True)
    clips = clips[: params.max_clips] if params.max_clips else clips
    return quantize_clips(clips, params, sr)


def quantize_clips(clips: list[Clip], params: ClipParams = IDENTIFY_PARAMS,
                   sr: int = SAMPLE_RATE) -> list[Clip]:
    """Centre-crop each clip to a multiple of params.quant_s; drop those that fall
    below min_net_s. Same rule, order (after sort + max_clips cap) and rounding as
    the offline evaluation, so CLI and evaluation embeddings match."""
    if not params.quant_s:
        return clips
    q = int(round(params.quant_s * sr))
    out = []
    for c in clips:
        length = len(c.wave) if c.wave is not None else int(round(c.net_dur * sr))
        n = (length // q) * q
        if n < int(params.min_net_s * sr):
            continue
        if n < length:
            if c.wave is not None:
                a = (length - n) // 2
                c.wave = c.wave[a:a + n]
            c.net_dur = round(n / sr, 3)
        out.append(c)
    return out


def speech_mask_for(wave: np.ndarray, params: ClipParams = IDENTIFY_PARAMS,
                    info: dict | None = None):
    """Silero mask over the whole decoded wave (once per meeting), or None → energy-only.

    Falls back to energy VAD with a warning if sherpa-onnx / silero_vad_v5.onnx is missing.
    """
    info = info if info is not None else {}
    if params.neural_vad == "energy":
        info["vad"] = "energy"
        return None
    if params.neural_vad != "silero":
        raise ValueError(f"neural_vad must be 'silero' or 'energy', got {params.neural_vad!r}")
    from . import vad as nvad
    if not nvad.available():
        msg = ("Silero VAD unavailable (sherpa-onnx or silero_vad_v5.onnx missing; run "
               "fetch_models.py silero_vad) — falling back to energy VAD, which passes "
               "ringtones/hold music")
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        info.update(vad="energy (fallback)", vad_warning=msg)
        return None
    t = time.time()
    mask = nvad.silero_mask(wave)
    info.update(vad="silero_v5", vad_s=round(time.time() - t, 2),
                vad_audio_s=round(len(wave) / SAMPLE_RATE, 1),
                vad_speech_frac=round(mask.speech_fraction, 3), neural_min_frac=params.neural_min_frac)
    return mask


def select_meeting_clips(wave: np.ndarray, uset: UtteranceSet,
                         params: ClipParams = IDENTIFY_PARAMS,
                         speakers: list[str] | None = None,
                         sr: int = SAMPLE_RATE, speech_mask=None,
                         info: dict | None = None) -> dict[str, list[Clip]]:
    """select_clips for every cluster (or the given subset), sharing one noise floor
    and one Silero pass (computed here unless `speech_mask` is given)."""
    info = info if info is not None else {}
    floor = noise_floor_db(wave, params, sr)
    if speech_mask is None:
        speech_mask = speech_mask_for(wave, params, info)
    return {spk: select_clips(wave, uset, spk, params, floor, sr, speech_mask=speech_mask, info=info)
            for spk in (speakers or uset.speakers)}


def total_speech_s(uset: UtteranceSet, speaker: str) -> float:
    return sum(u.end_ms - u.start_ms for u in uset.utterances if u.speaker == speaker) / 1000


# ─── Embedding (duck-typed Embedder) ────────────────────────────────────────

EMBED_BATCH = 1   # measured (ReDimNet2-B6, CPU, 8 s clips): peak RSS ≈ 1.4 GB at
                  # batch 1 vs 4.5 GB at 4 and 16.5 GB at 16 — and batch 1 was also fastest per clip.


def _embed_one_batch(embedder, waves: list[np.ndarray]) -> np.ndarray:
    if hasattr(embedder, "embed_batch"):
        return np.asarray(embedder.embed_batch(waves), dtype=np.float32)
    return np.stack([np.asarray(embedder.embed(w), dtype=np.float32) for w in waves])


def embedder_device(embedder) -> str:
    """Device an embedder runs on (set as `vp_device` by identify_speakers.load_embedder)."""
    return getattr(embedder, "vp_device", None) or getattr(embedder, "_device", None) or "cpu"


def embed_clips(embedder, clips: list[Clip], batch_size: int = EMBED_BATCH,
                sr: int = SAMPLE_RATE, cap_s: float | None = None) -> np.ndarray:
    """L2-normalised embeddings aligned with `clips`; frees clip waves afterwards.

    Guards (embedder contract): waves longer than `cap_s` (default: the
    embedder's device cap, see DEVICE_CAP_S) are centre-cropped
    (select_clips never produces them; this protects library callers); clips
    that are < MIN_EMBED_S, non-finite, or rejected by the embedder
    (ValueError) are DROPPED — `clips` is filtered in place so rows stay aligned.
    Never pads (ReDimNet2 z-scores per clip; padding changes embeddings).
    """
    dim = int(getattr(embedder, "dim", 0))
    cap = int((cap_s if cap_s else cap_for(embedder_device(embedder))) * sr)
    for c in clips:
        w = np.asarray(c.wave, dtype=np.float32).ravel()
        if w.size > cap:
            a = (w.size - cap) // 2
            w = w[a:a + cap]
            c.net_dur = round(cap / sr, 3)
        c.wave = w
    clips[:] = [c for c in clips if c.wave.size >= MIN_EMBED_S * sr and np.isfinite(c.wave).all()]
    kept, rows = [], []
    for i in range(0, len(clips), batch_size):
        chunk = clips[i:i + batch_size]
        try:
            rows.append(_embed_one_batch(embedder, [c.wave for c in chunk]))
            kept += chunk
        except ValueError:
            for c in chunk:                                   # isolate the bad clip
                try:
                    rows.append(_embed_one_batch(embedder, [c.wave]))
                    kept.append(c)
                except ValueError:
                    pass
    clips[:] = kept
    for c in clips:
        c.wave = None
    if not rows:
        return np.zeros((0, dim), dtype=np.float32)
    embs = np.concatenate(rows, axis=0)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    return embs


def embed_meeting(embedder, audio_path: str | Path, uset: UtteranceSet,
                  params: ClipParams = IDENTIFY_PARAMS,
                  speakers: list[str] | None = None,
                  wave: np.ndarray | None = None,
                  batch_size: int = EMBED_BATCH,
                  offset_s: float = 0.0,
                  info: dict | None = None) -> dict[str, tuple[list[Clip], np.ndarray]]:
    """Decode (unless `wave` given), select clips per cluster, embed.

    Returns {cluster: (clips_meta, embeddings)}. Library entry point for the
    LOMO evaluation — touches no on-disk store. `params.device` is set from the
    embedder, so the clip cap follows the device (CPU 12 s, MPS 60 s) unless
    `params.hard_cap_s` overrides it.

    Without `wave`, only the span covered by the selected speakers' utterances
    is decoded (a 20 h file whose meeting is 50 min decodes ~50 min). With
    `wave`, `offset_s` is the file time of wave[0]. Clip start/end_ms are
    always file times.
    """
    if wave is None:
        wave, offset_s = decode_for(audio_path, uset, speakers)
    off_ms = int(round(offset_s * 1000))
    local = shift_utterances(uset, -off_ms) if off_ms else uset
    params = replace(params, device=embedder_device(embedder))
    cap = clip_cap_s(params)
    per = select_meeting_clips(wave, local, params, speakers, info=info)
    for clips in per.values():
        for c in clips:
            c.start_ms += off_ms
            c.end_ms += off_ms
    return {spk: (clips, embed_clips(embedder, clips, batch_size, cap_s=cap)) for spk, clips in per.items()}
