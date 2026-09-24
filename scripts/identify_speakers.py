#!/usr/bin/env python3
"""
Voiceprint speaker identification (meeting-documenter Step 2.5, advisory only)
-----------------------------------------------------------------------------
Scores each diarized cluster (AssemblyAI Speaker A/B/…) against locally enrolled
voiceprints and proposes a cluster → name mapping with margins and a confidence
flag. It never renames anything; Claude reconciles it with the roster and
content cues, and asks the user when voice and content disagree.

Usage:
    identify_speakers.py identify --audio X.m4a --utterances X.utterances.json
        [--roster "Alice Example,Bob Example"] [--labels "A=Bob Example,B=Alice Example"]
        [--model redimnet2] [--channel phone] [--json out.json|-]
    identify_speakers.py enroll --name "Canonical Name" --audio X --utterances X.json
        --cluster A [--channel phone] [--source confirmed_meeting|user_ear|self_check]
    identify_speakers.py cohort-add --audio X --utterances X.json [--clusters A,C]
    identify_speakers.py calibrate [--model-id ID] [--dry-run] [--json out.json|-]
    identify_speakers.py track --audio X.m4a [--roster "Alice Example,Bob Example"]
        [--backend auto|nemotron3|seg3] [--channel phone] [--out X.timeline.json] [--rttm X.rttm]
    identify_speakers.py track-calibrate [--dry-run] [--json out.json|-]
    identify_speakers.py track-check [--roster "Alice Example,Bob Example"] [--backend nemotron3]
    identify_speakers.py enroll --name "Canonical Name" --audio X --timeline X.timeline.json
        [--speaker GUEST_1] [--spans "12.5-20,01:05-01:30"] [--min-turn-s 3]
    identify_speakers.py list [--json]
    identify_speakers.py forget --name "Canonical Name" --yes
    identify_speakers.py audit sample --input X.m4a X.timeline.json [--input …] --out DIR
        [--n 300] [--seed 0] [--pair "Alice Example|Bob Example"] [--channel phone]
    identify_speakers.py audit score --manifest DIR/manifest.json --answers answers.json

Store: $VOICEPRINT_DIR (default ~/.local/share/meeting-documenter/voiceprints),
never inside the notes vault or this clone. `identify` is a graceful no-op
(exit 0) when the store is empty or the voiceprint dependencies/models are
not installed; it never imports torch on that path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPTS_DIR.parent
KNOWN_SPEAKERS = SKILL_DIR / "references" / "KNOWN_SPEAKERS.yaml"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from voiceprint import config as vp_config  # noqa: E402  (stdlib only)

try:  # numpy/scipy are optional voiceprint deps; torch is only imported inside load_embedder
    import numpy as np  # noqa: E402

    from voiceprint import audio as vp_audio  # noqa: E402
    from voiceprint import scoring as vp_scoring  # noqa: E402
    from voiceprint import store as vp_store  # noqa: E402
    DEPS_ERROR: str | None = None
except ImportError as _e:  # graceful no-op for identify; other commands explain what to install
    np = vp_audio = vp_scoring = vp_store = None  # type: ignore[assignment]
    DEPS_ERROR = f"{type(_e).__name__}: {_e}"

DEFAULT_MODEL = "redimnet2"


# ─── Names ──────────────────────────────────────────────────────────────────

class NameResolver:
    """Canonicalise names via KNOWN_SPEAKERS.yaml aliases (case-insensitive).

    The private registry is references/KNOWN_SPEAKERS.yaml (gitignored), or the file named
    by MEETING_DOCUMENTER_KNOWN_SPEAKERS. Falls back to exact names when PyYAML or the
    registry is unavailable.
    """

    def __init__(self, path: Path | None = None):
        self.map: dict[str, str] = {}
        self.loaded = False
        path = path or Path(os.environ.get("MEETING_DOCUMENTER_KNOWN_SPEAKERS") or KNOWN_SPEAKERS)
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for sp in data.get("speakers") or []:
                canon = sp.get("canonical_name")
                if not canon:
                    continue
                for n in [canon, *(sp.get("aliases") or [])]:
                    self.map.setdefault(str(n).strip().casefold(), canon)
            self.loaded = True
        except Exception:
            pass

    def canonical(self, name: str | None) -> str | None:
        if name is None:
            return None
        return self.map.get(name.strip().casefold(), name.strip())

    def known(self, name: str) -> bool:
        return (not self.loaded) or name.strip().casefold() in self.map

    def split(self, csv: str | None) -> list[str]:
        return [self.canonical(x) for x in (csv or "").split(",") if x.strip()]


def parse_labels(arg: str | None, uset: vp_audio.UtteranceSet, names: NameResolver) -> dict | None:
    """--labels "A=Name,B=Name" wins over the sidecar speaker_map."""
    if arg:
        out = {}
        for part in arg.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = names.canonical(v)
        return out
    if uset.speaker_map:
        return {k: names.canonical(v) for k, v in uset.speaker_map.items() if v}
    return None


# ─── Helpers ────────────────────────────────────────────────────────────────

def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


PARITY_MIN_COS = 0.999
_PARITY: dict[str, dict] = {}     # model → {"ok", "cos", "cpu"}: first-MPS-use self-test, per process


def resolve_device(requested: str, model: str, torch_models) -> str:
    """'auto' → 'mps' for torch-backed models when torch reports MPS available, else 'cpu'.

    Measured on Apple silicon (one hour-long meeting, 40 clips, 4 distinct shapes with
    quant_s=0.5, fresh process incl. parity self-test, busy machine): identify wall
    MPS 36–51 s vs CPU 76–90 s, RSS 1.5–1.7 vs 1.9 GB. MPS wins for a single
    identify, so no workload-size rule: fixed MPS overhead ≈ +1.6 s load/parity
    + ~0.3 s per new shape, break-even ≈ 24 clips even at idle CPU speed (~190 ms/clip).

    Imports torch only here, and only after the model is known to be available.
    """
    if requested != "auto":
        return requested
    if model not in set(torch_models or ()):
        return "cpu"                               # sherpa backends: cpu (coreml only if asked)
    try:
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:
        return "cpu"


def _parity_wave(sec: float = 4.0, sr: int = 16000) -> np.ndarray:
    """Deterministic voice-like test signal (harmonics + noise), float32 in [-1, 1]."""
    rng = np.random.default_rng(20260922)
    t = np.arange(int(sec * sr)) / sr
    f0 = 140 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum((0.3 / k) * np.sin(k * phase) for k in range(1, 8)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2)
    return np.clip(x + 0.01 * rng.normal(size=t.size), -1, 1).astype(np.float32)


def mps_parity(embedders, model: str, emb_mps) -> dict:
    """One-clip MPS-vs-CPU parity check, cached per model for the process lifetime."""
    if model in _PARITY:
        return _PARITY[model]
    res: dict = {"ok": False, "cos": None, "cpu": None}
    try:
        cpu = embedders.get_embedder(model, device="cpu")
        res["cpu"] = cpu
        w = _parity_wave()
        a, b = np.asarray(emb_mps.embed(w), np.float64), np.asarray(cpu.embed(w), np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        res.update(ok=cos >= PARITY_MIN_COS, cos=cos)
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    _PARITY[model] = res
    return res


def load_embedder(model: str, device: str = "auto"):
    """(embedder, None) or (None, reason). Never raises.

    device: auto | cpu | mps. On the first MPS use per process a one-clip CPU
    parity self-test runs; cos < PARITY_MIN_COS (or any error) → fall back to CPU.
    The chosen device is recorded as `embedder.vp_device` (drives the clip cap).
    """
    try:
        from voiceprint import embedders  # lazy: torch/sherpa imported inside get_embedder
    except Exception as e:  # module missing or broken
        return None, f"voiceprint.embedders not importable ({type(e).__name__}: {e})"
    try:
        avail = embedders.available_embedders()
    except Exception as e:
        return None, f"available_embedders() failed ({type(e).__name__}: {e})"
    if model not in avail:
        return None, f"embedder '{model}' unavailable (available: {avail or 'none — torch/sherpa-onnx missing?'})"
    dev = resolve_device(device, model, getattr(embedders, "TORCH_SPECS", {}))
    try:
        emb = embedders.get_embedder(model, device=dev)
    except Exception as e:
        return None, f"get_embedder('{model}', device='{dev}') failed ({type(e).__name__}: {e})"
    if dev == "mps":
        par = mps_parity(embedders, model, emb)
        if not par["ok"]:
            detail = f"cos {par['cos']:.6f}" if par["cos"] is not None else par.get("error", "error")
            print(f"⚠️  MPS parity self-test failed for {model} ({detail} < {PARITY_MIN_COS}); using CPU",
                  file=sys.stderr)
            if par["cpu"] is None:
                try:
                    par["cpu"] = embedders.get_embedder(model, device="cpu")
                except Exception as e:
                    return None, f"CPU fallback failed ({type(e).__name__}: {e})"
            emb, dev = par["cpu"], "cpu"
    try:
        emb.vp_device = dev
    except Exception:
        pass
    return emb, None


# calibration.json clip key → ClipParams field. Accepted both inside a `clip` block and flat
# at the model-entry top level (older calibration files wrote them flat, with field names).
CALIB_CLIP_KEYS = {
    "clip_mode": "clip_mode", "hard_cap_s": "hard_cap_s", "quant_s": "quant_s",
    "min_speech_frac": "neural_min_frac", "neural_min_frac": "neural_min_frac",
    "vad": "neural_vad", "neural_vad": "neural_vad", "window_s": "window_s",
    "window_hop_s": "window_hop_s", "max_clip_s": "max_clip_s", "min_net_s": "min_net_s",
    "max_clips": "max_clips",
}
# Fields that change the per-clip embeddings (compared by the enroll guard). max_clips only
# changes how many clips form an identify cluster embedding, so it applies to identify only.
IDENTIFY_ONLY_FIELDS = {"max_clips"}
CPU_60S_PEAK_GB = 6.7   # measured peak RSS (ReDimNet2, CPU, one 60 s clip)


def _cli_clip_overrides(args) -> dict:
    """Explicit CLI flags only (argparse defaults are None) → ClipParams fields."""
    kw = {}
    if getattr(args, "clip_mode", None):
        kw["clip_mode"] = args.clip_mode
    if getattr(args, "window_s", None):             # sweep convention: hop = w/2, max_clip = w
        kw.update(window_s=args.window_s, window_hop_s=args.window_s / 2, max_clip_s=args.window_s)
    if getattr(args, "max_clip_s", None):           # also lifts the device cap (≤ SAFETY_CEILING_S)
        kw.update(max_clip_s=args.max_clip_s, hard_cap_s=args.max_clip_s)
    if getattr(args, "hard_cap_s", None):
        kw["hard_cap_s"] = args.hard_cap_s
    if getattr(args, "quant_s", None) is not None:
        kw["quant_s"] = args.quant_s
    if getattr(args, "max_clips", None):
        kw["max_clips"] = args.max_clips
    if getattr(args, "min_net", None):
        kw["min_net_s"] = args.min_net
    if getattr(args, "vad", None):
        kw["neural_vad"] = args.vad
    if getattr(args, "min_speech_frac", None) is not None:
        kw["neural_min_frac"] = args.min_speech_frac
    return kw


def calib_clip_block(calib: dict | None, identify: bool = True) -> dict:
    """Calibration clip settings mapped to ClipParams fields: flat top-level keys, then the
    `clip` block over them. Unknown keys ignored; identify-only fields dropped for enroll."""
    calib = calib or {}
    out = {CALIB_CLIP_KEYS[k]: v for k, v in calib.items() if k in CALIB_CLIP_KEYS}
    out.update({CALIB_CLIP_KEYS[k]: v for k, v in (calib.get("clip") or {}).items() if k in CALIB_CLIP_KEYS})
    if not identify:
        out = {k: v for k, v in out.items() if k not in IDENTIFY_ONLY_FIELDS}
    return out


def resolve_clip_config(args, base: vp_audio.ClipParams, calib: dict | None,
                        identify: bool = True) -> tuple[vp_audio.ClipParams, dict]:
    """Precedence: explicit CLI flag > calibration.json `clip` block > built-in `base`.

    Returns (params, source) where source maps each overridden field to
    "calibration" or "cli" (absent = built-in default).
    """
    source: dict[str, str] = {}
    cal = calib_clip_block(calib, identify)
    for k in cal:
        source[k] = "calibration"
    cli_kw = _cli_clip_overrides(args)
    for k in cli_kw:
        source[k] = "cli"
    return replace(base, **{**cal, **cli_kw}), source


def clip_summary(params: vp_audio.ClipParams, device: str, source: dict | None = None) -> dict:
    """Effective clip config (what actually shapes the embeddings) + provenance."""
    eff = replace(params, device=device)
    d = {"clip_mode": params.clip_mode, "effective_cap_s": vp_audio.clip_cap_s(eff),
         "hard_cap_s": params.hard_cap_s, "quant_s": params.quant_s, "vad": params.neural_vad,
         "min_speech_frac": params.neural_min_frac, "min_net_s": params.min_net_s,
         "max_clips": params.max_clips, "device": device}
    if params.clip_mode == "window":
        d.update(window_s=params.window_s, window_hop_s=params.window_hop_s, max_clip_s=params.max_clip_s)
    src = source or {}
    d["source"] = {k: src.get(k, "default") for k in
                   ("clip_mode", "hard_cap_s", "quant_s", "neural_vad", "neural_min_frac",
                    "window_s", "max_clip_s", "min_net_s", "max_clips") if k in src}
    d["from_calibration"] = any(v == "calibration" for v in src.values())
    return d


def clip_signature(summary: dict) -> str:
    """Short, device-independent id of the clip config (effective cap, not the device)."""
    keys = ["clip_mode", "effective_cap_s", "quant_s", "vad", "min_speech_frac", "min_net_s"]
    if summary["clip_mode"] == "window":
        keys += ["window_s", "window_hop_s", "max_clip_s"]
    blob = json.dumps({k: summary.get(k) for k in keys}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def clip_config_mismatch(params: vp_audio.ClipParams, calib: dict | None) -> list[str]:
    """Embedding-relevant fields where `params` differs from the calibration clip settings."""
    out = []
    for field_, v in calib_clip_block(calib, identify=False).items():
        mine = getattr(params, field_)
        if (isinstance(v, (int, float)) and isinstance(mine, (int, float))
                and not isinstance(v, bool) and abs(float(v) - float(mine)) < 1e-9):
            continue
        if mine != v:
            out.append(f"{field_}: {mine!r} ≠ calibration {v!r}")
    return out


def note_cap_choice(params: vp_audio.ClipParams, device: str, source: dict) -> str | None:
    """Log when an explicit/calibrated hard cap exceeds the device default (never silently shorten)."""
    dev_cap = vp_audio.DEVICE_CAP_S.get(device, vp_audio.DEVICE_CAP_S["cpu"])
    eff = vp_audio.clip_cap_s(replace(params, device=device))
    if params.hard_cap_s and eff > dev_cap:
        msg = (f"clip cap {eff:g} s from {source.get('hard_cap_s', 'default')} kept on {device} "
               f"(device default {dev_cap:g} s) so clips match the calibration")
        if device != "mps" and eff >= 30:
            msg += f"; expect up to ~{CPU_60S_PEAK_GB} GB peak RSS for 60 s clips on CPU"
        print(f"ℹ️  {msg}", file=sys.stderr)
        return msg
    return None


def match_existing(cv, enr: dict, cohort, calib: dict | None, channel: str | None = None,
                   use_asnorm: bool = True) -> dict:
    """Does cluster vector `cv` match an already-enrolled person?

    Uses AS-norm when the cohort is large enough, else raw cosine. The bar is the calibrated
    τ_low for that score kind; without calibration it is the (stricter) prior τ, so an
    uncalibrated store refuses fewer genuinely new voices while still catching clear matches.
    """
    co_vecs, co_labels = cohort
    co = vp_scoring.build_cohort(enr, co_vecs if len(co_vecs) else None, co_labels)
    sm = vp_scoring.score_matrix([cv], enr, cohort=co, channel=channel, use_asnorm=use_asnorm)
    row = sm.used[0]
    j = int(np.nanargmax(row))
    th = vp_scoring.load_thresholds(calib, sm.kind)
    bar = th.tau_low if th.calibrated else th.tau
    score = float(row[j])
    return {"best": sm.persons[j], "score": round(score, 3), "kind": sm.kind, "bar": round(bar, 3),
            "calibrated": th.calibrated, "matches": score >= bar}


def use_asnorm_for(args, calib: dict | None) -> bool:
    """AS-norm unless disabled, or the calibration has cosine thresholds but no AS-norm ones
    (a small store calibrates only raw cosine; scoring AS-norm then would be uncalibrated)."""
    if getattr(args, "no_asnorm", False):
        return False
    return not (calib and isinstance(calib.get("cosine"), dict) and not isinstance(calib.get("asnorm"), dict))


def resolve_score_mode(args, calib: dict | None) -> tuple[str, str]:
    if getattr(args, "score_mode", None):
        return args.score_mode, "cli"
    m = (calib or {}).get("score_mode")
    if m in ("centroid", "top5"):
        return m, "calibration"
    if m:
        print(f"⚠️  calibration score_mode {m!r} not supported by the CLI; using centroid", file=sys.stderr)
    return "centroid", "default"


def _vad_note(info: dict) -> str:
    if info.get("vad") == "silero_v5":
        rej = sum((info.get("rejected_non_speech") or {}).values())
        return (f"VAD silero_v5 ({info['vad_s']} s for {info['vad_audio_s']:.0f} s audio; "
                f"{rej} non-speech clip(s) rejected at < {info['neural_min_frac']})")
    return f"VAD {info.get('vad', '?')}"


def emit(result: dict, json_arg: str | None) -> None:
    if not json_arg:
        return
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if json_arg == "-":
        print(text)
    else:
        Path(json_arg).write_text(text + "\n", encoding="utf-8")
        print(f"JSON → {json_arg}", file=sys.stderr)


def noop(status: str, reason: str, json_arg: str | None, **extra) -> int:
    print(f"ℹ️  voiceprint identify skipped ({status}): {reason}", file=sys.stderr)
    emit({"status": status, "reason": reason, **extra}, json_arg)
    return 0


# ─── identify ───────────────────────────────────────────────────────────────

def cmd_identify(args) -> int:
    try:
        return _identify(args)
    except Exception as e:  # advisory step: never break the transcribe flow
        traceback.print_exc(file=sys.stderr)
        return noop("error", f"{type(e).__name__}: {e}", args.json)


def _identify(args) -> int:
    t0 = time.time()
    names = NameResolver()
    store = vp_store.VoiceprintStore.open(args.store, create=False)
    if store.count() == 0:
        return noop("no_voiceprints", f"store empty at {store.root}", args.json)

    embedder, why = load_embedder(args.model, args.device)
    if embedder is None:
        return noop("no_embedder", why, args.json)
    model_id = embedder.model_id
    device = vp_audio.embedder_device(embedder)
    calib = store.load_calibration(model_id, getattr(embedder, "name", args.model))

    uset = vp_audio.load_utterances(args.utterances)
    mk = args.meeting_key or vp_audio.meeting_key_from(args.utterances, args.audio)
    exclude = [] if args.include_self or not mk else [mk]
    exemplars = store.exemplars(model_id, exclude_meetings=exclude)
    if not exemplars:
        others = [m for m in store.model_ids() if m != model_id]
        return noop("no_voiceprints", f"no exemplars for model_id {model_id}"
                    + (f" (store has: {others})" if others else ""), args.json)

    params, source = resolve_clip_config(args, vp_audio.IDENTIFY_PARAMS, calib)
    cap_note = note_cap_choice(params, device, source)
    clip_cfg = clip_summary(params, device, source)
    clip_cfg["signature"] = sig = clip_signature(clip_cfg)
    score_mode, score_mode_src = resolve_score_mode(args, calib)
    vad_info: dict = {}
    per = vp_audio.embed_meeting(embedder, args.audio, uset, params, info=vad_info)
    cluster_vecs = [vp_scoring.cluster_embedding(spk, embs, [c.net_dur for c in clips],
                                                 min_net_s=args.min_cluster_s)
                    for spk, (clips, embs) in per.items()]

    enrollment = vp_scoring.build_enrollment(exemplars)
    co_vecs, co_labels = store.cohort(model_id, exclude_meetings=exclude)
    cohort = vp_scoring.build_cohort(enrollment, co_vecs if len(co_vecs) else None, co_labels)
    roster = names.split(args.roster) or None
    labels = parse_labels(args.labels, uset, names)

    result = vp_scoring.identify(cluster_vecs, enrollment, calibration=calib, roster=roster,
                                 labels=labels, cohort=cohort, channel=args.channel,
                                 mode=score_mode, top_n=args.top_n, min_cohort=args.min_cohort,
                                 use_asnorm=use_asnorm_for(args, calib))
    result.update({
        "model": args.model, "model_id": model_id, "device": vp_audio.embedder_device(embedder),
        "meeting_key": mk,
        "excluded_meetings": exclude, "audio": str(args.audio), "utterances": str(args.utterances),
        "utterance_source": uset.source, "approx_timestamps": uset.approx, "roster": roster,
        "labels_in": labels, "enrolled": {p: m.info() for p, m in enrollment.items()},
        "elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": peak_rss_mb(),
        "vad": vad_info, "clip_config": clip_cfg, "score_mode_source": score_mode_src,
        "calibration_meta": {k: calib.get(k) for k in ("params_hash", "cohort", "cohort_type", "calibrated_on")
                             if k in calib} if calib else None,
    })
    if vad_info.get("vad_warning"):
        result["warnings"].append(vad_info["vad_warning"])
    if cap_note:
        result["warnings"].append(cap_note)
    mism = clip_config_mismatch(params, calib)
    if mism:
        result["warnings"].append("clip config differs from calibration (thresholds may not apply): "
                                  + "; ".join(mism))
    other = sum(1 for e in exemplars if e.clip_sig != sig)
    if other:
        result["warnings"].append(f"{other}/{len(exemplars)} enrolled exemplars were made with a different "
                                  f"(or unrecorded) clip config than this run ({sig}); re-enroll for consistency")
    if uset.approx:
        result["warnings"].append("gemini-approx timestamps: clips cut from turn middles; lower reliability")
    print(format_report(result))
    emit(result, args.json)
    return 0


# ─── track (known-speaker tracking: who speaks when) ───────────────────────

def cmd_track(args) -> int:
    try:
        return _track(args)
    except Exception as e:  # advisory step: never break the transcribe flow
        traceback.print_exc(file=sys.stderr)
        return noop("error", f"{type(e).__name__}: {e}", args.json)


def _track(args) -> int:
    from voiceprint import timeline as vp_timeline
    from voiceprint import track as vp_track

    names = NameResolver()
    store = vp_store.VoiceprintStore.open(args.store, create=False)
    if store.count() == 0:
        return noop("no_voiceprints", f"store empty at {store.root}", args.json)
    embedder, why = load_embedder(args.model, args.device)
    if embedder is None:
        return noop("no_embedder", why, args.json)
    if not store.exemplars(embedder.model_id):
        return noop("no_voiceprints", f"no exemplars for model_id {embedder.model_id}", args.json)
    mk = args.meeting_key or vp_audio.meeting_key_from(args.audio)
    exclude = [] if args.include_self or not mk else [mk]
    roster = names.split(args.roster) or None
    params = vp_track.TrackParams()
    over = {k: v for k, v in (("theta", args.theta), ("switch", args.switch),
                              ("p_uncertain", args.p_uncertain)) if v is not None}
    if over:
        params = replace(params, **over)
    info: dict = {}
    tl = vp_track.track(args.audio, roster, store, embedder, backend=args.backend, params=params,
                        channel=args.channel, exclude_meetings=exclude, info=info)
    if args.channel:
        tl.meta.setdefault("channel", args.channel)             # the audit's channel stratum
    if args.out:
        Path(args.out).write_text(vp_timeline.to_json(tl) + "\n", encoding="utf-8")
    if args.rttm:
        uri = Path(args.audio).stem.replace(" ", "_")
        Path(args.rttm).write_text(vp_timeline.to_rttm(tl, uri), encoding="utf-8")
    speech: dict[str, float] = {}
    for t in tl.exclusive:
        speech[t.speaker] = speech.get(t.speaker, 0.0) + t.dur
    result = {"status": "ok", "audio": str(args.audio), "meeting_key": mk, "excluded_meetings": exclude,
              "backend": tl.meta.get("backend"), "roster": tl.meta.get("roster"),
              "calibration": tl.meta.get("calibration", {}).get("source"),
              "speakers_s": {k: round(v, 1) for k, v in sorted(speech.items(), key=lambda kv: -kv[1])},
              "turns": len(tl.turns), "info": info, "out": args.out, "rttm": args.rttm}
    print(f"track: {len(tl.turns)} turns, backend {result['backend']}, "
          f"{info.get('xrt', '?')}x real time, peak footprint {info.get('peak_footprint_gb')} GB", file=sys.stderr)
    for k, v in result["speakers_s"].items():
        print(f"  {k:<32} {v:8.1f} s", file=sys.stderr)
    emit(result, args.json)
    return 0


def track_readiness(args) -> dict:
    """Can `track` name speakers on this machine? Cheap: no torch, no audio, no model load.

    Ready needs the embedder and boundary model installed, enrolled voiceprints for that
    embedder, the store's own track calibration (`track-calibrate`; the built-in public-data
    calibration is not trusted for naming people), and, given a roster, at least one enrolled
    roster member. transcribe_pipeline.py falls back to cloud diarization when not ready.
    """
    from voiceprint import embedders
    from voiceprint import segment as vp_segment
    from voiceprint import track as vp_track

    def no(status: str, reason: str, **extra) -> dict:
        return {"ready": False, "status": status, "reason": reason, **extra}

    if args.model not in embedders.available_embedders():
        return no("no_embedder", f"embedder '{args.model}' or its dependencies are not installed")
    backend = vp_track.resolve_backend(args.backend)
    if not vp_segment.available(backend):
        return no("no_boundary", f"boundary model '{backend}' is not installed")
    store = vp_store.VoiceprintStore.open(args.store, create=False)
    model_id = embedders.model_id_for(args.model)
    enrolled = {s["person"] for s in store.summary() if s["model_id"] == model_id}
    if not enrolled:
        return no("no_voiceprints", f"no voiceprints for {args.model} in {store.root}")
    track_cal = (store.load_calibration(model_id, args.model) or {}).get("track")
    if not track_cal:
        return no("no_calibration", "the store has no track calibration; run identify-speakers.sh "
                                    "track-calibrate after enrolling a few meetings")
    names = NameResolver()
    roster = names.split(args.roster)
    present = [n for n in roster if n in enrolled]
    if roster and not present:
        return no("roster_not_enrolled", "nobody on the roster is enrolled", roster_missing=roster)
    return {"ready": True, "status": "ready", "reason": "", "backend": backend, "model_id": model_id,
            "calibration": track_cal.get("source"), "enrolled_people": len(enrolled),
            "roster_enrolled": present, "roster_missing": [n for n in roster if n not in enrolled]}


def cmd_track_check(args) -> int:
    """Always exit 0 and print one JSON object on stdout (the pipeline parses it)."""
    try:
        res = track_readiness(args)
    except vp_config.VaultPathError as e:
        res = {"ready": False, "status": "store_refused", "reason": str(e)}
    except Exception as e:  # noqa: BLE001  (a readiness probe never fails the caller)
        res = {"ready": False, "status": "error", "reason": f"{type(e).__name__}: {e}"}
    print(json.dumps(res, ensure_ascii=False))
    return 0


def cmd_track_calibrate(args) -> int:
    from voiceprint import calibrate_track as vp_ct
    from voiceprint import track as vp_track

    store = vp_store.VoiceprintStore.open(args.store, create=False)
    embedder, why = load_embedder(args.model, "cpu")
    if embedder is None:
        print(f"❌ {why}", file=sys.stderr)
        return 1
    cal, rep = vp_ct.calibrate_store(store, embedder.model_id, vp_track.DEFAULT_CALIBRATION, args.top_n)
    result = {"model_id": embedder.model_id, "calibration": cal.to_dict(), "report": rep,
              "written": not args.dry_run}
    if not args.dry_run:
        entry = store.load_calibration(embedder.model_id) or {}
        entry["track"] = cal.to_dict()
        path = store.save_calibration(embedder.model_id, entry)
        print(f"track calibration → {path}", file=sys.stderr)
    r0, r1 = rep["base"]["all"], rep["refit"]["all"]
    print(f"track-calibrate: {r1['n_target']} target / {r1['n_nontarget']} non-target LOMO trials; "
          f"Cllr {r0['cllr']:.3f} → {r1['cllr']:.3f} (min {r1['min_cllr']:.3f}), EER {r1['eer']:.3f}",
          file=sys.stderr)
    emit(result, args.json)
    return 0


def _fmt(x, w=7):
    return f"{'—':>{w}}" if x is None else f"{x:>{w}.2f}"


def format_report(r: dict) -> str:
    lines = []
    kind = r["score_kind"]
    th = r["thresholds"]
    cal = "calibrated" if r["calibrated"] else "UNCALIBRATED"
    lines.append(f"Voiceprint identify — {r.get('meeting_key') or ''} · model {r.get('model')} · "
                 f"scores: {kind} ({cal}; τ={th['tau']:.2f} τ_low={th['tau_low']:.2f} δ={th['delta']:.2f})")
    persons = r["persons"]
    w = max([8, *[len(p) for p in persons]]) + 1
    info = {c["cluster"]: c for c in r["cluster_info"]}
    for title, mat in (("raw cosine", r["raw_cosine"]), ("AS-norm", r["asnorm"])):
        if mat is None:
            continue
        lines.append(f"\n{title}:")
        lines.append(f"{'cluster':<10}{'net s':>7}  " + "".join(f"{p[:w-1]:>{w}}" for p in persons))
        for c, row in zip(r["clusters"], mat):
            ci = info[c]
            lines.append(f"{c:<10}{ci['net_dur']:>7.1f}  " + "".join(_fmt(x, w) for x in row)
                         + ("  (insufficient)" if ci["insufficient"] else ""))
    lines.append("\nProposed mapping:")
    for d in r["decisions"]:
        name = d["proposed"] or f"Speaker {d['cluster']} (unknown)"
        extra = f" [tentative: {d['tentative']}]" if d.get("tentative") else ""
        gap = f" gap {d['gap']:.2f} vs {d['second_name']}" if d.get("gap") is not None else ""
        lab = ""
        if d.get("existing_label"):
            lab = f" | label: {d['existing_label']} ({'agrees' if d['label_agrees'] else 'DISAGREES' if d['label_agrees'] is False else 'n/a'})"
        lines.append(f"  {d['cluster']:<4} → {name}  [{d['confidence']}]{extra}{gap}{lab}")
        for reason in d["reasons"]:
            lines.append(f"         · {reason}")
    if r.get("two_speaker_margin") is not None:
        lines.append(f"Two-speaker permutation margin Δ = {r['two_speaker_margin']:.2f} (δ = {th['delta']:.2f})")
    lc = r["label_check"]
    if lc["status"] != "no_labels":
        lines.append(f"Label check: {lc['status'].upper()}"
                     + (f" — inverted pairs {lc['inverted_pairs']}" if lc.get("inverted_pairs") else ""))
        for p in lc.get("pairs", []):
            lines.append(f"  labels {p['clusters']}={p['labels']}: label margin {p['label_margin']:+.2f}"
                         " (negative ⇒ swapped labels fit the voices better)")
    if r.get("roster_not_enrolled"):
        lines.append(f"Roster not enrolled (cannot be voice-matched): {', '.join(r['roster_not_enrolled'])}")
    for wmsg in r["warnings"]:
        lines.append(f"⚠️  {wmsg}")
    cc = r.get("clip_config")
    if cc:
        extra = (f" window {cc['window_s']:g}/{cc['window_hop_s']:g} s" if cc["clip_mode"] == "window" else "")
        lines.append(f"Clip config: {cc['clip_mode']}{extra} · cap {cc['effective_cap_s']:g} s · quant "
                     f"{cc['quant_s']:g} s · vad {cc['vad']}≥{cc['min_speech_frac']:g} · score "
                     f"{r.get('score_mode')} [{'from calibration' if cc['from_calibration'] else 'built-in defaults'}"
                     f"{'; cli overrides: ' + ','.join(k for k, v in cc['source'].items() if v == 'cli') if 'cli' in cc['source'].values() else ''}]"
                     f" sig {cc.get('signature')}")
    if r.get("vad"):
        lines.append(_vad_note(r["vad"]))
    lines.append(f"(cohort {r['cohort_size']} clips · {r.get('elapsed_s')} s · peak RSS {r.get('peak_rss_mb')} MB)")
    return "\n".join(lines)


# ─── enroll ─────────────────────────────────────────────────────────────────

TIMELINE_MIN_TURN_S = 3.0   # enroll only timeline turns at least this long (short turns carry edge errors)


def _seconds(ts: str) -> float:
    parts = [float(x) for x in ts.strip().split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def parse_spans(arg: str | None) -> list[tuple[float, float]]:
    """"12.5-20,01:05-01:30" → [(12.5, 20.0), (65.0, 90.0)]: seconds, MM:SS or H:MM:SS."""
    spans = []
    for part in (arg or "").split(","):
        if part.strip():
            a, _, b = part.partition("-")
            spans.append((_seconds(a), _seconds(b)))
            if not spans[-1][1] > spans[-1][0]:
                raise ValueError(f"span {part.strip()!r} does not end after it starts")
    return spans


def timeline_enroll_source(args, names: NameResolver, name: str):
    """(UtteranceSet, label), or (None, reason): the confirmed Timeline turns of `--speaker`
    (default: the name itself) that last at least `--min-turn-s`, clipped to `--spans` when
    given, as one cluster; every other label's turns are crosstalk that clip selection cuts
    out (± the overlap guard). A named label must be `name` unless --force (like the sidecar
    speaker_map check); UNCERTAIN turns need --spans the user confirmed by ear; UNKNOWN and
    OVERLAP are never enrolled."""
    from voiceprint import timeline as vp_timeline

    tl = vp_timeline.from_json(Path(args.timeline).read_text(encoding="utf-8"))
    label = args.speaker or name
    kind, _ = vp_timeline.parse_label(label)
    try:
        spans = parse_spans(args.spans)
    except ValueError as e:
        return None, f"--spans: {e}"
    if kind in ("unknown", "overlap"):
        return None, f"{label} turns cannot be enrolled; confirm the speaker by ear and pass --spans"
    if kind == "uncertain" and not spans:
        return None, f"{label} turns need --spans that the user confirmed by ear"
    if kind == "name" and names.canonical(label) != name and not args.force:
        return None, f"the timeline labels these turns {label}, not {name}. Resolve the label (or --force)."
    turns = [t for t in tl.turns if t.speaker == label]
    if not turns:
        have = sorted({t.speaker for t in tl.turns})
        return None, f"no turns labelled {label!r} in the timeline (have {have})"
    pieces = []
    for t in turns:
        for a, b in (spans or [(t.start, t.end)]):
            a, b = max(a, t.start), min(b, t.end)
            if b - a >= args.min_turn_s:
                pieces.append((a, b))
    if not pieces:
        return None, (f"no {label} turn of at least {args.min_turn_s:g} s"
                      + (" inside --spans" if spans else "") + " to enroll")
    utts = [vp_audio.Utterance(label, int(a * 1000), int(b * 1000)) for a, b in pieces]
    utts += [vp_audio.Utterance(t.speaker, int(t.start * 1000), int(t.end * 1000))
             for t in (*tl.turns, *tl.exclusive) if t.speaker != label]
    utts.sort(key=lambda u: u.start_ms)
    total = sum(b - a for a, b in pieces)
    print(f"Timeline: {len(pieces)} {label} turn(s) ≥ {args.min_turn_s:g} s, {total:.1f} s before VAD",
          file=sys.stderr)
    return vp_audio.UtteranceSet(utts, source="timeline", path=str(args.timeline)), label


def cmd_enroll(args) -> int:
    if (args.timeline is None) == (args.utterances is None):
        print("❌ pass exactly one of --utterances (with --cluster) or --timeline", file=sys.stderr)
        return 2
    names = NameResolver()
    name = names.canonical(args.name)
    if not names.known(name) and not args.allow_unregistered:
        print(f"❌ '{args.name}' is not in KNOWN_SPEAKERS.yaml — add it first or pass --allow-unregistered",
              file=sys.stderr)
        return 2
    if args.timeline:
        uset, cluster = timeline_enroll_source(args, names, name)
        if uset is None:
            print(f"❌ {cluster}", file=sys.stderr)
            return 2
    else:
        cluster = args.cluster
        if not cluster:
            print("❌ --cluster is required with --utterances", file=sys.stderr)
            return 2
        uset = vp_audio.load_utterances(args.utterances)
        if cluster not in uset.speakers:
            print(f"❌ cluster {cluster!r} not in utterances (have {uset.speakers})", file=sys.stderr)
            return 2
        labels = parse_labels(None, uset, names) or {}
        if labels.get(cluster) and labels[cluster] != name and not args.force:
            print(f"❌ sidecar speaker_map says {cluster}={labels[cluster]}, not {name}. "
                  "Resolve the label (or --force).", file=sys.stderr)
            return 2
    mk = args.meeting_key or vp_audio.meeting_key_from(args.timeline or args.utterances, args.audio)
    if not mk:
        print("❌ cannot derive meeting key from file names; pass --meeting-key YYYY-MM-DD-HHMM", file=sys.stderr)
        return 2

    embedder, why = load_embedder(args.model, args.device)
    if embedder is None:
        print(f"❌ {why}", file=sys.stderr)
        return 1
    store = vp_store.VoiceprintStore.open(args.store)
    device = vp_audio.embedder_device(embedder)
    calib = store.load_calibration(embedder.model_id, getattr(embedder, "name", args.model))
    params, source = resolve_clip_config(args, vp_audio.ENROLL_PARAMS, calib, identify=False)
    mism = clip_config_mismatch(params, calib)
    if mism:
        msg = ("clip config differs from calibration.json — mixed configs break the calibrated "
               "thresholds: " + "; ".join(mism))
        if not args.force:
            print(f"❌ {msg}. Drop the override flags, or --force.", file=sys.stderr)
            return 4
        print(f"⚠️⚠️  --force: {msg}", file=sys.stderr)
    note_cap_choice(params, device, source)
    clip_cfg = clip_summary(params, device, source)
    sig = clip_signature(clip_cfg)
    print(f"Clip config: {json.dumps({k: v for k, v in clip_cfg.items() if k != 'source'})} sig {sig}")
    vad_info: dict = {}
    per = vp_audio.embed_meeting(embedder, args.audio, uset, params, speakers=[cluster], info=vad_info)
    print(_vad_note(vad_info))
    clips, embs = per[cluster]
    net = sum(c.net_dur for c in clips)
    if not clips:
        print(f"❌ no clean clips for cluster {cluster} (turn lengths / overlap / VAD)", file=sys.stderr)
        return 1

    # Self-check against what's already enrolled (other meetings only)
    prior = store.exemplars(embedder.model_id, exclude_meetings=[mk])
    check = {}
    if prior:
        enr = vp_scoring.build_enrollment(prior)
        cv = vp_scoring.cluster_embedding(cluster, embs, [c.net_dur for c in clips], min_net_s=0)
        sims = {p: round(float(m.reference(args.channel) @ cv.vec), 3) for p, m in enr.items()}
        best = max(sims, key=sims.get)
        check = {"cosine_vs_enrolled": sims, "best": best}
        print(f"Self-check (raw cosine vs enrolled): {sims}")
        if name in enr and best != name:
            msg = (f"new clips are closer to {best} ({sims[best]}) than to {name} ({sims[name]}): "
                   "possible inverted label")
            if not args.force:
                print(f"❌ {msg}. Verify by ear, then --force.", file=sys.stderr)
                return 3
            print(f"⚠️⚠️  --force: {msg}", file=sys.stderr)
        if name not in enr:
            co_vecs, co_labels = store.cohort(embedder.model_id, exclude_meetings=[mk])
            m = match_existing(cv, enr, (co_vecs, co_labels), calib, args.channel,
                               use_asnorm=use_asnorm_for(args, calib))
            check["first_enroll_match"] = m
            if m["matches"]:
                msg = (f"'{name}' is not enrolled yet, but these clips match enrolled {m['best']} "
                       f"({m['kind']} {m['score']:.2f} ≥ {m['bar']:.2f}): possible mislabelled cluster — "
                       f"storing it would file {m['best']}'s voice under {name}")
                if not args.force:
                    print(f"❌ {msg}. Verify by ear, then --force.", file=sys.stderr)
                    return 3
                print(f"⚠️⚠️  --force: {msg}", file=sys.stderr)

    rows = [vp_store.Exemplar(person=name, embedding=e, channel=args.channel, language=args.language,
                              meeting_key=mk, cluster=cluster, start_ms=c.start_ms, end_ms=c.end_ms,
                              net_dur=c.net_dur, source=args.source, clip_sig=sig)
            for c, e in zip(clips, embs)]
    if args.dry_run:
        print(f"(dry run) would enroll {len(rows)} clips / {net:.1f} s for {name} from {mk}")
        return 0
    res = store.add_exemplars(embedder.model_id, rows, cap=args.cap)
    print(f"✅ enrolled {name}: {res['added']} clips, {net:.1f} s net from {mk} cluster {cluster} "
          f"[{args.channel}, {args.source}] model {embedder.model_id}"
          + (f" (replaced {res['replaced']})" if res["replaced"] else "")
          + (f" (trimmed to cap: {res['trimmed']})" if res["trimmed"] else ""))
    if net < 20:
        print(f"⚠️  only {net:.1f} s net speech from this meeting; research target ≥60 s across ≥3 meetings")
    emit({"status": "enrolled", "person": name, "meeting_key": mk, "clips": len(rows),
          "net_s": round(net, 1), "model_id": embedder.model_id, "self_check": check,
          "clip_config": {**clip_cfg, "signature": sig}, **res}, args.json)
    return 0


# ─── cohort-add ─────────────────────────────────────────────────────────────

def cmd_cohort_add(args) -> int:
    names = NameResolver()
    uset = vp_audio.load_utterances(args.utterances)
    mk = args.meeting_key or vp_audio.meeting_key_from(args.utterances, args.audio)
    labels = parse_labels(args.labels, uset, names) or {}
    clusters = [c.strip() for c in args.clusters.split(",")] if args.clusters else uset.speakers
    embedder, why = load_embedder(args.model, args.device)
    if embedder is None:
        print(f"❌ {why}", file=sys.stderr)
        return 1
    store = vp_store.VoiceprintStore.open(args.store)
    device = vp_audio.embedder_device(embedder)
    calib = store.load_calibration(embedder.model_id, getattr(embedder, "name", args.model))
    params, source = resolve_clip_config(args, replace(vp_audio.ENROLL_PARAMS, max_clips=20), calib,
                                         identify=False)
    mism = clip_config_mismatch(params, calib)
    if mism:
        print(f"⚠️  cohort clip config differs from calibration: {'; '.join(mism)}", file=sys.stderr)
    note_cap_choice(params, device, source)
    sig = clip_signature(clip_summary(params, device, source))
    per = vp_audio.embed_meeting(embedder, args.audio, uset, params, speakers=clusters, info={})
    rows = []
    for spk, (clips, embs) in per.items():
        person = labels.get(spk)
        label = person or f"{mk}:{spk}"
        rows += [vp_store.CohortClip(label=label, embedding=e, person=person, meeting_key=mk,
                                     start_ms=c.start_ms, end_ms=c.end_ms, net_dur=c.net_dur, clip_sig=sig)
                 for c, e in zip(clips, embs)]
    n = store.add_cohort(embedder.model_id, rows)
    print(f"✅ cohort +{n} clips from {mk} ({', '.join(clusters)}) model {embedder.model_id}")
    return 0


# ─── list / forget ──────────────────────────────────────────────────────────

def cmd_list(args) -> int:
    store = vp_store.VoiceprintStore.open(args.store, create=False)
    summ, coh = store.summary(), store.cohort_summary()
    calib = store.calibration_path()
    if args.json:
        emit({"store": str(store.root), "people": summ, "cohort": coh,
              "calibration": str(calib) if calib.exists() else None}, args.json)
        return 0
    print(f"Voiceprint store: {store.root}{'' if store.exists else ' (not created yet)'}")
    if not summ:
        print("  (no enrolled people)")
    for s in summ:
        print(f"  {s['person']:<22} {s['samples']:>4} clips  {s['net_speech_s']:>7.1f} s  "
              f"{s['meetings']:>2} mtgs  channels={','.join(s['channels'])}  "
              f"sources={','.join(s['sources'])}  {s['first_enrolled']}→{s['last_enrolled']}  [{s['model_id']}]")
    for c in coh:
        print(f"  cohort: {c['clips']} clips / {c['speakers']} labels  [{c['model_id']}]")
    print(f"  calibration: {calib if calib.exists() else 'none (uncalibrated: confidence capped at low; run calibrate)'}")
    return 0


def cmd_forget(args) -> int:
    names = NameResolver()
    name = names.canonical(args.name)
    store = vp_store.VoiceprintStore.open(args.store, create=False)
    have = [s for s in store.summary() if s["person"] == name]
    if not args.yes:
        n = sum(s["samples"] for s in have)
        print(f"Would delete {n} exemplar rows (+ cohort rows) for {name}. Re-run with --yes.")
        return 0
    res = store.forget(name)
    print(f"🗑️  forgot {name}: {res}")
    return 0


# ─── calibrate ──────────────────────────────────────────────────────────────

CALIBRATION_METHOD = ("leave-one-meeting-out over the local store: tau = p99 and tau_low = p95 of "
                      "stranger scores (unenrolled / leave-one-speaker-out / cohort voices), delta = p05 "
                      "of true pair margins, gap = max(0, p05 of target minus best other)")


def calibration_clip_block(enroll: "vp_audio.ClipParams", device: str, identify_max_clips: int) -> dict:
    """Clip settings as calibration.json records them: the hard cap is explicit, so identify
    and enroll reproduce the same clips (and clip signature) on any device."""
    d = {"clip_mode": enroll.clip_mode, "hard_cap_s": vp_audio.clip_cap_s(replace(enroll, device=device)),
         "quant_s": enroll.quant_s, "vad": enroll.neural_vad, "min_speech_frac": enroll.neural_min_frac,
         "min_net_s": enroll.min_net_s, "max_clips": identify_max_clips}
    if enroll.clip_mode == "window":
        d.update(window_s=enroll.window_s, window_hop_s=enroll.window_hop_s, max_clip_s=enroll.max_clip_s)
    return d


def match_enrolled_clip_config(args, calib: dict | None, sigs) -> tuple:
    """(enroll params, device, signature, problems) for the config the exemplars were enrolled with."""
    params, source = resolve_clip_config(args, vp_audio.ENROLL_PARAMS, calib, identify=False)
    known = {s: n for s, n in sigs.items() if s}
    problems = []
    if len(known) > 1:
        problems.append(f"exemplars use {len(known)} different clip configs; re-enroll so they match")
    if sigs.get(None):
        problems.append(f"{sigs[None]} exemplar(s) have no recorded clip config")
    target = max(known, key=known.get) if known else None
    devices = [args.device] if args.device in ("cpu", "mps") else ["cpu", "mps"]
    for dev in devices:
        sig = clip_signature(clip_summary(params, dev, source))
        if target is None or sig == target:
            return params, dev, sig, problems
    problems.append("the current clip flags/calibration do not reproduce the enrolled clip config "
                    f"({target}); pass the flags used at enrollment")
    return params, devices[0], target, problems


def _fmt_nested(n: dict) -> str:
    keys = ("enrolled_units", "enrolled_correct_high", "enrolled_correct_low", "enrolled_unknown",
            "enrolled_wrong_high", "enrolled_wrong_low", "stranger_units", "stranger_unknown",
            "stranger_wrong_high", "stranger_wrong_low")
    return ", ".join(f"{k} {n.get(k, 0)}" for k in keys)


def cmd_calibrate(args) -> int:
    from collections import Counter
    from datetime import date

    from voiceprint import calibrate as vp_calibrate

    store = vp_store.VoiceprintStore.open(args.store, create=False)
    models = [args.model_id] if args.model_id else [m for m in store.model_ids() if store.count(m)]
    if not models:
        print(f"❌ no enrolled voiceprints in {store.root}; enroll confirmed speakers first", file=sys.stderr)
        return 5
    results, rc = {}, 0
    for model_id in models:
        exemplars = store.exemplars(model_id)
        calib = store.load_calibration(model_id)
        sigs = Counter(e.clip_sig for e in exemplars)
        params, device, sig, problems = match_enrolled_clip_config(args, calib, sigs)
        for msg in problems:
            print(f"⚠️  {model_id}: {msg}", file=sys.stderr)
        blocking = [m for m in problems if "different clip configs" in m or "do not reproduce" in m]
        if blocking and not args.force:
            print(f"❌ {model_id}: clip config problem (see above); fix it or pass --force", file=sys.stderr)
            rc = rc or 4
            continue
        ident, _ = resolve_clip_config(args, vp_audio.IDENTIFY_PARAMS, calib)
        score_mode, _ = resolve_score_mode(args, calib)
        res = vp_calibrate.calibrate(exemplars, store.cohort_clips(model_id), mode=score_mode,
                                     test_clips=ident.max_clips, min_net_s=args.min_cluster_s,
                                     top_n=args.top_n, min_cohort=args.min_cohort)
        results[model_id] = res
        print(f"Calibration — {model_id}: {res['folds']} meetings, {res['people']} people "
              f"({res['people_multi_meeting']} with ≥2 meetings); score mode {score_mode}")
        entry = {"clip": calibration_clip_block(params, device, ident.max_clips), "clip_sig": sig,
                 "score_mode": score_mode, "cohort": "clip", "calibrated_on": date.today().isoformat(),
                 "method": CALIBRATION_METHOD, "diagnostics": {}}
        for kind, k in res["kinds"].items():
            if not k["calibrated"]:
                print(f"  {kind}: not calibrated ({k.get('reason')})")
                continue
            entry[kind] = k["thresholds"]
            entry["diagnostics"][kind] = {x: k[x] for x in ("counts", "sources", "eer_all", "eer_stranger",
                                                            "target_accept_at_tau", "nested")}
            th = k["thresholds"]
            print(f"  {kind}: τ={th['tau']:.3f} τ_low={th['tau_low']:.3f} δ={th['delta']:.3f} "
                  f"gap={th['gap']:.3f} · trials {k['counts']} · EER {k['eer_all']}")
            print(f"    sources: {k['sources']}")
            print(f"    nested LOMO (thresholds from other meetings): {_fmt_nested(k['nested'])}")
        if not any(kind in entry for kind in vp_calibrate.KINDS):
            print(f"❌ {model_id}: not enough data to calibrate; enroll more meetings or add other "
                  "voices with cohort-add. Nothing written.", file=sys.stderr)
            rc = rc or 5
            continue
        if "asnorm" not in entry:
            print("  ℹ️  AS-norm not calibrated: identify will score raw cosine for this model")
        if args.dry_run:
            print("  (dry run) calibration.json not written")
        else:
            path = store.save_calibration(model_id, entry)
            print(f"  ✅ wrote {path} [clip sig {sig}, device cap {entry['clip']['hard_cap_s']:g} s]")
    emit({"status": "ok" if rc == 0 else "incomplete", "store": str(store.root), "models": results}, args.json)
    return rc


# ─── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", help="override VOICEPRINT_DIR (must be outside the notes vault and this clone)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    caps = vp_config.DEVICE_CAP_S

    def common_audio(p, utterances_required=True):
        p.add_argument("--audio", required=True, type=Path)
        p.add_argument("--utterances", required=utterances_required, type=Path)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto",
                       help="auto = MPS if available (torch models; CPU-parity self-tested), else CPU")
        p.add_argument("--max-clip-s", type=float,
                       help=f"max clip length, s (default 10; device cap CPU {caps['cpu']:g} / "
                            f"MPS {caps['mps']:g}; ceiling {vp_config.SAFETY_CEILING_S:g})")
        p.add_argument("--meeting-key", help="YYYY-MM-DD-HHMM (default: from file names)")
        p.add_argument("--max-clips", type=int)
        p.add_argument("--min-net", type=float, help="min net speech per clip, s (default 2.0)")
        p.add_argument("--vad", choices=["silero", "energy"],
                       help="speech gate: silero (built-in default; rejects ringtones/hold music) or energy-only")
        p.add_argument("--clip-mode", choices=["window", "longest_run"],
                       help="clip selection (default: calibration.json clip block, else window)")
        p.add_argument("--window-s", type=float,
                       help="window length for clip-mode window (sets hop = w/2 and max clip = w)")
        p.add_argument("--hard-cap-s", type=float,
                       help=f"hard clip cap, s (overrides the device cap; ≤ {vp_config.SAFETY_CEILING_S:g})")
        p.add_argument("--quant-s", type=float, help="clip length grid, s (default 0.5; 0 = off)")
        p.add_argument("--min-speech-frac", type=float,
                       help="Silero speech fraction to keep a clip (default 0.6)")
        p.add_argument("--json", help="write JSON result to path, or '-' for stdout")

    p = sub.add_parser("identify", help="score clusters against enrolled voiceprints")
    common_audio(p)
    p.add_argument("--roster", help="comma-separated attendee names (closed set)")
    p.add_argument("--labels", help="existing mapping to check, e.g. 'A=Bob Example,B=Alice Example' "
                                    "(default: sidecar speaker_map)")
    p.add_argument("--channel", choices=vp_config.CHANNELS, help="use per-channel sub-centroids")
    p.add_argument("--score-mode", choices=["centroid", "top5"],
                   help="default: calibration.json score_mode, else centroid")
    p.add_argument("--top-n", type=int, default=100, help="AS-norm top-N cohort")
    p.add_argument("--min-cohort", type=int, default=50, help="below this, fall back to raw cosine")
    p.add_argument("--min-cluster-s", type=float, default=8.0, help="cluster net speech for 'sufficient'")
    p.add_argument("--no-asnorm", action="store_true")
    p.add_argument("--include-self", action="store_true",
                   help="also use exemplars enrolled from this same meeting (default: excluded)")
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("enroll", help="enroll a verified cluster, or confirmed timeline turns, as a person")
    common_audio(p, utterances_required=False)
    p.add_argument("--timeline", type=Path,
                   help="enroll from a speaker Timeline JSON (from track) instead of --utterances/--cluster")
    p.add_argument("--name", required=True)
    p.add_argument("--cluster", help="utterances cluster to enroll (with --utterances)")
    p.add_argument("--speaker", help="timeline label to enroll, e.g. GUEST_1 (default: --name)")
    p.add_argument("--spans", help="only these user-confirmed spans, e.g. '12.5-20,01:05-01:30' "
                                   "(required for an UNCERTAIN label)")
    p.add_argument("--min-turn-s", type=float, default=TIMELINE_MIN_TURN_S,
                   help=f"timeline turns shorter than this are skipped (default {TIMELINE_MIN_TURN_S:g})")
    p.add_argument("--channel", choices=vp_config.CHANNELS, default="unknown")
    p.add_argument("--language", help="zh | en | mixed")
    p.add_argument("--source", choices=vp_config.SOURCES, default="confirmed_meeting")
    p.add_argument("--cap", type=int, default=vp_config.PERSON_CAP)
    p.add_argument("--allow-unregistered", action="store_true")
    p.add_argument("--force", action="store_true", help="override label/self-check refusals")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("cohort-add", help="add clips of other voices for AS-norm")
    common_audio(p)
    p.add_argument("--clusters", help="comma-separated clusters (default: all)")
    p.add_argument("--labels", help="cluster=name map (default: sidecar speaker_map)")
    p.set_defaults(func=cmd_cohort_add)

    p = sub.add_parser("calibrate", help="derive thresholds from the enrolled store (leave-one-meeting-out)",
                       description="Derive τ/τ_low/δ/gap from your own enrolled meetings and write "
                                   "calibration.json (with the clip config) into the store.")
    p.add_argument("--model-id", help="calibrate one model_id (default: every model in the store)")
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto",
                   help="device the exemplars were enrolled on (auto: infer from their clip signature)")
    for flag, kw in (("--clip-mode", {"choices": ["window", "longest_run"]}), ("--window-s", {"type": float}),
                     ("--max-clip-s", {"type": float}), ("--hard-cap-s", {"type": float}),
                     ("--quant-s", {"type": float}), ("--max-clips", {"type": int}),
                     ("--min-net", {"type": float}), ("--vad", {"choices": ["silero", "energy"]}),
                     ("--min-speech-frac", {"type": float})):
        p.add_argument(flag, help="clip flag used at enrollment (only if you overrode it then)", **kw)
    p.add_argument("--score-mode", choices=["centroid", "top5"], help="default: existing calibration, else centroid")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--min-cohort", type=int, default=50)
    p.add_argument("--min-cluster-s", type=float, default=8.0)
    p.add_argument("--force", action="store_true", help="calibrate despite mixed/unmatched clip configs")
    p.add_argument("--dry-run", action="store_true", help="report only; do not write calibration.json")
    p.add_argument("--json", help="write JSON result to path, or '-' for stdout")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("track", help="who speaks when: track enrolled speakers through a recording")
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--roster", help="comma-separated attendee names (default: everyone enrolled)")
    p.add_argument("--channel", choices=vp_config.CHANNELS, help="use per-channel sub-centroids")
    p.add_argument("--backend", choices=["auto", "nemotron3", "seg3"], default="auto",
                   help="boundary layer (auto: nemotron3 when installed, else seg3)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    p.add_argument("--meeting-key", help="YYYY-MM-DD-HHMM (default: from the file name)")
    p.add_argument("--include-self", action="store_true",
                   help="also use exemplars enrolled from this same meeting (default: excluded)")
    p.add_argument("--theta", type=float, help="UNKNOWN log-odds (default: TrackParams)")
    p.add_argument("--switch", type=float, help="HMM switch penalty λ")
    p.add_argument("--p-uncertain", type=float, help="posterior below which a name becomes UNCERTAIN")
    p.add_argument("--out", help="write the Timeline JSON here")
    p.add_argument("--rttm", help="write the turns as RTTM here")
    p.add_argument("--json", help="write a JSON summary to path, or '-' for stdout")
    p.set_defaults(func=cmd_track)

    p = sub.add_parser("track-check", help="can `track` name speakers here? (JSON on stdout; cheap)")
    p.add_argument("--roster", help="comma-separated attendee names")
    p.add_argument("--backend", choices=["auto", "nemotron3", "seg3"], default="nemotron3",
                   help="boundary layer that must be installed (default nemotron3: the seg3 fallback "
                        "names far more strangers)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.set_defaults(func=cmd_track_check)

    p = sub.add_parser("track-calibrate", help="refit the tracker's score calibration on the store (LOMO)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", help="write JSON result to path, or '-' for stdout")
    p.set_defaults(func=cmd_track_calibrate)

    p = sub.add_parser("list", help="enrolled people, counts, channels, model ids (no vectors)")
    p.add_argument("--json", nargs="?", const="-")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("forget", help="delete all voiceprint data for a person")
    p.add_argument("--name", required=True)
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_forget)

    from voiceprint import audit as vp_audit  # stdlib at import; clip rendering imports numpy lazily
    vp_audit.add_cli(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if DEPS_ERROR:
        if args.cmd == "track-check":
            print(json.dumps({"ready": False, "status": "no_deps",
                              "reason": f"voiceprint dependencies not installed ({DEPS_ERROR})"}))
            return 0
        if args.cmd in ("identify", "track"):
            return noop("no_deps", f"voiceprint dependencies not installed ({DEPS_ERROR}); "
                        "see README § Voiceprint speaker ID", args.json)
        print(f"❌ voiceprint dependencies not installed ({DEPS_ERROR}). Install them with "
              "MEETING_DOCUMENTER_VOICEPRINT=1 scripts/transcribe.sh --help, or "
              "uv pip install -r scripts/voiceprint/requirements.txt", file=sys.stderr)
        return 1
    try:
        return args.func(args)
    except vp_config.VaultPathError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 0 if args.cmd in ("identify", "track") else 2


if __name__ == "__main__":
    sys.exit(main())
