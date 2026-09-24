#!/usr/bin/env python3
"""Embedding memory check: peak physical footprint on a long mixed-length input (not run in CI).

    python scripts/voiceprint/tools/perf_embed.py [--device mps] [--minutes 30]
        [--max-footprint-gb 12] [--json out.json]

Embeds synthetic voice-like clips of many distinct lengths (0.5-12 s, the worst case for the
MPS caching allocator, which keeps buffers per input shape) through `audio.embed_clips`, the
path every caller uses. Prints speed (× real time of embedded audio) and the process's
lifetime peak *physical footprint*, which, unlike RSS, includes MPS buffers. Exits 1 when
the peak exceeds --max-footprint-gb. Before the bounded-batch fix, this workload's footprint
grew without bound on MPS (74-85 GB were observed on real meetings).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from voiceprint import audio as au  # noqa: E402
from voiceprint import embedders as em  # noqa: E402
from voiceprint.footprint import gb, peak_footprint, phys_footprint  # noqa: E402

SR = au.SAMPLE_RATE


def voice_like(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f0 = rng.uniform(90, 260)
    phase = 2 * np.pi * np.cumsum(f0 + 15 * np.sin(2 * np.pi * rng.uniform(0.3, 1.5) * t)) / SR
    x = sum((0.3 / k) * np.sin(k * phase) for k in range(1, 8))
    return np.clip(x + 0.01 * rng.normal(size=n), -1, 1).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    ap.add_argument("--minutes", type=float, default=30.0, help="total audio to embed")
    ap.add_argument("--batch", type=int, default=64, help="clips per embed_clips batch")
    ap.add_argument("--max-footprint-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    emb = em.get_embedder("redimnet2", device=a.device)
    base = phys_footprint()
    clips, total = [], 0.0
    while total < a.minutes * 60:
        n = int(rng.integers(SR // 2, 12 * SR))             # ~all lengths distinct
        clips.append(au.Clip("s", 0, 0, n / SR, voice_like(n, rng)))
        total += n / SR
    t0 = time.time()
    E = au.embed_clips(emb, clips, batch_size=a.batch)
    dt = time.time() - t0
    peak = peak_footprint()
    res = {"device": a.device, "clips": len(E), "audio_s": round(total, 1), "embed_s": round(dt, 1),
           "xrt": round(total / dt, 1), "footprint_before_gb": gb(base), "peak_footprint_gb": gb(peak),
           "limit_gb": a.max_footprint_gb, "model_id": emb.model_id}
    print(json.dumps(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    return 0 if peak is None or peak <= a.max_footprint_gb * 1e9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
