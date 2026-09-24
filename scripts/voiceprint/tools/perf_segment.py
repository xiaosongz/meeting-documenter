#!/usr/bin/env python3
"""Speed / memory smoke benchmark for the boundary layer (not run in CI).

    python scripts/voiceprint/tools/perf_segment.py AUDIO [--backend seg3|nemotron3]
        [--step 2.5] [--threads 4] [--device cpu] [--minutes 120] [--json out.json]

Decodes AUDIO (any ffmpeg-readable file; use public audio such as AMI), runs
segment() + local_regions(), and prints real-time factor and peak RSS. Peak RSS is
the process high-water mark (decode + model + activations), which is what the
≤ 3 GB budget is about.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from voiceprint import segment as sg  # noqa: E402
from voiceprint.audio import decode_audio  # noqa: E402


def peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20     # bytes on macOS, KiB on Linux


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--backend", default="seg3", choices=sg.BACKENDS)
    ap.add_argument("--step", type=float, default=sg.DEFAULT_STEP_S, help="seg3 chunk step (s)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="nemotron3 only: cpu | mps")
    ap.add_argument("--minutes", type=float, default=None, help="decode only the first N minutes")
    ap.add_argument("--min-clean", type=float, default=1.0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    wave = decode_audio(a.audio, dur_s=a.minutes * 60 if a.minutes else None)
    t_dec = time.perf_counter() - t0
    rss_decoded = peak_rss_gb()
    kw = {"step": a.step, "num_threads": a.threads} if a.backend == "seg3" else \
         {"device": a.device, "num_threads": a.threads}
    t1 = time.perf_counter()
    seg = sg.segment(wave, backend=a.backend, **kw)
    t_seg = time.perf_counter() - t1
    t2 = time.perf_counter()
    regions = sg.local_regions(seg, a.min_clean)
    t_reg = time.perf_counter() - t2
    dur = len(wave) / sg.SAMPLE_RATE
    res = {
        "backend": a.backend, "audio_s": round(dur, 1), "decode_s": round(t_dec, 2),
        "segment_s": round(t_seg, 2), "x_realtime": round(dur / t_seg, 1),
        "local_regions_s": round(t_reg, 2), "n_chunks": seg.n_chunks, "n_regions": len(regions),
        "activations_mb": round(seg.activations.nbytes / 2**20, 1),
        "peak_rss_after_decode_gb": round(rss_decoded, 2), "peak_rss_gb": round(peak_rss_gb(), 2),
        **({"step": a.step} if a.backend == "seg3" else {"device": a.device}), "threads": a.threads,
    }
    print(json.dumps(res))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
