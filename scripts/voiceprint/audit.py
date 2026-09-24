"""
Tier-1 listening audit of speaker timelines (plan P3; research §7 "Tier 1").

    audit sample   (audio, Timeline) pairs → manifest.json + clips/*.wav + index.html
    audit score    manifest.json + answers.json → report.json + report.md

The audit samples the *system output* (the exclusive timeline), so it measures
how often a label is right, per reference-free stratum. It needs no reference
and no tracker internals: any producer that writes a Timeline (JSON or RTTM)
can be audited.

Population (one unit per exclusive turn ≥ min_turn_s, plus system gaps):
    stratum = turn-duration bucket (<1s, 1-3s, 3-10s, >=10s)
            × confidence tercile (cut points over all units; "all" when conf has < 3 distinct values)
            × channel tag (per input > Timeline.meta["channel"] > --channel > "unknown")
Pools (disjoint; each unit belongs to exactly one, first match wins):
    uncertain   kind "uncertain" turns                              default 10 % of N
    switch      named turns of a close pair (A, B) next to the other
                member within switch_gap_s                          default 10 % of N
    gap         system silence ≥ gap_min_s between turns (catches
                missed speech, B0's dominant error)                 default 5 % of N
    cell        everything else, one pool per stratum
The rest of N goes to duration buckets by share (default <1s 35 %, 1-3s 25 %,
3-10s 20 %, >=10s 20 %: short turns oversampled), then to the cells of a
bucket ∝ sqrt(cell size), so small cells stay represented. Every pool is a
simple random sample without replacement; an item's design weight is
N_pool / n_pool, and `score` reports weighted proportions with Kish effective
n. The close pair comes from --pair, else the pair with the most UNCERTAIN
time, else the pair with the most direct switches (recorded in the manifest).

Target window: the whole unit when ≤ max_target_s, else a max_target_s window
placed uniformly inside it (at the switch edge for switch items). All draws are
deterministic in the seed (each pool and item has its own string-seeded RNG).

Clips: [target − context_s, target + context_s], 16 kHz mono 16-bit, one gain
per clip (BS.1770-style gated RMS to TARGET_DB, peak ≤ PEAK_DB, gain ≤
MAX_GAIN_DB), then a soft TONE_S tone is *inserted* (not overlaid) just before
and just after the target: rising pitch = start, falling = end. Inserting the
tones masks no speech at the boundaries, which is where errors are; the page
also shades the target span on the clip's progress bar and can replay the
target alone (T).

Answers: name:<Name> | guest (someone not on the roster) | overlap | unsure | nospeech.
Outcome per item (speech units; "unsure" answers are excluded from rates):
    system \\ answer   same name   other name / guest   overlap          no speech
    name              correct     wrong_name           overlap_missed   false_alarm
    guest / unknown   abstain*    correct (guest)      overlap_missed   false_alarm
    uncertain(A|B)    abstain     abstain              overlap_missed   false_alarm
    overlap(A|B)      abstain if the answer is one of A|B, else wrong_name; overlap → correct
  (* a named answer on guest/unknown: the system withheld a name it should have given.)
Uncertain items also report the hit rate (answer ∈ alts). Gap units score
"correct" for no speech and "missed" for any speech answer.

Everything written by this module (clips, manifest, page, reports) holds
private audio or names, so every output directory passes the same guard as the
voiceprint store (config.assert_outside_vault: never inside this clone, a
notes root or an Obsidian vault). Standard library only, except clip
rendering (numpy + ffmpeg via audio.py), which is imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import timeline as tlm
from .config import CHANNELS, assert_outside_vault
from .timeline import Timeline

SCHEMA = 1
BUCKETS = (("<1s", 0.0, 1.0), ("1-3s", 1.0, 3.0), ("3-10s", 3.0, 10.0), (">=10s", 10.0, math.inf))
BUCKET_NAMES = tuple(b[0] for b in BUCKETS)
TERCILES = ("low", "mid", "high")
DEFAULT_BUCKET_SHARE = {"<1s": 0.35, "1-3s": 0.25, "3-10s": 0.20, ">=10s": 0.20}
ANSWER_CODES = ("guest", "overlap", "unsure", "nospeech")   # plus "name:<Name>"
OUTCOMES = ("correct", "wrong_name", "abstain", "overlap_missed", "false_alarm")
Z95 = 1.959963984540054

SR = 16000
TARGET_DB = -23.0     # dBFS, gated RMS of the clip's speech
PEAK_DB = -1.0        # dBFS sample peak after gain
MAX_GAIN_DB = 30.0    # don't blow up near-silent clips
TONE_S = 0.12
TONE_DB = -26.0       # dBFS RMS: audible under normalised speech, not startling
TONE_HZ = {"start": (660.0, 990.0), "end": (990.0, 660.0)}   # linear glide: rising = start, falling = end
EDGE_FADE_S = 0.005   # fade speech out/in around an inserted tone (no clicks)
PAGE_TEMPLATE = Path(__file__).with_name("audit_page.html")
DATA_MARK = "/*__AUDIT_DATA__*/null"


@dataclass
class SampleParams:
    n: int = 300
    seed: int = 0
    min_turn_s: float = 0.25
    max_target_s: float = 3.0
    context_s: float = 2.0
    switch_gap_s: float = 1.0
    gap_min_s: float = 1.0
    uncertain_frac: float = 0.10
    switch_frac: float = 0.10
    gap_frac: float = 0.05
    bucket_share: dict = field(default_factory=lambda: dict(DEFAULT_BUCKET_SHARE))


@dataclass
class AuditInput:
    audio: str
    timeline: str
    channel: str = ""        # per-input tag; "" = Timeline meta "channel", else the default
    tl: Timeline | None = None
    roster: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return Path(self.audio).stem


@dataclass
class Unit:
    input: int
    start: float
    end: float
    label: str               # "" for gap units
    kind: str                # timeline kind, or "gap"
    conf: float
    alts: tuple[str, ...]
    bucket: str = ""
    tercile: str = ""
    channel: str = "unknown"
    pool: str = ""
    switch_edge: str | None = None    # "start" | "end": side where the close-pair switch is

    @property
    def dur(self) -> float:
        return self.end - self.start


# ─── Population and strata ──────────────────────────────────────────────────

def bucket_of(dur: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= dur < hi:
            return name
    return BUCKETS[-1][0]


def tercile_cuts(confs: list[float]) -> tuple[float, float] | None:
    """Cut points (q1, q2) of the finite confidences, or None when they can't split three ways."""
    vals = sorted(c for c in confs if math.isfinite(c))
    if len(set(vals)) < 3:
        return None
    q1, q2 = statistics.quantiles(vals, n=3, method="inclusive")
    return q1, q2


def tercile_of(conf: float, cuts: tuple[float, float] | None) -> str:
    if cuts is None or not math.isfinite(conf):
        return "all"
    return "low" if conf < cuts[0] else "mid" if conf < cuts[1] else "high"


def is_name(label: str) -> bool:
    return bool(label) and tlm.parse_label(label)[0] == "name"


def roster_of(tl: Timeline, extra: list[str] | None = None) -> list[str]:
    """Enrolled names the listener may pick for this input: named labels, named alts, meta roster, extra."""
    names = set(extra or ()) | set(tl.meta.get("roster") or ())
    for t in list(tl.turns) + list(tl.exclusive):
        names.update(c for c in tlm.candidates(t.speaker) if is_name(c))
        names.update(a for a in t.alts if is_name(a))
    return sorted(n for n in names if is_name(n))


def auto_pair(inputs: list[AuditInput], switch_gap_s: float) -> tuple[tuple[str, str] | None, str]:
    """The pair with the most UNCERTAIN time; else the most direct switches; else None."""
    unc: Counter = Counter()
    sw: Counter = Counter()
    for inp in inputs:
        ex = inp.tl.exclusive
        for t in ex:
            if t.kind == "uncertain":
                names = sorted({a for a in t.alts if is_name(a)})
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        unc[(names[i], names[j])] += t.dur
        for a, b in zip(ex, ex[1:]):
            if a.kind == b.kind == "name" and a.speaker != b.speaker and b.start - a.end <= switch_gap_s:
                sw[tuple(sorted((a.speaker, b.speaker)))] += 1
    for counts, source in ((unc, "uncertain_time"), (sw, "switch_count")):
        if counts:
            return max(sorted(counts), key=lambda k: counts[k]), source
    return None, "none"


def build_units(inputs: list[AuditInput], params: SampleParams,
                pairs: list[tuple[str, str]]) -> tuple[list[Unit], dict]:
    """All sampling units with strata and pool; plus population stats for the manifest."""
    units: list[Unit] = []
    skipped = Counter()
    for i, inp in enumerate(inputs):
        ex = sorted(inp.tl.exclusive, key=lambda t: (t.start, t.end))
        prev_end = 0.0
        for k, t in enumerate(ex):
            if t.start - prev_end >= params.gap_min_s:
                units.append(Unit(i, round(prev_end, 3), round(t.start, 3), "", "gap", math.nan, (),
                                  bucket="gap", channel=inp.channel, pool="gap"))
            prev_end = max(prev_end, t.end)
            if t.dur < params.min_turn_s:
                skipped["short"] += 1
                continue
            u = Unit(i, round(t.start, 3), round(t.end, 3), t.speaker, t.kind, t.conf, tuple(t.alts),
                     bucket=bucket_of(t.dur), channel=inp.channel)
            if t.kind == "name":
                for a, b in pairs:
                    if t.speaker not in (a, b):
                        continue
                    other = b if t.speaker == a else a
                    near_prev = k > 0 and ex[k - 1].speaker == other and t.start - ex[k - 1].end <= params.switch_gap_s
                    near_next = (k + 1 < len(ex) and ex[k + 1].speaker == other
                                 and ex[k + 1].start - t.end <= params.switch_gap_s)
                    if near_prev or near_next:
                        u.switch_edge = "start" if near_prev and not near_next else "end" if near_next and not near_prev \
                            else random.Random(f"{params.seed}:edge:{i}:{u.start:.3f}").choice(("start", "end"))
                        break
            units.append(u)
    cuts = tercile_cuts([u.conf for u in units if u.kind != "gap"])
    for u in units:
        if u.kind == "gap":
            continue
        u.tercile = tercile_of(u.conf, cuts)
        u.pool = ("uncertain" if u.kind == "uncertain" else "switch" if u.switch_edge
                  else f"cell:{u.bucket}|{u.tercile}|{u.channel}")
    stats = {"units": len(units), "skipped_short_turns": skipped["short"],
             "tercile_cuts": list(cuts) if cuts else None,
             "pool_sizes": dict(sorted(Counter(u.pool for u in units).items()))}
    return units, stats


# ─── Allocation and sampling ────────────────────────────────────────────────

def allocate(total: int, sizes: dict, weights: dict) -> dict:
    """Split `total` over keys ∝ weight, capped at size (largest remainder; capped keys' excess flows on)."""
    alloc = {k: 0 for k in sizes}
    remaining = min(total, sum(sizes.values()))
    open_keys = sorted((k for k in sizes if sizes[k] > 0 and weights.get(k, 0) > 0), key=str)
    while remaining > 0 and open_keys:
        wsum = sum(weights[k] for k in open_keys)
        quota = {k: remaining * weights[k] / wsum for k in open_keys}
        give = {k: min(int(quota[k]), sizes[k] - alloc[k]) for k in open_keys}
        left = remaining - sum(give.values())
        for k in sorted(open_keys, key=lambda k: (-(quota[k] - int(quota[k])), str(k))):
            if left <= 0:
                break
            if alloc[k] + give[k] < sizes[k]:
                give[k] += 1
                left -= 1
        given = sum(give.values())
        if given == 0:
            break
        for k, g in give.items():
            alloc[k] += g
        remaining -= given
        open_keys = [k for k in open_keys if alloc[k] < sizes[k]]
    return alloc


def plan_allocation(units: list[Unit], params: SampleParams) -> dict[str, int]:
    """Items per pool: fixed fractions for the oversampled pools, the rest by bucket share then sqrt(cell)."""
    sizes = Counter(u.pool for u in units)
    n = min(params.n, len(units))
    alloc: dict[str, int] = {}
    for pool, frac in (("uncertain", params.uncertain_frac), ("switch", params.switch_frac),
                       ("gap", params.gap_frac)):
        alloc[pool] = min(sizes.get(pool, 0), int(round(n * frac)))
    cells = {p: s for p, s in sizes.items() if p.startswith("cell:")}
    rest = n - sum(alloc.values())
    by_bucket = {b: sum(s for p, s in cells.items() if p[5:].split("|")[0] == b) for b in BUCKET_NAMES}
    bucket_n = allocate(rest, by_bucket, params.bucket_share)
    left = rest - sum(bucket_n.values())
    if left > 0:   # buckets with zero share or too few units: spread what's left by cell size
        extra = allocate(left, {b: by_bucket[b] - bucket_n[b] for b in BUCKET_NAMES},
                         {b: max(by_bucket[b] - bucket_n[b], 0) for b in BUCKET_NAMES})
        bucket_n = {b: bucket_n[b] + extra[b] for b in BUCKET_NAMES}
    for b in BUCKET_NAMES:
        mine = {p: s for p, s in cells.items() if p[5:].split("|")[0] == b}
        alloc.update(allocate(bucket_n[b], mine, {p: math.sqrt(s) for p, s in mine.items()}))
    left = n - sum(alloc.values())
    if left > 0:   # oversampled pools were capped by their size and cells are full: top up anywhere
        spare = {p: sizes[p] - alloc.get(p, 0) for p in sizes}
        for p, g in allocate(left, spare, dict(spare)).items():
            alloc[p] = alloc.get(p, 0) + g
    return {p: k for p, k in sorted(alloc.items()) if k > 0}


def target_window(u: Unit, max_target_s: float, seed: int) -> tuple[float, float]:
    if u.dur <= max_target_s:
        return u.start, u.end
    if u.switch_edge == "start":
        return u.start, round(u.start + max_target_s, 3)
    if u.switch_edge == "end":
        return round(u.end - max_target_s, 3), u.end
    s = u.start + random.Random(f"{seed}:target:{u.input}:{u.start:.3f}").uniform(0, u.dur - max_target_s)
    return round(s, 3), round(s + max_target_s, 3)


def draw(units: list[Unit], alloc: dict[str, int], params: SampleParams) -> list[dict]:
    """Simple random sample per pool; items in presentation order (meetings shuffled, items shuffled within)."""
    by_pool: dict[str, list[Unit]] = defaultdict(list)
    for u in sorted(units, key=lambda u: (u.input, u.start, u.end)):
        by_pool[u.pool].append(u)
    picked: list[tuple[Unit, float]] = []
    for pool, k in alloc.items():
        pop = by_pool[pool]
        weight = len(pop) / k
        picked += [(u, weight) for u in random.Random(f"{params.seed}:pool:{pool}").sample(pop, k)]
    rng = random.Random(f"{params.seed}:order")
    meetings = sorted({u.input for u, _ in picked})
    rng.shuffle(meetings)
    ordered: list[tuple[Unit, float]] = []
    for m in meetings:
        block = sorted((p for p in picked if p[0].input == m), key=lambda p: (p[0].start, p[0].end))
        rng.shuffle(block)
        ordered += block
    items = []
    for order, (u, weight) in enumerate(ordered, 1):
        t0, t1 = target_window(u, params.max_target_s, params.seed)
        items.append({"id": f"i{order:03d}", "order": order, "input": u.input, "pool": u.pool,
                      "weight": weight, "bucket": u.bucket, "tercile": u.tercile, "channel": u.channel,
                      "unit": {"start": u.start, "end": u.end, "label": u.label, "kind": u.kind,
                               "conf": u.conf if math.isfinite(u.conf) else None, "alts": list(u.alts),
                               "switch_edge": u.switch_edge},
                      "target": {"start": t0, "end": t1}})
    return items


# ─── Clip rendering (numpy + ffmpeg, imported lazily) ───────────────────────

def gated_level_db(x, sr: int = SR, frame_s: float = 0.4, hop_s: float = 0.1) -> float:
    """BS.1770-style gated mean power (dBFS) without the K-weighting filter: -70 dB absolute gate,
    then a relative gate 10 dB under the absolute-gated mean. Enough for even listening levels."""
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    f, h = int(frame_s * sr), int(hop_s * sr)
    if x.size < f:
        p = np.array([np.mean(x ** 2)]) if x.size else np.array([0.0])
    else:
        c = np.concatenate([[0.0], np.cumsum(x ** 2)])
        starts = np.arange(0, x.size - f + 1, h)
        p = (c[starts + f] - c[starts]) / f
    p = p[p > 10 ** (-70 / 10)]
    if p.size == 0:
        return -120.0
    rel = 10 * math.log10(float(p.mean())) - 10
    p = p[10 * np.log10(p) > rel]
    return 10 * math.log10(float(p.mean()))


def tone(which: str, sr: int = SR):
    import numpy as np
    n = int(TONE_S * sr)
    f0, f1 = TONE_HZ[which]
    freq = np.linspace(f0, f1, n)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    amp = math.sqrt(2) * 10 ** (TONE_DB / 20)
    return (amp * np.sin(phase) * np.hanning(n)).astype(np.float32)


def render_clip(wave, start_s: float, t0: float, t1: float, context_s: float,
                sr: int = SR) -> tuple[object, dict]:
    """Cut [t0 − context, t1 + context] from `wave` (which starts at `start_s`), normalise, insert tones.

    Returns (int16 samples, geometry): clip_dur and the target's span inside the rendered clip.
    """
    import numpy as np
    c0 = max(0.0, t0 - context_s)
    a = int(round((c0 - start_s) * sr))
    b = int(round((t0 - start_s) * sr))
    c = int(round((t1 - start_s) * sr))
    d = int(round((t1 + context_s - start_s) * sr))
    n = len(wave)
    a, b, c, d = (min(max(v, 0), n) for v in (a, b, c, d))
    seg = np.asarray(wave[a:d], dtype=np.float32).copy()
    if seg.size:
        gain_db = min(TARGET_DB - gated_level_db(seg, sr), MAX_GAIN_DB)
        seg *= 10 ** (gain_db / 20)
        peak = float(np.max(np.abs(seg)))
        if peak > 10 ** (PEAK_DB / 20):
            seg *= 10 ** (PEAK_DB / 20) / peak
    pre, tgt, post = seg[: b - a].copy(), seg[b - a: c - a].copy(), seg[c - a:].copy()
    for part, head, tail in ((pre, False, True), (tgt, True, True), (post, True, False)):
        k = min(int(EDGE_FADE_S * sr), part.size // 2)
        if k and head:
            part[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)
        if k and tail:
            part[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
    out = np.concatenate([pre, tone("start", sr), tgt, tone("end", sr), post])
    geom = {"clip_dur": round(out.size / sr, 3),
            "t0": round((pre.size + int(TONE_S * sr)) / sr, 3),
            "t1": round((pre.size + int(TONE_S * sr) + tgt.size) / sr, 3),
            "context_start": round(start_s + a / sr, 3)}
    pcm = np.clip(np.round(out * 32767.0), -32768, 32767).astype("<i2")
    return pcm, geom


def write_wav(path: Path, pcm, sr: int = SR) -> None:
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def render_clips(inputs: list[AuditInput], items: list[dict], out_dir: Path, context_s: float,
                 log=print) -> dict:
    """Decode each input once, write clips/<id>.wav, fill item['clip']. Returns timing stats."""
    from . import audio as vp_audio
    clips = out_dir / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    audio_s = 0.0
    for i, inp in enumerate(inputs):
        mine = [it for it in items if it["input"] == i]
        if not mine:
            continue
        lo = max(0.0, min(it["target"]["start"] for it in mine) - context_s)
        hi = max(it["target"]["end"] for it in mine) + context_s
        wave = vp_audio.decode_audio(inp.audio, SR, start_s=lo, dur_s=hi - lo)
        audio_s += len(wave) / SR
        for it in mine:
            pcm, geom = render_clip(wave, lo, it["target"]["start"], it["target"]["end"], context_s)
            write_wav(clips / f"{it['id']}.wav", pcm)
            it["clip"] = {"file": f"clips/{it['id']}.wav", **geom}
        log(f"  rendered {len(mine)} clips from {inp.name}")
        del wave
    return {"decoded_audio_s": round(audio_s, 1), "render_s": round(time.perf_counter() - t_start, 2)}


# ─── Page ───────────────────────────────────────────────────────────────────

def keymaps(inputs: list[AuditInput]) -> tuple[str, list[list[str]]]:
    """One shared 1–9 map when the union roster fits, else one per input (first 9 names get keys)."""
    union = sorted(set().union(*(inp.roster for inp in inputs))) if inputs else []
    if len(union) <= 9:
        return "shared", [union for _ in inputs]
    return "per_input", [inp.roster for inp in inputs]


def page_data(manifest: dict) -> dict:
    items = []
    for it in manifest["items"]:
        clip = it.get("clip") or {}
        items.append({"id": it["id"], "input": it["input"], "clip": clip.get("file"),
                      "dur": clip.get("clip_dur"), "t0": clip.get("t0"), "t1": clip.get("t1"),
                      "system": it["unit"]["label"] or "(no speech)", "kind": it["unit"]["kind"]})
    return {"schema": SCHEMA, "audit_id": manifest["audit_id"], "created": manifest["created"],
            "keymap_mode": manifest["keymap_mode"],
            "inputs": [{"name": inp["name"], "roster": inp["roster"], "keys": inp["keys"]}
                       for inp in manifest["inputs"]],
            "items": items}


def write_page(manifest: dict, out_dir: Path) -> Path:
    html = PAGE_TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(page_data(manifest), ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/").replace("<!--", "<\\!--")
    if DATA_MARK not in html:
        raise RuntimeError(f"{PAGE_TEMPLATE.name} lost its data placeholder")
    path = out_dir / "index.html"
    path.write_text(html.replace(DATA_MARK, data), encoding="utf-8")
    return path


def check_page(out_dir: Path) -> list[str]:
    """Static check of a rendered audit page: no network references, data parses, clips exist."""
    import re
    out_dir = Path(out_dir)
    html = (out_dir / "index.html").read_text(encoding="utf-8")
    problems = []
    for pat in (r"""(?:src|href|action)\s*=\s*["']?\s*(?:https?:)?//""", r"url\(\s*['\"]?\s*(?:https?:)?//",
                r"@import", r"""\bfetch\s*\(""", r"XMLHttpRequest", r"WebSocket", r"<link\b", r"<script[^>]+src="):
        if re.search(pat, html, re.I):
            problems.append(f"external reference or network API: /{pat}/")
    m = re.search(r'<script id="audit-data" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        return problems + ["no audit-data block"]
    data = json.loads(m.group(1))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if data["audit_id"] != manifest["audit_id"] or len(data["items"]) != len(manifest["items"]):
        problems.append("page data does not match manifest.json")
    for it in data["items"]:
        if not it["clip"] or not (out_dir / it["clip"]).is_file():
            problems.append(f"missing clip for {it['id']}")
    return problems


# ─── sample ─────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_timeline(path: str | Path, uri: str | None = None) -> Timeline:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".rttm":
        uris = {ln.split()[1] for ln in text.splitlines() if ln.startswith("SPEAKER ") and len(ln.split()) > 1}
        tl = tlm.from_rttm(text, uri=uri if len(uris) > 1 else None)
    else:
        tl = tlm.from_json(text)
    if tl.turns and not tl.exclusive:
        tl.exclusive = tlm.derive_exclusive(tl.turns)
    return tl


def sample(inputs: list[AuditInput], out_dir: str | Path, params: SampleParams = SampleParams(), *,
           pairs: list[tuple[str, str]] | None = None, roster: list[str] | None = None,
           default_channel: str | None = None, render: bool = True, force: bool = False, log=print) -> dict:
    """Draw the audit sample, render clips and the page; returns the manifest (also written to out_dir)."""
    out = Path(out_dir).expanduser().resolve()
    assert_outside_vault(out)
    if (out / "manifest.json").exists() and not force:
        raise FileExistsError(f"{out}/manifest.json exists (an audit may be in progress); use --force to replace it")
    for inp in inputs:
        if inp.tl is None:
            inp.tl = load_timeline(inp.timeline, uri=Path(inp.audio).stem)
        inp.channel = inp.channel or str(inp.tl.meta.get("channel") or default_channel or "unknown")
        inp.roster = roster_of(inp.tl, roster)
    pair_source = "cli"
    if not pairs:
        pair, pair_source = auto_pair(inputs, params.switch_gap_s)
        pairs = [pair] if pair else []
    units, stats = build_units(inputs, params, pairs)
    alloc = plan_allocation(units, params)
    items = draw(units, alloc, params)
    fingerprint = json.dumps({"params": asdict(params), "pairs": pairs,
                              "inputs": [[str(Path(i.audio).resolve()), _sha256(Path(i.timeline)), i.channel]
                                         for i in inputs]}, sort_keys=True)
    mode, maps = keymaps(inputs)
    manifest = {
        "schema": SCHEMA, "type": "speaker-audit-manifest",
        "audit_id": hashlib.sha256(fingerprint.encode()).hexdigest()[:12],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": asdict(params), "pairs": [list(p) for p in pairs], "pair_source": pair_source,
        "keymap_mode": mode,
        "inputs": [{"name": inp.name, "audio": str(Path(inp.audio).resolve()),
                    "timeline": str(Path(inp.timeline).resolve()), "timeline_sha256": _sha256(Path(inp.timeline)),
                    "channel": inp.channel, "roster": inp.roster, "keys": maps[i][:9]}
                   for i, inp in enumerate(inputs)],
        "population": stats, "allocation": alloc, "items": items,
    }
    out.mkdir(parents=True, exist_ok=True)
    if render:
        manifest["render"] = render_clips(inputs, items, out, params.context_s, log=log)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    if render:
        write_page(manifest, out)
    return manifest


# ─── score ──────────────────────────────────────────────────────────────────

def outcome(kind: str, label: str, alts, answer: str) -> str:
    """Outcome of one judged item (table in the module docstring); 'excluded' for unsure answers."""
    if answer == "unsure":
        return "excluded"
    if kind == "gap":
        return "correct" if answer == "nospeech" else "missed"
    if answer == "nospeech":
        return "false_alarm"
    if answer == "overlap":
        return "correct" if kind == "overlap" else "overlap_missed"
    name = answer[5:] if answer.startswith("name:") else None     # None: guest / someone else
    if kind == "name":
        return "correct" if name == label else "wrong_name"
    if kind in ("guest", "unknown"):
        return "correct" if name is None else "abstain"
    if kind == "uncertain":
        return "abstain"
    if kind == "overlap":
        hit = name in alts if name else any(not is_name(a) for a in alts)
        return "abstain" if hit else "wrong_name"
    raise ValueError(f"unknown kind {kind!r}")


def uncertain_hit(alts, answer: str) -> bool:
    name = answer[5:] if answer.startswith("name:") else None
    return name in alts if name else any(not is_name(a) for a in alts)


def wilson(p: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for proportion p on (effective) sample size n."""
    if n <= 0:
        return (math.nan, math.nan)
    z2 = z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def weighted_rate(rows: list[tuple[float, bool]]) -> dict:
    """Weighted proportion with Kish effective n and a Wilson 95 % CI on it."""
    if not rows:
        return {"p": None, "lo": None, "hi": None, "n": 0, "n_eff": 0.0}
    sw = sum(w for w, _ in rows)
    sw2 = sum(w * w for w, _ in rows)
    p = sum(w for w, y in rows if y) / sw
    n_eff = sw * sw / sw2
    lo, hi = wilson(p, n_eff)
    return {"p": p, "lo": lo, "hi": hi, "n": len(rows), "n_eff": round(n_eff, 1)}


def load_answers(path: str | Path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("type") != "speaker-audit-answers":
        raise ValueError(f"{path}: not an audit answers file")
    for iid, a in obj.get("answers", {}).items():
        ans = a.get("answer", "")
        if not (ans in ANSWER_CODES or (ans.startswith("name:") and len(ans) > 5)):
            raise ValueError(f"{path}: item {iid}: bad answer {ans!r}")
    return obj


_ORDER = {k: i for i, k in enumerate((*BUCKET_NAMES, "gap", *TERCILES, "all", *CHANNELS))}


def _stratum_order(key) -> tuple:
    return tuple((_ORDER.get(part, len(_ORDER)), part) for part in str(key).split(" | "))


def _stratum_table(rows: list[dict], key) -> dict:
    groups: dict = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    table = {}
    for k in sorted(groups, key=_stratum_order):
        g = groups[k]
        judged = [r for r in g if r["outcome"] != "excluded"]
        entry = {"items": len(g), "judged": len(judged), "unsure": len(g) - len(judged)}
        for o in OUTCOMES:
            entry[o] = weighted_rate([(r["weight"], r["outcome"] == o) for r in judged])
        table[k] = entry
    return table


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def score(manifest: dict, answers: dict, *, extra: list[dict] | None = None, force: bool = False) -> dict:
    if answers.get("audit_id") != manifest["audit_id"] and not force:
        raise ValueError(f"answers are for audit {answers.get('audit_id')}, manifest is {manifest['audit_id']} "
                         "(use --force to score anyway)")
    ans = answers.get("answers", {})
    speech, gaps, missing = [], [], 0
    for it in manifest["items"]:
        a = ans.get(it["id"])
        if not a:
            missing += 1
            continue
        u = it["unit"]
        row = {"id": it["id"], "weight": it["weight"], "bucket": it["bucket"], "tercile": it["tercile"],
               "channel": it["channel"], "pool": it["pool"].split(":")[0], "kind": u["kind"],
               "label": u["label"], "alts": u["alts"], "answer": a["answer"],
               "outcome": outcome(u["kind"], u["label"], u["alts"], a["answer"]),
               "changed_after_reveal": bool(a.get("changed_after_reveal"))}
        (gaps if u["kind"] == "gap" else speech).append(row)
    everyone = [dict(r, all="all") for r in speech]
    tables = {dim: _stratum_table(everyone, lambda r, d=dim: r[d])
              for dim in ("all", "bucket", "tercile", "channel", "pool", "kind")}
    tables["cell"] = _stratum_table(everyone, lambda r: f"{r['bucket']} | {r['tercile']} | {r['channel']}")
    judged_gaps = [r for r in gaps if r["outcome"] != "excluded"]
    gap_summary = {"items": len(gaps), "judged": len(judged_gaps),
                   "missed_speech": weighted_rate([(r["weight"], r["outcome"] == "missed") for r in judged_gaps])}
    unc = [r for r in speech if r["kind"] == "uncertain" and r["outcome"] != "excluded"]
    unc_summary = {"judged": len(unc),
                   "hit": weighted_rate([(r["weight"], uncertain_hit(r["alts"], r["answer"])) for r in unc])}

    def sys_col(r):
        return r["label"] if r["kind"] == "name" else r["kind"].upper()

    def truth_row(r):
        a = r["answer"]
        return a[5:] if a.startswith("name:") else {"guest": "(guest/other)", "overlap": "(overlap)",
                                                     "nospeech": "(no speech)"}[a]
    judged = [r for r in speech if r["outcome"] != "excluded"]
    confusion: dict = defaultdict(Counter)
    for r in judged:
        confusion[truth_row(r)][sys_col(r)] += 1
    wrong = Counter((truth_row(r), r["label"]) for r in judged if r["outcome"] == "wrong_name")
    pairs = []
    for a, b in manifest.get("pairs", []):
        sub = [r for r in judged if truth_row(r) in (a, b)]
        pairs.append({"pair": [a, b], "judged": len(sub),
                      f"{a}->{b}": sum(1 for r in sub if truth_row(r) == a and r["label"] == b),
                      f"{b}->{a}": sum(1 for r in sub if truth_row(r) == b and r["label"] == a),
                      "correct": sum(1 for r in sub if r["outcome"] == "correct")})
    times = [a["ms"] / 1000 for a in ans.values() if isinstance(a.get("ms"), (int, float)) and a["ms"] > 0]
    listener = {"answered": len(speech) + len(gaps), "missing": missing,
                "unsure": sum(r["outcome"] == "excluded" for r in speech + gaps),
                "changed_after_reveal": sum(r["changed_after_reveal"] for r in speech + gaps),
                "median_s_per_item": round(statistics.median(times), 1) if times else None,
                "total_listening_min": round(sum(times) / 60, 1) if times else None}
    agreement = []
    for other in extra or []:
        oa = other.get("answers", {})
        common = sorted(set(ans) & set(oa))
        x, y = [ans[i]["answer"] for i in common], [oa[i]["answer"] for i in common]
        agreement.append({"annotator": other.get("annotator") or "second", "common_items": len(common),
                          "agreement": (sum(p == q for p, q in zip(x, y)) / len(common)) if common else None,
                          "kappa": kappa(x, y)})
    return {"schema": SCHEMA, "type": "speaker-audit-report", "audit_id": manifest["audit_id"],
            "scored": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "annotator": answers.get("annotator"), "listener": listener, "strata": tables,
            "uncertain": unc_summary, "gaps": gap_summary,
            "confusion": {k: dict(v) for k, v in sorted(confusion.items())},
            "top_wrong_names": [{"truth": t, "system": s, "items": n} for (t, s), n in wrong.most_common(15)],
            "pairs": pairs, "agreement": agreement,
            "notes": ["Rates are design-weighted (weight = pool size / pool sample); CIs are Wilson 95 % on Kish "
                      "effective n.", "Unit = system turn (long turns: one window inside), so bucket rates are "
                      "per turn, not per second; speech the system missed shows up only in the gap pool."]}


def _pct(r: dict) -> str:
    if not r or r["p"] is None:
        return "–"
    return f"{100 * r['p']:.1f} [{100 * r['lo']:.1f}–{100 * r['hi']:.1f}]"


def report_markdown(rep: dict, manifest: dict) -> str:
    L = [f"# Speaker audit report ({rep['audit_id']})", "",
         f"Manifest: {len(manifest['items'])} items from {len(manifest['inputs'])} input(s), seed "
         f"{manifest['params']['seed']}; close pair(s): "
         f"{', '.join(' / '.join(p) for p in manifest.get('pairs', [])) or 'none'} ({manifest.get('pair_source')}).",
         f"Listener: {rep['listener']['answered']} answered, {rep['listener']['missing']} unanswered, "
         f"{rep['listener']['unsure']} unsure, {rep['listener']['changed_after_reveal']} changed after reveal; "
         f"median {rep['listener']['median_s_per_item'] or '–'} s/item, "
         f"total {rep['listener']['total_listening_min'] or '–'} min.",
         "", "Rates in % with Wilson 95 % CIs (design-weighted, Kish n_eff). "
         "wrong name = a name was given and it was wrong; abstain = UNCERTAIN / guest / unknown where a name was due.",
         ""]
    titles = {"all": "Overall", "bucket": "By system turn duration", "tercile": "By system confidence tercile",
              "channel": "By channel", "pool": "By sampling pool", "kind": "By system label kind",
              "cell": "By stratum (duration | confidence | channel)"}
    for dim in ("all", "bucket", "tercile", "channel", "pool", "kind", "cell"):
        L += [f"## {titles[dim]}", "",
              "| stratum | items | judged | correct % | wrong name % | abstain % | overlap missed % | false alarm % |",
              "|---|---|---|---|---|---|---|---|"]
        for k, e in rep["strata"][dim].items():
            L.append(f"| {k} | {e['items']} | {e['judged']} | {_pct(e['correct'])} | {_pct(e['wrong_name'])} | "
                     f"{_pct(e['abstain'])} | {_pct(e['overlap_missed'])} | {_pct(e['false_alarm'])} |")
        L.append("")
    g, u = rep["gaps"], rep["uncertain"]
    L += ["## System gaps and UNCERTAIN", "",
          f"- System silence ≥ {manifest['params']['gap_min_s']:g} s: {g['judged']} judged; speech heard in "
          f"{_pct(g['missed_speech'])} %.",
          f"- UNCERTAIN(A|B): {u['judged']} judged; the listener's answer was one of the alternatives in "
          f"{_pct(u['hit'])} %.", ""]
    L += ["## Confusion (listener answer → system label, items)", ""]
    cols = sorted({c for row in rep["confusion"].values() for c in row})
    if cols:
        L += ["| listener \\ system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for t, row in rep["confusion"].items():
            L.append(f"| {t} | " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    L += ["", "Top wrong names (truth → system, items):", ""]
    L += [f"- {w['truth']} → {w['system']}: {w['items']}" for w in rep["top_wrong_names"]] or ["- none"]
    for p in rep["pairs"]:
        a, b = p["pair"]
        L.append(f"\nClose pair {a} / {b}: {p['judged']} judged, {p['correct']} correct, "
                 f"{p[f'{a}->{b}']} {a}→{b}, {p[f'{b}->{a}']} {b}→{a}.")
    for ag in rep["agreement"]:
        L.append(f"\nSecond annotator ({ag['annotator']}): {ag['common_items']} common items, agreement "
                 f"{ag['agreement']}, Cohen's κ {ag['kappa']}.")
    L += ["", "## Notes", ""] + [f"- {n}" for n in rep["notes"]]
    return "\n".join(L) + "\n"


# ─── CLI ────────────────────────────────────────────────────────────────────

def _parse_share(text: str | None) -> dict:
    if not text:
        return dict(DEFAULT_BUCKET_SHARE)
    share = {}
    for part in text.split(","):
        k, _, v = part.partition("=")
        if k.strip() not in BUCKET_NAMES:
            raise SystemExit(f"--bucket-share: unknown bucket {k!r} (use {', '.join(BUCKET_NAMES)})")
        share[k.strip()] = float(v)
    return share


def _read_list(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 2:
            raise SystemExit(f"{path}: expected 'audio<TAB>timeline[<TAB>channel]' per line")
        rows.append((f[0].strip(), f[1].strip(), f[2].strip() if len(f) > 2 else ""))
    return rows


def cmd_sample(args) -> int:
    rows = [(a, t, "") for a, t in (args.input or [])]
    if args.list:
        rows += _read_list(args.list)
    if not rows:
        print("❌ give at least one --input AUDIO TIMELINE (or --list FILE)", file=sys.stderr)
        return 2
    inputs = []
    for audio, tl_path, ch in rows:
        if ch and ch not in CHANNELS:
            print(f"❌ channel {ch!r} not in {CHANNELS}", file=sys.stderr)
            return 2
        inputs.append(AuditInput(audio, tl_path, ch))
    for inp in inputs:
        if not args.no_clips and not Path(inp.audio).is_file():
            print(f"❌ audio not found: {inp.audio}", file=sys.stderr)
            return 2
    params = SampleParams(n=args.n, seed=args.seed, min_turn_s=args.min_turn_s, max_target_s=args.max_target_s,
                          context_s=args.context_s, switch_gap_s=args.switch_gap_s, gap_min_s=args.gap_min_s,
                          uncertain_frac=args.uncertain_frac, switch_frac=args.switch_frac,
                          gap_frac=args.gap_frac, bucket_share=_parse_share(args.bucket_share))
    pairs = []
    for p in args.pair or []:
        a, sep, b = p.partition("|")
        if not sep or not a.strip() or not b.strip():
            print(f"❌ --pair expects 'Name A|Name B', got {p!r}", file=sys.stderr)
            return 2
        pairs.append((a.strip(), b.strip()))
    roster = [r.strip() for r in args.roster.split(",") if r.strip()] if args.roster else None
    try:
        m = sample(inputs, args.out, params, pairs=pairs, roster=roster, default_channel=args.channel,
                   render=not args.no_clips,
                   force=args.force, log=lambda s: print(s, file=sys.stderr))
    except FileExistsError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 3
    out = Path(args.out).expanduser().resolve()
    print(f"✅ audit {m['audit_id']}: {len(m['items'])} items from {len(inputs)} input(s) → {out}")
    print(f"   pools: {m['allocation']}")
    print(f"   close pair(s): {m['pairs'] or 'none'} ({m['pair_source']})")
    if not args.no_clips:
        problems = check_page(out)
        for p in problems:
            print(f"⚠️  page check: {p}", file=sys.stderr)
        print(f"   open {out / 'index.html'} in a local browser; export answers.json when done")
        return 4 if problems else 0
    return 0


def cmd_score(args) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    answers = load_answers(args.answers[0])
    extra = [load_answers(p) for p in args.answers[1:]]
    out = Path(args.out).expanduser().resolve() if args.out else Path(args.manifest).expanduser().resolve().parent
    assert_outside_vault(out)
    try:
        rep = score(manifest, answers, extra=extra, force=args.force)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 3
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(report_markdown(rep, manifest), encoding="utf-8")
    a = rep["strata"]["all"].get("all")
    if a:
        print(f"✅ {a['judged']} judged: correct {_pct(a['correct'])}, wrong name {_pct(a['wrong_name'])}, "
              f"abstain {_pct(a['abstain'])} → {out / 'report.md'}")
    else:
        print(f"✅ no judged speech items yet → {out / 'report.md'}")
    return 0


def add_cli(sub) -> None:
    """Register `audit sample` / `audit score` on an argparse subparsers object."""
    d = SampleParams()
    p = sub.add_parser("audit", help="Tier-1 listening audit of speaker timelines (sample / score)",
                       description=__doc__.split("\n\n")[1])
    asub = p.add_subparsers(dest="audit_cmd", required=True)

    s = asub.add_parser("sample", help="draw stratified windows, render clips and the answering page")
    s.add_argument("--input", nargs=2, action="append", metavar=("AUDIO", "TIMELINE"),
                   help="audio file and its Timeline (.json, or .rttm); repeatable")
    s.add_argument("--list", type=Path, help="TSV: audio<TAB>timeline[<TAB>channel] per line")
    s.add_argument("--channel", choices=CHANNELS, help="channel tag for inputs without one "
                                                       "(default: Timeline meta 'channel', else unknown)")
    s.add_argument("--out", required=True, help="output dir (private: outside the vault and this clone)")
    s.add_argument("--n", type=int, default=d.n, help=f"items (default {d.n}, about 1 h of listening)")
    s.add_argument("--seed", type=int, default=d.seed)
    s.add_argument("--pair", action="append", metavar="'A|B'",
                   help="close pair to oversample at switches (default: most UNCERTAIN time, else most switches)")
    s.add_argument("--roster", help="comma-separated names to offer besides those in the timelines")
    s.add_argument("--min-turn-s", type=float, default=d.min_turn_s)
    s.add_argument("--max-target-s", type=float, default=d.max_target_s)
    s.add_argument("--context-s", type=float, default=d.context_s)
    s.add_argument("--switch-gap-s", type=float, default=d.switch_gap_s)
    s.add_argument("--gap-min-s", type=float, default=d.gap_min_s)
    s.add_argument("--uncertain-frac", type=float, default=d.uncertain_frac)
    s.add_argument("--switch-frac", type=float, default=d.switch_frac)
    s.add_argument("--gap-frac", type=float, default=d.gap_frac)
    s.add_argument("--bucket-share", help="e.g. '<1s=.35,1-3s=.25,3-10s=.2,>=10s=.2'")
    s.add_argument("--no-clips", action="store_true", help="manifest only (no audio decode, no page)")
    s.add_argument("--force", action="store_true", help="replace an existing manifest in --out")
    s.set_defaults(func=cmd_sample)

    c = asub.add_parser("score", help="score exported answers against the sample manifest")
    c.add_argument("--manifest", required=True, type=Path)
    c.add_argument("--answers", required=True, nargs="+", type=Path,
                   help="answers.json from the page; a second file adds annotator agreement")
    c.add_argument("--out", help="report dir (default: the manifest's dir)")
    c.add_argument("--force", action="store_true", help="score answers exported for another audit id")
    c.set_defaults(func=cmd_score)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="audit")
    add_cli(ap.add_subparsers(dest="cmd", required=True))
    args = ap.parse_args(["audit", *(argv if argv is not None else sys.argv[1:])])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
