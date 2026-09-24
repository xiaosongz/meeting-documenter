"""Score system RTTMs against a benchmark split.

    python -m bench.run --dataset ami --split test --system <rttm-dir> --out <dir>
    python -m bench.run --dataset ami --split test --sanity reference --out <dir>

System layout (labels: enrolled speaker IDs or GUEST_* / UNKNOWN / UNCERTAIN(...)):
    <rttm-dir>/<meeting>.rttm               closed roster (or <rttm-dir>/closed/<meeting>.rttm)
    <rttm-dir>/loo/<meeting>/<speaker>.rttm optional: roster without <speaker> (open set)
`--system` may also be a single RTTM file holding every meeting (closed roster).

Writes <out>/report.json and <out>/report.md. `--sanity reference` scores the reference
against itself (IER must be 0); `--sanity shuffled` permutes speaker labels within each
meeting (IER must be > 0).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from . import metrics as M
from .protocol import Meeting, enrollment, load_meetings, rosters
from .rttm import Seg, read_rttm, write_rttm

SUM_KEYS = ("total", "correct", "confusion", "missed detection", "false alarm")


def _load_system(system: Path, meetings: list[Meeting]) -> tuple[dict, dict]:
    """→ (closed {meeting: segs}, loo {meeting: {speaker: segs}})."""
    closed: dict[str, list[Seg]] = {}
    loo: dict[str, dict[str, list[Seg]]] = {}
    if system.is_file():
        allsegs = read_rttm(system)
        for m in meetings:
            closed[m.id] = allsegs.get(m.id, [])
        return closed, loo
    base = system / "closed" if (system / "closed").is_dir() else system
    for m in meetings:
        p = base / f"{m.id}.rttm"
        if not p.is_file():
            raise FileNotFoundError(f"system RTTM missing for {m.id}: {p}")
        closed[m.id] = read_rttm(p).get(m.id, [])
        d = system / "loo" / m.id
        if d.is_dir():
            loo[m.id] = {p.stem: read_rttm(p).get(m.id, []) for p in sorted(d.glob("*.rttm"))}
    return closed, loo


def sanity_system(meetings: list[Meeting], kind: str, seed: int = 0) -> dict[str, list[Seg]]:
    rng = random.Random(seed)
    out = {}
    for m in meetings:
        ref = m.reference()
        if kind == "reference":
            out[m.id] = ref
            continue
        spk = sorted({s.speaker for s in ref})
        perm = spk[:]
        while len(spk) > 1 and any(a == b for a, b in zip(spk, perm)):
            rng.shuffle(perm)                    # derangement: no speaker keeps its label
        mp = dict(zip(spk, perm))
        out[m.id] = [Seg(s.start, s.end, mp[s.speaker]) for s in ref]
    return out


def _pool(acc: dict, comp: dict) -> None:
    for k in SUM_KEYS:
        acc[k] = acc.get(k, 0.0) + comp[k]


def _rate(acc: dict) -> float | None:
    if not acc.get("total"):
        return None
    return (acc["confusion"] + acc["missed detection"] + acc["false alarm"]) / acc["total"]


def score(meetings: list[Meeting], closed: dict, loo: dict, collars: list[float],
          roster_of: dict[str, set[str]], turn_gap: float = 0.5) -> dict:
    report: dict = {"meetings": {}, "pooled": {}}
    for c in collars:
        ck = f"collar_{c:g}"
        ier_acc, der_acc, jer_vals = {}, {}, []
        bd_all = M.Breakdown()
        loo_ier_acc, bd_loo = {}, M.Breakdown()
        for m in meetings:
            ref, uem, hyp = m.reference(), m.uem(), closed[m.id]
            roster = roster_of[m.id]
            h = M.headline(ref, hyp, uem, roster, c, uri=m.id)
            bd = M.breakdown(ref, hyp, uem, roster, c, turn_gap)
            bd_all.add(bd)
            _pool(ier_acc, h["ier_components"])
            _pool(der_acc, h["der_components"])
            if "jer" in h:
                jer_vals.append(h["jer"])
            entry = {"ier": h["ier"], "der": h["der"], "jer": h.get("jer"),
                     "breakdown": M.summarize(bd, top_pairs=5)}
            for spk, segs in loo.get(m.id, {}).items():
                r = roster - {spk}
                hl = M.headline(ref, segs, uem, r, c, uri=m.id)
                _pool(loo_ier_acc, hl["ier_components"])
                b = M.breakdown(ref, segs, uem, r, c, turn_gap)
                bd_loo.add(b)
                entry.setdefault("loo", {})[spk] = {"ier": hl["ier"],
                                                    "false_name_rate": M.summarize(b)["false_name_rate"]}
            report["meetings"].setdefault(m.id, {})[ck] = entry
        pooled = {"ier": _rate(ier_acc), "der": _rate(der_acc),
                  "jer_macro": sum(jer_vals) / len(jer_vals) if jer_vals else None,
                  "ier_components": ier_acc, "breakdown": M.summarize(bd_all)}
        if loo_ier_acc:
            s = M.summarize(bd_loo)
            pooled["loo"] = {"ier": _rate(loo_ier_acc), "false_name_rate": s["false_name_rate"],
                             "unenrolled": s["unenrolled"], "variants": sum(len(v) for v in loo.values())}
        report["pooled"][ck] = pooled
    return report


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}"


def render_md(rep: dict) -> str:
    L = [f"# Speaker-tracking benchmark: {rep['dataset']} {rep['split']}", "",
         f"System: `{rep['system']}` · meetings: {len(rep['meetings'])} · roster: {rep['roster']}"
         f" · collars (± s): {', '.join(f'{c:g}' for c in rep['collars'])} · turn gap {rep['turn_gap']:g} s", ""]
    for ck, p in rep["pooled"].items():
        b = p["breakdown"]
        L += [f"## {ck.replace('_', ' ')} s", "",
              "| IER % | DER % | JER % (macro) | abstain % | named FA s | false-name % (LOO) | LOO IER % |",
              "|---|---|---|---|---|---|---|",
              f"| {_pct(p['ier'])} | {_pct(p['der'])} | {_pct(p['jer_macro'])} | {_pct(b['abstain_rate'])} "
              f"| {b['named_false_alarm_s']:.1f} | {_pct(p.get('loo', {}).get('false_name_rate'))} "
              f"| {_pct(p.get('loo', {}).get('ier'))} |", "",
              "Enrolled reference speech, by reference-turn duration:", "",
              "| bucket | ref s | correct % | wrong name % | abstain % | missed % |", "|---|---|---|---|---|---|"]
        rows = list(b["by_bucket"].items()) + list(b["by_overlap"].items()) + [("all", b["enrolled"])]
        for k, r in rows:
            L.append(f"| {k} | {r['seconds']:.0f} | {_pct(r['correct'])} | {_pct(r['wrong_name'])} "
                     f"| {_pct(r['abstain'])} | {_pct(r['missed'])} |")
        if p.get("loo"):
            u = p["loo"]["unenrolled"]
            L += ["", f"Unenrolled speech (leave-one-out, {p['loo']['variants']} variants): "
                      f"{u['seconds']:.0f} s; abstained {_pct(u['correct'])} %, false name "
                      f"{_pct(u['wrong_name'])} %, missed {_pct(u['missed'])} %."]
        if b["pair_confusion_s"]:
            L += ["", "Top wrong-name pairs (reference → hypothesis, seconds):", ""]
            L += [f"- {x['ref']} → {x['hyp']}: {x['seconds']:.1f}" for x in b["pair_confusion_s"]]
        L += ["", "Per meeting:", "", "| meeting | IER % | DER % | JER % |", "|---|---|---|---|"]
        for mid, e in rep["meetings"].items():
            x = e[ck]
            L.append(f"| {mid} | {_pct(x['ier'])} | {_pct(x['der'])} | {_pct(x['jer'])} |")
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=["ami", "aishell4"])
    ap.add_argument("--split", default="test")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--system", type=Path, help="directory of per-meeting RTTMs, or one RTTM file")
    src.add_argument("--sanity", choices=["reference", "shuffled"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--collars", type=float, nargs="+", default=[0.0, 0.25])
    ap.add_argument("--turn-gap", type=float, default=0.5)
    ap.add_argument("--meetings", nargs="*", help="restrict to these meeting IDs")
    a = ap.parse_args(argv)

    t0 = time.time()
    pool = load_meetings(a.dataset, a.split)
    meetings = [m for m in pool if not a.meetings or m.id in a.meetings]
    if not meetings:
        ap.error(f"no meetings for {a.dataset}/{a.split}")
    roster_of = {m.id: rosters(m, list(enrollment(m, pool)))["closed"] for m in meetings}
    if a.sanity:
        closed, loo = sanity_system(meetings, a.sanity), {}
        sysdir = a.out / "system"
        for mid, segs in closed.items():
            write_rttm(sysdir / f"{mid}.rttm", mid, segs)
        system_name = f"sanity:{a.sanity}"
    else:
        closed, loo = _load_system(a.system, meetings)
        system_name = str(a.system)
    rep = {"dataset": a.dataset, "split": a.split, "system": system_name,
           "roster": "closed + leave-one-out" if loo else "closed",
           "collars": a.collars, "turn_gap": a.turn_gap,
           "rosters": {k: sorted(v) for k, v in roster_of.items()}}
    rep.update(score(meetings, closed, loo, a.collars, roster_of, a.turn_gap))
    rep["elapsed_s"] = round(time.time() - t0, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    (a.out / "report.md").write_text(render_md(rep), encoding="utf-8")
    p = rep["pooled"][f"collar_{a.collars[0]:g}"]
    print(f"{a.dataset}/{a.split} {system_name}: IER {_pct(p['ier'])} % DER {_pct(p['der'])} % "
          f"(collar ±{a.collars[0]:g}) → {a.out / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
