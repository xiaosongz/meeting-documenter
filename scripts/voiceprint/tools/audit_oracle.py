#!/usr/bin/env python3
"""
Oracle listener for the audit tool on public data: answer an audit from reference RTTMs, and check
the audit's weighted estimates against the exact population rates (every unit judged by the oracle).

    python scripts/voiceprint/tools/audit_oracle.py --manifest OUT/manifest.json --ref-dir RTTM_DIR \
        [--answers-out OUT/answers.oracle.json] [--check]

Reference file per input: <audio stem>.rttm, else <stem up to the first dot>.rttm (AMI: ES2004a.rttm
for ES2004a.Array1-01.wav). Oracle rule for a target window: reference speech covering < 30 % of it
→ nospeech; two or more speakers each covering ≥ 30 % → overlap; else the speaker with the most
time → name:<speaker> when on the input's roster, else guest. Public benchmark data only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from voiceprint import audit as am  # noqa: E402
from voiceprint import timeline as tlm  # noqa: E402

MIN_SPEECH = 0.3
MIN_OVERLAP = 0.3


def ref_for(inp: dict, ref_dir: Path) -> list[tuple[float, float, str]]:
    stem = Path(inp["audio"]).stem
    for name in (f"{stem}.rttm", f"{stem.split('.')[0]}.rttm"):
        p = ref_dir / name
        if p.is_file():
            tl = tlm.from_rttm(p.read_text(encoding="utf-8"))
            return [(t.start, t.end, t.speaker) for t in tl.turns]
    raise FileNotFoundError(f"no reference RTTM for {stem} in {ref_dir}")


def oracle(ref: list[tuple[float, float, str]], t0: float, t1: float, roster: list[str]) -> str:
    dur = max(t1 - t0, 1e-9)
    per: dict = defaultdict(list)
    for s, e, spk in ref:
        if e > t0 and s < t1:
            per[spk].append((max(s, t0), min(e, t1)))
    cover = {spk: _union(iv) for spk, iv in per.items()}
    speech = _union([iv for ivs in per.values() for iv in ivs])
    if speech / dur < MIN_SPEECH:
        return "nospeech"
    if sum(c / dur >= MIN_OVERLAP for c in cover.values()) >= 2:
        return "overlap"
    top = max(sorted(cover), key=lambda k: cover[k])
    return f"name:{top}" if top in roster else "guest"


def _union(iv: list[tuple[float, float]]) -> float:
    total, end = 0.0, -1e18
    for s, e in sorted(iv):
        if e > end:
            total += e - max(s, end)
            end = e
    return total


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--ref-dir", required=True, type=Path)
    ap.add_argument("--answers-out", type=Path)
    ap.add_argument("--check", action="store_true", help="compare audit estimates with exact population rates")
    args = ap.parse_args(argv)
    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    refs = [ref_for(inp, args.ref_dir) for inp in m["inputs"]]
    answers = {}
    for it in m["items"]:
        inp = m["inputs"][it["input"]]
        answers[it["id"]] = {"answer": oracle(refs[it["input"]], it["target"]["start"], it["target"]["end"],
                                              inp["roster"]), "ms": 0}
    ans = {"schema": 1, "type": "speaker-audit-answers", "audit_id": m["audit_id"], "annotator": "oracle",
           "answers": answers}
    out = args.answers_out or args.manifest.with_name("answers.oracle.json")
    am.assert_outside_vault(out.parent)
    out.write_text(json.dumps(ans, indent=1), encoding="utf-8")
    print(f"wrote {len(answers)} oracle answers → {out}")
    if not args.check:
        return 0

    rep = am.score(m, ans)
    params = am.SampleParams(**m["params"])
    inputs = [am.AuditInput(i["audio"], i["timeline"], i["channel"],
                            am.load_timeline(i["timeline"], uri=Path(i["audio"]).stem), i["roster"])
              for i in m["inputs"]]
    units, _ = am.build_units(inputs, params, [tuple(p) for p in m["pairs"]])
    rows = []
    for u in units:
        t0, t1 = am.target_window(u, params.max_target_s, params.seed)
        a = oracle(refs[u.input], t0, t1, inputs[u.input].roster)
        rows.append((u, am.outcome(u.kind, u.label, u.alts, a)))
    covered = total = 0
    print("\n| stratum | outcome | population % | audit estimate % [95 % CI] | covered |\n|---|---|---|---|---|")
    for dim, key in (("all", lambda u: "all"), ("bucket", lambda u: u.bucket)):
        groups = defaultdict(list)
        for u, o in rows:
            if u.kind != "gap" and o != "excluded":
                groups[key(u)].append(o)
        for k, outs in sorted(groups.items(), key=lambda kv: am._stratum_order(kv[0])):
            est = rep["strata"][dim].get(k)
            if not est:
                continue
            for o in ("correct", "wrong_name", "abstain", "overlap_missed", "false_alarm"):
                pop = outs.count(o) / len(outs)
                e = est[o]
                ok = e["lo"] - 1e-9 <= pop <= e["hi"] + 1e-9
                covered += ok
                total += 1
                print(f"| {k} | {o} | {100 * pop:.1f} | {am._pct(e)} | {'yes' if ok else 'NO'} |")
    gap = [o for u, o in rows if u.kind == "gap" and o != "excluded"]
    if gap:
        pop = gap.count("missed") / len(gap)
        e = rep["gaps"]["missed_speech"]
        ok = e["lo"] - 1e-9 <= pop <= e["hi"] + 1e-9
        covered += ok
        total += 1
        print(f"| gaps | missed_speech | {100 * pop:.1f} | {am._pct(e)} | {'yes' if ok else 'NO'} |")
    print(f"\nCI coverage: {covered}/{total} population rates inside the audit's 95 % CI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
