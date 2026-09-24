"""Listening-audit tests: synthetic timelines, synthetic audio (tones + noise) and synthetic answers."""

import json
import math
import random
import shutil
import wave

import numpy as np
import pytest

import identify_speakers as cli
from voiceprint.tools import audit_oracle as oracle
from voiceprint import audit as am
from voiceprint import config
from voiceprint.timeline import Timeline, Turn, to_json, to_rttm

NAMES = ["Alice Example", "Bob Example", "Carol Example", "Dave Example"]
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def synth_timeline(seed=1, total_s=600.0, meta=None):
    """Alternating turns with a realistic spread of durations, confidences and kinds."""
    rng = random.Random(seed)
    turns, t, prev = [], 0.5, None
    while t < total_s:
        dur = rng.choice([0.4, 0.7, 1.5, 2.5, 4.0, 7.0, 12.0, 20.0]) * rng.uniform(0.8, 1.2)
        r = rng.random()
        if r < 0.08:
            a, b = rng.sample(NAMES[:2], 2)
            turns.append(Turn.from_label(t, t + dur, f"UNCERTAIN({a}|{b})", rng.uniform(0.2, 0.5)))
        elif r < 0.12:
            turns.append(Turn.from_label(t, t + dur, "GUEST_1", rng.uniform(0.4, 0.9)))
        elif r < 0.14:
            turns.append(Turn.from_label(t, t + dur, "UNKNOWN", rng.uniform(0.1, 0.4)))
        else:
            name = rng.choice([n for n in NAMES if n != prev])
            prev = name
            turns.append(Turn.from_label(t, t + dur, name, rng.uniform(0.3, 1.0)))
        t += dur + (rng.choice([0.0, 0.1, 0.3, 2.0]) if rng.random() < 0.9 else 3.0)
    return Timeline(turns, list(turns), dict(meta or {}))


def write_audio(path, dur_s, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * sr)) / sr
    x = 0.05 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.3 * t)) + 0.01 * rng.normal(size=t.size)
    pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(pcm.tobytes())


def inputs_on_disk(tmp_path, n=2, meta=None, audio=False):
    rows = []
    for i in range(n):
        tl = synth_timeline(seed=i + 1, meta=meta)
        tp = tmp_path / f"meeting{i}.timeline.json"
        tp.write_text(to_json(tl), encoding="utf-8")
        ap = tmp_path / f"meeting{i}.wav"
        if audio:
            write_audio(ap, max(t.end for t in tl.turns) + 3, seed=i)
        rows.append(am.AuditInput(str(ap), str(tp)))
    return rows


# ─── Allocation and sampling ────────────────────────────────────────────────

def test_allocate_respects_caps_and_total():
    alloc = am.allocate(10, {"a": 2, "b": 100, "c": 100}, {"a": 5, "b": 1, "c": 1})
    assert alloc == {"a": 2, "b": 4, "c": 4}
    assert sum(am.allocate(7, {"a": 3, "b": 3}, {"a": 1, "b": 1}).values()) == 6   # capped by population
    assert am.allocate(5, {"a": 5, "b": 5}, {"a": 1, "b": 0}) == {"a": 5, "b": 0}


def test_sample_is_deterministic_stratified_and_weighted(tmp_path):
    out = tmp_path.parent / (tmp_path.name + "-out")
    m1 = am.sample(inputs_on_disk(tmp_path), out / "a", am.SampleParams(n=120), render=False)
    m2 = am.sample(inputs_on_disk(tmp_path), out / "b", am.SampleParams(n=120), render=False)
    m3 = am.sample(inputs_on_disk(tmp_path), out / "c", am.SampleParams(n=120, seed=7), render=False)
    key = lambda m: [(it["input"], it["unit"]["start"], it["target"]["start"]) for it in m["items"]]
    assert key(m1) == key(m2) and key(m1) != key(m3)
    assert m1["audit_id"] == m2["audit_id"] != m3["audit_id"]
    items = m1["items"]
    assert len(items) == 120 == sum(m1["allocation"].values())
    assert len({(it["input"], it["unit"]["start"]) for it in items}) == 120          # no duplicates
    sizes = m1["population"]["pool_sizes"]
    for it in items:                                                                  # weight = N_pool / n_pool
        assert it["weight"] == pytest.approx(sizes[it["pool"]] / m1["allocation"][it["pool"]])
    pools = {it["pool"].split(":")[0] for it in items}
    assert pools == {"uncertain", "switch", "gap", "cell"}
    assert m1["pair_source"] == "uncertain_time" and m1["pairs"] == [["Alice Example", "Bob Example"]]
    # short turns are oversampled relative to their share of the population
    speech = [it for it in items if it["unit"]["kind"] != "gap"]
    pop_short = sum(s for p, s in sizes.items() if p.startswith("cell:<1s"))
    pop_cells = sum(s for p, s in sizes.items() if p.startswith("cell:"))
    got_short = sum(1 for it in speech if it["pool"].startswith("cell:<1s"))
    got_cells = sum(1 for it in speech if it["pool"].startswith("cell:"))
    assert got_short / got_cells > pop_short / pop_cells
    assert {it["tercile"] for it in speech} == {"low", "mid", "high"}
    assert {it["bucket"] for it in speech} == set(am.BUCKET_NAMES)
    # presentation order: meetings in blocks, not in time order within a block
    blocks = [it["input"] for it in items]
    assert blocks == sorted(blocks, key=blocks.index)
    starts = [it["unit"]["start"] for it in items if it["input"] == items[0]["input"]]
    assert starts != sorted(starts)


def test_target_windows_stay_inside_their_unit(tmp_path):
    m = am.sample(inputs_on_disk(tmp_path), tmp_path.parent / (tmp_path.name + "-o"),
                  am.SampleParams(n=200, max_target_s=3.0), render=False)
    for it in m["items"]:
        u, t = it["unit"], it["target"]
        assert u["start"] - 1e-6 <= t["start"] < t["end"] <= u["end"] + 1e-6
        assert t["end"] - t["start"] <= 3.0 + 1e-6
        if u["end"] - u["start"] <= 3.0:
            assert (t["start"], t["end"]) == (u["start"], u["end"])
        if it["pool"] == "switch" and u["end"] - u["start"] > 3.0:
            edge = u["switch_edge"]
            assert (t["start"] == u["start"]) if edge == "start" else (t["end"] == u["end"])


def test_switch_pool_and_explicit_pair():
    turns = [Turn.from_label(0, 5, "Alice Example", .9), Turn.from_label(5.5, 12, "Carol Example", .9),
             Turn.from_label(12, 20, "Dave Example", .9), Turn.from_label(20.2, 21, "Carol Example", .9),
             Turn.from_label(25, 30, "Dave Example", .9)]
    inp = am.AuditInput("x.wav", "x.json", "zoom", Timeline(turns, turns, {}))
    units, stats = am.build_units([inp], am.SampleParams(), [("Carol Example", "Dave Example")])
    by_start = {u.start: u for u in units}
    assert by_start[5.5].switch_edge == "end" and by_start[12].switch_edge == "start"
    assert by_start[20.2].switch_edge == "start"     # next Dave is 4 s later: beyond switch_gap_s
    assert by_start[25].switch_edge is None and by_start[0].switch_edge is None
    assert by_start[5.5].pool == "switch" and by_start[0].pool.startswith("cell:3-10s")
    gaps = [u for u in units if u.kind == "gap"]
    assert [(g.start, g.end) for g in gaps] == [(21.0, 25.0)]


def test_auto_pair_falls_back_to_switch_counts():
    turns = [Turn.from_label(i, i + 1, NAMES[i % 2 + 2], .9) for i in range(6)]
    inp = am.AuditInput("x.wav", "x.json", "", Timeline(turns, turns, {}))
    assert am.auto_pair([inp], 1.0) == (("Carol Example", "Dave Example"), "switch_count")


def test_constant_confidence_gives_a_single_tercile(tmp_path):
    tl = synth_timeline()
    rttm = tmp_path / "m.rttm"
    rttm.write_text(to_rttm(Timeline([Turn(t.start, t.end, t.speaker, t.kind, 1.0, t.alts) for t in tl.turns], []),
                            "m"), encoding="utf-8")
    m = am.sample([am.AuditInput(str(tmp_path / "m.wav"), str(rttm))], tmp_path.parent / (tmp_path.name + "-o"),
                  am.SampleParams(n=50), render=False)
    assert m["population"]["tercile_cuts"] is None
    assert {it["tercile"] for it in m["items"] if it["bucket"] != "gap"} == {"all"}


def test_channel_precedence_and_keymaps(tmp_path):
    rows = inputs_on_disk(tmp_path, n=2, meta={"channel": "zoom"})
    rows[1].channel = "phone"
    m = am.sample(rows, tmp_path.parent / (tmp_path.name + "-o"), am.SampleParams(n=40), render=False,
                  default_channel="room", roster=["Visitor Example"])
    assert [i["channel"] for i in m["inputs"]] == ["zoom", "phone"]
    assert m["keymap_mode"] == "shared" and m["inputs"][0]["keys"] == sorted(NAMES + ["Visitor Example"])
    many = [am.AuditInput("a", "a", roster=[f"P{i:02d} Example" for i in range(7)]),
            am.AuditInput("b", "b", roster=[f"P{i:02d} Example" for i in range(5, 12)])]
    mode, maps = am.keymaps(many)
    assert mode == "per_input" and maps[1][0] == "P05 Example"


def test_outputs_are_refused_inside_vaults_and_the_clone(tmp_path):
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    rows = inputs_on_disk(tmp_path)
    with pytest.raises(config.VaultPathError):
        am.sample(rows, vault / "audit", render=False)
    with pytest.raises(config.VaultPathError):
        am.sample(rows, config.REPO_ROOT / "audit-out", render=False)
    assert not (config.REPO_ROOT / "audit-out").exists()


def test_existing_manifest_needs_force(tmp_path):
    out = tmp_path.parent / (tmp_path.name + "-o")
    am.sample(inputs_on_disk(tmp_path), out, am.SampleParams(n=20), render=False)
    with pytest.raises(FileExistsError):
        am.sample(inputs_on_disk(tmp_path), out, am.SampleParams(n=20), render=False)
    am.sample(inputs_on_disk(tmp_path), out, am.SampleParams(n=20), render=False, force=True)


# ─── Clips and page ─────────────────────────────────────────────────────────

def test_render_clip_geometry_tones_and_level():
    sr = am.SR
    rng = np.random.default_rng(0)
    wave_ = (0.002 * rng.normal(size=sr * 20)).astype(np.float32)          # quiet recording
    pcm, g = am.render_clip(wave_, 100.0, 105.0, 106.5, 2.0)             # wave starts at t = 100 s
    x = pcm.astype(np.float32) / 32767
    assert g["context_start"] == 103.0
    assert g["clip_dur"] == pytest.approx(2.0 + 1.5 + 2.0 + 2 * am.TONE_S, abs=1e-3)
    assert g["t0"] == pytest.approx(2.0 + am.TONE_S, abs=1e-3) and g["t1"] == pytest.approx(g["t0"] + 1.5, abs=1e-3)
    speech = np.concatenate([x[: int(2.0 * sr)], x[int(g["t0"] * sr): int(g["t1"] * sr)]])
    assert am.gated_level_db(speech) == pytest.approx(am.TARGET_DB, abs=1.0)
    assert np.max(np.abs(x)) <= 10 ** (am.PEAK_DB / 20) + 1e-3
    tone = x[int(2.0 * sr): int((2.0 + am.TONE_S) * sr)]
    spec = np.abs(np.fft.rfft(tone))
    peak_hz = np.fft.rfftfreq(tone.size, 1 / sr)[np.argmax(spec)]
    assert 600 < peak_hz < 1050
    # near the file start the context is clipped, and the geometry says so
    _, g0 = am.render_clip(wave_, 100.0, 100.5, 101.0, 2.0)
    assert g0["context_start"] == 100.0 and g0["t0"] == pytest.approx(0.5 + am.TONE_S, abs=1e-3)


@needs_ffmpeg
def test_sample_renders_clips_and_a_self_contained_page(tmp_path):
    out = tmp_path.parent / (tmp_path.name + "-o")
    m = am.sample(inputs_on_disk(tmp_path, audio=True), out, am.SampleParams(n=30))
    assert am.check_page(out) == []
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "__AUDIT_DATA__" not in html and m["audit_id"] in html
    for it in m["items"]:
        with wave.open(str(out / it["clip"]["file"])) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
            assert w.getnframes() / 16000 == pytest.approx(it["clip"]["clip_dur"], abs=1e-3)
        assert 0 < it["clip"]["t0"] < it["clip"]["t1"] <= it["clip"]["clip_dur"]
    assert m["render"]["decoded_audio_s"] > 0


def test_check_page_flags_network_references(tmp_path):
    out = tmp_path.parent / (tmp_path.name + "-o")
    m = am.sample(inputs_on_disk(tmp_path), out, am.SampleParams(n=5), render=False)
    for it in m["items"]:
        it["clip"] = {"file": f"clips/{it['id']}.wav", "clip_dur": 1, "t0": 0.1, "t1": 0.5}
        (out / "clips").mkdir(exist_ok=True)
        (out / it["clip"]["file"]).write_bytes(b"")
    (out / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    page = am.write_page(m, out)
    assert am.check_page(out) == []
    page.write_text(page.read_text().replace("</head>", '<script src="https://cdn.example/x.js"></script></head>'))
    assert am.check_page(out)


def test_page_template_is_offline():
    html = am.PAGE_TEMPLATE.read_text(encoding="utf-8")
    assert html.count(am.DATA_MARK) == 1
    assert "http://" not in html and "https://" not in html


# ─── Scoring ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind,label,alts,answer,expected", [
    ("name", "Alice Example", [], "name:Alice Example", "correct"),
    ("name", "Alice Example", [], "name:Bob Example", "wrong_name"),
    ("name", "Alice Example", [], "guest", "wrong_name"),
    ("name", "Alice Example", [], "overlap", "overlap_missed"),
    ("name", "Alice Example", [], "nospeech", "false_alarm"),
    ("name", "Alice Example", [], "unsure", "excluded"),
    ("guest", "GUEST_1", [], "guest", "correct"),
    ("guest", "GUEST_1", [], "name:Bob Example", "abstain"),
    ("unknown", "UNKNOWN", [], "guest", "correct"),
    ("uncertain", "UNCERTAIN(Alice Example|Bob Example)", ["Alice Example", "Bob Example"], "name:Carol Example",
     "abstain"),
    ("overlap", "OVERLAP(Alice Example|Bob Example)", ["Alice Example", "Bob Example"], "overlap", "correct"),
    ("overlap", "OVERLAP(Alice Example|Bob Example)", ["Alice Example", "Bob Example"], "name:Bob Example", "abstain"),
    ("overlap", "OVERLAP(Alice Example|Bob Example)", ["Alice Example", "Bob Example"], "name:Carol Example",
     "wrong_name"),
    ("gap", "", [], "nospeech", "correct"),
    ("gap", "", [], "name:Carol Example", "missed"),
])
def test_outcome_table(kind, label, alts, answer, expected):
    assert am.outcome(kind, label, alts, answer) == expected


def test_wilson_and_weighted_rate():
    lo, hi = am.wilson(0.8, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
    assert am.wilson(0.0, 20)[0] == 0.0 and am.wilson(1.0, 20)[1] == 1.0
    r = am.weighted_rate([(1.0, True)] * 8 + [(1.0, False)] * 2)
    assert r["p"] == pytest.approx(0.8) and r["n_eff"] == 10
    r = am.weighted_rate([(3.0, True), (1.0, False), (1.0, False), (1.0, False)])
    assert r["p"] == pytest.approx(0.5) and r["n_eff"] == pytest.approx(36 / 12)


def oracle_answers(manifest, truth, rng, unsure=0.0, ms=4000):
    """Answer every item with `truth(item)`; a share of them 'unsure'."""
    ans = {}
    for it in manifest["items"]:
        a = "unsure" if rng.random() < unsure else truth(it)
        ans[it["id"]] = {"answer": a, "time": "2026-01-01T00:00:00Z", "ms": ms}
    return {"schema": 1, "type": "speaker-audit-answers", "audit_id": manifest["audit_id"], "answers": ans}


def test_score_end_to_end(tmp_path):
    out = tmp_path.parent / (tmp_path.name + "-o")
    m = am.sample(inputs_on_disk(tmp_path), out, am.SampleParams(n=150), render=False)
    rng = random.Random(3)

    def truth(it):                     # the system is right on named turns except Bob → heard as Alice
        u = it["unit"]
        if u["kind"] == "gap":
            return "nospeech"
        if u["kind"] == "name":
            return "name:Alice Example" if u["label"] == "Bob Example" else "name:" + u["label"]
        if u["kind"] == "uncertain":
            return "name:" + u["alts"][0]
        return "guest"
    answers = oracle_answers(m, truth, rng, unsure=0.05)
    rep = am.score(m, answers)
    allr = rep["strata"]["all"]["all"]
    bob = [it for it in m["items"] if it["unit"]["label"] == "Bob Example"]
    assert allr["judged"] < allr["items"] == 150 - sum(1 for it in m["items"] if it["unit"]["kind"] == "gap")
    assert 0 < allr["wrong_name"]["p"] < 0.5 and allr["wrong_name"]["lo"] <= allr["wrong_name"]["p"]
    assert rep["uncertain"]["hit"]["p"] == 1.0
    assert rep["gaps"]["missed_speech"]["p"] == 0.0
    assert rep["top_wrong_names"][0]["truth"] == "Alice Example" and rep["top_wrong_names"][0]["system"] == "Bob Example"
    assert rep["confusion"]["Alice Example"]["Bob Example"] == sum(
        1 for it in bob if answers["answers"][it["id"]]["answer"] != "unsure")
    pair = rep["pairs"][0]
    assert pair["Alice Example->Bob Example"] > 0 and pair["Bob Example->Alice Example"] == 0
    assert set(rep["strata"]["bucket"]) == set(am.BUCKET_NAMES)
    assert rep["listener"]["median_s_per_item"] == 4.0
    md = am.report_markdown(rep, m)
    assert "By system turn duration" in md and "Wilson" in md
    # a wrong audit id is refused unless forced; a second annotator adds agreement
    bad = dict(answers, audit_id="000000000000")
    with pytest.raises(ValueError):
        am.score(m, bad)
    second = oracle_answers(m, truth, random.Random(9), unsure=0.05)
    rep2 = am.score(m, answers, extra=[second])
    assert rep2["agreement"][0]["common_items"] == 150 and rep2["agreement"][0]["kappa"] > 0.8


def test_design_weights_recover_the_population_rate(tmp_path):
    """Oversampled pools must not bias the overall estimate: the weighted rate tracks the population rate."""
    out = tmp_path.parent / (tmp_path.name + "-o")
    rows = inputs_on_disk(tmp_path, n=3)
    m = am.sample(rows, out, am.SampleParams(n=250), render=False)
    wrong = lambda u: u["kind"] == "name" and (u["end"] - u["start"] < 1.0 or u["label"] == "Bob Example")

    def truth(it):
        u = it["unit"]
        if u["kind"] != "name":
            return "unsure"
        return "name:Carol Example" if wrong(u) and u["label"] != "Carol Example" else (
            "name:Dave Example" if wrong(u) else "name:" + u["label"])
    rep = am.score(m, oracle_answers(m, truth, random.Random(0)))
    units, _ = am.build_units(rows, am.SampleParams(n=250), [tuple(p) for p in m["pairs"]])
    named = [u for u in units if u.kind == "name"]
    pop = sum(wrong({"kind": u.kind, "start": u.start, "end": u.end, "label": u.label}) for u in named) / len(named)
    est = rep["strata"]["all"]["all"]["wrong_name"]
    unweighted = sum(1 for it in m["items"] if wrong(it["unit"])) / sum(
        1 for it in m["items"] if it["unit"]["kind"] == "name")
    assert est["lo"] <= pop <= est["hi"]
    assert abs(est["p"] - pop) < abs(unweighted - pop)


# ─── CLI ────────────────────────────────────────────────────────────────────

def test_cli_sample_and_score(tmp_path, capsys):
    rows = inputs_on_disk(tmp_path, audio=shutil.which("ffmpeg") is not None)
    out = tmp_path.parent / (tmp_path.name + "-cli")
    argv = ["audit", "sample", "--out", str(out), "--n", "25", "--seed", "3", "--pair", "Carol Example|Dave Example"]
    for r in rows:
        argv += ["--input", r.audio, r.timeline]
    if shutil.which("ffmpeg") is None:
        argv.append("--no-clips")
    assert cli.main(argv) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["pairs"] == [["Carol Example", "Dave Example"]] and m["pair_source"] == "cli"
    assert cli.main(argv) == 3                                      # refuses to clobber an audit
    answers = oracle_answers(m, lambda it: "nospeech" if it["unit"]["kind"] == "gap" else "guest", random.Random(1))
    ap = tmp_path / "answers.json"
    ap.write_text(json.dumps(answers))
    assert cli.main(["audit", "score", "--manifest", str(out / "manifest.json"), "--answers", str(ap)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["audit_id"] == m["audit_id"] and (out / "report.md").exists()
    assert cli.main(["audit", "sample", "--out", str(tmp_path / "x")]) == 2          # no inputs
    vault = tmp_path / "v"
    (vault / ".obsidian").mkdir(parents=True)
    assert cli.main(["audit", "score", "--manifest", str(out / "manifest.json"), "--answers", str(ap),
                     "--out", str(vault / "r")]) == 2


def test_oracle_listener_rule():
    ref = [(0.0, 2.0, "A"), (1.5, 4.0, "B"), (6.0, 7.0, "C")]
    assert oracle.oracle(ref, 0.0, 1.4, ["A", "B"]) == "name:A"
    assert oracle.oracle(ref, 1.0, 3.0, ["A", "B"]) == "overlap"          # A 1.0 s, B 1.5 s of 2 s
    assert oracle.oracle(ref, 3.0, 4.5, ["A", "B"]) == "name:B"
    assert oracle.oracle(ref, 4.2, 5.8, ["A", "B"]) == "nospeech"
    assert oracle.oracle(ref, 6.0, 7.0, ["A", "B"]) == "guest"             # C is not on the roster
