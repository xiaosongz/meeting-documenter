"""Clip config from calibration.json: precedence, CPU-cap override, enroll mismatch guard, clip_sig."""

import argparse
import json
import sqlite3

import numpy as np
import pytest

import identify_speakers as cli
from voiceprint import audio as au
from voiceprint import store as st
from voiceprint.tests.test_cli import SpectralEmbedder, make_meeting, run
from voiceprint.tests.test_device import RecordingEmbedder, _monologue

CAL_CLIP = {"clip_mode": "longest_run", "hard_cap_s": 60, "quant_s": 0.5, "min_speech_frac": 0.6,
            "vad": "silero"}


def ns(**kw):
    base = dict(clip_mode=None, window_s=None, max_clip_s=None, hard_cap_s=None, quant_s=None,
                max_clips=None, min_net=None, vad=None, min_speech_frac=None, score_mode=None)
    return argparse.Namespace(**{**base, **kw})


# ─── precedence ─────────────────────────────────────────────────────────────

def test_defaults_without_calibration():
    p, src = cli.resolve_clip_config(ns(), au.IDENTIFY_PARAMS, None)
    assert p == au.IDENTIFY_PARAMS and src == {}
    summ = cli.clip_summary(p, "cpu", src)
    assert not summ["from_calibration"] and summ["effective_cap_s"] == 12.0


def test_calibration_beats_defaults():
    p, src = cli.resolve_clip_config(ns(), au.IDENTIFY_PARAMS, {"clip": CAL_CLIP})
    assert (p.clip_mode, p.hard_cap_s, p.quant_s, p.neural_min_frac, p.neural_vad) == \
        ("longest_run", 60, 0.5, 0.6, "silero")
    assert src["clip_mode"] == "calibration" and src["neural_vad"] == "calibration"
    assert cli.clip_summary(p, "mps", src)["from_calibration"]
    assert cli.clip_config_mismatch(p, {"clip": CAL_CLIP}) == []


def test_cli_beats_calibration():
    p, src = cli.resolve_clip_config(ns(clip_mode="window", window_s=20, hard_cap_s=30),
                                     au.IDENTIFY_PARAMS, {"clip": CAL_CLIP})
    assert p.clip_mode == "window" and p.window_s == 20 and p.window_hop_s == 10 and p.max_clip_s == 20
    assert p.hard_cap_s == 30 and src["clip_mode"] == "cli" and src["hard_cap_s"] == "cli"
    assert p.quant_s == 0.5 and src["quant_s"] == "calibration"          # untouched keys keep calibration
    mism = cli.clip_config_mismatch(p, {"clip": CAL_CLIP})
    assert any(m.startswith("clip_mode") for m in mism) and any(m.startswith("hard_cap_s") for m in mism)


def test_score_mode_precedence():
    assert cli.resolve_score_mode(ns(), None) == ("centroid", "default")
    assert cli.resolve_score_mode(ns(), {"score_mode": "top5"}) == ("top5", "calibration")
    assert cli.resolve_score_mode(ns(score_mode="centroid"), {"score_mode": "top5"}) == ("centroid", "cli")


def test_signature_is_device_independent_when_cap_is_explicit():
    p, src = cli.resolve_clip_config(ns(), au.IDENTIFY_PARAMS, {"clip": CAL_CLIP})
    assert cli.clip_signature(cli.clip_summary(p, "cpu", src)) == cli.clip_signature(cli.clip_summary(p, "mps", src))
    q, s2 = cli.resolve_clip_config(ns(), au.IDENTIFY_PARAMS, None)          # device default cap
    assert cli.clip_signature(cli.clip_summary(q, "cpu", s2)) != cli.clip_signature(cli.clip_summary(q, "mps", s2))


# ─── CPU keeps the calibrated 60 s cap ──────────────────────────────────────

def test_calibrated_cap_not_shortened_on_cpu(capsys):
    p, src = cli.resolve_clip_config(ns(vad="energy"), au.IDENTIFY_PARAMS, {"clip": CAL_CLIP})
    assert au.clip_cap_s(au.ClipParams(**{**p.__dict__, "device": "cpu"})) == 60.0
    msg = cli.note_cap_choice(p, "cpu", src)
    assert msg and "60 s from calibration kept on cpu" in msg and "6.7 GB" in msg
    assert "6.7 GB" in capsys.readouterr().err
    assert cli.note_cap_choice(p, "mps", src) is None                     # within the MPS default
    wave, u = _monologue(55)
    emb = RecordingEmbedder("cpu")
    au.embed_meeting(emb, None, u, p, wave=wave)
    assert emb.max_len > 50 * au.SAMPLE_RATE                             # ~54.5 s clip on CPU


# ─── CLI: identify reports provenance; enroll guard; clip_sig ───────────────

@pytest.fixture
def cal_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_embedder", lambda model, device: (SpectralEmbedder(), None))
    store = tmp_path / "store"
    st.VoiceprintStore.open(store).close()
    # tone "voices" aren't speech to Silero → the calibrated config here uses energy VAD
    (store / "calibration.json").write_text(json.dumps({"models": {"fake": {
        "clip": {**CAL_CLIP, "vad": "energy"}, "score_mode": "centroid", "params_hash": "abc123",
        "cosine": {"tau": 0.8, "tau_low": 0.7, "delta": 0.3, "gap": 0.05}}}}))
    return tmp_path, store


def test_enroll_guard_and_identify_provenance(cal_env, capsys):
    tmp, store = cal_env
    wav, side = make_meeting(tmp, "2026-01-01-1000", [150, 260])
    common = ["--audio", str(wav), "--utterances", str(side)]
    assert run(store, "enroll", "--name", "Alice", "--cluster", "A", *common, "--clip-mode", "window") == 4
    assert "differs from calibration" in capsys.readouterr().err
    assert run(store, "enroll", "--name", "Alice", "--cluster", "A", *common, "--hard-cap-s", "12") == 4
    assert run(store, "enroll", "--name", "Alice", "--cluster", "A", *common,
               "--clip-mode", "window", "--force") == 0                 # loud warning, allowed
    assert "--force" in capsys.readouterr().err
    assert run(store, "enroll", "--name", "Alice", "--cluster", "A", *common) == 0   # matches → ok
    assert run(store, "enroll", "--name", "Bob", "--cluster", "B", *common) == 0
    s = st.VoiceprintStore.open(store)
    sigs = {e.clip_sig for e in s.exemplars("fake-spectral@test")}
    assert len(sigs) == 1 and None not in sigs                          # the matching enroll replaced the forced one

    wav2, side2 = make_meeting(tmp, "2026-01-05-1000", [260, 150], seed=3)
    out = tmp / "r.json"
    assert run(store, "identify", "--audio", str(wav2), "--utterances", str(side2), "--json", str(out)) == 0
    r = json.loads(out.read_text())
    cc = r["clip_config"]
    assert cc["from_calibration"] and cc["clip_mode"] == "longest_run" and cc["effective_cap_s"] == 60
    assert cc["source"]["clip_mode"] == "calibration" and cc["signature"] in sigs
    assert r["score_mode"] == "centroid" and r["score_mode_source"] == "calibration"
    assert r["calibration_meta"]["params_hash"] == "abc123"
    assert not any("different (or unrecorded) clip config" in w for w in r["warnings"])
    assert "from calibration" in capsys.readouterr().out
    # a CLI override at identify time is reported, not refused
    out2 = tmp / "r2.json"
    assert run(store, "identify", "--audio", str(wav2), "--utterances", str(side2), "--clip-mode", "window",
               "--json", str(out2)) == 0
    r2 = json.loads(out2.read_text())
    assert r2["clip_config"]["source"]["clip_mode"] == "cli"
    assert any("differs from calibration" in w for w in r2["warnings"])
    assert any("different (or unrecorded) clip config" in w for w in r2["warnings"])


def test_old_store_migrates_and_unrecorded_sig_warns(tmp_path):
    db_dir = tmp_path / "old"
    db_dir.mkdir()
    con = sqlite3.connect(db_dir / st.DB_NAME)
    con.executescript(st._SCHEMA)                                       # pre-clip_sig schema
    con.execute("INSERT INTO exemplars (model_id, person, source, enrolled_date, dim, embedding)"
                " VALUES ('m', 'X', 'confirmed_meeting', '2026-01-01', 4, ?)",
                (np.ones(4, np.float32).tobytes(),))
    con.commit()
    con.close()
    s = st.VoiceprintStore.open(db_dir, create=False)
    ex = s.exemplars("m")
    assert len(ex) == 1 and ex[0].clip_sig is None


def test_flat_calibration_keys_and_identify_only_max_clips():
    """Older calibration files: clip settings flat, ClipParams field names, max_clips."""
    flat = {"clip_mode": "longest_run", "hard_cap_s": 60, "quant_s": 0.5, "neural_vad": "silero",
            "neural_min_frac": 0.6, "max_clips": 20, "score_mode": "centroid", "cohort": "clip"}
    p, src = cli.resolve_clip_config(ns(), au.IDENTIFY_PARAMS, flat)
    assert (p.clip_mode, p.hard_cap_s, p.neural_vad, p.max_clips) == ("longest_run", 60, "silero", 20)
    e, esrc = cli.resolve_clip_config(ns(), au.ENROLL_PARAMS, flat, identify=False)
    assert e.clip_mode == "longest_run" and e.max_clips == au.ENROLL_PARAMS.max_clips   # enroll keeps 60
    assert cli.clip_config_mismatch(e, flat) == []                       # max_clips not compared
    both = {**flat, "clip": {"clip_mode": "window"}}                     # block wins over flat keys
    assert cli.resolve_clip_config(ns(), au.IDENTIFY_PARAMS, both)[0].clip_mode == "window"
