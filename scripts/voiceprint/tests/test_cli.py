"""End-to-end CLI tests: synthetic WAV 'voices' (harmonic tones) + a fake spectral embedder.

Exercises real ffmpeg decode → clip selection → enroll → identify, with no torch.
Everything lives under pytest's tmp_path; names come from the fictional conftest registry.
"""

import json

import numpy as np
import pytest
from scipy.io import wavfile

import identify_speakers as cli
from voiceprint import audio as au

SR = au.SAMPLE_RATE


class SpectralEmbedder:
    """Toy 'speaker embedding': mean-removed log power in 48 bands up to 2.4 kHz."""
    name, model_id, dim, sample_rate = "fake", "fake-spectral@test", 48, SR

    def embed(self, w):
        spec = np.abs(np.fft.rfft(w[: SR * 2] * np.hanning(min(len(w), SR * 2)))) ** 2
        freqs = np.fft.rfftfreq(min(len(w), SR * 2), 1 / SR)
        edges = np.linspace(50, 2450, self.dim + 1)
        bands = np.array([spec[(freqs >= a) & (freqs < b)].sum() for a, b in zip(edges[:-1], edges[1:])])
        v = np.log(bands + 1e-6)
        v -= v.mean()
        return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)

    def embed_batch(self, waves):
        return np.stack([self.embed(w) for w in waves])


def voice(f0, sec, rng):
    t = np.arange(int(sec * SR)) / SR
    x = sum((0.25 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) for k in range(1, 6))
    return x + 0.002 * rng.normal(size=t.size)


def make_meeting(tmp, key, f0s, turns=6, turn_s=10, seed=0, speaker_map=None):
    """Alternate clusters A, B, … each speaking `turn_s` seconds; returns (wav, sidecar)."""
    rng = np.random.default_rng(seed)
    clusters = [chr(ord("A") + i) for i in range(len(f0s))]
    pieces, utts, t = [], [], 0
    for k in range(turns):
        c = k % len(f0s)
        pieces.append(voice(f0s[c], turn_s, rng))
        utts.append({"speaker": clusters[c], "start": t, "end": t + turn_s * 1000, "text": "…"})
        t += turn_s * 1000
        pieces.append(0.002 * rng.normal(size=SR // 2))           # 0.5 s gap
        t += 500
    wav = tmp / f"{key} Test (rec-0000).wav"
    wavfile.write(wav, SR, (np.concatenate(pieces) * 32767).astype(np.int16))
    side = tmp / f"{key} Test-Transcript.utterances.json"
    side.write_text(json.dumps({"schema": 1, "source": "assemblyai", "transcript_id": None,
                                "audio_duration_ms": t, "speaker_map": speaker_map, "utterances": utts}))
    return wav, side


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "load_embedder", lambda model, device: (SpectralEmbedder(), None))
    store = tmp_path / "store"
    return tmp_path, store


def run(store, *argv):
    """Synthetic harmonic-tone 'voices' are (correctly) not speech to Silero → energy VAD here."""
    argv = list(argv)
    if argv and argv[0] in ("enroll", "identify", "cohort-add"):
        argv += ["--vad", "energy"]
    return cli.main(["--store", str(store), *argv])


def test_identify_noop_on_empty_store(tmp_path):
    wav, side = make_meeting(tmp_path, "2026-01-01-1000", [150, 260], turns=2)
    out = tmp_path / "r.json"
    rc = run(tmp_path / "empty", "identify", "--audio", str(wav), "--utterances", str(side), "--json", str(out))
    assert rc == 0
    assert json.loads(out.read_text())["status"] == "no_voiceprints"
    assert not (tmp_path / "empty").exists()                   # read path never creates the store


def test_identify_noop_when_embedder_unavailable(env):
    tmp, store = env
    wav, side = make_meeting(tmp, "2026-01-01-1000", [150, 260])
    assert run(store, "enroll", "--name", "Alice", "--audio", str(wav), "--utterances", str(side),
               "--cluster", "A") == 0
    cli.load_embedder = lambda m, d: (None, "torch missing")    # monkeypatch fixture restores it
    out = tmp / "r.json"
    rc = run(store, "identify", "--audio", str(wav), "--utterances", str(side), "--json", str(out))
    assert rc == 0 and json.loads(out.read_text())["status"] == "no_embedder"


def test_enroll_identify_inversion_stranger_list_forget(env, capsys):
    tmp, store = env
    A, B, Z = 150, 260, 410                                     # Alice, Bob, stranger f0
    for i, key in enumerate(["2026-01-01-1000", "2026-01-05-1000"]):
        wav, side = make_meeting(tmp, key, [A, B], seed=i)
        assert run(store, "enroll", "--name", "Alice", "--audio", str(wav), "--utterances", str(side),
                   "--cluster", "A", "--channel", "phone") == 0
        assert run(store, "enroll", "--name", "鲍勃", "--audio", str(wav), "--utterances", str(side),
                   "--cluster", "B", "--channel", "phone") == 0
    (store / "calibration.json").write_text(json.dumps(
        {"models": {"fake": {"cosine": {"tau": 0.8, "tau_low": 0.7, "delta": 0.3, "gap": 0.05}}}}))

    # new meeting: A is Bob's voice, B Alice's — but the labels say the opposite (inverted mapping)
    wav, side = make_meeting(tmp, "2026-01-15-1000", [B, A], seed=7,
                             speaker_map={"A": "Alice", "B": "Bob"})
    out = tmp / "r.json"
    assert run(store, "identify", "--audio", str(wav), "--utterances", str(side),
               "--roster", "Alice,Bob", "--json", str(out)) == 0
    r = json.loads(out.read_text())
    assert r["status"] == "ok" and r["calibrated"] and r["score_kind"] == "cosine"
    props = {d["cluster"]: d["proposed"] for d in r["decisions"]}
    assert props == {"A": "Bob Example", "B": "Alice Example"}
    assert r["label_check"]["status"] == "inversion"
    assert "INVERSION" in capsys.readouterr().out

    # a stranger stays Speaker X (open set)
    wav, side = make_meeting(tmp, "2026-01-20-1000", [Z, A], seed=9)
    out2 = tmp / "r2.json"
    assert run(store, "identify", "--audio", str(wav), "--utterances", str(side), "--json", str(out2)) == 0
    r2 = json.loads(out2.read_text())
    props2 = {d["cluster"]: d["proposed"] for d in r2["decisions"]}
    assert props2["A"] is None and props2["B"] == "Alice Example"

    # enroll self-check refuses an inverted label
    rc = run(store, "enroll", "--name", "Bob", "--audio", str(wav), "--utterances", str(side),
             "--cluster", "B")
    assert rc == 3

    assert run(store, "list", "--json", str(tmp / "l.json")) == 0
    people = {p["person"]: p for p in json.loads((tmp / "l.json").read_text())["people"]}
    assert set(people) == {"Alice Example", "Bob Example"} and people["Bob Example"]["meetings"] == 2

    assert run(store, "forget", "--name", "Bob") == 0          # no --yes → dry
    assert run(store, "forget", "--name", "Bob", "--yes") == 0
    assert run(store, "list", "--json", str(tmp / "l2.json")) == 0
    assert [p["person"] for p in json.loads((tmp / "l2.json").read_text())["people"]] == ["Alice Example"]


def test_enroll_refuses_speaker_map_conflict_and_unregistered(env):
    tmp, store = env
    wav, side = make_meeting(tmp, "2026-01-01-1000", [150, 260], speaker_map={"A": "Bob"})
    assert run(store, "enroll", "--name", "Alice", "--audio", str(wav), "--utterances", str(side),
               "--cluster", "A") == 2
    assert run(store, "enroll", "--name", "Nobody Known", "--audio", str(wav), "--utterances", str(side),
               "--cluster", "B") == 2


def test_cli_refuses_vault_store(tmp_path, monkeypatch):
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "notes"))
    rc = cli.main(["--store", str(tmp_path / "notes" / "tmpvp"), "list"])
    assert rc == 2
    wav, side = make_meeting(tmp_path, "2026-01-01-1000", [150, 260], turns=2)
    out = tmp_path / "r.json"
    assert cli.main(["--store", str(tmp_path / "notes" / "vp"), "identify", "--audio", str(wav),
                     "--utterances", str(side), "--json", str(out)]) == 0           # identify: no-op
    assert json.loads(out.read_text())["status"] == "error"
    assert not (tmp_path / "notes").exists()


def test_first_time_enroll_refuses_cluster_matching_enrolled_person(env, capsys):
    """§9b: a mislabelled cluster must not be filed under a NEW name."""
    tmp, store = env
    A, B, Z = 150, 260, 410
    wav, side = make_meeting(tmp, "2026-01-01-1000", [A, B])
    assert run(store, "enroll", "--name", "Alice", "--audio", str(wav), "--utterances", str(side),
               "--cluster", "A") == 0
    # new meeting: cluster A is Alice's voice but is labelled as Carol (not yet enrolled)
    wav2, side2 = make_meeting(tmp, "2026-01-05-1000", [A, Z], seed=5)
    args = ["enroll", "--name", "Carol", "--audio", str(wav2), "--utterances", str(side2), "--cluster", "A"]
    assert run(store, *args) == 3
    err = capsys.readouterr().err
    assert "not enrolled yet" in err and "Alice Example" in err
    assert run(store, *args, "--force") == 0                               # by-ear verified override
    assert "--force" in capsys.readouterr().err
    # a genuinely new voice enrolls without complaint
    assert run(store, "enroll", "--name", "Dave", "--audio", str(wav2), "--utterances", str(side2),
               "--cluster", "B") == 0


def test_track_noop_on_empty_store(tmp_path):
    wav, _ = make_meeting(tmp_path, "2026-01-01-1000", [150, 260], turns=2)
    out = tmp_path / "r.json"
    assert run(tmp_path / "empty", "track", "--audio", str(wav), "--json", str(out)) == 0
    assert json.loads(out.read_text())["status"] == "no_voiceprints"


def test_track_end_to_end_with_oracle_boundaries(env, monkeypatch):
    """enroll two voices from one meeting, then `track` another meeting. The boundary layer is
    replaced by oracle slots (the synthetic tones are not speech to the real models)."""
    from voiceprint import segment as sg
    from voiceprint import timeline as tlm
    tmp, store = env
    for key, seed in (("2026-01-01-1000", 1), ("2026-01-03-1000", 3)):
        wav1, side1 = make_meeting(tmp, key, [150, 260], seed=seed)
        for c, name in (("A", "Alice"), ("B", "Bob")):
            assert run(store, "enroll", "--name", name, "--audio", str(wav1), "--utterances", str(side1),
                       "--cluster", c) == 0
    cal = tmp / "cal.json"
    assert run(store, "track-calibrate", "--json", str(cal)) == 0
    assert json.loads(cal.read_text())["calibration"]["source"].startswith("store-lomo")
    wav2, side2 = make_meeting(tmp, "2026-01-02-1000", [260, 150, 400], turns=6, seed=2)
    utts = json.loads(side2.read_text())["utterances"]

    def oracle(wave, sr=SR, backend="seg3", **kw):
        n = int(np.ceil(len(wave) / sr / 0.01))
        P = np.zeros((n, 3), np.float32)
        for u in utts:
            P[u["start"] // 10:u["end"] // 10, ord(u["speaker"]) - ord("A")] = 0.95
        return sg.LocalSegmentation(np.zeros(1), len(wave) / sr, 0.01, 0.01, P[None], "oracle",
                                    {"duration": len(wave) / sr})
    monkeypatch.setattr(sg, "segment", oracle)
    out, js = tmp / "tl.json", tmp / "r.json"
    assert run(store, "track", "--audio", str(wav2), "--backend", "seg3", "--out", str(out),
               "--json", str(js)) == 0
    res = json.loads(js.read_text())
    assert res["status"] == "ok" and res["excluded_meetings"] == ["2026-01-02-1000"]
    tl = tlm.from_json(out.read_text())
    got = {}
    for t in tl.turns:
        got.setdefault(t.speaker, 0.0)
        got[t.speaker] += t.dur
    assert got.get("Bob Example", 0) > 15 and got.get("Alice Example", 0) > 15   # A=260 Hz, B=150 Hz
    assert any(t.speaker.startswith("GUEST_") for t in tl.turns)                  # 400 Hz is unenrolled
    c_turns = [(21.0, 31.0), (52.5, 62.5)]                                         # voice C (k = 2, 5)
    assert not any(t.kind == "name" and a < (t.start + t.end) / 2 < b for t in tl.turns for a, b in c_turns)
