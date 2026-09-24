import json
import os
import stat

import numpy as np
import pytest

from voiceprint import config
from voiceprint import store as st


def rows(person, n, mk="2026-01-02-1000", dim=16, seed=0, dur=3.0):
    rng = np.random.default_rng(seed)
    return [st.Exemplar(person=person, embedding=rng.normal(size=dim).astype(np.float32),
                        channel="phone", meeting_key=mk, cluster="A", start_ms=i * 1000,
                        end_ms=i * 1000 + 3000, net_dur=dur + i * 0.01) for i in range(n)]


def test_round_trip_and_permissions(tmp_path):
    root = tmp_path / "vp"
    s = st.VoiceprintStore.open(root)
    r = rows("Bob Example", 4)
    s.add_exemplars("m1", r)
    got = s.exemplars("m1")
    assert len(got) == 4
    assert np.allclose(got[0].embedding, r[0].embedding)
    assert got[0].channel == "phone" and got[0].source == "confirmed_meeting"
    assert s.exemplars("other-model") == []
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / st.DB_NAME).stat().st_mode) == 0o600
    summ = s.summary()
    assert summ[0]["person"] == "Bob Example" and summ[0]["samples"] == 4 and summ[0]["channels"] == ["phone"]
    assert "embedding" not in json.dumps(summ)


def test_replace_meeting_idempotent_and_exclude(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "vp")
    s.add_exemplars("m1", rows("X", 5))
    res = s.add_exemplars("m1", rows("X", 3))
    assert res["replaced"] == 5 and s.count("m1") == 3
    s.add_exemplars("m1", rows("X", 2, mk="2026-01-03-1000"))
    assert len(s.exemplars("m1", exclude_meetings=["2026-01-02-1000"])) == 2


def test_cap_trims_most_represented_meeting(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "vp")
    s.add_exemplars("m1", rows("X", 8, mk="2026-01-01-0000"), cap=10)
    res = s.add_exemplars("m1", rows("X", 4, mk="2026-02-01-0000"), cap=10)
    assert s.count("m1") == 10 and res["trimmed"] == {"X": 2}
    mks = [e.meeting_key for e in s.exemplars("m1")]
    assert mks.count("2026-02-01-0000") == 4          # small meeting kept whole


def test_forget_removes_exemplars_and_cohort(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "vp")
    s.add_exemplars("m1", rows("X", 3) + rows("Y", 2, seed=1))
    s.add_cohort("m1", [st.CohortClip(label="X", person="X", embedding=np.ones(16)),
                        st.CohortClip(label="2026-01-01-0000:C", embedding=np.ones(16))])
    res = s.forget("X")
    assert res["exemplars"] == 3 and res["cohort"] == 1
    assert {e.person for e in s.exemplars("m1")} == {"Y"}
    vecs, labels = s.cohort("m1")
    assert labels == ["2026-01-01-0000:C"]


def test_missing_store_is_empty_not_created(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "nope", create=False)
    assert not s.exists and s.count() == 0 and s.summary() == [] and s.exemplars("m") == []
    assert not (tmp_path / "nope").exists()


def test_refuses_vault_paths(tmp_path, monkeypatch):
    fake_vault = tmp_path / "vault"
    (fake_vault / ".obsidian").mkdir(parents=True)
    with pytest.raises(st.VaultPathError):
        st.VoiceprintStore.open(fake_vault / "sub" / "vp")
    monkeypatch.setenv("VOICEPRINT_DIR", str(fake_vault / "x"))
    with pytest.raises(st.VaultPathError):
        st.resolve_store_dir()


def test_forbidden_roots_come_from_config(tmp_path, monkeypatch):
    notes, shared, ok = tmp_path / "notes", tmp_path / "shared", tmp_path / "private"
    for d in (notes, shared, ok):
        d.mkdir()
    assert st.resolve_store_dir(notes / "vp") == (notes / "vp").resolve()   # nothing configured yet
    monkeypatch.setenv("VAULT_PATH", str(notes))
    monkeypatch.setenv("VOICEPRINT_FORBIDDEN_ROOTS", os.pathsep.join([str(shared), ""]))
    for bad in (notes, notes / "deep" / "vp", shared / "vp"):
        with pytest.raises(st.VaultPathError):
            st.resolve_store_dir(bad)
    monkeypatch.setenv("MEETING_NOTES_DIR", str(ok / "MeetingNotes"))
    with pytest.raises(st.VaultPathError):
        st.resolve_store_dir(ok / "MeetingNotes" / "vp")
    assert st.resolve_store_dir(ok / "vp") == (ok / "vp").resolve()
    with pytest.raises(st.VaultPathError):                               # never inside this clone
        st.resolve_store_dir(config.REPO_ROOT / "voiceprints")


def test_default_dirs_follow_xdg(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("VOICEPRINT_MODEL_DIR", raising=False)
    assert st.resolve_store_dir() == (tmp_path / "data" / "meeting-documenter" / "voiceprints").resolve()
    assert config.model_dir() == tmp_path / "cache" / "meeting-documenter" / "models"
    monkeypatch.setenv("VOICEPRINT_MODEL_DIR", str(tmp_path / "m"))
    assert config.model_dir() == tmp_path / "m"


def test_bad_source_rejected(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "vp")
    r = rows("X", 1)
    r[0].source = "guess"
    with pytest.raises(ValueError):
        s.add_exemplars("m1", r)
    assert s.count() == 0                                # transaction rolled back


def test_calibration_lookup(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "vp")
    assert s.load_calibration("redimnet2-x@abc", "redimnet2") is None
    s.calibration_path().write_text(json.dumps({"models": {"redimnet2": {"asnorm": {"tau": 5}}}}))
    e = s.load_calibration("redimnet2-x@abc", "redimnet2")
    assert e["asnorm"]["tau"] == 5 and e["_source"].endswith("calibration.json")


def test_save_calibration_merges_atomically_owner_only(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "vp")
    s.calibration_path().write_text(json.dumps({"models": {"other": {"cosine": {"tau": 0.5}}}}))
    path = s.save_calibration("m1", {"cosine": {"tau": 0.7}, "_source": "ignored"})
    data = json.loads(path.read_text())
    assert data["models"]["other"]["cosine"]["tau"] == 0.5
    assert data["models"]["m1"] == {"cosine": {"tau": 0.7}}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in path.parent.iterdir() if p.name.startswith(".calibration-")] == []


def test_backfill_clip_sig_only_null(tmp_path):
    s = st.VoiceprintStore.open(tmp_path / "vp")
    r = rows("X", 3)
    r[0].clip_sig = "keepme"
    s.add_exemplars("m1", r)
    s.add_cohort("m1", [st.CohortClip(label="c", embedding=np.ones(16))])
    assert s.backfill_clip_sig("m1", "abc") == {"exemplars": 2, "cohort": 1}
    assert sorted(e.clip_sig for e in s.exemplars("m1")) == ["abc", "abc", "keepme"]
    assert s.backfill_clip_sig("other", "abc") == {"exemplars": 0, "cohort": 0}
