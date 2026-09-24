"""Voiceprint hooks in the transcription workflow; synthetic data and mocked providers only.

The voiceprint package's own tests live in scripts/voiceprint/tests (pytest).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock

import test_pipeline as support

pipeline = support.pipeline
ROOT = support.ROOT
CLI = ROOT / "scripts" / "identify_speakers.py"


class UtterancesSidecarTests(unittest.TestCase):
    def case(self):
        case = support.AssemblyAIIntegrationTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_assemblyai_run_writes_sidecar_with_exact_timestamps_and_no_names(self):
        case = self.case()
        case.poll_result["audio_duration"] = 3.2
        case.run_main()
        sidecar = pipeline.utterances_sidecar_path(case.output)
        self.assertEqual(sidecar.name, "transcript.utterances.json")
        data = json.loads(sidecar.read_text())
        self.assertEqual(data["schema"], 1)
        self.assertEqual(data["source"], "assemblyai")
        self.assertEqual(data["transcript_id"], "synthetic-job")
        self.assertEqual(data["audio_duration_ms"], 3200)
        self.assertIsNone(data["speaker_map"])
        self.assertEqual(data["utterances"][1],
                         {"speaker": "B", "start": 1000, "end": 3000, "text": "Synthetic response."})

    def test_existing_sidecar_fails_before_upload_and_is_preserved(self):
        case = self.case()
        case.output.parent.mkdir()
        sidecar = pipeline.utterances_sidecar_path(case.output)
        sidecar.write_text("previous sidecar")
        with self.assertRaises(FileExistsError):
            case.run_main()
        case.post.assert_not_called()
        self.assertEqual(sidecar.read_text(), "previous sidecar")
        self.assertFalse(case.output.exists())

    def test_sidecar_created_during_provider_work_does_not_fail_transcript(self):
        case = self.case()
        sidecar = pipeline.utterances_sidecar_path(case.output)

        def poll(*args, **kwargs):
            case.output.parent.mkdir(exist_ok=True)
            sidecar.write_text("concurrent sidecar")
            return support.response(case.poll_result)
        case.get.side_effect = poll
        case.run_main()
        self.assertTrue(case.output.exists())
        self.assertEqual(sidecar.read_text(), "concurrent sidecar")
        self.assertIn("Utterances sidecar not written", case.stdout.getvalue())


# Stands in for scripts/identify_speakers.py: answers `track-check` from stub.json and, for
# `track`, writes the configured timeline and status where the pipeline asked for them.
STUB_TRACK_CLI = r"""
import json, sys, time
from pathlib import Path
here = Path(__file__).parent
cfg = json.loads((here / "stub.json").read_text())
with (here / "calls.jsonl").open("a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
cmd, argv = sys.argv[1], sys.argv[2:]
opt = lambda k: argv[argv.index(k) + 1] if k in argv else None
if cmd == "track-check":
    print("progress noise before the JSON line")
    print(json.dumps(cfg["check"]))
elif cmd == "track":
    time.sleep(cfg.get("sleep", 0))
    if cfg.get("timeline"):
        Path(opt("--out")).write_text(cfg["timeline"])
    Path(opt("--json")).write_text(json.dumps(cfg["status"]))
"""

READY = {"ready": True, "status": "ready", "reason": "", "backend": "nemotron3",
         "model_id": "fake-model@test", "calibration": "store-lomo-test",
         "roster_enrolled": ["Alice Example", "Bob Example"], "roster_missing": []}


def synthetic_timeline(turns, **meta):
    tlm = pipeline._voiceprint_module("timeline")
    return tlm.to_json(tlm.Timeline.from_turns(
        [tlm.Turn.from_label(a, b, label, conf) for a, b, label, conf in turns],
        {"backend": "nemotron3", "model_id": "fake-model@test",
         "calibration": {"source": "store-lomo-test"}, **meta}))


class LocalSpeakersTests(unittest.TestCase):
    """--local-speakers / MEETING_DOCUMENTER_LOCAL_SPEAKERS: AssemblyAI words only, speakers from a
    local tracker (a stub CLI here), and cloud diarization as the fallback."""

    WORDS = [
        {"text": "Synthetic", "start": 0, "end": 400, "confidence": 0.98, "speaker": None},
        {"text": "opening.", "start": 400, "end": 900, "confidence": 0.95, "speaker": None},
        {"text": "Synthetic", "start": 4000, "end": 4400, "confidence": 0.97, "speaker": None},
        {"text": "response.", "start": 4400, "end": 5000, "confidence": 0.93, "speaker": None},
        {"text": "Guest", "start": 6000, "end": 6400, "confidence": 0.9, "speaker": None},
        {"text": "remark.", "start": 6400, "end": 7000, "confidence": 0.9, "speaker": None},
        {"text": "Unclear", "start": 8000, "end": 8500, "confidence": 0.9, "speaker": None},
        {"text": "voice.", "start": 8500, "end": 9000, "confidence": 0.9, "speaker": None},
    ]
    TURNS = [(0.0, 1.2, "Alice Example", 0.97), (3.8, 5.3, "Bob Example", 0.95),
             (5.8, 7.2, "GUEST_1", 0.8), (7.9, 9.1, "UNCERTAIN(Alice Example|Bob Example)", 0.5)]

    def base_case(self):
        case = support.AssemblyAIIntegrationTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        self.stub_dir = case.directory / "stub"
        self.stub_dir.mkdir()
        stub = self.stub_dir / "identify_speakers.py"
        stub.write_text(STUB_TRACK_CLI)
        self.timelines = case.directory / "private-timelines"
        case.stack.enter_context(unittest.mock.patch.object(pipeline, "TRACK_CLI", stub))
        case.stack.enter_context(unittest.mock.patch.dict(os.environ, {
            pipeline.TIMELINE_DIR_ENV: str(self.timelines), pipeline.LOCAL_SPEAKERS_ENV: ""}))
        self.stub(READY, timeline=synthetic_timeline(self.TURNS))
        return case

    def stub(self, check, timeline=None, status=None, sleep=0):
        status = status or {"status": "ok", "info": {"xrt": 99.0, "peak_footprint_gb": 1.0}}
        (self.stub_dir / "stub.json").write_text(json.dumps(
            {"check": check, "timeline": timeline, "status": status, "sleep": sleep}))

    def calls(self):
        path = self.stub_dir / "calls.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

    def case(self, words=None):
        case = self.base_case()
        case.poll_result.pop("utterances")
        case.poll_result["words"] = list(self.WORDS if words is None else words)
        case.poll_result["audio_duration"] = 9.2
        return case

    def timeline_path(self, case):
        return self.timelines / (case.output.stem + ".timeline.json")

    def check_words_only_request(self, case):
        submitted = case.posts[1][1]["json"]
        self.assertIs(submitted["speaker_labels"], False)
        self.assertNotIn("speakers_expected", submitted)
        self.assertFalse(pipeline.utterances_sidecar_path(case.output).exists())
        sidecar = pipeline.words_sidecar_path(case.output)
        self.assertEqual(sidecar.name, "transcript.words.json")
        data = json.loads(sidecar.read_text())
        self.assertEqual((data["schema"], data["type"], data["time_unit"]), (1, "words", "ms"))
        self.assertEqual(data["transcript_id"], "synthetic-job")
        self.assertEqual(data["audio_duration_ms"], 9200)
        self.assertEqual(data["words"][1], {"text": "opening.", "start": 400, "end": 900, "confidence": 0.95})
        self.assertNotIn("speaker", sidecar.read_text())

    def check_tracked_run(self, case):
        self.check_words_only_request(case)
        text = case.output.read_text()
        body = text.split("\n---\n\n")[-1]
        self.assertEqual(body, "**[00:00] Alice Example:** Synthetic opening.\n\n"
                               "**[00:04] Bob Example:** Synthetic response.\n\n"
                               "**[00:06] Guest 1:** Guest remark.\n\n"
                               "**[00:08] Alice Example or Bob Example (uncertain):** Unclear voice.\n")
        self.assertIn("words only, no cloud diarization) + local speaker tracking", text)
        self.assertIn("**Local Speaker Tracking:** `nemotron3` boundaries · `fake-model@test` voiceprints · "
                      "calibration `store-lomo-test` · timeline `transcript.timeline.json`", text)
        self.assertIn("4 turns · Bob Example 2 s, Guest 1 1 s, Alice Example 1 s, "
                      "Alice Example or Bob Example (uncertain) 1 s", text)   # most speech first
        self.assertIn("> [!warning]- Speaker review: 2 turn(s) to spot-check\n"
                      "> - [00:06] Guest 1, 1.0 s\n"
                      "> - [00:08] Alice Example or Bob Example (uncertain), 1.0 s\n", text)
        self.assertNotIn("Speaker A", text)
        self.assertEqual(pipeline.extract_last_timestamp(text), 8)
        timeline = self.timeline_path(case)
        self.assertEqual(timeline.stat().st_mode & 0o777, 0o600)
        tl = pipeline._voiceprint_module("timeline").from_json(timeline.read_text())
        self.assertEqual(tl.meta["transcript"], "transcript.md")
        self.assertEqual(tl.meta["words"], "transcript.words.json")
        self.assertEqual(tl.meta["audio"], case.source.name)
        self.assertEqual([t.speaker for t in tl.turns][:2], ["Alice Example", "Bob Example"])
        return text

    def test_cli_flag_tracks_speakers_locally_end_to_end(self):
        case = self.case()
        context = case.directory / "context.txt"
        context.write_text("Date: 2026-01-01\nPossible speakers:\n- Alice Example (host)\n"
                           "- Bob Example\n\nDomain terms: synthetic\n")
        case.run_main("--local-speakers", "--speakers-expected", "2", "--context-file", str(context),
                      "--channel", "zoom")
        self.check_tracked_run(case)
        self.assertIn("omitting speakers_expected", case.stdout.getvalue())
        check, track = self.calls()
        self.assertEqual(check, ["track-check", "--backend", "nemotron3",
                                 "--roster", "Alice Example,Bob Example"])
        self.assertEqual(track[:2], ["track", "--audio"])
        self.assertEqual(track[track.index("--roster") + 1], "Alice Example,Bob Example")
        self.assertEqual(track[track.index("--channel") + 1], "zoom")
        self.assertNotIn("--meeting-key", track)          # no YYYY-MM-DD-HHMM in the file names
        tl = pipeline._voiceprint_module("timeline").from_json(self.timeline_path(case).read_text())
        self.assertEqual(tl.meta["channel"], "zoom")
        self.assertIn("**Roster:** Alice Example, Bob Example", case.output.read_text())

    def test_explicit_roster_and_meeting_key_reach_the_tracker(self):
        case = self.case()
        case.output = case.directory / "output" / "2026-01-02-0930 Sync-Transcript.md"
        case.run_main("--local-speakers", "--roster", "Alice Example, Bob Example")
        track = self.calls()[1]
        self.assertEqual(track[track.index("--roster") + 1], "Alice Example,Bob Example")
        self.assertEqual(track[track.index("--meeting-key") + 1], "2026-01-02-0930")
        self.assertTrue((self.timelines / "2026-01-02-0930 Sync-Transcript.timeline.json").exists())

    def test_environment_enables_local_speakers_and_flag_disables_it(self):
        case = self.case()
        with unittest.mock.patch.dict(os.environ, {pipeline.LOCAL_SPEAKERS_ENV: "1"}):
            case.run_main()
        self.check_tracked_run(case)
        case = self.base_case()
        with unittest.mock.patch.dict(os.environ, {pipeline.LOCAL_SPEAKERS_ENV: "1"}):
            case.run_main("--no-local-speakers")
        self.assertIs(case.posts[1][1]["json"]["speaker_labels"], True)

    def test_default_is_unchanged_until_the_maintainer_flips_it(self):
        case = self.base_case()
        case.run_main()
        self.assertIs(case.posts[1][1]["json"]["speaker_labels"], True)
        self.assertTrue(pipeline.utterances_sidecar_path(case.output).exists())
        self.assertFalse(pipeline.words_sidecar_path(case.output).exists())
        self.assertEqual(self.calls(), [])
        with unittest.mock.patch.object(pipeline, "LOCAL_SPEAKERS_DEFAULT", True):
            self.assertTrue(pipeline.local_speakers_default())
            with unittest.mock.patch.dict(os.environ, {pipeline.LOCAL_SPEAKERS_ENV: "0"}):
                self.assertFalse(pipeline.local_speakers_default())

    def test_not_ready_falls_back_to_cloud_diarization_with_a_warning(self):
        case = self.base_case()
        self.stub({"ready": False, "status": "no_calibration",
                   "reason": "the store has no track calibration; run track-calibrate"})
        case.run_main("--local-speakers", "--speakers-expected", "2")
        submitted = case.posts[1][1]["json"]
        self.assertIs(submitted["speaker_labels"], True)
        self.assertEqual(submitted["speakers_expected"], 2)
        text = case.output.read_text()
        self.assertIn("> [!warning] Local speaker tracking was requested but is unavailable (no_calibration: "
                      "the store has no track calibration; run track-calibrate). AssemblyAI cloud "
                      "diarization was used instead.", text)
        self.assertIn("**[00:01] Speaker B:** Synthetic response.", text)
        self.assertIn("speaker diarization", text)
        self.assertTrue(pipeline.utterances_sidecar_path(case.output).exists())
        self.assertFalse(pipeline.words_sidecar_path(case.output).exists())
        self.assertEqual([c[0] for c in self.calls()], ["track-check"])   # the tracker never started
        self.assertFalse(self.timelines.exists())

    def test_broken_readiness_check_falls_back(self):
        case = self.base_case()
        with unittest.mock.patch.object(pipeline, "TRACK_CLI", case.directory / "missing.py"):
            case.run_main("--local-speakers")
        self.assertIs(case.posts[1][1]["json"]["speaker_labels"], True)
        self.assertIn("readiness check failed", case.output.read_text())

    def test_tracker_failure_after_transcription_keeps_words_and_warns(self):
        case = self.case()
        self.stub(READY, status={"status": "error", "reason": "RuntimeError: synthetic failure"})
        case.run_main("--local-speakers")
        self.check_words_only_request(case)
        text = case.output.read_text()
        self.assertIn("> [!warning] Local speaker tracking failed (error: RuntimeError: synthetic failure); "
                      "speakers are not labelled.", text)
        self.assertIn("**[00:00]** Synthetic opening.\n\n**[00:04]** Synthetic response.", text)
        self.assertFalse(self.timeline_path(case).exists())

    def test_tracker_timeout_is_stopped_and_reported(self):
        case = self.case()
        self.stub(READY, timeline=synthetic_timeline(self.TURNS), sleep=30)
        case.run_main("--local-speakers", "--track-timeout", "0.5")
        self.assertIn("Local speaker tracking failed (timed out after 0.5 s)", case.output.read_text())
        self.assertFalse(self.timeline_path(case).exists())

    def test_existing_sidecar_or_timeline_fails_before_upload(self):
        for existing in (pipeline.words_sidecar_path, self.timeline_path):
            case = self.case()
            path = existing(case.output) if existing is pipeline.words_sidecar_path else existing(case)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("previous")
            with self.assertRaises(FileExistsError):
                case.run_main("--local-speakers")
            case.post.assert_not_called()
            self.assertEqual(path.read_text(), "previous")
            self.assertEqual([c[0] for c in self.calls()], ["track-check"])

    def test_timeline_inside_the_notes_vault_is_refused(self):
        case = self.base_case()
        with unittest.mock.patch.dict(os.environ, {"VAULT_PATH": str(case.directory)}):
            case.run_main("--local-speakers")
        self.assertIs(case.posts[1][1]["json"]["speaker_labels"], True)
        self.assertIn("timeline location refused", case.output.read_text())
        self.assertEqual(self.calls(), [])

    def test_no_words_fails_without_transcript(self):
        case = self.case(words=[])
        with self.assertRaisesRegex(RuntimeError, "no words"):
            case.run_main("--local-speakers")
        self.assertFalse(case.output.exists())
        self.assertFalse(self.timeline_path(case).exists())

    def test_provider_failure_stops_the_tracker(self):
        case = self.case()
        self.stub(READY, timeline=synthetic_timeline(self.TURNS), sleep=30)
        case.poll_result["status"] = "error"
        with self.assertRaisesRegex(RuntimeError, "transcription failed"):
            case.run_main("--local-speakers")
        self.assertFalse(case.output.exists())
        self.assertFalse(self.timeline_path(case).exists())

    def test_sidecar_failure_does_not_fail_transcript(self):
        case = self.case()
        vp_words = pipeline._voiceprint_words()
        with unittest.mock.patch.object(vp_words, "dump_sidecar", side_effect=OSError("disk")):
            case.run_main("--local-speakers")
        self.assertTrue(case.output.exists())
        self.assertIn("Words sidecar not written", case.stdout.getvalue())
        self.assertTrue(self.timeline_path(case).exists())

    def test_mandarin_shaped_words_join_without_spaces_and_split_long_turns(self):
        # Shape only: 1-3 character tokens, contiguous timing, almost no punctuation.
        tokens = ["我们", "今天", "讨论", "一下", "方案", "然后", "看看", "进度"] * 10
        words, t = [], 0
        for tok in tokens:
            words.append({"text": tok, "start": t, "end": t + 1000, "confidence": 0.9})
            t += 1000
        case = self.case(words=words)
        self.stub(READY, timeline=synthetic_timeline([(0.0, 80.0, "Alice Example", 0.9)]))
        case.run_main("--local-speakers")
        body = case.output.read_text().split("\n---\n\n")[-1]
        paragraphs = body.strip().split("\n\n")
        self.assertGreater(len(paragraphs), 1)  # 80 s without sentence ends still splits
        self.assertTrue(paragraphs[0].startswith("**[00:00] Alice Example:** 我们今天讨论一下方案然后看看进度我们"))
        self.assertNotIn(" 今天", body)

    def test_unlabelled_mandarin_paragraphs_still_split(self):
        tokens = ["我们", "今天", "讨论", "一下"] * 20
        words = [{"text": tok, "start": i * 1000, "end": i * 1000 + 1000, "confidence": 0.9}
                 for i, tok in enumerate(tokens)]
        paragraphs = pipeline._aai_local_paragraphs(words)
        self.assertGreater(len(paragraphs), 1)
        self.assertTrue(paragraphs[0]["text"].startswith("我们今天讨论一下"))

    def test_roster_from_context(self):
        roster = pipeline.roster_from_context
        self.assertEqual(roster("Possible speakers:\n- [Canonical name] ([role])\n"), [])
        self.assertEqual(roster("Date: x\nPossible speakers:\n- Alice Example (host)\n* Bob Example\n"
                                "- Alice Example\n\nAttendees: Carol Example, Dave Example (guest)\n"),
                         ["Alice Example", "Bob Example", "Carol Example", "Dave Example"])
        self.assertEqual(roster("Speakers: 鲍勃（主持）\n"), ["鲍勃"])
        self.assertEqual(roster(None), [])

    def test_timestamp_format_matches_assemblyai_lines(self):
        vp_words = pipeline._voiceprint_words()
        for ms in (0, 999, 61_900, 599_990, 3_600_000, 3_725_200, 36_000_000):
            self.assertEqual(vp_words.format_ts(ms / 1000), pipeline._ms_to_ts(ms))


class IdentifyNoOpTests(unittest.TestCase):
    """The advisory step must never break the workflow: exit 0, torch never imported."""

    def run_cli(self, *argv, block=("numpy",)):
        code = (
            "import sys\n"
            f"for m in {list(block)!r}: sys.modules[m] = None\n"
            f"sys.argv = ['identify_speakers.py', *{list(argv)!r}]\n"
            f"sys.path.insert(0, {str(CLI.parent)!r})\n"
            "import identify_speakers as c\n"
            "try:\n"
            "    rc = c.main()\n"
            "except SystemExit as e:\n"
            "    rc = e.code\n"
            "print('RC', rc, 'TORCH', 'torch' in sys.modules)\n"
        )
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)

    def test_identify_without_dependencies_is_a_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            result = self.run_cli("--store", str(Path(tmp) / "store"), "identify", "--audio", "x.m4a",
                                  "--utterances", "x.utterances.json", "--json", str(out))
            self.assertIn("RC 0 TORCH False", result.stdout, result.stderr)
            self.assertEqual(json.loads(out.read_text())["status"], "no_deps")
            self.assertFalse((Path(tmp) / "store").exists())

    def test_help_and_other_commands_without_dependencies(self):
        result = self.run_cli("--help")
        self.assertIn("RC 0", result.stdout)
        self.assertIn("calibrate", result.stdout)
        result = self.run_cli("list")
        self.assertIn("RC 1 TORCH False", result.stdout)
        self.assertIn("voiceprint dependencies not installed", result.stderr)


class IdentifyWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.wrapper = scripts / "identify-speakers.sh"
        shutil.copy2(ROOT / "scripts" / "identify-speakers.sh", self.wrapper)
        python = self.root / ".venv" / "bin" / "python3"
        python.parent.mkdir(parents=True)
        python.write_text('#!/usr/bin/env bash\nprintf "CLI:%s VAULT:%s\\n" "$2" "${VAULT_PATH:-unset}"\n')
        python.chmod(0o700)
        self.env_file = self.root / ".env"
        self.env_file.write_text('VAULT_PATH="/notes/root"\n')
        self.env_file.chmod(0o600)
        self.environment = {k: v for k, v in os.environ.items()
                            if k not in {"MEETING_DOCUMENTER_ENV_FILE", "VAULT_PATH"}}

    def run_wrapper(self):
        return subprocess.run(["bash", str(self.wrapper), "list"], env=self.environment,
                              capture_output=True, text=True, timeout=10)

    def test_loads_trusted_config_for_the_store_guard(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CLI:list VAULT:/notes/root", result.stdout)

    def test_group_writable_config_is_refused(self):
        self.env_file.chmod(0o660)
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to source", result.stderr)
        self.assertNotIn("CLI:", result.stdout)


if __name__ == "__main__":
    unittest.main()
