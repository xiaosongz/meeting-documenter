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


class LocalSpeakersTests(unittest.TestCase):
    """MEETING_DOCUMENTER_LOCAL_SPEAKERS / --local-speakers: words only, no cloud diarization."""

    WORDS = [
        {"text": "Synthetic", "start": 0, "end": 400, "confidence": 0.98, "speaker": None},
        {"text": "opening.", "start": 400, "end": 900, "confidence": 0.95, "speaker": None},
        {"text": "Synthetic", "start": 4000, "end": 4400, "confidence": 0.97, "speaker": None},
        {"text": "response.", "start": 4400, "end": 5000, "confidence": 0.93, "speaker": None},
    ]

    def base_case(self):
        case = support.AssemblyAIIntegrationTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def case(self, words=None):
        case = self.base_case()
        case.poll_result.pop("utterances")
        case.poll_result["words"] = list(self.WORDS if words is None else words)
        case.poll_result["audio_duration"] = 5.2
        return case

    def check_local_run(self, case):
        submitted = case.posts[1][1]["json"]
        self.assertIs(submitted["speaker_labels"], False)
        self.assertNotIn("speakers_expected", submitted)
        text = case.output.read_text()
        self.assertIn("**[00:00]** Synthetic opening.\n\n**[00:04]** Synthetic response.\n", text)
        self.assertIn("words only, no cloud diarization", text)
        self.assertIn("Speakers are not labelled yet", text)
        self.assertNotIn("Speaker A", text)
        self.assertEqual(pipeline.extract_last_timestamp(text), 4)
        self.assertFalse(pipeline.utterances_sidecar_path(case.output).exists())
        sidecar = pipeline.words_sidecar_path(case.output)
        self.assertEqual(sidecar.name, "transcript.words.json")
        data = json.loads(sidecar.read_text())
        self.assertEqual((data["schema"], data["type"], data["time_unit"]), (1, "words", "ms"))
        self.assertEqual(data["transcript_id"], "synthetic-job")
        self.assertEqual(data["audio_duration_ms"], 5200)
        self.assertEqual(data["words"][1], {"text": "opening.", "start": 400, "end": 900, "confidence": 0.95})
        self.assertNotIn("speaker", sidecar.read_text())

    def test_cli_flag_requests_words_only_and_writes_sidecar(self):
        case = self.case()
        case.run_main("--local-speakers", "--speakers-expected", "2")
        self.check_local_run(case)
        self.assertIn("omitting speakers_expected", case.stdout.getvalue())

    def test_environment_enables_local_speakers(self):
        case = self.case()
        with unittest.mock.patch.dict(os.environ, {pipeline.LOCAL_SPEAKERS_ENV: "1"}):
            case.run_main()
        self.check_local_run(case)

    def test_default_is_unchanged(self):
        case = self.base_case()
        with unittest.mock.patch.dict(os.environ, {pipeline.LOCAL_SPEAKERS_ENV: "0"}):
            case.run_main()
        self.assertIs(case.posts[1][1]["json"]["speaker_labels"], True)
        self.assertTrue(pipeline.utterances_sidecar_path(case.output).exists())
        self.assertFalse(pipeline.words_sidecar_path(case.output).exists())

    def test_existing_words_sidecar_fails_before_upload(self):
        case = self.case()
        case.output.parent.mkdir()
        sidecar = pipeline.words_sidecar_path(case.output)
        sidecar.write_text("previous sidecar")
        with self.assertRaisesRegex(FileExistsError, "Words sidecar"):
            case.run_main("--local-speakers")
        case.post.assert_not_called()
        self.assertEqual(sidecar.read_text(), "previous sidecar")

    def test_no_words_fails_without_transcript(self):
        case = self.case(words=[])
        with self.assertRaisesRegex(RuntimeError, "no words"):
            case.run_main("--local-speakers")
        self.assertFalse(case.output.exists())

    def test_sidecar_failure_does_not_fail_transcript(self):
        case = self.case()
        vp_words = pipeline._voiceprint_words()
        with unittest.mock.patch.object(vp_words, "dump_sidecar", side_effect=OSError("disk")):
            case.run_main("--local-speakers")
        self.assertTrue(case.output.exists())
        self.assertIn("Words sidecar not written", case.stdout.getvalue())

    def test_mandarin_shaped_words_join_without_spaces_and_split_long_paragraphs(self):
        # Shape only: 1-3 character tokens, contiguous timing, almost no punctuation.
        tokens = ["我们", "今天", "讨论", "一下", "方案", "然后", "看看", "进度"] * 10
        words, t = [], 0
        for tok in tokens:
            words.append({"text": tok, "start": t, "end": t + 1000, "confidence": 0.9})
            t += 1000
        case = self.case(words=words)
        case.run_main("--local-speakers")
        body = case.output.read_text().split("---\n\n")[-1]
        paragraphs = body.strip().split("\n\n")
        self.assertGreater(len(paragraphs), 1)  # 80 s without pauses or sentence ends still splits
        self.assertTrue(paragraphs[0].startswith("**[00:00]** 我们今天讨论一下方案然后看看进度我们"))
        self.assertNotIn(" 今天", body)

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
