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
