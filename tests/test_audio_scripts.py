"""Local integration checks for non-destructive archiving and env loading."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from audio_fixture import write_tone

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe", "bc")),
    "audio checks require ffmpeg, ffprobe and bc",
)
class ArchiveTests(unittest.TestCase):
    def test_compress_and_copy_ogg_preserve_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = write_tone(directory / "synthetic.wav")
            original = source.read_bytes()
            archive = directory / "archives" / "synthetic.ogg"
            result = self.compress(source, archive)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(source.read_bytes(), original)
            self.assertGreater(archive.stat().st_size, 0)
            decoded = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(archive), "-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(decoded.returncode, 0, decoded.stderr)
            self.assertEqual(decoded.stderr, "")
            copied = directory / "another archive" / "copy.ogg"
            result = self.compress(archive, copied)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(copied.read_bytes(), archive.read_bytes())
            self.assertEqual(source.read_bytes(), original)

    def test_invalid_audio_preserves_source_and_existing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "invalid.ogg"
            source.write_bytes(b"Synthetic invalid audio fixture.\n")
            archive = directory / "existing.ogg"
            archive.write_bytes(b"Existing archive must survive a failed conversion.\n")
            source_before = source.read_bytes()
            archive_before = archive.read_bytes()
            result = self.compress(source, archive)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(archive.read_bytes(), archive_before)

    def test_backup_preserves_adjacent_ogg_and_refuses_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = write_tone(directory / "Recording 20260101120000.wav")
            source_bytes = source.read_bytes()
            archive = directory / "archives" / "meeting.ogg"
            self.assertEqual(self.compress(source, archive).returncode, 0)
            archive_bytes = archive.read_bytes()
            adjacent = source.with_suffix(".ogg")
            adjacent.write_bytes(b"Unrelated neighboring file")
            environment = dict(os.environ, MEETING_AUDIO_BACKUP_DIR=str(directory / "backups"))
            command = ["bash", str(ROOT / "scripts" / "cleanup-source-audio.sh"),
                       str(source), str(archive), "--meeting-name", "Team Sync", "--meeting-time", "1130"]
            first = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            backups = list((directory / "backups").rglob("*.wav"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), source_bytes)
            self.assertFalse(source.exists())
            self.assertEqual(adjacent.read_bytes(), b"Unrelated neighboring file")
            self.assertEqual(archive.read_bytes(), archive_bytes)
            source.write_bytes(source_bytes)
            second = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(backups[0].read_bytes(), source_bytes)

    def test_backup_requires_distinct_source_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ogg"
            source.write_bytes(b"Original must stay")
            result = subprocess.run(
                ["bash", str(ROOT / "scripts" / "cleanup-source-audio.sh"),
                 str(source), str(source), "--meeting-name", "Team Sync", "--meeting-time", "1130"],
                env=dict(os.environ, MEETING_AUDIO_BACKUP_DIR=str(Path(directory) / "backups")),
                capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(source.read_bytes(), b"Original must stay")

    @staticmethod
    def compress(source, archive):
        return subprocess.run(
            ["bash", str(ROOT / "scripts" / "compress-audio.sh"), str(source),
             "--output", str(archive)],
            capture_output=True, text=True, timeout=60,
        )


class EnvironmentLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.wrapper = scripts / "transcribe.sh"
        shutil.copy2(ROOT / "scripts" / "transcribe.sh", self.wrapper)
        bin_dir = self.root / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        python = bin_dir / "python3"
        python.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "${1:-}" == "-c" ]]; then exit 0; fi\n'
            'printf "PIPELINE_CALLED:%s\\n" "${MEETING_DOCUMENTER_TEST_MARKER:-unset}"\n'
        )
        python.chmod(0o700)
        self.env_file = self.root / ".env"
        self.env_file.write_text("MEETING_DOCUMENTER_TEST_MARKER=loaded\n")
        self.env_file.chmod(0o600)
        self.environment = {
            key: value for key, value in os.environ.items()
            if key not in {"MEETING_DOCUMENTER_ENV_FILE", "MEETING_DOCUMENTER_TEST_MARKER"}
        }

    def run_wrapper(self):
        return subprocess.run(
            ["bash", str(self.wrapper), "--help"], env=self.environment,
            capture_output=True, text=True, timeout=10,
        )

    def test_owner_only_env_file_loads(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PIPELINE_CALLED:loaded", result.stdout)

    def test_group_writable_env_file_is_rejected_before_sourcing(self):
        marker = self.root / "must-not-exist"
        self.env_file.write_text(f"touch '{marker}'\n")
        self.env_file.chmod(0o660)
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to source", result.stderr)
        self.assertFalse(marker.exists())
        self.assertNotIn("PIPELINE_CALLED", result.stdout)

    def test_gnu_stat_output_does_not_contaminate_permission_mode(self):
        bin_dir = self.root / "stat-bin"
        bin_dir.mkdir()
        stat = bin_dir / "stat"
        stat.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "$1" == "-c" ]]; then echo 600; exit 0; fi\n'
            'echo "File system details from GNU stat -f"; exit 1\n'
        )
        stat.chmod(0o700)
        self.environment["PATH"] = str(bin_dir) + os.pathsep + self.environment["PATH"]
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PIPELINE_CALLED:loaded", result.stdout)

    def test_wrapper_checks_only_selected_backend_dependency(self):
        python = self.root / ".venv" / "bin" / "python3"
        log = self.root / "imports.log"
        python.write_text(
            '#!/usr/bin/env bash\n'
            'if [[ "$1" == "-c" ]]; then printf "%s\\n" "$2" >> "$MEETING_DOCUMENTER_IMPORT_LOG"; fi\n'
        )
        self.environment["MEETING_DOCUMENTER_IMPORT_LOG"] = str(log)
        for flags, expected in [([], "import requests"), (["--backend", "gemini"], "import google.genai"), (["--backend=gemini"], "import google.genai")]:
            with self.subTest(flags=flags):
                log.unlink(missing_ok=True)
                result = subprocess.run(["bash", str(self.wrapper), "--help", *flags], env=self.environment, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(log.read_text().strip(), expected)

    def test_explicit_empty_env_override_skips_default_file(self):
        self.environment["MEETING_DOCUMENTER_ENV_FILE"] = ""
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PIPELINE_CALLED:unset", result.stdout)


if __name__ == "__main__":
    unittest.main()
