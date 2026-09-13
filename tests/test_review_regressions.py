"""Reproductions for the public pull-request review; never calls a provider."""
from contextlib import redirect_stdout
import builtins
import io
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from audio_fixture import write_tone
import test_pipeline as support

pipeline = support.pipeline
ROOT = support.ROOT


class ReviewRegressions(unittest.TestCase):
    def case(self):
        case = support.AssemblyAIIntegrationTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        return case

    def test_existing_transcript_and_input_aliases_fail_before_upload(self):
        for kind in ('existing', 'input', 'symlink', 'hardlink', 'dangling'):
            with self.subTest(kind=kind):
                case = self.case()
                case.output.parent.mkdir()
                if kind == 'existing':
                    case.output.write_text('Previous transcript')
                elif kind == 'input':
                    case.output = case.source
                elif kind == 'symlink':
                    case.output.symlink_to(case.source)
                elif kind == 'hardlink':
                    os.link(case.source, case.output)
                else:
                    case.output.symlink_to(case.directory / 'absent')
                with self.assertRaises((FileExistsError, ValueError)):
                    case.run_main()
                case.post.assert_not_called()
                self.assertEqual(case.source.read_bytes(), case.source_bytes)
                if kind == 'existing':
                    self.assertEqual(case.output.read_text(), 'Previous transcript')

    def test_known_archive_collision_fails_before_upload_or_output(self):
        case = self.case()
        archives = case.directory / 'archives'
        archives.mkdir()
        archive = archives / (case.source.stem + '.ogg')
        archive.write_bytes(b'Previous archive')
        with self.assertRaises(FileExistsError):
            case.run_main('--archive-dir', str(archives))
        case.post.assert_not_called()
        self.assertFalse(case.output.exists())
        self.assertEqual(archive.read_bytes(), b'Previous archive')

    def test_transcript_created_during_provider_work_is_not_overwritten(self):
        case = self.case()
        def poll(*args, **kwargs):
            case.output.parent.mkdir()
            case.output.write_text('Concurrent transcript')
            return support.response(case.poll_result)
        case.get.side_effect = poll
        with self.assertRaises(FileExistsError):
            case.run_main()
        self.assertEqual(case.output.read_text(), 'Concurrent transcript')

    def test_requested_language_survives_different_detected_language(self):
        case = self.case()
        case.poll_result['language_code'] = 'fr'
        case.run_main('--language-code', 'en')
        text = case.output.read_text()
        self.assertIn('**AAI Requested Language:** `en`', text)
        self.assertIn('**AAI Detected Language:** `fr (0.9900)`', text)

    def test_default_backend_runs_without_google_package(self):
        case = self.case()
        original_import = builtins.__import__
        def without_google(name, *args, **kwargs):
            if name == 'google' or name.startswith('google.'):
                raise ImportError('Google package deliberately unavailable')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=without_google), patch.object(sys, 'argv', [
            'transcribe_pipeline.py', str(case.source), '--output', str(case.output)
        ]):
            runpy.run_path(str(ROOT / 'scripts/transcribe_pipeline.py'), run_name='__main__')
        self.assertTrue(case.output.exists())

    def test_empty_gemini_results_never_create_transcript(self):
        for segments in ([], [pipeline.TranscriptSegment(' \n', 0, 3)]):
            with self.subTest(segments=len(segments)), tempfile.TemporaryDirectory() as directory:
                source = write_tone(Path(directory) / 'source.wav')
                output = Path(directory) / 'transcript.md'
                with patch.object(sys, 'argv', ['pipeline', str(source), '--backend', 'gemini', '--output', str(output)]), patch.object(
                    pipeline, 'prepare_audio', return_value=(source, 3)
                ), patch('google.genai.Client', return_value=Mock()), patch.object(
                    pipeline, 'get_api_key', return_value='test-only-value'
                ), patch.object(pipeline, 'transcribe_with_retry', return_value=segments), redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        pipeline.main()
                self.assertFalse(output.exists())

    def test_compressor_refuses_existing_destinations_and_source_hardlinks(self):
        for kind in ('existing', 'symlink', 'hardlink', 'dangling'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                source = write_tone(Path(directory) / 'source.wav')
                original = source.read_bytes()
                output = Path(directory) / 'archive.ogg'
                target = Path(directory) / 'target.ogg'
                target.write_bytes(b'Existing archive')
                if kind == 'existing':
                    output.write_bytes(b'Existing archive')
                elif kind == 'symlink':
                    output.symlink_to(target)
                elif kind == 'hardlink':
                    os.link(source, output)
                else:
                    output.symlink_to(Path(directory) / 'absent')
                result = subprocess.run(['bash', str(ROOT / 'scripts/compress-audio.sh'), str(source), '--output', str(output)], capture_output=True, timeout=30)
                self.assertEqual(source.read_bytes(), original, 'Source bytes changed')
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(target.read_bytes(), b'Existing archive')
                if kind == 'existing':
                    self.assertEqual(output.read_bytes(), b'Existing archive')

    def test_compressor_publish_race_and_failed_encode_preserve_files(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = write_tone(directory / 'source.wav')
            original = source.read_bytes()
            output = directory / 'archive.ogg'
            bin_dir = directory / 'bin'
            bin_dir.mkdir()
            shim = bin_dir / 'python3'
            shim.write_text('#!/usr/bin/env bash\nprintf "Concurrent archive" > "$3"\nexec "' + sys.executable + '" "$@"\n')
            shim.chmod(0o700)
            environment = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'])
            result = subprocess.run(['bash', str(ROOT / 'scripts/compress-audio.sh'), str(source), '--output', str(output)], env=environment, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), 'Concurrent archive')
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(list(directory.glob('.meeting-archive.*')), [])
            invalid = directory / 'invalid.wav'
            invalid.write_bytes(b'Invalid synthetic audio')
            result = subprocess.run(['bash', str(ROOT / 'scripts/compress-audio.sh'), str(invalid), '--output', str(directory / 'failed.ogg')], capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((directory / 'failed.ogg').exists())
            self.assertEqual(list(directory.glob('.meeting-archive.*')), [])
            self.assertEqual(invalid.read_bytes(), b'Invalid synthetic audio')

    def test_pipeline_rejects_non_ogg_archive_hardlink_before_compressor(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_tone(Path(directory) / 'source.wav')
            output = source.with_suffix('.ogg')
            os.link(source, output)
            with patch.object(pipeline.subprocess, 'run') as compress, self.assertRaises(FileExistsError):
                pipeline.archive_to_ogg(source, source.parent)
            compress.assert_not_called()


if __name__ == '__main__':
    unittest.main()
