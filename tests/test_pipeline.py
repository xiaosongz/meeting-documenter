"""Regression tests using synthetic audio and mocked provider responses only."""

from contextlib import ExitStack, redirect_stdout, redirect_stderr
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from audio_fixture import write_tone

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "transcribe_pipeline", ROOT / "scripts" / "transcribe_pipeline.py"
)
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def response(payload, status=200):
    return SimpleNamespace(status_code=status, json=lambda: payload)


def frontmatter_scalar(text, field):
    header = text.split("---", 2)[1]
    line = next(line for line in header.splitlines() if line.startswith(field + ": "))
    return json.loads(line.split(": ", 1)[1])


@unittest.skipUnless(shutil.which("ffprobe"), "pipeline integration checks require ffprobe")
class AssemblyAIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.source = write_tone(self.directory / 'synthetic "quoted": sample.wav')
        self.source_bytes = self.source.read_bytes()
        self.output = self.directory / "output" / "transcript.md"
        self.stdout = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(patch.dict(os.environ, {"ASSEMBLYAI_API_KEY": "test-only-value"}))
        self.posts = []
        self.poll_result = {
            "id": "synthetic-job", "status": "completed",
            "speech_models": ["universal-3-5-pro"],
            "speech_model_used": "universal-3-5-pro", "language_code": "en",
            "language_confidence": 0.99,
            "utterances": [
                {"speaker": "A", "start": 0, "end": 1000, "text": "Synthetic opening."},
                {"speaker": "B", "start": 1000, "end": 3000, "text": "Synthetic response."},
            ],
        }
        self.post = self.stack.enter_context(patch("requests.post", side_effect=self.fake_post))
        self.get = self.stack.enter_context(patch(
            "requests.get", side_effect=lambda *a, **k: response(self.poll_result)
        ))
        self.archive = self.stack.enter_context(patch.object(pipeline, "archive_to_ogg"))
        self.prepare = self.stack.enter_context(patch.object(pipeline, "prepare_audio"))
        self.split = self.stack.enter_context(patch.object(pipeline, "split_audio"))

    def fake_post(self, url, **kwargs):
        if url.endswith("/upload"):
            self.assertNotIn("files", kwargs)
            self.assertEqual(kwargs["data"].read(), self.source_bytes)
            self.posts.append(("upload", kwargs))
            return response({"upload_url": "https://example.invalid/synthetic-audio"})
        self.assertTrue(url.endswith("/transcript"), url)
        self.posts.append(("submit", kwargs))
        return response({"id": "synthetic-job"})

    def run_main(self, *arguments):
        with patch.object(sys, "argv", [
            "transcribe_pipeline.py", str(self.source), "--output", str(self.output), *arguments
        ]):
            pipeline.main()

    def test_default_main_writes_original_source_diarization_and_provenance(self):
        description = 'Synthetic "quotes": first line\nsecond line'
        self.run_main("--description", description)
        submitted = self.posts[1][1]["json"]
        self.assertEqual(submitted["speech_models"], ["universal-3-5-pro"])
        self.assertTrue(submitted["language_detection"])
        self.assertTrue(submitted["speaker_labels"])
        self.assertNotIn("speakers_expected", submitted)
        self.assertNotIn("language_code", submitted)
        self.assertNotIn("prompt", submitted)
        self.assertEqual(self.posts[0][1]["headers"]["authorization"], "test-only-value")
        text = self.output.read_text()
        self.assertEqual(frontmatter_scalar(text, "source"), self.source.name)
        self.assertEqual(frontmatter_scalar(text, "description"), description)
        self.assertIn("**[00:01] Speaker B:** Synthetic response.", text)
        self.assertIn("**AAI Transcript ID:** `synthetic-job`", text)
        self.assertIn("**AAI Submitted Models:** `universal-3-5-pro`", text)
        self.assertIn("**AAI Returned Models:** `universal-3-5-pro`", text)
        self.assertIn("**AAI Model Used:** `universal-3-5-pro`", text)
        self.assertIn("**AAI Detected Language:** `en (0.9900)`", text)
        self.assertEqual(self.source.read_bytes(), self.source_bytes)
        self.assertTrue(self.posts[0][1]["data"].closed)
        self.archive.assert_not_called()
        self.prepare.assert_not_called()
        self.split.assert_not_called()

    def test_explicit_options_and_keyterms_omit_incompatible_prompt(self):
        terms = self.directory / "terms.txt"
        terms.write_text("Synthetic terminology\n\n  Example term  \n")
        context = self.directory / "context.txt"
        context.write_text("Synthetic context for a provider mock.")
        archive_dir = self.directory / "archives"
        self.run_main(
            "--keyterms-file", str(terms), "--context-file", str(context),
            "--language-code", "en", "--speakers-expected", "2",
            "--aai-speech-models", "universal-3-5-pro", "universal-2",
            "--archive-dir", str(archive_dir),
        )
        submitted = self.posts[1][1]["json"]
        self.assertEqual(submitted["keyterms_prompt"], ["Synthetic terminology", "Example term"])
        self.assertEqual(submitted["speech_models"], ["universal-3-5-pro", "universal-2"])
        self.assertEqual(submitted["speakers_expected"], 2)
        self.assertEqual(submitted["language_code"], "en")
        self.assertNotIn("language_detection", submitted)
        self.assertNotIn("prompt", submitted)
        self.assertIn("**AAI Speakers Expected:** `2`", self.output.read_text())
        self.archive.assert_called_once_with(self.source.resolve(), archive_dir)

    def test_context_without_keyterms_uses_prompt_and_legacy_no_archive_is_accepted(self):
        context = self.directory / "context.txt"
        context.write_text("Synthetic provider context.")
        self.run_main("--context-file", str(context), "--no-archive")
        self.assertEqual(self.posts[1][1]["json"]["prompt"], context.read_text())
        self.assertNotIn("keyterms_prompt", self.posts[1][1]["json"])
        self.archive.assert_not_called()

    def test_http_failures_do_not_write_transcript_or_echo_response_body(self):
        for stage in ("upload", "submit", "poll"):
            with self.subTest(stage=stage):
                self.posts.clear()
                failed = response({"error": "PRIVATE_SERVICE_BODY_SENTINEL"}, status=401)
                original_post = self.fake_post

                def failing_post(url, **kwargs):
                    is_upload = url.endswith("/upload")
                    if (stage == "upload" and is_upload) or (stage == "submit" and not is_upload):
                        return failed
                    return original_post(url, **kwargs)

                with patch("requests.post", side_effect=failing_post), patch(
                    "requests.get", return_value=failed
                ), self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                    self.run_main()
                self.assertNotIn("PRIVATE_SERVICE_BODY_SENTINEL", str(caught.exception) + self.stdout.getvalue())
                self.assertFalse(self.output.exists())
                self.assertEqual(self.source.read_bytes(), self.source_bytes)
                self.archive.assert_not_called()

    def test_provider_job_error_and_empty_speech_fail_without_transcript(self):
        for result in (
            {"status": "error", "error": "PRIVATE_SERVICE_BODY_SENTINEL"},
            {"status": "completed", "utterances": []},
        ):
            with self.subTest(result=result["status"]):
                self.poll_result = result
                with self.assertRaises(RuntimeError) as caught:
                    self.run_main()
                self.assertNotIn("PRIVATE_SERVICE_BODY_SENTINEL", str(caught.exception) + self.stdout.getvalue())
                self.assertFalse(self.output.exists())
                self.archive.assert_not_called()

    def test_processing_job_has_finite_poll_deadline(self):
        self.poll_result = {"status": "processing"}
        with patch.object(pipeline.time, "monotonic", side_effect=[0, 0, 1, 2]), patch.object(
            pipeline.time, "sleep"
        ), self.assertRaisesRegex(TimeoutError, "polling timed out"):
            self.run_main("--aai-poll-timeout", "1")
        self.get.assert_called_once()
        self.assertLessEqual(self.get.call_args.kwargs["timeout"], 1)
        self.assertFalse(self.output.exists())
        self.archive.assert_not_called()

    def test_invalid_cli_options_fail_before_any_upload(self):
        cases = [
            ("--aai-poll-timeout", "0"), ("--aai-poll-timeout", "nan"),
            ("--aai-poll-timeout", "inf"), ("--speakers-expected", "0"),
            ("--archive-dir", str(self.directory), "--no-archive"),
            ("--keyterms-file", str(self.directory / "missing.txt")),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                self.run_main(*arguments)
            self.assertEqual(caught.exception.code, 2)
        self.post.assert_not_called()


class ArchiveDestinationTests(unittest.TestCase):
    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe", "bc")),
        "archive integration requires ffmpeg, ffprobe and bc",
    )
    def test_colliding_recording_names_preserve_archive_and_both_sources(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            first_dir, second_dir = directory / "first", directory / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = write_tone(first_dir / "recording.wav", seconds=2)
            second = write_tone(second_dir / "recording.wav", seconds=3)
            first_bytes, second_bytes = first.read_bytes(), second.read_bytes()
            archive_dir = directory / "archives"
            archive = pipeline.archive_to_ogg(first, archive_dir)
            archived_bytes = archive.read_bytes()
            with self.assertRaises(FileExistsError):
                pipeline.archive_to_ogg(second, archive_dir)
            self.assertEqual(archive.read_bytes(), archived_bytes)
            self.assertEqual(first.read_bytes(), first_bytes)
            self.assertEqual(second.read_bytes(), second_bytes)
            # The same OGG source and destination is a safe, verified no-op.
            self.assertEqual(pipeline.archive_to_ogg(archive, archive_dir), archive)
            self.assertEqual(archive.read_bytes(), archived_bytes)

    def test_archive_destination_symlinks_are_refused_before_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = write_tone(directory / "recording.wav")
            archive_dir = directory / "archives"
            archive_dir.mkdir()
            destination = archive_dir / "recording.ogg"
            target = directory / "target.ogg"
            for existing_target in (False, True):
                with self.subTest(existing_target=existing_target):
                    if existing_target:
                        target.write_bytes(b"Existing archive fixture.")
                    destination.symlink_to(target)
                    with patch.object(pipeline.subprocess, "run") as compress:
                        with self.assertRaises(FileExistsError):
                            pipeline.archive_to_ogg(source, archive_dir)
                        compress.assert_not_called()
                    self.assertTrue(destination.is_symlink())
                    if existing_target:
                        self.assertEqual(target.read_bytes(), b"Existing archive fixture.")
                    else:
                        self.assertFalse(target.exists())
                    destination.unlink()

    def test_failed_explicit_archive_raises_instead_of_reporting_success(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            directory = Path(directory)
            source = write_tone(directory / "recording.wav")
            original = source.read_bytes()
            failed = SimpleNamespace(returncode=1, stdout="", stderr="Synthetic encoder failure.")
            with patch.object(pipeline.subprocess, "run", return_value=failed) as compress:
                with self.assertRaises(RuntimeError):
                    pipeline.archive_to_ogg(source, directory / "archives")
            compress.assert_called_once()
            self.assertEqual(source.read_bytes(), original)


class GeminiTemporaryFileTests(unittest.TestCase):
    def test_failures_during_prepare_split_or_transcription_remove_all_temporary_audio(self):
        for stage in ("prepare", "split", "transcribe"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                source = write_tone(Path(directory) / "synthetic.wav")
                original = source.read_bytes()
                output = Path(directory) / "transcript.md"
                owned_directories = []
                reached = []

                def prepare(audio_path, duration, temp_dir):
                    owned_directories.append(temp_dir)
                    prepared = temp_dir / "prepared.wav"
                    prepared.write_bytes(original)
                    reached.append("prepare")
                    if stage == "prepare":
                        raise RuntimeError("injected prepare failure")
                    return prepared, duration

                def split(audio_path, target_duration=3000, *, output_dir):
                    output_dir.mkdir(parents=True)
                    chunk = output_dir / "chunk.wav"
                    chunk.write_bytes(original)
                    reached.append("split")
                    if stage == "split":
                        raise RuntimeError("injected split failure")
                    return [pipeline.AudioChunk(chunk, 0, 3)]

                def transcribe(*args, temp_dir, **kwargs):
                    retry = temp_dir / "retry"
                    retry.mkdir()
                    (retry / "partial.wav").write_bytes(original)
                    reached.append("transcribe")
                    raise RuntimeError("injected transcribe failure")

                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(sys, "argv", [
                    "transcribe_pipeline.py", str(source), "--backend", "gemini", "--output", str(output)
                ]))
                stack.enter_context(patch.object(pipeline, "get_audio_info", return_value={
                    "duration_int": 3, "duration": 3.0, "file_size": len(original), "mime_type": "audio/wav"
                }))
                stack.enter_context(patch.object(pipeline, "CHUNK_THRESHOLD", 1))
                stack.enter_context(patch.object(pipeline, "prepare_audio", side_effect=prepare))
                stack.enter_context(patch.object(pipeline, "split_audio", side_effect=split))
                stack.enter_context(patch.object(pipeline, "transcribe_with_retry", side_effect=transcribe))
                stack.enter_context(patch.object(pipeline, "get_api_key", return_value="test-only-value"))
                stack.enter_context(patch("google.genai.Client"))
                with self.assertRaisesRegex(RuntimeError, f"injected {stage} failure"):
                    pipeline.main()
                self.assertEqual(reached[-1], stage)
                self.assertEqual(len(owned_directories), 1)
                self.assertFalse(owned_directories[0].exists())
                self.assertEqual(source.read_bytes(), original)
                self.assertFalse(output.exists())

    @unittest.skipUnless(shutil.which("ffprobe"), "requires ffprobe")
    def test_gemini_success_records_original_source_after_preparation(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            source = write_tone(Path(directory) / 'synthetic "source".wav')
            output = Path(directory) / "transcript.md"
            original = source.read_bytes()
            prepared_paths = []

            def prepare(audio_path, duration, temp_dir):
                prepared = temp_dir / "prepared.wav"
                prepared.write_bytes(original)
                prepared_paths.append(prepared)
                return prepared, duration

            client = Mock()
            client.models.generate_content.return_value = SimpleNamespace(
                text="**[00:03] Speaker A:** Synthetic speech.",
                candidates=[SimpleNamespace(finish_reason="STOP")],
                usage_metadata=None,
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.object(sys, "argv", [
                "transcribe_pipeline.py", str(source), "--backend", "gemini", "--output", str(output)
            ]))
            stack.enter_context(patch.object(pipeline, "get_api_key", return_value="test-only-value"))
            stack.enter_context(patch("google.genai.Client", return_value=client))
            stack.enter_context(patch.object(pipeline, "prepare_audio", side_effect=prepare))
            archive = stack.enter_context(patch.object(pipeline, "archive_to_ogg"))
            pipeline.main()
            self.assertEqual(frontmatter_scalar(output.read_text(), "source"), source.name)
            self.assertIn("Synthetic speech.", output.read_text())
            self.assertNotIn("prepared.wav", output.read_text())
            self.assertFalse(prepared_paths[0].parent.exists())
            self.assertEqual(source.read_bytes(), original)
            client.models.generate_content.assert_called_once()
            archive.assert_not_called()

    def test_recursive_resplits_are_unique_inside_owned_directory_and_keep_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = write_tone(directory / "synthetic.wav")
            chunk = pipeline.AudioChunk(source, 3600, 120, depth=1)
            outputs = []

            def split(audio_path, target_duration, *, output_dir):
                outputs.append(output_dir)
                return [pipeline.AudioChunk(output_dir / "chunk.wav", 7, 30)]

            with patch.object(pipeline, "split_audio", side_effect=split):
                first = pipeline.resplit_chunk(chunk, 60, temp_dir=directory)
                second = pipeline.resplit_chunk(chunk, 60, temp_dir=directory)
            self.assertNotEqual(outputs[0], outputs[1])
            for output in outputs:
                self.assertEqual(output.parent, directory)
            for result in (first, second):
                self.assertEqual(result[0].offset_seconds, 3607)
                self.assertEqual(result[0].depth, 2)


if __name__ == "__main__":
    unittest.main()
