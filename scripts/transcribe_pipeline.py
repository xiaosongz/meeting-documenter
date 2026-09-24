#!/usr/bin/env python3
"""
Meeting Transcription Pipeline
-------------------------------
Transcribe via AssemblyAI (default) or Gemini (opt-in), preserve speaker
and model provenance, and optionally archive to an explicit directory.

Usage:
    transcribe_pipeline.py <audio-path> [--output <path>] [--context-file <path>] [--model <model>]

Environment:
    AssemblyAI: ASSEMBLYAI_API_KEY
    Gemini: GOOGLE_API_KEY or GEMINI_API_KEY
    transcribe.sh loads the configured per-user .env, never an implicit ~/.env.
"""

import os
import sys
import re
import json
import time
import tempfile
import math
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# ─── Constants ──────────────────────────────────────────────────────────────

SKILL_DIR = Path(__file__).parent.parent.resolve()
SPLIT_SCRIPT = SKILL_DIR / "scripts" / "split-audio.sh"
ARCHIVE_SCRIPT = SKILL_DIR / "scripts" / "compress-audio.sh"

MAX_OUTPUT_TOKENS = 65536  # Gemini Flash output cap
MAX_INLINE_SIZE = 20 * 1024 * 1024  # 20MB — above this, use Files API
CHUNK_THRESHOLD = 3600  # 60 minutes — split if longer
DEFAULT_MODEL = "gemini-2.5-flash"
ASSEMBLYAI_BASE = "https://api.assemblyai.com"
ASSEMBLYAI_MODELS = ["universal-3-5-pro"]
# Local speaker tracking: AssemblyAI returns words only (speaker_labels=false, no cloud voice
# embeddings), `identify_speakers.py track` works out who speaks when from enrolled voiceprints,
# and each word takes the speaker of the timeline it falls in. Opt-in per run (--local-speakers)
# or per install (MEETING_DOCUMENTER_LOCAL_SPEAKERS=1; =0 opts out). To make it the default for
# everyone, the maintainer sets LOCAL_SPEAKERS_DEFAULT = True.
LOCAL_SPEAKERS_ENV = "MEETING_DOCUMENTER_LOCAL_SPEAKERS"
LOCAL_SPEAKERS_DEFAULT = False
TIMELINE_DIR_ENV = "MEETING_DOCUMENTER_TIMELINE_DIR"
TRACK_CLI = SKILL_DIR / "scripts" / "identify_speakers.py"
TRACK_CHECK_TIMEOUT = 180    # s; the readiness probe loads no models
LOCAL_PARAGRAPH_PAUSE = 2.0  # s; unlabelled paragraphs break at longer pauses
LOCAL_PARAGRAPH_MAX = 30.0   # s; ...or after a sentence end once longer than this
REVIEW_MAX_ITEMS = 40        # uncertain/guest/unknown turns listed for spot-checking in the header
MEETING_KEY_RE = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{4})")

SUPPORTED_FORMATS = {
    ".ogg": "audio/ogg", ".mp3": "audio/mp3", ".wav": "audio/wav",
    ".m4a": "audio/mp4", ".aac": "audio/aac", ".flac": "audio/flac",
    ".aiff": "audio/aiff", ".webm": "audio/webm",
}

# Codec → container mapping for safe stream copy (-c copy).
# If the source codec doesn't match its container, ffmpeg -c copy will fail.
CODEC_CONTAINER_MAP = {
    "opus": ".ogg",
    "vorbis": ".ogg",
    "aac": ".m4a",
    "mp3": ".mp3",
    "flac": ".flac",
    "pcm_s16le": ".wav",
    "pcm_s24le": ".wav",
    "pcm_f32le": ".wav",
}

# Trailing silence detection thresholds
TRAILING_SILENCE_MIN_DURATION = 10.0  # seconds — minimum silence segment to detect
TRAILING_SILENCE_NOISE_DB = -30       # dB — noise floor threshold
TRAILING_SILENCE_THRESHOLD = 60       # seconds — only trim if trailing silence exceeds this
TRIM_BUFFER = 30                      # seconds — keep this much buffer after last speech

TRANSCRIPTION_PROMPT = """You are a professional transcription service. Transcribe this audio with the following requirements:

1. **Speaker Identification**: Identify distinct speakers. If names are mentioned or inferable from context, use names instead of "Speaker 1/2/3".

2. **Timestamps**: Provide timestamps at natural speaking boundaries (every few sentences or when speakers change). Use MM:SS format. Start from 00:00 for this audio segment.

3. **Formatting**:
   - Start each speaker segment with: `**[MM:SS] Speaker Name:**`
   - Use natural paragraph breaks
   - Preserve filler words sparingly (um, uh) only when intentional
   - Use proper punctuation and capitalization

4. **Quality**:
   - Transcribe verbatim but clean up obvious false starts
   - Mark unclear audio as [inaudible] or [unclear]
   - Note significant non-speech sounds like [laughter], [pause]

5. **CRITICAL**: Transcribe the ENTIRE audio from start to finish. Do not stop early.

6. **Output**: Return ONLY the formatted transcript in markdown, no additional commentary.

Begin transcription:"""


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class AudioChunk:
    path: Path
    offset_seconds: int
    duration: int
    depth: int = 0  # retry depth for truncation recovery


@dataclass
class TranscriptSegment:
    text: str
    offset_seconds: int
    duration: int


# ─── Utilities ──────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print("Error: GOOGLE_API_KEY not set. Check .env file.")
        sys.exit(1)
    return key


def get_audio_info(audio_path: Path) -> dict:
    """Get duration, format, and file size via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,format_name", "-show_entries", "stream=codec_name",
         "-of", "json", str(audio_path)],
        capture_output=True, text=True, timeout=30
    )
    info = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    file_size = audio_path.stat().st_size
    suffix = audio_path.suffix.lower()
    mime = SUPPORTED_FORMATS.get(suffix)
    if not mime:
        print(f"Error: Unsupported format: {suffix}")
        print(f"Supported: {', '.join(SUPPORTED_FORMATS.keys())}")
        sys.exit(1)
    return {
        "duration": duration,
        "duration_int": int(duration),
        "file_size": file_size,
        "mime_type": mime,
        "suffix": suffix,
    }


def extract_last_timestamp(text: str) -> int | None:
    """Extract last timestamp from transcript, return as seconds.

    Handles MM:SS and H:MM:SS:
      **[12:07]   → 727 seconds
      **[1:05:30] → 3930 seconds
    """
    matches = re.findall(r'\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\]', text)
    if not matches:
        return None
    last = matches[-1]
    parts = last.split(':')
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return None


def offset_timestamps(text: str, offset_seconds: int) -> str:
    """Add offset_seconds to all **[MM:SS] or **[H:MM:SS] timestamps."""
    if offset_seconds == 0:
        return text

    def replace_ts(match):
        ts_str = match.group(1)
        parts = ts_str.split(':')
        if len(parts) == 2:
            total = int(parts[0]) * 60 + int(parts[1]) + offset_seconds
        elif len(parts) == 3:
            total = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) + offset_seconds
        else:
            return match.group(0)

        if total >= 3600:
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            return f"**[{h}:{m:02d}:{s:02d}]"
        else:
            m, s = divmod(total, 60)
            return f"**[{m}:{s:02d}]"

    return re.sub(r'\*\*\[(\d{1,2}:\d{2}(?::\d{2})?)\]', replace_ts, text)


def fix_timestamp_monotonicity(text: str) -> tuple[str, int]:
    """Detect and fix timestamps that go backwards (Gemini hallucination artifact).

    Strategy: scan all **[MM:SS] timestamps. If one is less than its predecessor,
    replace it with the midpoint between the previous and next valid timestamps.
    Returns (fixed_text, number_of_fixes).
    """
    pattern = r'(\*\*\[)(\d{1,2}:\d{2}(?::\d{2})?)(\])'
    matches = list(re.finditer(pattern, text))
    if len(matches) < 2:
        return text, 0

    def ts_to_seconds(ts_str: str) -> int:
        parts = ts_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0

    def seconds_to_ts(total: int) -> str:
        if total >= 3600:
            h, rem = divmod(total, 3600)
            m, s = divmod(rem, 60)
            return f"{h}:{m:02d}:{s:02d}"
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"

    # Parse all timestamps with positions
    entries = []
    for m in matches:
        ts_str = m.group(2)
        entries.append({
            'start': m.start(2), 'end': m.end(2),
            'original': ts_str, 'seconds': ts_to_seconds(ts_str),
        })

    # Find regressions and fix them
    fixes = []
    for i in range(1, len(entries)):
        if entries[i]['seconds'] < entries[i - 1]['seconds']:
            prev_ts = entries[i - 1]['seconds']
            # Find next valid timestamp (one that's >= prev)
            next_ts = None
            for j in range(i + 1, len(entries)):
                if entries[j]['seconds'] >= prev_ts:
                    next_ts = entries[j]['seconds']
                    break
            if next_ts is None:
                next_ts = prev_ts + 2  # fallback: just after previous
            # Interpolate
            fixed_ts = prev_ts + (next_ts - prev_ts) // 2
            entries[i]['seconds'] = fixed_ts
            entries[i]['fixed'] = seconds_to_ts(fixed_ts)
            fixes.append((entries[i]['original'], entries[i]['fixed']))

    if not fixes:
        return text, 0

    # Apply fixes in reverse order to preserve positions
    result = text
    for i in range(len(entries) - 1, 0, -1):
        if 'fixed' in entries[i]:
            result = result[:entries[i]['start']] + entries[i]['fixed'] + result[entries[i]['end']:]

    return result, len(fixes)


def format_duration(seconds: int) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


# ─── Audio Preparation ─────────────────────────────────────────────────────

def detect_audio_codec(audio_path: Path) -> str:
    """Detect the audio stream codec of a file via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0",
         str(audio_path)],
        capture_output=True, text=True, timeout=30
    )
    return result.stdout.strip()


def get_compatible_extension(codec: str, current_ext: str) -> str:
    """Return a container extension compatible with the given codec.

    Returns current_ext unchanged if already compatible or codec is unknown.
    """
    target = CODEC_CONTAINER_MAP.get(codec)
    if target is None or current_ext == target:
        return current_ext
    return target


def prepare_audio(audio_path: Path, duration_seconds: int, temp_dir: Path) -> tuple[Path, int]:
    """Detect trailing silence and trim; fix codec/container mismatches.

    This step runs automatically before chunking/transcription to:
    1. Avoid sending minutes of dead air to the transcription API (cost + noise).
    2. Ensure the container format supports the source codec for stream-copy
       operations in split-audio.sh.

    Returns (prepared_path, effective_duration).
    If no changes needed, returns (original_path, original_duration).
    """
    codec = detect_audio_codec(audio_path)
    current_ext = audio_path.suffix.lower()
    target_ext = get_compatible_extension(codec, current_ext)
    needs_remux = (target_ext != current_ext)

    if needs_remux:
        print(f"   🔄 Container mismatch: {current_ext} has {codec} codec → {target_ext}")

    # --- Detect trailing silence ---
    trim_to = None
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-af", f"silencedetect=noise={TRAILING_SILENCE_NOISE_DB}dB"
                    f":d={TRAILING_SILENCE_MIN_DURATION}",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=300
        )
        silence_starts = [
            float(m.group(1))
            for line in result.stderr.split('\n')
            if (m := re.search(r'silence_start:\s*([\d.]+)', line))
        ]
        silence_ends = [
            float(m.group(1))
            for line in result.stderr.split('\n')
            if (m := re.search(r'silence_end:\s*([\d.]+)', line))
        ]
        # Only trim silence that truly extends to the end of the file.
        # If there are more silence_starts than silence_ends, the last
        # silence segment runs to EOF (ffmpeg never emitted silence_end).
        if silence_starts and len(silence_starts) > len(silence_ends):
            last_start = silence_starts[-1]
            trailing = duration_seconds - last_start
            if trailing > TRAILING_SILENCE_THRESHOLD:
                trim_to = min(int(last_start) + TRIM_BUFFER, duration_seconds)
                print(f"   🔇 Trailing silence: {format_duration(int(trailing))}")
                print(f"   ✂️  Trimming at {format_duration(trim_to)} "
                      f"(last speech at {format_duration(int(last_start))})")
    except (subprocess.TimeoutExpired, Exception) as exc:
        print(f"   ⚠ Silence detection failed ({exc}), skipping trim")

    needs_trim = trim_to is not None

    if not needs_remux and not needs_trim:
        return audio_path, duration_seconds

    # --- Create prepared file (stream copy first, re-encode as fallback) ---
    prepared_path = temp_dir / f"prepared{target_ext}"

    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path)]
    if needs_trim:
        cmd.extend(["-t", str(trim_to)])
    cmd.extend(["-c", "copy", str(prepared_path), "-y"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # Stream copy failed — fall back to re-encode as OGG Vorbis
        print(f"   ⚠ Stream copy failed, re-encoding to OGG...")
        prepared_path.unlink(missing_ok=True)
        prepared_path = prepared_path.with_suffix(".ogg")
        cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path)]
        if needs_trim:
            cmd.extend(["-t", str(trim_to)])
        cmd.extend(["-c:a", "libvorbis", "-q:a", "4", str(prepared_path), "-y"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"   ❌ Re-encode failed: {result.stderr.strip()}")
            print(f"   Proceeding with original audio")
            prepared_path.unlink(missing_ok=True)
            return audio_path, duration_seconds

    effective_duration = trim_to if needs_trim else duration_seconds
    reasons = []
    if needs_trim:
        reasons.append(f"trimmed {format_duration(duration_seconds - effective_duration)} silence")
    if needs_remux:
        reasons.append(f"remuxed {current_ext} → {target_ext} ({codec})")
    print(f"   ✅ {', '.join(reasons)}")

    return prepared_path, effective_duration


# ─── Splitting ──────────────────────────────────────────────────────────────

def split_audio(audio_path: Path, target_duration: int = 3000,
                *, output_dir: Path) -> list[AudioChunk]:
    """Split audio inside the caller-owned private temporary directory."""

    result = subprocess.run(
        [str(SPLIT_SCRIPT), str(audio_path),
         "--target-duration", str(target_duration),
         "--output-dir", str(output_dir)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"Error: split-audio.sh failed: {result.stderr}")
        sys.exit(1)

    manifest = json.loads(result.stdout)
    return [
        AudioChunk(
            path=Path(item["path"]),
            offset_seconds=item["offset_seconds"],
            duration=item["duration"],
            depth=0,
        )
        for item in manifest
    ]


def resplit_chunk(chunk: AudioChunk, target_duration: int, *, temp_dir: Path) -> list[AudioChunk]:
    """Re-split a chunk at a shorter target duration for truncation recovery."""
    output_dir = Path(tempfile.mkdtemp(prefix="resplit_", dir=temp_dir))
    sub_chunks = split_audio(chunk.path, target_duration, output_dir=output_dir)
    # Adjust offsets relative to original audio and increment depth
    for sc in sub_chunks:
        sc.offset_seconds += chunk.offset_seconds
        sc.depth = chunk.depth + 1
    return [sc for sc in sub_chunks if sc.duration > 0]


# ─── Transcription ──────────────────────────────────────────────────────────

def _finish_name(reason) -> str:
    """Normalize FinishReason enum to bare member name.

    google-genai's FinishReason is a `str`-subclass Enum, but `str(member)`
    returns `"FinishReason.MAX_TOKENS"` (Enum's __str__), not the bare name.
    `.name` always returns the bare name on Enum instances; fall back to
    `str()` for anything weird upstream (e.g., None or a plain string).
    """
    if reason is None:
        return "NONE"
    return getattr(reason, "name", str(reason))


def check_truncation(response, chunk_duration_seconds: int) -> bool:
    """Returns True if transcript appears truncated."""
    # No candidates = safety filter / empty response. Reach here BEFORE
    # is_transient_failure() in transcribe_with_recovery; guarding inline
    # keeps the retry loop engaged instead of crashing the pipeline.
    if not response.candidates:
        print("   ⚠ No candidates in response (safety filter or empty)")
        return True
    candidate = response.candidates[0]
    finish_str = _finish_name(candidate.finish_reason)

    # Explicit truncation
    if finish_str == "MAX_TOKENS":
        print("   ⚠ Truncation detected: MAX_TOKENS finish reason")
        return True

    # RECITATION = content filter refusal — no text produced
    if finish_str == "RECITATION" or response.text is None:
        reason = finish_str if finish_str != "STOP" else "empty response"
        print(f"   ⚠ No transcript produced: {reason}")
        return True

    # Implicit truncation: model stopped but transcript too short
    last_ts = extract_last_timestamp(response.text)
    if last_ts is not None and chunk_duration_seconds > 0:
        coverage = last_ts / chunk_duration_seconds
        if coverage < 0.8:
            print(f"   ⚠ Truncation detected: coverage {coverage:.0%} "
                  f"(last ts {last_ts}s / chunk {chunk_duration_seconds}s)")
            return True

    return False


TRANSIENT_REASONS = {"RECITATION", "SAFETY", "OTHER"}
MAX_TRANSIENT_RETRIES = 2

def is_transient_failure(response) -> bool:
    """True if failure is transient (retry same chunk), False if structural (re-split)."""
    if not response.candidates:
        return True
    finish_str = _finish_name(response.candidates[0].finish_reason)
    if finish_str in TRANSIENT_REASONS:
        return True
    if response.text is None and finish_str != "MAX_TOKENS":
        return True
    return False


def transcribe_chunk(client, chunk: AudioChunk, model: str,
                     context: str | None) -> str:
    """Transcribe a single audio chunk via Gemini API."""
    from google.genai import types

    audio_path = chunk.path
    file_size = audio_path.stat().st_size
    mime_type = SUPPORTED_FORMATS.get(audio_path.suffix.lower(), "audio/mp3")

    # Build prompt
    prompt_parts = []
    if context:
        prompt_parts.append(context)
    prompt_parts.append(TRANSCRIPTION_PROMPT)
    prompt = "\n\n".join(prompt_parts)

    # Upload or inline
    if file_size > MAX_INLINE_SIZE:
        print(f"   Uploading via Files API ({file_size / 1048576:.1f} MB)...")
        uploaded = client.files.upload(file=audio_path, config={"mime_type": mime_type})
        contents = [prompt, uploaded]
    else:
        audio_bytes = audio_path.read_bytes()
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        contents = [prompt, audio_part]

    # Generate
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    )

    # Log stats
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count or 0
        candidates_tokens = usage.candidates_token_count or 0
        print(f"   Tokens: {prompt_tokens:,} in, "
              f"{candidates_tokens:,} out")
    finish = _finish_name(response.candidates[0].finish_reason) if response.candidates else "UNKNOWN"
    print(f"   Finish reason: {finish}")

    return response


def transcribe_with_retry(client, chunks: list[AudioChunk], model: str,
                          context: str | None, max_depth: int = 2, *, temp_dir: Path
                          ) -> list[TranscriptSegment]:
    """Transcribe chunks with transient-error retry and truncation re-split."""
    results = []
    for i, chunk in enumerate(chunks):
        label = f"Chunk {i + 1}/{len(chunks)}"
        print(f"\n📝 {label}: {format_duration(chunk.duration)} "
              f"(offset {format_duration(chunk.offset_seconds)}, depth {chunk.depth})")

        response = None
        succeeded = False

        for attempt in range(1, MAX_TRANSIENT_RETRIES + 2):  # +2: 1-indexed, includes first attempt
            if attempt > 1:
                print(f"   🔁 Retry {attempt - 1}/{MAX_TRANSIENT_RETRIES} (transient error)...")

            response = transcribe_chunk(client, chunk, model, context)

            if not check_truncation(response, chunk.duration):
                succeeded = True
                break

            if not is_transient_failure(response) or attempt > MAX_TRANSIENT_RETRIES:
                break

        if succeeded:
            last_ts = extract_last_timestamp(response.text)
            coverage = (last_ts / chunk.duration * 100) if last_ts and chunk.duration else 0
            print(f"   ✅ Coverage: {coverage:.0f}%")
            results.append(TranscriptSegment(
                text=response.text, offset_seconds=chunk.offset_seconds,
                duration=chunk.duration,
            ))
        elif chunk.depth >= max_depth:
            if response.text is None:
                finish_str = (
                    _finish_name(response.candidates[0].finish_reason)
                    if response.candidates else "no-candidates"
                )
                print(f"   ❌ No text after {max_depth} re-splits + "
                      f"{MAX_TRANSIENT_RETRIES} retries "
                      f"(finish: {finish_str}). Skipping chunk.")
            else:
                actual_ts = extract_last_timestamp(response.text)
                print(f"   ❌ Still truncated after {max_depth} re-splits. "
                      f"Last ts: {actual_ts}s, expected: {chunk.duration}s")
                print(f"   Keeping partial transcript for this chunk.")
                results.append(TranscriptSegment(
                    text=response.text, offset_seconds=chunk.offset_seconds,
                    duration=chunk.duration,
                ))
        else:
            new_target = chunk.duration // 2
            print(f"   🔄 Re-splitting at {format_duration(new_target)} target...")
            sub_chunks = resplit_chunk(chunk, new_target, temp_dir=temp_dir)
            results.extend(
                transcribe_with_retry(client, sub_chunks, model, context, max_depth, temp_dir=temp_dir)
            )

    return results


# ─── Combination ────────────────────────────────────────────────────────────

def combine_segments(segments: list[TranscriptSegment], audio_path: Path,
                     model: str, description: str | None = None) -> str:
    """Combine transcript segments into a single document with frontmatter."""
    now = datetime.now()
    total_dur = sum(s.duration for s in segments)

    description = description or (
        f"Transcript of {audio_path.stem}, {format_duration(total_dur)}, {len(segments)} segment(s)"
    )
    desc_line = "\ndescription: " + json.dumps(description, ensure_ascii=False)

    header = f"""---
title: {json.dumps("Meeting Transcript - " + audio_path.stem, ensure_ascii=False)}
date: {now.strftime('%Y-%m-%d')}
type: transcript
source: {json.dumps(audio_path.name, ensure_ascii=False)}
created: {now.strftime('%Y-%m-%d')}{desc_line}
tags:
  - "#transcript"
  - "#meeting"
---

# Meeting Transcript

**Source:** `{audio_path.name}`
**Transcribed:** {now.strftime('%Y-%m-%d %H:%M')}
**Tool:** Gemini API ({model}), {len(segments)}-segment pipeline
**Duration:** ~{format_duration(total_dur)}

---

"""
    body_parts = []
    for seg in segments:
        adjusted = offset_timestamps(seg.text, seg.offset_seconds)
        body_parts.append(adjusted)

    return header + "\n\n".join(body_parts)


# ─── Archival ───────────────────────────────────────────────────────────────

def archive_destination(audio_path: Path, output_dir: Path) -> Path:
    """Preflight without writes; only an exact source OGG path may be reused."""
    ogg_path = output_dir.expanduser().resolve() / (audio_path.stem + ".ogg")
    in_place = audio_path.suffix.lower() == ".ogg" and ogg_path == audio_path.resolve()
    if ogg_path.is_symlink() or (ogg_path.exists() and not in_place):
        raise FileExistsError("Archive destination already exists; choose a different directory or named output.")
    return ogg_path


def write_transcript(output_path: Path, transcript: str) -> None:
    """Exclusive creation also rejects a file or symlink created after preflight."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        output.write(transcript)


def archive_to_ogg(audio_path: Path, output_dir: Path) -> Path:
    """Create a verified OGG archive without replacing existing files."""
    ogg_path = archive_destination(audio_path, output_dir)

    print(f"\n📦 Archiving to OGG: {ogg_path}")
    result = subprocess.run(
        [str(ARCHIVE_SCRIPT), str(audio_path), "--output", str(ogg_path)],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        print(result.stdout.strip())
        return ogg_path
    raise RuntimeError("OGG archival failed; the transcript and original audio have been preserved.")


# ─── AssemblyAI Backend (default) ──────────────────────

def _ms_to_ts(ms: int) -> str:
    """Milliseconds → MM:SS (or H:MM:SS past an hour), matching the Gemini format."""
    s = int(ms // 1000)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def _aai_body_line(u: dict) -> str:
    """`**[MM:SS] Label:** text`: a readable local label, `Speaker X` for an AssemblyAI cluster,
    or no label (words-only paragraphs whose speakers are not attributed)."""
    ts, text = _ms_to_ts(u["start"]), u["text"].strip()
    if u.get("label"):
        return f"**[{ts}] {u['label']}:** {text}"
    if u["speaker"] is None:
        return f"**[{ts}]** {text}"
    return f"**[{ts}] Speaker {u['speaker']}:** {text}"


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min" if seconds >= 60 else f"{seconds:.0f} s"


def _tracking_header(tracking: dict, utterances: list) -> tuple[str, str, str]:
    """(extra header lines, count text, note) for a transcript labelled by local tracking."""
    speakers = ", ".join(f"{label} {_minutes(sec)}" for label, sec in tracking["speakers"][:10])
    extra = (f"**Local Speaker Tracking:** `{tracking['backend']}` boundaries · "
             f"`{tracking['model_id']}` voiceprints · calibration `{tracking['calibration']}` · "
             f"timeline `{tracking['timeline']}`\n")
    if tracking.get("roster"):
        missing = tracking.get("roster_missing") or []
        extra += (f"**Roster:** {', '.join(tracking['roster'])}"
                  + (f" (no voiceprints: {', '.join(missing)})" if missing else "") + "\n")
    count_text = f"{len(utterances)} turns · {speakers or 'no speech attributed'}"
    note = """> [!note] Speakers were attributed locally from enrolled voiceprints (who speaks when).
> AssemblyAI ran without diarization, so no cloud voice embeddings were created.
> "A or B (uncertain)": the voices could not be told apart; "Guest n": an unenrolled
> voice; "Unknown speaker": speech no voiceprint matched. Check names against the
> content and roster (SKILL.md Step 2.5) and spot-check the review list."""
    review = [u for u in utterances if u.get("kind") in ("uncertain", "guest", "unknown", "overlap")]
    if review:
        lines = [f"> - [{_ms_to_ts(u['start'])}] {u['label']}, {(u['end'] - u['start']) / 1000:.1f} s"
                 for u in review[:REVIEW_MAX_ITEMS]]
        if len(review) > REVIEW_MAX_ITEMS:
            lines.append(f"> - …and {len(review) - REVIEW_MAX_ITEMS} more (see the timeline)")
        note += (f"\n\n> [!warning]- Speaker review: {len(review)} turn(s) to spot-check\n"
                 + "\n".join(lines))
    return extra, count_text, note


def _aai_format_transcript(utterances: list, audio_path: Path,
                           description: str | None,
                           metadata: dict | None = None,
                           *, local_speakers: bool = False, tracking: dict | None = None,
                           warning: str | None = None) -> str:
    """Format AssemblyAI utterances into the same markdown shape as the Gemini path.

    Default: speakers stay as acoustic clusters (Speaker A/B/C/D) — AssemblyAI does
    diarization, not voice-ID; SKILL.md Step 2.5 maps clusters → names.
    With local_speakers, AssemblyAI ran without diarization: `tracking` (a summary of the
    local speaker timeline) labels the utterances; without it they are unlabelled paragraphs.
    `warning` adds a warning callout (local tracking unavailable or failed).
    """
    now = datetime.now()
    total_dur = int(utterances[-1]["end"] / 1000) if utterances else 0
    n_speakers = len({u["speaker"] for u in utterances if u["speaker"] is not None})
    metadata = metadata or {}
    transcript_id = metadata.get("id") or "unknown"
    submitted_models = metadata.get("speech_models") or []
    submitted_models_text = ", ".join(submitted_models) if submitted_models else "not returned"
    language_code = metadata.get("language_code") or "unknown"
    language_confidence = metadata.get("language_confidence")
    speakers_expected = metadata.get("speakers_expected")
    language_text = language_code
    if language_confidence is not None:
        language_text += f" ({language_confidence:.4f})"
    expected_text = str(speakers_expected) if speakers_expected is not None else "not set"

    if local_speakers and tracking:
        stt_text = "AssemblyAI pre-recorded STT (words only) with local speaker tracking"
    elif local_speakers:
        stt_text = "AssemblyAI pre-recorded STT (words only; speakers attributed locally)"
    else:
        stt_text = f"AssemblyAI pre-recorded STT ({n_speakers} diarized speakers)"
    description = description or f"Transcript of {audio_path.stem}, {format_duration(total_dur)}, {stt_text}"
    desc_line = "\ndescription: " + json.dumps(description, ensure_ascii=False)
    requested_models_text = ", ".join(metadata.get("requested_speech_models") or submitted_models) or "not recorded"
    model_used = metadata.get("speech_model_used") or "not returned"
    requested_language = metadata.get("requested_language_code") or "not recorded"

    extra = ""
    if local_speakers and tracking:
        tool_text = "AssemblyAI pre-recorded STT (words only, no cloud diarization) + local speaker tracking"
        extra, count_text, note = _tracking_header(tracking, utterances)
    elif local_speakers:
        tool_text = "AssemblyAI pre-recorded STT (words only, no cloud diarization)"
        count_text = f"{len(utterances)} paragraphs · speakers not yet attributed"
        note = """> [!note] Speakers are not labelled yet. AssemblyAI ran without diarization
> (no cloud voice embeddings); speakers are attributed locally from the
> word sidecar (`<transcript stem>.words.json`) next to this transcript."""
    else:
        tool_text = "AssemblyAI pre-recorded STT (speaker diarization)"
        count_text = f"{len(utterances)} utterances · {n_speakers} speaker clusters"
        note = """> [!note] Speakers below are acoustic clusters (Speaker A/B/C/D), not names.
> AssemblyAI does diarization, not voice identification. Map clusters → real
> names using the attendee roster + content cues (SKILL.md Step 2.5) before
> writing the summary."""
    if warning:
        note = f"> [!warning] {warning}\n\n{note}"

    header = f"""---
title: {json.dumps("Meeting Transcript - " + audio_path.stem, ensure_ascii=False)}
date: {now.strftime('%Y-%m-%d')}
type: transcript
source: {json.dumps(audio_path.name, ensure_ascii=False)}
created: {now.strftime('%Y-%m-%d')}{desc_line}
tags:
  - "#transcript"
  - "#meeting"
---

# Meeting Transcript

**Source:** `{audio_path.name}`
**Transcribed:** {now.strftime('%Y-%m-%d %H:%M')}
**Tool:** {tool_text}
**AAI Transcript ID:** `{transcript_id}`
**AAI Submitted Models:** `{requested_models_text}`
**AAI Returned Models:** `{submitted_models_text}`
**AAI Model Used:** `{model_used}`
**AAI Requested Language:** `{requested_language}`
**AAI Detected Language:** `{language_text}`
**AAI Speakers Expected:** `{expected_text}`
{extra}**Duration:** ~{format_duration(total_dur)} · {count_text}

{note}

---

"""
    body = "\n\n".join(_aai_body_line(u) for u in utterances)
    return header + body + "\n"


def utterances_sidecar_path(transcript_path: Path) -> Path:
    """`<transcript stem>.utterances.json` next to the transcript (voiceprint Step 2.5a input)."""
    return transcript_path.with_name(transcript_path.stem + ".utterances.json")


def _aai_utterances_sidecar(utterances: list, metadata: dict | None = None) -> dict:
    """Utterances sidecar (schema 1): exact AssemblyAI millisecond timestamps per cluster.

    `speaker_map` stays null; confirmed names are added after Step 2.5. It holds meeting
    text but no biometrics, so it lives next to the transcript with the same privacy.
    """
    metadata = metadata or {}
    duration_s = metadata.get("audio_duration")
    return {
        "schema": 1,
        "source": "assemblyai",
        "transcript_id": metadata.get("id"),
        "audio_duration_ms": (int(duration_s * 1000) if duration_s is not None
                              else (int(utterances[-1]["end"]) if utterances else 0)),
        "speaker_map": None,
        "utterances": [
            {"speaker": u["speaker"], "start": int(u["start"]),
             "end": int(u["end"]), "text": (u.get("text") or "").strip()}
            for u in utterances
        ],
    }


def write_utterances_sidecar(output_path: Path, utterances: list, metadata: dict) -> Path | None:
    """Fail-soft: a sidecar problem never fails a completed transcription."""
    sidecar = utterances_sidecar_path(output_path)
    try:
        with sidecar.open("x", encoding="utf-8") as handle:
            json.dump(_aai_utterances_sidecar(utterances, metadata), handle, ensure_ascii=False, indent=1)
            handle.write("\n")
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠ Utterances sidecar not written ({type(e).__name__}); voiceprint check will be skipped.")
        return None
    print(f"   Utterances sidecar: {sidecar.name}")
    return sidecar


def words_sidecar_path(transcript_path: Path) -> Path:
    """`<transcript stem>.words.json` next to the transcript (local speaker attribution input)."""
    return transcript_path.with_name(transcript_path.stem + ".words.json")


def _voiceprint_words():
    """voiceprint.words is standard-library only, so this import needs no voiceprint extras."""
    scripts_dir = str(SKILL_DIR / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from voiceprint import words
    return words


def write_words_sidecar(output_path: Path, words: list, metadata: dict) -> Path | None:
    """Fail-soft like the utterances sidecar: the transcript is already written."""
    sidecar = words_sidecar_path(output_path)
    try:
        vp_words = _voiceprint_words()
        with sidecar.open("x", encoding="utf-8") as handle:
            handle.write(vp_words.dump_sidecar(vp_words.sidecar_from_assemblyai(words, metadata)))
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠ Words sidecar not written ({type(e).__name__}); local speaker attribution will be skipped.")
        return None
    print(f"   Words sidecar: {sidecar.name}")
    return sidecar


def _aai_local_paragraphs(words: list) -> list:
    """Unlabelled paragraphs in the utterance shape `_aai_format_transcript` renders."""
    vp_words = _voiceprint_words()
    turns = vp_words.rebuild_turns(vp_words.unattributed(vp_words.words_from_assemblyai(words)),
                                   max_pause=LOCAL_PARAGRAPH_PAUSE, max_turn_s=LOCAL_PARAGRAPH_MAX)
    return [{"speaker": None, "start": int(round(t.start * 1000)), "end": int(round(t.end * 1000)),
             "text": t.text} for t in turns if t.text]


# ─── Local speaker tracking ─────────────────────────────────────────────────

def _voiceprint_module(name: str):
    """A standard-library voiceprint module (timeline, words, config): no voiceprint extras."""
    _voiceprint_words()                          # puts scripts/ on sys.path
    import importlib
    return importlib.import_module(f"voiceprint.{name}")


def local_speakers_default() -> bool:
    value = os.environ.get(LOCAL_SPEAKERS_ENV)
    return LOCAL_SPEAKERS_DEFAULT if value not in ("0", "1") else value == "1"


ROSTER_HEADING = re.compile(r"^\s*(possible speakers|speakers|attendees|participants|roster)\s*:\s*(.*)$", re.I)


def roster_from_context(context: str | None) -> list[str]:
    """Attendee names from a context file (CONTEXT_TEMPLATE.md): the `- Name (role)` bullets
    under a "Possible speakers:" (or Speakers/Attendees/Participants/Roster) line, or a
    comma-separated list on that line. Bracketed template placeholders are ignored."""
    names: list[str] = []
    lines = (context or "").splitlines()
    for i, line in enumerate(lines):
        m = ROSTER_HEADING.match(line)
        if not m:
            continue
        found = m.group(2).split(",")
        for nxt in lines[i + 1:]:
            b = re.match(r"^\s*[-*•]\s+(.*)$", nxt)
            if not b:
                break
            found.append(b.group(1))
        for item in found:
            name = re.sub(r"\s*[(（].*$", "", item).strip().strip("*_`")
            if name and not name.startswith("[") and name not in names:
                names.append(name)
    return names


def timeline_path_for(output_path: Path) -> Path:
    """`<transcript stem>.timeline.json` in MEETING_DOCUMENTER_TIMELINE_DIR (default
    ${XDG_DATA_HOME:-~/.local/share}/meeting-documenter/timelines). Refused inside the notes
    vault or this clone: the timeline is private working data, not a note."""
    raw = os.environ.get(TIMELINE_DIR_ENV)
    if raw:
        directory = Path(raw).expanduser()
    else:
        base = os.environ.get("XDG_DATA_HOME")
        directory = (Path(base).expanduser() if base else Path.home() / ".local" / "share") \
            / "meeting-documenter" / "timelines"
    path = directory.absolute() / (output_path.stem + ".timeline.json")
    _voiceprint_module("config").assert_outside_vault(path)
    return path


@dataclass
class LocalTrackingPlan:
    roster: list[str]
    channel: str | None
    backend: str
    meeting_key: str | None
    timeline_path: Path
    check: dict


def _track_cli_args(args, roster: list[str]) -> list[str]:
    extra = ["--roster", ",".join(roster)] if roster else []
    return extra + (["--channel", args.channel] if args.channel else [])


def plan_local_tracking(args, audio_path: Path, output_path: Path,
                        context: str | None) -> tuple[LocalTrackingPlan | None, str]:
    """(plan, "") when local tracking can name speakers, else (None, reason). Runs the cheap
    `track-check` probe (no models, no audio) before anything is uploaded."""
    roster = [x.strip() for x in (args.roster or "").split(",") if x.strip()] or roster_from_context(context)
    try:
        timeline_path = timeline_path_for(output_path)
    except ValueError as e:
        return None, f"timeline location refused: {e}"
    cmd = [sys.executable, str(TRACK_CLI), "track-check", "--backend", args.track_backend,
           *(["--roster", ",".join(roster)] if roster else [])]
    try:
        probe = subprocess.run(cmd, capture_output=True, text=True, timeout=TRACK_CHECK_TIMEOUT)
        check = json.loads(probe.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as e:
        return None, f"readiness check failed ({type(e).__name__})"
    if not check.get("ready"):
        return None, f"{check.get('status', 'not ready')}: {check.get('reason', '')}".rstrip(": ")
    if timeline_path.exists() or timeline_path.is_symlink():
        raise FileExistsError("Speaker timeline for this transcript already exists; choose a new output path.")
    key = next((m.group(1) for p in (output_path, audio_path) if (m := MEETING_KEY_RE.search(p.name))), None)
    return LocalTrackingPlan(roster, args.channel, check.get("backend") or args.track_backend, key,
                             timeline_path, check), ""


class LocalTracker:
    """`identify_speakers.py track` in a child process, started before the upload so it runs
    while AssemblyAI transcribes. Its timeline, status and log stay in a private temporary
    directory until `result()`; `close()` stops a still-running child and removes them."""

    def __init__(self, plan: LocalTrackingPlan, audio_path: Path, args):
        self.plan = plan
        self._tmp = tempfile.TemporaryDirectory(prefix="meeting_documenter_track_")
        tmp = Path(self._tmp.name)
        self.out, self.status, self.log_path = tmp / "timeline.json", tmp / "status.json", tmp / "track.log"
        cmd = [sys.executable, str(TRACK_CLI), "track", "--audio", str(audio_path),
               "--backend", plan.backend, "--out", str(self.out), "--json", str(self.status),
               *(["--meeting-key", plan.meeting_key] if plan.meeting_key else []),
               *_track_cli_args(args, plan.roster)]
        self._log = self.log_path.open("w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)

    def result(self, timeout: float):
        """(Timeline, status dict) or (None, reason)."""
        try:
            rc = self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.close()
            return None, f"timed out after {timeout:g} s"
        status = {}
        if self.status.exists():
            try:
                status = json.loads(self.status.read_text(encoding="utf-8"))
            except ValueError:
                pass
        if status.get("status") != "ok" or not self.out.exists():
            reason = status.get("reason") or f"exit code {rc}"
            return None, f"{status.get('status', 'failed')}: {reason}"[:300]
        try:
            return _voiceprint_module("timeline").from_json(self.out.read_text(encoding="utf-8")), status
        except (OSError, ValueError, KeyError) as e:
            return None, f"unreadable timeline ({type(e).__name__})"

    def log_tail(self, n: int = 5) -> str:
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()
        self._tmp.cleanup()


def _aai_tracked_turns(words: list, tl) -> list:
    """Words attributed to the local timeline, regrouped into turns with readable labels."""
    vp_words = _voiceprint_words()
    turns = vp_words.rebuild_turns(vp_words.assign_words(vp_words.words_from_assemblyai(words), tl),
                                   max_turn_s=LOCAL_PARAGRAPH_MAX)
    return [{"speaker": t.speaker, "kind": t.kind, "label": vp_words.display_label(t.speaker),
             "start": int(round(t.start * 1000)), "end": int(round(t.end * 1000)), "text": t.text}
            for t in turns if t.text]


def _tracking_summary(plan: LocalTrackingPlan, tl) -> dict:
    vp_words = _voiceprint_words()
    speech: dict[str, float] = {}
    for t in tl.exclusive:
        speech[t.speaker] = speech.get(t.speaker, 0.0) + (t.end - t.start)
    ranked = sorted(speech.items(), key=lambda kv: -kv[1])
    calibration = tl.meta.get("calibration") or {}
    return {"backend": tl.meta.get("backend") or plan.backend,
            "model_id": tl.meta.get("model_id") or plan.check.get("model_id") or "unknown",
            "calibration": calibration.get("source") or plan.check.get("calibration") or "unknown",
            "timeline": plan.timeline_path.name, "roster": plan.roster,
            "roster_missing": plan.check.get("roster_missing") or [],
            "speakers": [(vp_words.display_label(k), v) for k, v in ranked]}


def write_timeline(plan: LocalTrackingPlan, tl, audio_path: Path, output_path: Path) -> Path | None:
    """Fail-soft like the sidecars: the transcript is already written."""
    try:
        tl.meta.update(transcript=output_path.name, audio=audio_path.name,
                       words=words_sidecar_path(output_path).name)
        if plan.channel:
            tl.meta.setdefault("channel", plan.channel)
        plan.timeline_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(plan.timeline_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_voiceprint_module("timeline").to_json(tl) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"   ⚠ Speaker timeline not written ({type(e).__name__}); enrollment from it will be unavailable.")
        return None
    print(f"   Speaker timeline: {plan.timeline_path}")
    return plan.timeline_path


def run_assemblyai_backend(args, audio_path: Path, output_path: Path,
                           context: str | None, plan: LocalTrackingPlan | None = None) -> None:
    """Transcribe via AssemblyAI pre-recorded API (raw HTTP per integration guide).

    Sends the original file in one request; Gemini preparation and splitting
    are not applied. Provider file limits apply. With a local tracking `plan`,
    AssemblyAI returns words only and the local speaker timeline labels them.
    """
    try:
        import requests
    except ImportError:
        print("Error: requests not installed. Run transcribe.sh (auto-installs).")
        sys.exit(1)

    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        print("Error: ASSEMBLYAI_API_KEY not set. Set it in your environment or configured .env file.")
        sys.exit(1)
    headers = {"authorization": key}  # raw key, not a Bearer token

    info = get_audio_info(audio_path)
    print(f"🎵 Audio: {audio_path.name}")
    print(f"   Duration: {format_duration(info['duration_int'])}")
    print(f"   Size: {info['file_size'] / 1048576:.1f} MB")
    speech_models = args.aai_speech_models or ASSEMBLYAI_MODELS
    local_speakers = plan is not None
    print(f"   Backend: AssemblyAI ({' → '.join(speech_models)})")
    if local_speakers:
        print("   Speakers: local speaker tracking (AssemblyAI speaker_labels off, word sidecar)")

    # Optional high-precision keyterm boost (names + domain jargon)
    keyterms: list[str] = []
    if args.keyterms_file:
        kt_path = args.keyterms_file.expanduser().resolve()
        if kt_path.exists():
            keyterms = [ln.strip() for ln in kt_path.read_text().splitlines() if ln.strip()]
            print(f"   Keyterms: {len(keyterms)} terms")
        else:
            print(f"   Warning: keyterms file not found: {kt_path}")
    if context:
        print(f"   Prompt context: {len(context)} chars")

    # The request (the upload URL is added once the audio is uploaded).
    body = {
        "speech_models": speech_models,
        "speaker_labels": not local_speakers,
    }
    if args.language_code:
        body["language_code"] = args.language_code
    else:
        body["language_detection"] = True
    if args.speakers_expected is not None and local_speakers:
        print("   Note: omitting speakers_expected because AssemblyAI diarization is off.")
    elif args.speakers_expected is not None:
        body["speakers_expected"] = args.speakers_expected
    if keyterms:
        body["keyterms_prompt"] = keyterms
    if context and not keyterms:
        body["prompt"] = context  # natural-language conditioning (U3 Pro)
    elif context and keyterms:
        print("   Note: omitting AAI prompt because keyterms_prompt and prompt cannot be combined.")
    speaker_msg = (
        f", speakers_expected={args.speakers_expected}"
        if args.speakers_expected is not None
        else ""
    )
    labels_msg = "speaker_labels=false" if local_speakers else f"speaker_labels{speaker_msg}"

    # Local tracking runs on this machine while AssemblyAI transcribes.
    tracker = LocalTracker(plan, audio_path, args) if plan else None
    tracked = None
    try:
        if tracker:
            print(f"🎙️  Local speaker tracking started ({plan.backend}"
                  + (f", roster: {', '.join(plan.roster)}" if plan.roster else ", everyone enrolled") + ")")
        r = _aai_transcribe(requests, headers, audio_path, body, args.aai_poll_timeout,
                            f"{' → '.join(speech_models)}, {labels_msg}")
        if tracker:
            print("⏳ Waiting for local speaker tracking...")
            tracked = tracker.result(args.track_timeout)
            if tracked[0] is None:
                print(f"   ⚠ Local speaker tracking failed ({tracked[1]})")
                tail = tracker.log_tail()
                if tail:
                    print("   " + tail.replace("\n", "\n   "))
    finally:
        if tracker:
            tracker.close()

    warning = getattr(args, "local_fallback", None)
    tracking = None
    if local_speakers:
        words = r.get("words") or []
        if not words:
            raise RuntimeError("AssemblyAI returned no words; no transcript was written.")
        tl = tracked[0] if tracked else None
        if tl is not None:
            status = tracked[1]
            utterances = _aai_tracked_turns(words, tl)
            tracking = _tracking_summary(plan, tl)
            info = status.get("info") or {}
            print(f"   Words: {len(words)} | turns: {len(utterances)} | local tracking "
                  f"{info.get('xrt', '?')}x real time, peak footprint {info.get('peak_footprint_gb', '?')} GB")
            for label, sec in tracking["speakers"]:
                print(f"     {label:<32} {sec:8.1f} s")
        else:
            utterances = _aai_local_paragraphs(words)
            warning = (f"Local speaker tracking failed ({tracked[1] if tracked else 'not run'}); speakers "
                       "are not labelled. Map speakers from content, or re-run the transcription.")
            print(f"   Words: {len(words)} | paragraphs: {len(utterances)} (speakers not attributed)")
        speakers = []
    else:
        utterances = r.get("utterances") or []
        if not utterances:
            raise RuntimeError("AssemblyAI returned no diarized speech; no transcript was written.")
        speakers = sorted({u["speaker"] for u in utterances})
        print(f"   Utterances: {len(utterances)} | speaker clusters: {len(speakers)} {speakers}")
    print(f"   Detected language: {r.get('language_code')} ({r.get('language_confidence')})")
    print(f"   Submitted models returned by AAI: {r.get('speech_models')}")
    r["requested_speech_models"] = speech_models
    r["requested_language_code"] = args.language_code or "automatic detection"
    r["speakers_expected"] = args.speakers_expected

    transcript = _aai_format_transcript(utterances, audio_path, args.description, r,
                                        local_speakers=local_speakers, tracking=tracking,
                                        warning=warning)
    write_transcript(output_path, transcript)
    print(f"\n✅ Transcript saved: {output_path}")
    if local_speakers:
        write_words_sidecar(output_path, words, r)
        if tracking:
            write_timeline(plan, tl, audio_path, output_path)
    else:
        write_utterances_sidecar(output_path, utterances, r)

    if args.archive_dir is not None:
        archive_to_ogg(audio_path, args.archive_dir)
    else:
        print("\n📦 Archival skipped (no --archive-dir). Use SKILL.md Step 2b for named archive.")

    print(f"\n{'=' * 50}")
    print(f"📊 AssemblyAI backend complete")
    if tracking:
        review = sum(1 for u in utterances if u["kind"] in ("uncertain", "guest", "unknown", "overlap"))
        print(f"   Speakers: tracked locally; {review} uncertain/guest/unknown turn(s) to review (Step 2.5)")
    elif local_speakers:
        print("   Speakers: not labelled (local tracking failed; see the warning in the transcript)")
    else:
        print(f"   Speakers: {len(speakers)} clusters (map A/B/C/D → names in Step 2.5)")
    if warning:
        print(f"   ⚠ {warning}")
    print(f"   Output: {output_path}")
    print(f"{'=' * 50}")


def _aai_transcribe(requests, headers: dict, audio_path: Path, body: dict, poll_timeout: float,
                    submit_msg: str) -> dict:
    """Upload the raw audio, submit `body` with its upload URL, and poll until completed."""
    # 1) Upload the raw binary body, not multipart form data.
    print("\n⬆️  Uploading (raw binary)...")
    with open(audio_path, "rb") as f:
        up = requests.post(f"{ASSEMBLYAI_BASE}/v2/upload", headers=headers,
                           data=f, timeout=600)
    if up.status_code >= 400:
        raise RuntimeError(f"AssemblyAI upload failed (HTTP {up.status_code}).")
    upload_url = up.json()["upload_url"]

    # 2) Submit the explicitly selected speech models.
    print(f"📤 Submitting ({submit_msg})...")
    sub = requests.post(f"{ASSEMBLYAI_BASE}/v2/transcript",
                        headers={**headers, "content-type": "application/json"},
                        json={"audio_url": upload_url, **body}, timeout=60)
    if sub.status_code >= 400:
        raise RuntimeError(f"AssemblyAI submission failed (HTTP {sub.status_code}).")
    tid = sub.json()["id"]
    print(f"   Transcript ID: {tid}")

    # 3) Poll
    print("⏳ Polling...")
    t0 = time.monotonic()
    while True:
        remaining = poll_timeout - (time.monotonic() - t0)
        if remaining <= 0:
            raise TimeoutError("AssemblyAI polling timed out; the remote job may still be running.")
        poll = requests.get(f"{ASSEMBLYAI_BASE}/v2/transcript/{tid}",
                            headers=headers, timeout=min(60, remaining))
        if poll.status_code >= 400:
            raise RuntimeError(f"AssemblyAI polling failed (HTTP {poll.status_code}).")
        r = poll.json()
        st = r.get("status")
        if st == "completed":
            print(f"   ✅ Completed in {time.monotonic() - t0:.0f}s")
            return r
        if st == "error":
            raise RuntimeError("AssemblyAI transcription failed; inspect the job in your provider dashboard.")
        if st not in {"queued", "processing"}:
            raise RuntimeError("AssemblyAI returned an unexpected job status.")
        time.sleep(min(5, max(0, poll_timeout - (time.monotonic() - t0))))


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe meeting audio with automatic chunking and truncation recovery",
        allow_abbrev=False
    )
    parser.add_argument("audio_file", type=Path, help="Path to audio file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output transcript path")
    parser.add_argument("--backend", choices=["assemblyai", "gemini"], default="assemblyai",
                        help="Transcription backend (default: assemblyai)")
    parser.add_argument("--keyterms-file", type=Path, default=None,
                        help="AssemblyAI: newline-separated names and domain terms")
    parser.add_argument("--aai-speech-models", nargs="+", default=None,
                        help="AssemblyAI: ordered speech model list")
    parser.add_argument("--speakers-expected", type=int, default=None,
                        help="AssemblyAI: expected speaker count")
    parser.add_argument("--language-code", default=None,
                        help="AssemblyAI: language code; omit for automatic detection")
    parser.add_argument("--aai-poll-timeout", type=float, default=3600,
                        help="AssemblyAI: maximum polling time in seconds (default: 3600)")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Speaker/domain context: Gemini prefix or AssemblyAI prompt")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--description", default=None,
                        help="Optional description for transcript frontmatter (~150 chars)")
    parser.add_argument("--local-speakers", action=argparse.BooleanOptionalAction,
                        default=local_speakers_default(),
                        help="AssemblyAI (beta): words only (speaker_labels=false), speakers tracked "
                             "locally from enrolled voiceprints; falls back to cloud diarization when "
                             f"local tracking is not set up (default from {LOCAL_SPEAKERS_ENV}=1|0)")
    parser.add_argument("--roster", default=None,
                        help="Local speakers: comma-separated attendee names (default: the context "
                             "file's 'Possible speakers:' list, else everyone enrolled)")
    parser.add_argument("--channel", choices=["phone", "zoom", "room"], default=None,
                        help="Local speakers: recording channel (per-channel voiceprints, audit stratum)")
    parser.add_argument("--track-backend", choices=["nemotron3", "seg3", "auto"], default="nemotron3",
                        help="Local speakers: boundary model that must be installed (default nemotron3)")
    parser.add_argument("--track-timeout", type=float, default=3600,
                        help="Local speakers: maximum wait for local tracking after AssemblyAI finishes, s")
    archive_options = parser.add_mutually_exclusive_group()
    archive_options.add_argument("--archive-dir", type=Path, default=None,
                                 help="Opt in to OGG archival in this directory")
    archive_options.add_argument("--no-archive", action="store_true",
                                 help="Compatibility option; archival is already off by default")
    args = parser.parse_args()
    if args.aai_poll_timeout <= 0 or not math.isfinite(args.aai_poll_timeout):
        parser.error("--aai-poll-timeout must be a positive finite number")
    if args.speakers_expected is not None and args.speakers_expected < 1:
        parser.error("--speakers-expected must be positive")
    if args.track_timeout <= 0 or not math.isfinite(args.track_timeout):
        parser.error("--track-timeout must be a positive finite number")
    for context_path in (args.context_file, args.keyterms_file):
        if context_path is not None and not context_path.expanduser().is_file():
            parser.error("The specified context or keyterms file does not exist")

    audio_path = args.audio_file.expanduser().resolve()
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    output_path = args.output
    if output_path:
        output_path = output_path.expanduser().absolute()
    else:
        output_path = audio_path.with_name(audio_path.stem + "-Transcript.md")

    # Preserve previous transcripts and source aliases; do this before provider work.
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("Transcript destination already exists; choose a new output path.")

    # Load context
    context = None
    if args.context_file:
        ctx_path = args.context_file.expanduser().resolve()
        if ctx_path.exists():
            context = ctx_path.read_text()
        else:
            print(f"Warning: Context file not found: {ctx_path}")

    # Local speaker tracking must be ready before anything is uploaded; otherwise AssemblyAI
    # diarizes as before and the transcript says why.
    plan = None
    if args.local_speakers and args.backend != "assemblyai":
        print("Note: --local-speakers applies to the AssemblyAI backend only; ignored.")
        args.local_speakers = False
    elif args.local_speakers:
        plan, why = plan_local_tracking(args, audio_path, output_path, context)
        if plan is None:
            args.local_speakers = False
            args.local_fallback = (f"Local speaker tracking was requested but is unavailable ({why}). "
                                   "AssemblyAI cloud diarization was used instead.")
            print(f"⚠️  {args.local_fallback}")
        elif plan.check.get("roster_missing"):
            print(f"   Roster without voiceprints (their speech will show as Guest n): "
                  f"{', '.join(plan.check['roster_missing'])}")
    if args.backend == "assemblyai":
        sidecar_path = (words_sidecar_path if args.local_speakers else utterances_sidecar_path)(output_path)
        if sidecar_path.exists() or sidecar_path.is_symlink():
            kind = "Words" if args.local_speakers else "Utterances"
            raise FileExistsError(f"{kind} sidecar for this transcript already exists; choose a new output path.")
    if args.archive_dir is not None:
        archive_path = archive_destination(audio_path, args.archive_dir)
        if archive_path == output_path.resolve():
            raise ValueError("Transcript and archive destinations must be distinct.")

    if args.backend == "assemblyai":
        run_assemblyai_backend(args, audio_path, output_path, context, plan)
        return

    # Own all prepared audio and retry chunks from before the first write.
    original_audio_path = audio_path
    with tempfile.TemporaryDirectory(prefix="meeting_documenter_") as temp_name:
        temp_dir = Path(temp_name)
        print(f"🎵 Audio: {audio_path.name}")
        info = get_audio_info(audio_path)
        print(f"   Duration: {format_duration(info['duration_int'])}")
        print(f"   Size: {info['file_size'] / 1048576:.1f} MB")
        print(f"   Format: {info['mime_type']}")
        print(f"   Model: {args.model}")
        if context:
            print(f"   Context: {len(context)} chars")

        # ── Step 1b: PREPARE (trim trailing silence, fix codec/container) ──
        print(f"\n🔧 Preparing audio...")
        audio_path, effective_duration = prepare_audio(audio_path, info["duration_int"], temp_dir)
        if audio_path != original_audio_path:
            info["duration_int"] = effective_duration
            info["duration"] = float(effective_duration)
        else:
            print(f"   ✅ No preparation needed")

        # ── Step 2: DECIDE ──
        if info["duration_int"] > CHUNK_THRESHOLD:
            print(f"\n✂️  Audio exceeds {CHUNK_THRESHOLD}s — splitting at silence points...")
            chunks = split_audio(audio_path, output_dir=temp_dir / "chunks")
            print(f"   Split into {len(chunks)} chunks")
        else:
            print(f"\n📝 Audio under {CHUNK_THRESHOLD}s — single-chunk transcription")
            chunks = [AudioChunk(path=audio_path, offset_seconds=0,
                                 duration=info["duration_int"])]

        # ── Step 3-4: TRANSCRIBE with retry ──
        from google import genai

        client = genai.Client(api_key=get_api_key())
        segments = transcribe_with_retry(client, chunks, args.model, context, temp_dir=temp_dir)

        if not any(segment.text.strip() for segment in segments):
            raise RuntimeError("Gemini returned no transcript text; no output was written.")

        # ── Step 5: COMBINE ──
        print(f"\n📋 Combining {len(segments)} segments...")
        transcript = combine_segments(segments, original_audio_path, args.model, args.description)

        # ── Step 5b: MONOTONICITY CHECK ──
        transcript, ts_fixes = fix_timestamp_monotonicity(transcript)
        if ts_fixes > 0:
            print(f"\n🔧 Fixed {ts_fixes} timestamp regression(s) (Gemini artifact)")

        # Write output
        write_transcript(output_path, transcript)
        print(f"\n✅ Transcript saved: {output_path}")

        # ── Step 6: ARCHIVE (from original, not trimmed) ──
        if args.archive_dir is not None:
            archive_to_ogg(original_audio_path, args.archive_dir)
        else:
            print("\n📦 Archival skipped. Use --archive-dir or SKILL.md Step 2b for named archive.")

        # ── Step 7: REPORT ──
        total_duration = sum(s.duration for s in segments)
        print(f"\n{'='*50}")
        print(f"📊 Pipeline complete")
        print(f"   Duration: {format_duration(total_duration)}")
        print(f"   Chunks: {len(segments)}")
        print(f"   Output: {output_path}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
