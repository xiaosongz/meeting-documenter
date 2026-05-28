#!/usr/bin/env python3
"""
Meeting Transcription Pipeline
-------------------------------
Full pipeline: detect audio → decide chunking → transcribe via Gemini →
detect truncation with auto-retry → combine segments → archive to OGG.

Usage:
    transcribe_pipeline.py <audio-path> [--output <path>] [--context-file <path>] [--model <model>]

Environment:
    GOOGLE_API_KEY or GEMINI_API_KEY must be set (loaded by transcribe.sh from .env)
"""

import os
import sys
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai package not installed. Run transcribe.sh to auto-install.")
    sys.exit(1)


# ─── Constants ──────────────────────────────────────────────────────────────

SKILL_DIR = Path(__file__).parent.parent.resolve()
SPLIT_SCRIPT = SKILL_DIR / "scripts" / "split-audio.sh"
ARCHIVE_SCRIPT = SKILL_DIR / "scripts" / "compress-audio.sh"

MAX_OUTPUT_TOKENS = 65536  # gemini-3-flash-preview hard cap
MAX_INLINE_SIZE = 20 * 1024 * 1024  # 20MB — above this, use Files API
CHUNK_THRESHOLD = 3600  # 60 minutes — split if longer
DEFAULT_MODEL = "gemini-3-flash-preview"

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


def prepare_audio(audio_path: Path, duration_seconds: int) -> tuple[Path, int]:
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
    prepared_path = Path(f"/tmp/prepared_{os.getpid()}_{audio_path.stem}{target_ext}")

    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path)]
    if needs_trim:
        cmd.extend(["-t", str(trim_to)])
    cmd.extend(["-c", "copy", str(prepared_path), "-y"])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        # Stream copy failed — fall back to re-encode as OGG Vorbis
        print(f"   ⚠ Stream copy failed, re-encoding to OGG...")
        prepared_path = prepared_path.with_suffix(".ogg")
        cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path)]
        if needs_trim:
            cmd.extend(["-t", str(trim_to)])
        cmd.extend(["-c:a", "libvorbis", "-q:a", "4", str(prepared_path), "-y"])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"   ❌ Re-encode failed: {result.stderr.strip()}")
            print(f"   Proceeding with original audio")
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
                output_dir: Path | None = None) -> list[AudioChunk]:
    """Split audio using silence-aware split-audio.sh."""
    if output_dir is None:
        output_dir = Path(f"/tmp/meeting_chunks_{os.getpid()}")

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


def resplit_chunk(chunk: AudioChunk, target_duration: int) -> list[AudioChunk]:
    """Re-split a chunk at a shorter target duration for truncation recovery."""
    output_dir = chunk.path.parent / f"resplit_d{chunk.depth + 1}"
    sub_chunks = split_audio(chunk.path, target_duration, output_dir)
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
                          context: str | None, max_depth: int = 2
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
            sub_chunks = resplit_chunk(chunk, new_target)
            results.extend(
                transcribe_with_retry(client, sub_chunks, model, context, max_depth)
            )

    return results


# ─── Combination ────────────────────────────────────────────────────────────

def combine_segments(segments: list[TranscriptSegment], audio_path: Path,
                     model: str, description: str | None = None) -> str:
    """Combine transcript segments into a single document with frontmatter."""
    now = datetime.now()
    total_dur = sum(s.duration for s in segments)

    # Build description line for frontmatter
    desc_line = ""
    if description:
        desc_line = f'\ndescription: "{description}"'
    else:
        # Auto-generate a minimal description
        desc_line = f'\ndescription: "Transcript of {audio_path.stem}, {format_duration(total_dur)}, {len(segments)} segment(s)"'

    header = f"""---
title: "Meeting Transcript - {audio_path.stem}"
date: {now.strftime('%Y-%m-%d')}
type: transcript
source: {audio_path.name}
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

def archive_to_ogg(audio_path: Path) -> Path | None:
    """Create OGG archive copy if source isn't already OGG."""
    if audio_path.suffix.lower() == ".ogg":
        print("\n📦 Already OGG, skipping archival.")
        return None

    ogg_path = audio_path.with_suffix(".ogg")
    print(f"\n📦 Archiving to OGG: {ogg_path.name}")
    result = subprocess.run(
        [str(ARCHIVE_SCRIPT), str(audio_path), "--output", str(ogg_path)],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode == 0:
        print(result.stdout.strip())
        return ogg_path
    else:
        print(f"   Warning: OGG archival failed: {result.stderr.strip()}")
        return None


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe meeting audio with automatic chunking and truncation recovery"
    )
    parser.add_argument("audio_file", type=Path, help="Path to audio file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Output transcript path")
    parser.add_argument("--context-file", type=Path, default=None,
                        help="Speaker/domain context file for Gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Gemini model (default: {DEFAULT_MODEL})")
    parser.add_argument("--description", default=None,
                        help="Optional description for transcript frontmatter (~150 chars)")
    parser.add_argument("--no-archive", action="store_true",
                        help="Skip built-in OGG archival (use when SKILL.md Step 2b handles it)")
    args = parser.parse_args()

    audio_path = args.audio_file.expanduser().resolve()
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    output_path = args.output
    if output_path:
        output_path = output_path.expanduser().resolve()
    else:
        output_path = audio_path.with_name(audio_path.stem + "-Transcript.md")

    # Load context
    context = None
    if args.context_file:
        ctx_path = args.context_file.expanduser().resolve()
        if ctx_path.exists():
            context = ctx_path.read_text()
        else:
            print(f"Warning: Context file not found: {ctx_path}")

    # ── Step 1: DETECT ──
    original_audio_path = audio_path  # preserve for archival
    # PID-scoped chunk dir; split_audio() writes here when output_dir omitted.
    # Cleanup runs in finally below so chunks don't survive pipeline crashes —
    # meeting audio is private and /tmp persists across reboots on some FS.
    chunks_base_dir = Path(f"/tmp/meeting_chunks_{os.getpid()}")
    try:
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
        audio_path, effective_duration = prepare_audio(audio_path, info["duration_int"])
        if audio_path != original_audio_path:
            info["duration_int"] = effective_duration
            info["duration"] = float(effective_duration)
        else:
            print(f"   ✅ No preparation needed")

        # ── Step 2: DECIDE ──
        if info["duration_int"] > CHUNK_THRESHOLD:
            print(f"\n✂️  Audio exceeds {CHUNK_THRESHOLD}s — splitting at silence points...")
            chunks = split_audio(audio_path)
            print(f"   Split into {len(chunks)} chunks")
        else:
            print(f"\n📝 Audio under {CHUNK_THRESHOLD}s — single-chunk transcription")
            chunks = [AudioChunk(path=audio_path, offset_seconds=0,
                                 duration=info["duration_int"])]

        # ── Step 3-4: TRANSCRIBE with retry ──
        client = genai.Client(api_key=get_api_key())
        segments = transcribe_with_retry(client, chunks, args.model, context)

        # ── Step 5: COMBINE ──
        print(f"\n📋 Combining {len(segments)} segments...")
        transcript = combine_segments(segments, audio_path, args.model, args.description)

        # ── Step 5b: MONOTONICITY CHECK ──
        transcript, ts_fixes = fix_timestamp_monotonicity(transcript)
        if ts_fixes > 0:
            print(f"\n🔧 Fixed {ts_fixes} timestamp regression(s) (Gemini artifact)")

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript)
        print(f"\n✅ Transcript saved: {output_path}")

        # ── Step 6: ARCHIVE (from original, not trimmed) ──
        if not args.no_archive:
            archive_to_ogg(original_audio_path)
        else:
            print("\n📦 Archival skipped (--no-archive). Use SKILL.md Step 2b for named archive.")

        # ── Step 7: REPORT ──
        total_duration = sum(s.duration for s in segments)
        print(f"\n{'='*50}")
        print(f"📊 Pipeline complete")
        print(f"   Duration: {format_duration(total_duration)}")
        print(f"   Chunks: {len(segments)}")
        print(f"   Output: {output_path}")
        print(f"{'='*50}")
    finally:
        # ── Step 8: CLEANUP (runs on success and on exception) ──
        if audio_path != original_audio_path and audio_path.exists():
            try:
                audio_path.unlink()
                print(f"\n🧹 Cleaned up prepared temp file: {audio_path.name}")
            except OSError as e:
                print(f"\n⚠ Failed to remove prepared temp file {audio_path}: {e}")
        if chunks_base_dir.exists():
            shutil.rmtree(chunks_base_dir, ignore_errors=True)
            print(f"🧹 Removed chunk dir: {chunks_base_dir}")


if __name__ == "__main__":
    main()
