#!/usr/bin/env bash
# compress-audio.sh — Convert any audio format to OGG (Vorbis/Opus) for archival
# Usage: compress-audio.sh <input-audio> [--output <path>]
# Non-destructive: does NOT delete the source file.
# Exit codes: 0=success/skipped, 1=conversion failed, 2=verification failed
set -euo pipefail

# --- Parse arguments ---
INPUT=""
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    *) INPUT="$1"; shift ;;
  esac
done

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "Error: File not found: ${INPUT:-<not provided>}" >&2
  exit 1
fi

EXT="${INPUT##*.}"
EXT_LOWER=$(echo "$EXT" | tr '[:upper:]' '[:lower:]')

# Skip if already OGG with valid codec
if [[ "$EXT_LOWER" == "ogg" ]]; then
  EXISTING_CODEC=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$INPUT" 2>/dev/null)
  if [[ "$EXISTING_CODEC" == "vorbis" || "$EXISTING_CODEC" == "opus" ]]; then
    echo "Already OGG ($EXISTING_CODEC), skipping: $INPUT"
    exit 0
  fi
fi

OUTPUT="${OUTPUT:-${INPUT%.*}.ogg}"
INPUT_SIZE=$(stat -f%z "$INPUT" 2>/dev/null || stat -c%s "$INPUT" 2>/dev/null)

echo "Archiving: $INPUT → $OUTPUT"
echo "Source: $((INPUT_SIZE / 1048576)) MB ($EXT_LOWER)"

# Step 1: Convert to OGG — encoder fallback chain
ENCODER_USED=""

if ffmpeg -v error -i "$INPUT" -c:a libvorbis -q:a 4 "$OUTPUT" -y 2>/dev/null; then
  ENCODER_USED="libvorbis"
elif ffmpeg -v error -i "$INPUT" -c:a libopus -b:a 96k "$OUTPUT" -y 2>/dev/null; then
  ENCODER_USED="libopus"
else
  if ffmpeg -v error -i "$INPUT" -c:a copy "$OUTPUT" -y 2>/dev/null; then
    ENCODER_USED="copy"
  else
    echo "FAIL: no working encoder (tried libvorbis, libopus, stream copy)" >&2
    exit 1
  fi
fi

echo "Encoder: $ENCODER_USED"

OGG_SIZE=$(stat -f%z "$OUTPUT" 2>/dev/null || stat -c%s "$OUTPUT" 2>/dev/null)

# Step 2: Verify — full decode integrity
ERRORS=$(ffmpeg -v error -i "$OUTPUT" -f null - 2>&1)
if [[ -n "$ERRORS" ]]; then
  echo "FAIL: Decode errors detected: $ERRORS" >&2
  rm -f "$OUTPUT"
  exit 2
fi

# Step 3: Verify — duration match (±1.0s tolerance, relaxed for cross-format)
INPUT_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
OGG_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUTPUT")
DELTA=$(echo "$INPUT_DUR - $OGG_DUR" | bc | tr -d '-')
if (( $(echo "$DELTA > 1.0" | bc -l) )); then
  echo "FAIL: Duration mismatch — source=${INPUT_DUR}s, OGG=${OGG_DUR}s, delta=${DELTA}s" >&2
  rm -f "$OUTPUT"
  exit 2
fi

# Step 4: Verify — codec validation (vorbis or opus both valid for OGG)
CODEC=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$OUTPUT")
if [[ "$CODEC" != "vorbis" ]] && [[ "$CODEC" != "opus" ]]; then
  echo "FAIL: Unexpected codec: $CODEC (expected vorbis or opus)" >&2
  rm -f "$OUTPUT"
  exit 2
fi

# Report (no source deletion — archival is non-destructive)
if [[ "$INPUT_SIZE" -gt 0 ]]; then
  REDUCTION=$(echo "scale=1; (1 - $OGG_SIZE / $INPUT_SIZE) * 100" | bc)
  echo ""
  echo "Archived: $((INPUT_SIZE / 1048576)) MB $EXT_LOWER → $((OGG_SIZE / 1048576)) MB OGG (${REDUCTION}% reduction)"
fi
echo "Verification: decode clean, duration delta=${DELTA}s, codec=${CODEC}"
echo "Output: $OUTPUT"
