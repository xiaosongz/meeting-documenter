#!/usr/bin/env bash
# split-audio.sh — Silence-aware audio splitting for long recordings
# Usage: split-audio.sh <audio-path> [--target-duration 3000] [--output-dir /tmp/chunks]
# Outputs JSON manifest to stdout
#
# Splits audio at silence points near target boundaries (±5 min tolerance).
# Falls back to hard-split if no silence found in window.
set -euo pipefail

# --- Parse arguments ---
INPUT=""
TARGET_DURATION=3000  # 50 min default
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-duration) TARGET_DURATION="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) INPUT="$1"; shift ;;
  esac
done

if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "Error: Audio file not found: ${INPUT:-<not provided>}" >&2
  exit 1
fi

# Default output dir with PID for uniqueness
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/meeting_chunks_$$}"
mkdir -p "$OUTPUT_DIR"

# --- Get total duration ---
TOTAL_DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
TOTAL_DURATION_INT=$(printf "%.0f" "$TOTAL_DURATION")

# If audio is shorter than target, output single-chunk manifest
if (( TOTAL_DURATION_INT <= TARGET_DURATION + 300 )); then
  EXT="${INPUT##*.}"
  cp "$INPUT" "$OUTPUT_DIR/chunk_00.${EXT}"
  printf '[{"path": "%s/chunk_00.%s", "offset_seconds": 0, "duration": %s}]\n' \
    "$OUTPUT_DIR" "$EXT" "$TOTAL_DURATION_INT"
  exit 0
fi

# --- Detect silence points ---
SILENCE_FILE=$(mktemp "${OUTPUT_DIR}/silence_points.XXXXXX")
ffmpeg -i "$INPUT" -af silencedetect=noise=-30dB:d=1.0 -f null - 2>&1 \
  | grep "silence_end" | awk '{print $5}' | sed 's/[^0-9.]//g' > "$SILENCE_FILE"

# --- Calculate split points ---
TOLERANCE=300  # ±5 minutes
SPLIT_POINTS=()
BOUNDARY=$TARGET_DURATION

while (( BOUNDARY < TOTAL_DURATION_INT )); do
  LOW=$((BOUNDARY - TOLERANCE))
  HIGH=$((BOUNDARY + TOLERANCE))
  (( LOW < 0 )) && LOW=0

  # Find closest silence point within tolerance window
  BEST=""
  BEST_DIST=999999
  while IFS= read -r ts; do
    TS_INT=$(printf "%.0f" "$ts")
    if (( TS_INT >= LOW && TS_INT <= HIGH )); then
      DIST=$(( TS_INT > BOUNDARY ? TS_INT - BOUNDARY : BOUNDARY - TS_INT ))
      if (( DIST < BEST_DIST )); then
        BEST="$TS_INT"
        BEST_DIST=$DIST
      fi
    fi
  done < "$SILENCE_FILE"

  # Fallback to hard-split at boundary if no silence found
  if [[ -z "$BEST" ]]; then
    BEST=$BOUNDARY
    echo "Warning: No silence found near ${BOUNDARY}s, hard-splitting" >&2
  fi

  SPLIT_POINTS+=("$BEST")
  BOUNDARY=$((BEST + TARGET_DURATION))
done

# --- Split audio at chosen points ---
EXT="${INPUT##*.}"
MANIFEST="["
PREV=0
IDX=0

for POINT in "${SPLIT_POINTS[@]}"; do
  OUTFILE="$OUTPUT_DIR/chunk_$(printf '%02d' $IDX).${EXT}"
  CHUNK_DUR=$((POINT - PREV))
  ffmpeg -v error -i "$INPUT" -ss "$PREV" -to "$POINT" -c copy "$OUTFILE" -y
  MANIFEST+="{\"path\": \"${OUTFILE}\", \"offset_seconds\": ${PREV}, \"duration\": ${CHUNK_DUR}}"
  MANIFEST+=", "
  PREV=$POINT
  IDX=$((IDX + 1))
done

# Last chunk: from last split point to end
OUTFILE="$OUTPUT_DIR/chunk_$(printf '%02d' $IDX).${EXT}"
CHUNK_DUR=$((TOTAL_DURATION_INT - PREV))
ffmpeg -v error -i "$INPUT" -ss "$PREV" -c copy "$OUTFILE" -y
MANIFEST+="{\"path\": \"${OUTFILE}\", \"offset_seconds\": ${PREV}, \"duration\": ${CHUNK_DUR}}"
MANIFEST+="]"

# Cleanup
rm -f "$SILENCE_FILE"

# Output JSON manifest
echo "$MANIFEST"
