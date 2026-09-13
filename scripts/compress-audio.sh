#!/usr/bin/env bash
# Convert audio to a verified OGG without replacing source or destination files.
# Usage: compress-audio.sh <input-audio> [--output <path>]
# Requires ffmpeg, ffprobe, bc, and Python 3 (atomic no-overwrite publication).
set -euo pipefail
umask 077

INPUT=""
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT="$2"; shift 2 ;;
    *) INPUT="$1"; shift ;;
  esac
done
if [[ -z "$INPUT" || ! -f "$INPUT" ]]; then
  echo "Error: Audio input must be an existing file" >&2
  exit 1
fi
command -v python3 >/dev/null
EXT_LOWER=$(echo "${INPUT##*.}" | tr '[:upper:]' '[:lower:]')
OUTPUT="${OUTPUT:-${INPUT%.*}.ogg}"
IN_PLACE=false
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  # A different hard link is not an in-place archive, even with the same inode.
  INPUT_NAME="$(cd "$(dirname "$INPUT")" && pwd -P)/$(basename "$INPUT")"
  OUTPUT_NAME="$(cd "$(dirname "$OUTPUT")" && pwd -P)/$(basename "$OUTPUT")"
  if [[ ! -L "$OUTPUT" && "$EXT_LOWER" == "ogg" && "$INPUT_NAME" == "$OUTPUT_NAME" ]]; then
    IN_PLACE=true
  else
    echo "Refusing archive: destination already exists or is a symlink" >&2
    exit 1
  fi
fi

COPY_OGG=false
if [[ "$EXT_LOWER" == "ogg" ]]; then
  CODEC=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$INPUT")
  if [[ "$CODEC" == "vorbis" || "$CODEC" == "opus" ]]; then
    ERRORS=$(ffmpeg -v error -i "$INPUT" -f null - 2>&1)
    if [[ -n "$ERRORS" ]]; then
      echo "FAIL: Source OGG has decode errors" >&2
      exit 2
    fi
    if [[ "$IN_PLACE" == true ]]; then
      echo "Already OGG ($CODEC, verified), in place: $INPUT"
      exit 0
    fi
    COPY_OGG=true
  fi
fi
if [[ "$IN_PLACE" == true ]]; then
  echo "Refusing to re-encode over the original file" >&2
  exit 1
fi

# Encode/copy only inside our private staging directory. Failed work never
# appears at the final archive path, and the trap removes partial encodes.
mkdir -p "$(dirname "$OUTPUT")"
WORK_DIR=$(mktemp -d "$(dirname "$OUTPUT")/.meeting-archive.XXXXXX")
trap 'rm -rf -- "$WORK_DIR"' EXIT
WORK_OUTPUT="$WORK_DIR/archive.ogg"
if [[ "$COPY_OGG" == true ]]; then
  cp "$INPUT" "$WORK_OUTPUT"
elif ffmpeg -v error -i "$INPUT" -c:a libvorbis -q:a 4 "$WORK_OUTPUT" -y 2>/dev/null; then
  :
elif ffmpeg -v error -i "$INPUT" -c:a libopus -b:a 96k "$WORK_OUTPUT" -y 2>/dev/null; then
  :
elif ffmpeg -v error -i "$INPUT" -c:a copy "$WORK_OUTPUT" -y 2>/dev/null; then
  :
else
  echo "FAIL: no working encoder (tried libvorbis, libopus, stream copy)" >&2
  exit 1
fi

ERRORS=$(ffmpeg -v error -i "$WORK_OUTPUT" -f null - 2>&1)
if [[ -n "$ERRORS" ]]; then
  echo "FAIL: Archive has decode errors" >&2
  exit 2
fi
INPUT_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$INPUT")
OGG_DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WORK_OUTPUT")
DELTA=$(echo "$INPUT_DUR - $OGG_DUR" | bc | tr -d '-')
if (( $(echo "$DELTA > 1.0" | bc -l) )); then
  echo "FAIL: Archive duration mismatch" >&2
  exit 2
fi
CODEC=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of csv=p=0 "$WORK_OUTPUT")
if [[ "$CODEC" != "vorbis" && "$CODEC" != "opus" ]]; then
  echo "FAIL: Unexpected archive codec" >&2
  exit 2
fi

# os.link atomically refuses any existing destination, including symlinks and
# directories created during encoding. Staging shares the destination filesystem.
python3 - "$WORK_OUTPUT" "$OUTPUT" <<'PY'
import os
import sys
try:
    os.link(sys.argv[1], sys.argv[2])
except FileExistsError:
    sys.exit("Refusing archive: destination appeared during encoding")
PY
echo "Verification: decode clean, duration delta=${DELTA}s, codec=${CODEC}"
echo "Output: $OUTPUT"
