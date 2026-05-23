#!/usr/bin/env bash
# cleanup-source-audio.sh — Verify archive, backup original with meeting name, remove vault root copies
#
# Usage:
#   cleanup-source-audio.sh <original_audio> <archive_ogg> --meeting-name "Title" --meeting-time "HHMM"
#
# Backup naming:
#   YYYY-MM-DD-HHMM Title (rec-HHMM).ext
#   Meeting time = when the meeting started (from skill Step 0)
#   rec-HHMM     = when the recording started (extracted from original filename)
#
# Steps:
#   1. Verify archive OGG is valid (exists, decodable, vorbis/opus, duration > 0)
#   2. Rename + move original to ${MEETING_AUDIO_BACKUP_DIR}/YYYY-MM/
#   3. Remove any pipeline-created duplicate OGG from vault root
#   4. Report results
#
# Configuration:
#   MEETING_AUDIO_BACKUP_DIR  Base directory for archived originals
#                             (default: $HOME/audio-backups/meetings)
#
# Exit codes: 0 = success, 1 = verification failed (nothing removed)

set -euo pipefail

BACKUP_BASE="${MEETING_AUDIO_BACKUP_DIR:-$HOME/audio-backups/meetings}"

# ─── Args ────────────────────────────────────────────────────────────────────

MEETING_NAME=""
MEETING_TIME=""

if [ $# -lt 2 ]; then
    echo "Usage: $0 <original_audio> <archive_ogg> --meeting-name \"Title\" --meeting-time \"HHMM\""
    echo ""
    echo "  original_audio  Path to the source recording (e.g., vault root M4A)"
    echo "  archive_ogg     Path to the OGG archive in recordings/"
    echo "  --meeting-name  Confirmed meeting title (e.g., \"Project Sync\")"
    echo "  --meeting-time  Meeting start time as HHMM (e.g., \"1430\")"
    echo ""
    echo "Backup base directory: $BACKUP_BASE"
    echo "(override with MEETING_AUDIO_BACKUP_DIR env var)"
    exit 1
fi

ORIGINAL="$1"
ARCHIVE="$2"
shift 2

while [ $# -gt 0 ]; do
    case "$1" in
        --meeting-name) MEETING_NAME="$2"; shift 2 ;;
        --meeting-time) MEETING_TIME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$MEETING_NAME" ] || [ -z "$MEETING_TIME" ]; then
    echo "Both --meeting-name and --meeting-time are required"
    exit 1
fi

# ─── Verify archive ─────────────────────────────────────────────────────────

echo "Verifying archive: $(basename "$ARCHIVE")"

if [ ! -f "$ARCHIVE" ]; then
    echo "   Archive not found: $ARCHIVE"
    exit 1
fi

ARCHIVE_SIZE=$(stat -f%z "$ARCHIVE" 2>/dev/null || stat --format=%s "$ARCHIVE" 2>/dev/null)
if [ "$ARCHIVE_SIZE" -eq 0 ]; then
    echo "   Archive is empty (0 bytes)"
    exit 1
fi

if ! ffmpeg -v error -i "$ARCHIVE" -f null - 2>/dev/null; then
    echo "   Archive failed decode check (corrupted)"
    exit 1
fi

CODEC=$(ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$ARCHIVE" 2>/dev/null)
if [ "$CODEC" != "vorbis" ] && [ "$CODEC" != "opus" ]; then
    echo "   Unexpected codec: $CODEC (expected vorbis or opus)"
    exit 1
fi

DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$ARCHIVE" 2>/dev/null)
if [ "$(echo "$DURATION > 0" | bc -l)" -ne 1 ]; then
    echo "   Archive duration is 0 or invalid"
    exit 1
fi

DURATION_FMT=$(printf "%dm %ds" "$(echo "$DURATION/60" | bc)" "$(echo "$DURATION%60/1" | bc)")
ARCHIVE_MB=$(echo "scale=1; $ARCHIVE_SIZE/1048576" | bc)
echo "   Valid: ${DURATION_FMT}, ${ARCHIVE_MB}MB, ${CODEC}"

# ─── Build backup filename ───────────────────────────────────────────────────

ORIG_BASENAME=$(basename "$ORIGINAL")
REC_TIME=""
if [[ "$ORIG_BASENAME" =~ Recording[[:space:]]([0-9]{8})([0-9]{2})([0-9]{2}) ]]; then
    REC_TIME="${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
fi

if [[ "$ORIG_BASENAME" =~ Recording[[:space:]]([0-9]{4})([0-9]{2})([0-9]{2}) ]]; then
    FILE_YEAR="${BASH_REMATCH[1]}"
    FILE_MONTH="${BASH_REMATCH[2]}"
    FILE_DAY="${BASH_REMATCH[3]}"
    FILE_DATE="${FILE_YEAR}-${FILE_MONTH}-${FILE_DAY}"
    YEAR_MONTH="${FILE_YEAR}-${FILE_MONTH}"
else
    FILE_DATE=$(stat -f%Sm -t"%Y-%m-%d" "$ORIGINAL" 2>/dev/null || date -r "$ORIGINAL" +"%Y-%m-%d" 2>/dev/null)
    YEAR_MONTH=$(echo "$FILE_DATE" | cut -c1-7)
fi

ORIG_EXT="${ORIGINAL##*.}"

if [ -n "$REC_TIME" ]; then
    BACKUP_NAME="${FILE_DATE}-${MEETING_TIME} ${MEETING_NAME} (rec-${REC_TIME}).${ORIG_EXT}"
else
    BACKUP_NAME="${FILE_DATE}-${MEETING_TIME} ${MEETING_NAME}.${ORIG_EXT}"
fi

# ─── Backup original ────────────────────────────────────────────────────────

BACKUP_DIR="${BACKUP_BASE}/${YEAR_MONTH}"

if [ -f "$ORIGINAL" ]; then
    mkdir -p "$BACKUP_DIR"
    BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

    if [ -f "$BACKUP_PATH" ]; then
        echo "Backup already exists: ${BACKUP_NAME}"
    else
        mv "$ORIGINAL" "$BACKUP_PATH"
        echo "Backed up: ${ORIG_BASENAME} -> ${YEAR_MONTH}/${BACKUP_NAME}"
    fi

    ORIGINAL_SIZE=$(stat -f%z "$BACKUP_PATH" 2>/dev/null || stat --format=%s "$BACKUP_PATH" 2>/dev/null)
    ORIGINAL_MB=$(echo "scale=1; $ORIGINAL_SIZE/1048576" | bc)
else
    echo "Original not found (already removed?): $ORIGINAL"
    ORIGINAL_MB=0
fi

# ─── Remove pipeline duplicate OGG from vault root ──────────────────────────

ORIGINAL_DIR=$(dirname "$ORIGINAL")
ORIGINAL_STEM=$(basename "$ORIGINAL" | sed 's/\.[^.]*$//')
DUPLICATE_OGG="${ORIGINAL_DIR}/${ORIGINAL_STEM}.ogg"

if [ -f "$DUPLICATE_OGG" ]; then
    rm "$DUPLICATE_OGG"
    echo "Removed pipeline duplicate: $(basename "$DUPLICATE_OGG")"
fi

# ─── Report ──────────────────────────────────────────────────────────────────

echo ""
echo "Cleanup complete"
echo "   Archive:  $(basename "$ARCHIVE") (${ARCHIVE_MB}MB, ${DURATION_FMT})"
if [ "$ORIGINAL_MB" != "0" ]; then
    echo "   Backup:   ${BACKUP_NAME} (${ORIGINAL_MB}MB)"
    echo "   Location: ${BACKUP_DIR}/"
fi
