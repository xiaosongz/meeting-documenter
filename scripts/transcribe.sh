#!/usr/bin/env bash
# transcribe.sh — Bootstrap environment and run transcription pipeline
# Usage: transcribe.sh <audio-path> [--output <path>] [--context-file <path>] [--model <model>]
#
# Ensures:
#   1. .venv exists (created with uv if missing)
#   2. google-genai is installed
#   3. .env is loaded (GOOGLE_API_KEY)
#   4. Delegates to transcribe_pipeline.py
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export SKILL_DIR
VENV_DIR="${SKILL_DIR}/.venv"

# 1. Ensure .venv exists (idempotent, fast no-op if present)
if [[ ! -f "${VENV_DIR}/bin/python3" ]]; then
  echo "🔧 Creating persistent venv with uv..."
  uv venv "${VENV_DIR}" --seed
fi

# 2. Install google-genai only if not already present (warm start = instant)
if ! "${VENV_DIR}/bin/python3" -c "import google.genai" 2>/dev/null; then
  echo "📦 Installing google-genai..."
  VIRTUAL_ENV="${VENV_DIR}" uv pip install -q google-genai
fi

# 3. Load API key (.env must set GOOGLE_API_KEY=...)
# Allow override: MEETING_DOCUMENTER_ENV_FILE=/path/to/.env supports shared installs,
# read-only mounts, multi-user hosts. Default is ${SKILL_DIR}/.env.
# Note: bare `-` (not `:-`) so an explicit empty value opts out of file loading.
ENV_FILE="${MEETING_DOCUMENTER_ENV_FILE-${SKILL_DIR}/.env}"
if [[ -n "${ENV_FILE}" && -f "${ENV_FILE}" ]]; then
  # Defense: source executes arbitrary code. Refuse files we don't own or that
  # are group/world-writable — prevents env-file injection on shared hosts.
  if [[ ! -O "${ENV_FILE}" ]]; then
    echo "Refusing to source ${ENV_FILE}: not owned by current user (UID=${UID})." >&2
    exit 1
  fi
  perm=$(stat -f '%Lp' "${ENV_FILE}" 2>/dev/null || stat -c '%a' "${ENV_FILE}" 2>/dev/null)
  if (( 8#${perm:-0} & 022 )); then
    echo "Refusing to source ${ENV_FILE}: group or world writable (mode=${perm}). chmod 600 ${ENV_FILE} to fix." >&2
    exit 1
  fi
  set -a; source "${ENV_FILE}"; set +a
fi

# 4. Delegate to Python pipeline
exec "${VENV_DIR}/bin/python3" "${SKILL_DIR}/scripts/transcribe_pipeline.py" "$@"
