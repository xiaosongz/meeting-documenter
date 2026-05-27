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
ENV_FILE="${MEETING_DOCUMENTER_ENV_FILE:-${SKILL_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
fi

# 4. Delegate to Python pipeline
exec "${VENV_DIR}/bin/python3" "${SKILL_DIR}/scripts/transcribe_pipeline.py" "$@"
