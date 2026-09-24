#!/usr/bin/env bash
# identify-speakers.sh — Load the trusted config, then run the voiceprint CLI
# Usage: identify-speakers.sh identify|enroll|cohort-add|calibrate|list|forget [options]
#
# Loads the same guarded configuration as transcribe.sh, so VAULT_PATH, MEETING_NOTES_DIR
# and VOICEPRINT_* reach the "never store biometrics inside notes" guard. Installs nothing:
# `identify` is a graceful no-op (exit 0) without dependencies, models, or voiceprints.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export SKILL_DIR

# Load the configured paths (same checks as transcribe.sh).
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
  perm=$(stat -c '%a' "${ENV_FILE}" 2>/dev/null || stat -f '%Lp' "${ENV_FILE}" 2>/dev/null || true)
  # Fail closed: if neither stat dialect could read the mode, we can't verify
  # the file isn't world-writable, so refuse rather than source blind.
  if [[ ! "${perm}" =~ ^[0-7]{3,4}$ ]]; then
    echo "Refusing to source ${ENV_FILE}: could not read file permissions to verify it is not writable by others." >&2
    exit 1
  fi
  if (( 8#${perm} & 022 )); then
    echo "Refusing to source ${ENV_FILE}: group or world writable (mode=${perm}). chmod 600 ${ENV_FILE} to fix." >&2
    exit 1
  fi
  set -a; source "${ENV_FILE}"; set +a
fi

PYTHON="${SKILL_DIR}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi
exec "${PYTHON}" "${SKILL_DIR}/scripts/identify_speakers.py" "$@"
