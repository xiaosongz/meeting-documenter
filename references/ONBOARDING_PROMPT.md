# Onboarding Prompt — meeting-documenter

**Audience:** This document is written for a Claude Code agent running in the **user's** environment, not for the skill author. The user opens Claude Code in (or alongside) their notes directory and says something like:

> "Read `references/ONBOARDING_PROMPT.md` in `meeting-documenter` and follow it to set this skill up for my system."

Your job (as the agent) is to inspect the user's note-taking system, then either **adapt** the skill to fit it, or **bootstrap** a minimum structure that lets the skill run. Always start in **diagnostic** mode — never write files until the user picks adapt or bootstrap.

---

## Contract — what the skill requires to run

Probe these in order. Treat the first list as blockers; the second as adapters; the third as conventions you may need to negotiate.

### Hard requirements (skill cannot transcribe without these)

| Requirement | Check |
|---|---|
| `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) set in `.env` | `grep -E '^(GOOGLE\|GEMINI)_API_KEY=' .env` |
| `ffmpeg` on `$PATH` | `command -v ffmpeg` |
| `ffprobe` on `$PATH` | `command -v ffprobe` |
| `uv` on `$PATH` | `command -v uv` |
| Writable output directory | `test -w "$MEETING_RAW_DIR"` (or chosen path) |

### Soft requirements (needed for full pipeline; can be disabled per-step)

| Env var | Purpose | Disable by |
|---|---|---|
| `VAULT_PATH` | Notes root | Required if any soft path is unset |
| `MEETING_NOTES_DIR` | Summaries output | Defaults `${VAULT_PATH}/MeetingNotes` |
| `MEETING_RAW_DIR` | Transcripts output | Defaults `${MEETING_NOTES_DIR}/raw` |
| `MEETING_RECORDINGS_DIR` | OGG archive output | Defaults `${MEETING_NOTES_DIR}/recordings` |
| `DAILY_NOTES_DIR` | Daily notes root | Skip Step 5 if unset |
| `PROJECTS_DIR` | Project folders root | Skip Step 6 if unset |
| `MEETING_AUDIO_BACKUP_DIR` | Original audio backup | Defaults `$HOME/audio-backups/meetings` |
| `DAILY_NOTE_PATH_FORMAT` | strftime template for daily-note path | Defaults `%Y/%m-%B/%Y-%m-%d.md` |
| `PROJECT_MEETING_SUBDIR` | Per-project reference-note subdir | Defaults `Meeting`; empty = project root |
| `LINK_STYLE` | Output link form (`wikilink`/`markdown`/`plain`) | Defaults `wikilink` |
| `references/KNOWN_SPEAKERS.yaml` populated | Speaker resolution | Skip Step 0b if file absent or empty |
| `references/PROJECT_KEYWORDS.yaml` populated | Project detection | Skip Step 3 keyword matching if absent |

### Conventions the skill currently assumes

Each one may or may not fit the user's setup. Flag mismatches in diagnostic mode; resolve in adapt/bootstrap mode.

| Convention | Where it appears | Adapter strategy |
|---|---|---|
| Daily note path format | Step 5 | Detect actual format; set `DAILY_NOTE_PATH_FORMAT` in `.env` |
| Daily note has `## Meetings` section + table | Step 5 | Detect heading; offer to add it if missing |
| Daily note may have `## Carryover` section | Step 5 cross-ref | Soft check, skip cleanly if absent |
| | | **Carryover section definition:** a `## Carryover` Markdown heading followed by a checklist of unfinished tasks rolled over from previous days. Format: `- [ ] task description`. See `WORKFLOW_DETAILS.md § Step 5b` for how the skill cross-references it. |
| Link form (`[[Name]]` vs `[Name](path)` vs bare) | Steps 4, 5, 6 | Detect majority; set `LINK_STYLE` in `.env` |
| Per-project reference-note subdir | Step 6 | Detect actual subdir name; set `PROJECT_MEETING_SUBDIR` in `.env` (empty = project root) |
| Project root has `Dashboard.md` | Step 6 | Soft check, Dashboard update skipped if absent |
| Frontmatter taxonomy `meeting_outcome: decision\|update\|planning\|blocked\|cancelled` | Step 4 | If user prefers other taxonomy, edit `references/SUMMARY_FORMAT.md` |
| Markdown notes (UTF-8 `.md` files) | All steps | Notion/Bear/Logseq users: warn — skill writes `.md` directly to filesystem |

---

## Mode selection — ask the user first

After greeting, ask:

> Three setup modes are available:
>
> 1. **Diagnostic** — I inspect your system and report what's ready, what's missing, what doesn't match. No files are written. Recommended first.
> 2. **Adapt** — You already have a notes system. I generate `.env` and template files to make the skill fit your existing layout.
> 3. **Bootstrap** — You have nothing yet (or a clean slate). I scaffold a minimum folder + template structure so the skill can run end-to-end.
>
> Which mode? (default: diagnostic)

Use `AskUserQuestion` with these three options. Do not proceed past diagnostic without explicit user confirmation.

> **Fallback:** If your agent doesn't have an `AskUserQuestion` tool, ask the user in plain prose and wait for their reply before proceeding. The point is to never auto-pick an answer for a decision-point question. This applies to every `AskUserQuestion` reference in this doc.

---

## Mode 1 — Diagnostic (read-only)

**Goal:** produce a single status report covering every contract item. No filesystem writes. In particular, do NOT run `scripts/transcribe.sh --help` during Mode 1 — it creates `.venv/` inside the skill repo on first invocation. That check belongs in the post-adapt/post-bootstrap Verification section.

### Steps

1. **Locate the skill repo.** Ask: "What is the path to your `meeting-documenter` clone?" Verify `SKILL.md` and `scripts/transcribe.sh` exist there. Resolve and export `SKILL_DIR=<absolute path to the cloned skill>` for use in later steps (every `${SKILL_DIR}` reference in this doc resolves against that export).

   > **Shell persistence caveat:** Claude Code runs each Bash tool invocation in a fresh subshell — `export` does not carry across calls. Two options: (a) prefix every bash block in this doc with `export SKILL_DIR="<path>"` so it's set in that subshell, or (b) once adapt/bootstrap completes, add `SKILL_DIR="<path>"` to the generated `.env` and the verification snippet's `source .env` will populate it.

2. **Locate the notes root.** Ask: "What is your notes / vault root directory (where you keep your Markdown files)? Or type 'none' if you don't have one yet." Verify the path exists.

3. **Check hard requirements.** Run each shell check from the "Hard requirements" table. Capture results.

4. **Check soft requirements.** For each env var in the "Soft requirements" table:
   - Look in `${SKILL_DIR}/.env` if it exists
   - Otherwise check current shell env
   - Otherwise note as unset
   - For each *path* env var that resolves to something, verify the directory exists

5. **Probe conventions.** Inside the notes root:
   - Find candidate daily-notes directories. Heuristic: any directory containing files matching `[0-9]{4}-[0-9]{2}-[0-9]{2}.md`.
   - For one sample daily note, grep for `## Meetings` and `## Carryover`. Report which exist.
   - Find candidate projects directory. Heuristic: any directory with multiple subfolders each containing a `Dashboard.md` or `README.md`.
   - Check whether existing notes use `[[wikilinks]]` (grep `'\[\['`) or `[markdown](links)` (grep `'\]\('`). Report majority style. **Threshold:** Count occurrences of `[[...]]` vs `[...](...)` patterns. If one form is ≥75% of detected links, call it the majority. Otherwise report "mixed" and ask the user which style they want the skill to emit.

6. **Check registry files.**
   - `references/KNOWN_SPEAKERS.yaml` exists? Count entries.
   - `references/PROJECT_KEYWORDS.yaml` exists? Count entries.
   - If only the `.template.yaml` versions exist, note them as "template only — not yet populated".

7. **Emit report.** Format:

   > Status markers below use emoji by default. If the user's terminal or stated preferences disallow emoji, substitute ASCII: `[ OK ]` / `[FAIL]` / `[WARN]`.

   ```
   meeting-documenter — Diagnostic Report
   =======================================

   Hard requirements
   -----------------
   [✅ / ❌]  GOOGLE_API_KEY              : <found in .env | NOT SET>
   [✅ / ❌]  ffmpeg                      : <path | NOT FOUND>
   ...

   Soft requirements
   -----------------
   [✅ / ⚠️ / ❌]  VAULT_PATH              : <path | unset>
   ...

   Conventions (probed in $VAULT_PATH)
   -----------------------------------
   Daily notes dir candidate(s)    : <path(s) or none found>
   Daily note path format detected : <e.g. YYYY-MM-DD.md | YYYY/MM/YYYY-MM-DD.md | none>
   `## Meetings` section in sample : <yes | no | n/a>
   `## Carryover` section          : <yes | no | n/a>
   Link style majority             : <wikilink | markdown | mixed | unknown>
   Projects dir candidate(s)       : <path(s) or none>

   Registries
   ----------
   KNOWN_SPEAKERS.yaml             : <populated (N entries) | template only | missing>
   PROJECT_KEYWORDS.yaml           : <populated (N entries) | template only | missing>

   Recommendation
   --------------
   <one of:
     - "Adapt mode — your existing layout matches; only env wiring needed."
     - "Adapt mode with adjustments — N convention mismatches; review below."
     - "Bootstrap mode — minimum structure missing; scaffold recommended."
     - "Blocked — hard requirement(s) missing; install before either mode.">
   ```

8. **Stop.** Ask: "Proceed to adapt or bootstrap, or exit?" Do not write anything in diagnostic mode.

---

## Mode 2 — Adapt (writes `.env` and registry files only)

**Goal:** user already has a working notes system. Fit the skill to it without restructuring their notes.

### Steps

1. **Run diagnostic first** (Mode 1) if not already done in this session. Carry the findings forward.

2. **Resolve each soft env var.** For every path env var:
   - If diagnostic found a matching dir, propose it; confirm with user.
   - If no match, ask the user to either point at a path or skip the step the var enables.
   - Build the resolved `.env` in memory; do not write yet.

3. **Resolve conventions.** For each mismatch flagged in diagnostic:

   - **Daily note path format.** Ask user for their actual format string (e.g., `%Y-%m-%d.md`, `%Y/%m-%B/%Y-%m-%d.md`, `Journal/%Y/%Y-%m-%d.md`). Add `DAILY_NOTE_PATH_FORMAT="..."` to `.env`. The skill reads this var in Step 5 — no SKILL.md edit needed.

   - **`## Meetings` section missing.** Ask: "Add `## Meetings` (with empty table header) to your existing daily-note template?" If yes, find the template file (`templates/Daily.md`, `_templates/daily-note.md`, etc.) or ask user where it lives. Patch the template; do not retro-edit historical daily notes.

   - **Link style mismatch.** Determine the user's preferred form (`wikilink` / `markdown` / `plain`) — default to the majority detected in diagnostic, or ask if mixed. Add `LINK_STYLE="..."` to `.env`. The skill reads this var in Steps 4-6 and `SUMMARY_FORMAT.md § Link Styles` provides the substitution table.

   - **Project layout mismatch.** If projects dir uses different subfolder names (e.g. `Notes/` not `Meeting/`), ask for the actual subdir name. Add `PROJECT_MEETING_SUBDIR="Notes"` to `.env`. Set to `""` (empty string) to put reference notes directly into the project root. The skill reads this var in Step 6.

4. **Populate registries.**
   - Copy `references/KNOWN_SPEAKERS.template.yaml` → `references/KNOWN_SPEAKERS.yaml` if absent.
   - Walk the user through one example entry. Ask: "Who are the 2–3 people you most often record meetings with? I'll add them to the speaker registry; you can add more later."
   - Copy `references/PROJECT_KEYWORDS.template.yaml` → `references/PROJECT_KEYWORDS.yaml` if absent.
   - List subdirectories of `${PROJECTS_DIR}`. For each, ask the user: "Should this project be auto-detected from meeting transcripts? If yes, what 2–3 keywords or aliases should match it?"
   - Both registry files must stay gitignored. Confirm `.gitignore` covers them before writing.

5. **Write `.env`.** Show the assembled `.env` to the user for review. Test whether `${SKILL_DIR}` is writable (`test -w "${SKILL_DIR}"`):
   - **Writable:** write to `${SKILL_DIR}/.env` and verify the path is gitignored.
   - **Not writable** (read-only mount, shared install, multi-user host): ask the user for an external path (e.g., `$HOME/.config/meeting-documenter/.env`). Write `.env` there, then emit a shell-export snippet for the user to add to their `~/.zshrc` or `~/.bashrc`:
     ```bash
     export MEETING_DOCUMENTER_ENV_FILE="<chosen external path>"
     ```
     Both `scripts/transcribe.sh` and the SKILL.md detection check honor this override.

6. **Final report.** List:
   - `.env` written: <path>
   - `KNOWN_SPEAKERS.yaml`: <N entries>
   - `PROJECT_KEYWORDS.yaml`: <N entries>
   - Convention env vars resolved (`DAILY_NOTE_PATH_FORMAT`, `PROJECT_MEETING_SUBDIR`, `LINK_STYLE`)
   - Next step: "Try `transcribe.sh path/to/test.mp3` to confirm the pipeline runs end-to-end."

---

## Mode 3 — Bootstrap (scaffolds minimum structure)

**Goal:** user has nothing (or wants a clean structure dedicated to meeting notes). Create the smallest folder + template tree that makes the skill run end-to-end.

### Steps

1. **Confirm intent.** Ask: "I'll create a new notes structure under a path you choose. This will create folders and one example file each, but won't modify anything outside that path. Continue?"

2. **Choose root.** Ask for `VAULT_PATH` (default suggestion: `$HOME/notes`). Verify it does not already exist OR is empty. If non-empty, switch to adapt mode instead.

3. **Create structure.** Use today's date (computed via `date +%Y`, `date +%m`, `date +%B`, `date +%Y-%m-%d`) for the seeded daily note so the user's first real-run-today already finds the path it expects.

   ```
   ${VAULT_PATH}/
   ├── MeetingNotes/
   │   ├── raw/                         # transcripts
   │   └── recordings/                  # OGG archives
   ├── DailyNotes/
   │   └── <YYYY>/                      # e.g. 2026
   │       └── <MM-MonthName>/          # e.g. 05-May
   │           └── <YYYY-MM-DD>.md     # today's daily note (see template below)
   ├── Projects/
   │   └── _example-project/
   │       ├── Dashboard.md             # example project dashboard (see template below)
   │       └── Meeting/                 # per-project reference notes
   └── templates/
       ├── DailyNote.md                 # template Claude copies when a daily note is missing
       └── ProjectDashboard.md
   ```

   Use `mkdir -p` only; never `rm -rf`.

   Also create the audio backup directory (referenced by `MEETING_AUDIO_BACKUP_DIR` in the `.env` below; Step 8b of the skill writes here and fails on first run if it doesn't exist):

   ```bash
   mkdir -p "${MEETING_AUDIO_BACKUP_DIR}"
   ```

4. **Write today's daily note** (path computed from `date` honoring `DAILY_NOTE_PATH_FORMAT`, defaulting to `%Y/%m-%B/%Y-%m-%d.md`). Use a here-doc with today's date substituted:

   ```bash
   # LC_TIME=C pins %B to English (May, not Mai/Mayo) so path is stable across locales.
   DAILY_REL=$(LC_TIME=C date "+${DAILY_NOTE_PATH_FORMAT:-%Y/%m-%B/%Y-%m-%d.md}")
   TODAY=$(date +%Y-%m-%d)
   mkdir -p "$(dirname "${VAULT_PATH}/DailyNotes/${DAILY_REL}")"
   cat > "${VAULT_PATH}/DailyNotes/${DAILY_REL}" <<EOF
   ---
   date: ${TODAY}
   ---

   # ${TODAY}

   ## Meetings

   <!-- Standard variant per WORKFLOW_DETAILS.md § Step 5 — keep these columns. -->
   | Time | Meeting | Transcript | Summary |
   |------|---------|------------|---------|

   ## Carryover

   ## Notes

   EOF
   ```

   The `## Carryover` section is intentionally empty — do not seed a placeholder task. The skill's Step 5b cross-references this section; a stub item would become noise on first real run.

5. **Write example project dashboard** (`Projects/_example-project/Dashboard.md`):

   ```markdown
   ---
   project: _example-project
   status: active
   updated: 2026-01-01
   ---

   # _example-project

   ## Recent Meetings

   <!-- meeting-documenter appends rows here -->

   ## Overview

   (one-paragraph project description)
   ```

6. **Write daily-note template** (`templates/DailyNote.md`) — same structure as the example daily note but with `{{date}}` placeholder.

7. **Write `.env`.** Point all path vars at the scaffolded dirs and include the three convention env vars as commented defaults so the adopter can flip them later without learning the env-var names from scratch:

   ```bash
   VAULT_PATH="<chosen root>"
   MEETING_NOTES_DIR="${VAULT_PATH}/MeetingNotes"
   MEETING_RAW_DIR="${MEETING_NOTES_DIR}/raw"
   MEETING_RECORDINGS_DIR="${MEETING_NOTES_DIR}/recordings"
   DAILY_NOTES_DIR="${VAULT_PATH}/DailyNotes"
   PROJECTS_DIR="${VAULT_PATH}/Projects"
   MEETING_AUDIO_BACKUP_DIR="$HOME/audio-backups/meetings"
   GOOGLE_API_KEY="<paste your key here>"

   # Convention overrides — uncomment + edit if your setup differs from defaults
   # DAILY_NOTE_PATH_FORMAT="%Y/%m-%B/%Y-%m-%d.md"  # examples: "%Y-%m-%d.md" (flat), "Journal/%Y/%Y-%m-%d.md"
   # PROJECT_MEETING_SUBDIR="Meeting"               # empty string = put refs in project root
   # LINK_STYLE="wikilink"                          # wikilink | markdown | plain
   ```

   **Quote any value that could contain shell metachars.** If the value contains `<`, `>`, `*`, or any shell metachar, quote it: `GOOGLE_API_KEY="<paste your key here>"`. Placeholder tokens like `<chosen root>` and `<paste your key here>` MUST be inside double quotes in the written `.env`, otherwise the shell tries to redirect on `source`.

   Ask the user for the Gemini API key. If they don't have one, point them to https://aistudio.google.com/apikey and stop here; resume after they paste the key.

8. **Initialize registries.** Copy both `.template.yaml` files to their runtime `.yaml` counterparts. **Replace** the templated example entries with a single obviously-fake stub the user will edit. Do not leave templated names like `Alice Example` or `Bob Sample` in the populated `.yaml` — those will match real transcripts and corrupt speaker resolution on first run.

9. **Final report.** Same format as adapt mode plus a `tree` of the scaffolded directory (depth 3) so user sees what was created.

---

## Verification (run after adapt or bootstrap)

1. Source `.env` and confirm every required var is set. Honor `MEETING_DOCUMENTER_ENV_FILE` if the adapter wrote `.env` outside the skill repo (Mode 2 Step 5 read-only-mount branch). The `.env` must use quoted placeholder values (e.g. `GOOGLE_API_KEY="<paste your key here>"`) so `set -a; source` doesn't trip on shell metachars like `<`, `>`, or `*`:

   ```bash
   ENV_FILE="${MEETING_DOCUMENTER_ENV_FILE-${SKILL_DIR}/.env}"
   set -a; source "${ENV_FILE}"; set +a
   echo "GOOGLE_API_KEY      = ${GOOGLE_API_KEY:-UNSET}"
   echo "VAULT_PATH          = ${VAULT_PATH:-UNSET}"
   echo "MEETING_NOTES_DIR   = ${MEETING_NOTES_DIR:-UNSET}"
   echo "MEETING_RAW_DIR     = ${MEETING_RAW_DIR:-UNSET}"
   ```

   (Portable — each var expanded explicitly. Do NOT use bash indirect expansion of the form dollar-brace-bang-varname here; it fails in zsh and other POSIX-ish shells.)

2. Dry-run pipeline help:

   ```bash
   "${SKILL_DIR}/scripts/transcribe.sh" --help 2>&1 | head -20
   ```

   **Caveat:** Running `transcribe.sh --help` will create `.venv/` inside the skill repo if it doesn't exist. This is the skill's expected first-run side effect; it's gitignored. Skip this step if the skill repo is on a read-only mount.

   If this fails (and the mount is writable), the venv didn't bootstrap; check `uv` installation.

3. Suggest the user run one short test audio (≤2 minutes) to confirm the full pipeline. Do not generate or fabricate audio for them.

---

## Guardrails

- **Never write outside the user's chosen notes root or the skill repo.** No `$HOME/.zshrc` edits, no global config.
- **Never overwrite an existing file without showing the diff and confirming.** Especially `.env`, registries, and daily-note templates.
- **Never commit on the user's behalf.** Leave git decisions to them.
- **Never paste the user's API key into chat output or logs.** Read it once, write to `.env`, do not echo back.
- **Stop and ask if any probe step returns ambiguous results.** Better to ask than guess wrong about the user's layout.
- **Surface "known limitations" honestly.** The `meeting_outcome` taxonomy (`decision | update | planning | blocked | cancelled`) is hardcoded in `references/SUMMARY_FORMAT.md`. If the user prefers a different taxonomy, document the mismatch in the final report. Daily-note path, link form, and project subdir are now parameterized via `DAILY_NOTE_PATH_FORMAT`, `LINK_STYLE`, and `PROJECT_MEETING_SUBDIR` — set them in `.env` rather than calling out as limitations.

---

## Output schema — what "done" looks like

After adapt or bootstrap mode completes successfully, the user should have:

- `${SKILL_DIR}/.env` with all required + chosen-soft env vars
- `${SKILL_DIR}/references/KNOWN_SPEAKERS.yaml` (populated or with at least one example entry)
- `${SKILL_DIR}/references/PROJECT_KEYWORDS.yaml` (populated or with at least one example entry)
- Notes root containing the structure their `.env` points at
- A clear written list of any conventions that don't match and require manual `SKILL.md` edits (until those become env-configurable)

If any of these are missing, the run is incomplete; report it and stop.
