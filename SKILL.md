---
name: meeting-documenter
description: >-
  This skill should be used when the user asks to "document this meeting",
  "process this recording", "create meeting notes", "summarize this meeting",
  or provides a path to an audio file (OGG, MP3, WAV, M4A). End-to-end
  pipeline: audio preparation, transcribe via Gemini, detect project context,
  generate structured summary with action items, update daily note, and link
  to project folders.
version: 0.4.0
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - AskUserQuestion
---

# Meeting Documenter

End-to-end meeting documentation: audio compression, transcription, structured summarization, daily note integration, and project linking in a single workflow.

## When to Use

- User provides a path to an audio file (MP3, WAV, OGG, M4A, FLAC, WebM, AAC)
- User mentions "document this meeting", "process this recording", "create meeting notes"
- User provides an existing transcript and asks for summary + integration

## Configuration

The skill writes into the directories listed below. Defaults are neutral folder names — override via environment variables to match your vault layout (PARA, Johnny.Decimal, custom, etc.):

| Variable | Default | Purpose |
|----------|---------|---------|
| `VAULT_PATH` | (required) | Vault root |
| `MEETING_NOTES_DIR` | `${VAULT_PATH}/MeetingNotes` | Where summaries live |
| `MEETING_RAW_DIR` | `${MEETING_NOTES_DIR}/raw` | Transcripts |
| `MEETING_RECORDINGS_DIR` | `${MEETING_NOTES_DIR}/recordings` | Archived OGG audio |
| `DAILY_NOTES_DIR` | `${VAULT_PATH}/DailyNotes` | Date-organized daily notes |
| `PROJECTS_DIR` | `${VAULT_PATH}/Projects` | Active projects with `Dashboard.md` |
| `MEETING_AUDIO_BACKUP_DIR` | `$HOME/audio-backups/meetings` | Original recordings (post-archive) |
| `GOOGLE_API_KEY` | (required) | Gemini API key — set in `.env`. `GEMINI_API_KEY` also accepted. |

The skill reads two YAML registries from `references/`:
- `KNOWN_SPEAKERS.yaml` — your team's speaker registry (copy from `KNOWN_SPEAKERS.template.yaml` and edit)
- `PROJECT_KEYWORDS.yaml` — your project keyword map (copy from `PROJECT_KEYWORDS.template.yaml` and edit)

Both runtime `.yaml` files are gitignored to prevent accidental commits of real names/emails/project codenames. If they're missing at runtime, Step 0b falls back to asking the user for every name, and Step 3 asks for the project directly.

## Pipeline Overview

```
Step 0:  Gather Metadata (title, time, attendees)
Step 0b: Resolve Speaker Names (KNOWN_SPEAKERS.yaml + user confirmation)
  → Step 1: Audio Preparation (automatic: silence trim + codec/container fix)
    → Step 2: Transcribe (Gemini API with --context-file)
      → Step 2b: Archive Recording (trim + OGG compress → recordings/ folder)
        → Step 3: Detect/Confirm Project (keyword match + AskUserQuestion)
          → Step 4: Generate Summary (per references/SUMMARY_FORMAT.md)
            → Step 5: Update Daily Note (Meetings table row)
              → Step 6: Link to Project (reference note + Dashboard)
                → Step 7: Verify All Links (references/QUALITY_CHECKLIST.md)
                  → Step 8: Cleanup Temp Files (/tmp/meeting_context.txt)
```

**Fast-path:** For existing transcripts, skip Steps 0-2. See `references/WORKFLOW_DETAILS.md`.

## Workflow

### Progress Tracker (copy and update as each step completes)

```
Meeting Documentation Progress:
- [ ] Step 0: Gather metadata → title: ___, time: ___, attendees: ___
- [ ] Step 0b: Resolve speaker names → confirmed via KNOWN_SPEAKERS.yaml + user
- [ ] Step 1: Audio preparation (automatic) → silence trimmed, codec/container fixed
- [ ] Step 2: Transcribe audio → saved to raw/ folder
- [ ] Step 2b: Archive recording → trimmed OGG saved to recordings/ folder
- [ ] Step 3: Detect/confirm project → project: ___
- [ ] Step 4: Generate summary → saved to MEETING_NOTES_DIR
- [ ] Step 5: Update daily note → row added to Meetings table
- [ ] Step 6: Link to project → reference note created
- [ ] Step 7: Verify all links → confirmed via Read tool
- [ ] Step 8: Cleanup → temp files removed, source audio backed up, vault root cleaned
```

### Step 0: Gather Metadata

Collect metadata from user prompt or ask via AskUserQuestion:
- **Title**: From user prompt or audio filename
- **Time**: Recording filenames typically use the **save/stop** timestamp. Calculate `start = filename_timestamp − audio_duration`, round to nearest 15 min.
- **Attendees**: Ask if not provided ("Who was in this meeting?")

### Step 0b: Resolve Speaker Names (MANDATORY)

**CRITICAL — Do NOT skip this step.** Raw names from user input or AI-extracted attendee lists are often misspelled or inconsistent with the vault. This step prevents wrong names from propagating through the entire pipeline.

1. Read `references/KNOWN_SPEAKERS.yaml` — the canonical speaker registry
2. For each extracted name, match (case-insensitive) against `canonical_name` and `aliases`
3. If matched → use the `canonical_name` and `wikilink` from the registry
4. If unmatched → flag as `[NEW]` and present to user
5. **Always** present the resolved name list to the user via AskUserQuestion before proceeding:
   - Show: `Raw name → Resolved name (role)` for each speaker
   - Include option to correct any mismatches
   - For `[NEW]` speakers, ask for the correct full name
6. After confirmation, use resolved names for the context file and all downstream outputs

See `references/WORKFLOW_DETAILS.md § Speaker Name Resolution` for the detailed matching procedure.

If attendees are confirmed, generate a speaker diarization context file per `references/CONTEXT_TEMPLATE.md`. This significantly improves speaker identification in the transcript.

### Step 1: Audio Preparation (Automatic)

The transcription pipeline automatically handles two common audio issues before transcription:

1. **Trailing silence trimming**: Detects silence >60s at the end of recordings and trims it (with 30s buffer). This prevents sending dead air to the API, saving cost and processing time. Uses `silencedetect=noise=-30dB:d=10.0`.

2. **Codec/container compatibility**: Detects codec mismatches (e.g., Opus codec in M4A container) and remuxes to a compatible container via stream copy. Falls back to re-encoding if stream copy fails. Mapping: `opus→.ogg`, `aac→.m4a`, `mp3→.mp3`, `flac→.flac`, `pcm→.wav`.

**No manual action needed.** The pipeline handles all preparation automatically. Skip to Step 2.

**For manual archival only** (outside the pipeline):
```bash
.claude/skills/meeting-documenter/scripts/compress-audio.sh "/path/to/audio.mp3"
```

### Step 2: Transcribe

Run the transcription pipeline — a single command that handles everything:

```bash
"${VAULT_PATH}/.claude/skills/meeting-documenter/scripts/transcribe.sh" \
  "/path/to/audio.mp3" \
  --no-archive \
  --context-file /tmp/meeting_context.txt \
  --output "${MEETING_RAW_DIR}/YYYY-MM-DD-HHMM Title-Transcript.md"
```

**What happens automatically:**
- **Audio preparation**: trailing silence detection + trim, codec/container compatibility fix
- Format detection (MP3, WAV, OGG, M4A, etc.)
- Silence-aware splitting for audio >50 min (target ~40 min chunks)
- Gemini API transcription with speaker diarization
- Truncation detection (finish_reason + coverage check) with auto-retry
- Timestamp monotonicity validation (fixes Gemini timestamp regression artifacts)
- Timestamp offset and segment combination
- Auto-generated `description` field in transcript frontmatter
- Prepared temp file cleanup

**Optional flags:**
- `--no-archive` — skip built-in OGG archival (recommended; Step 2b handles named archive instead)
- `--description "..."` — custom description for transcript frontmatter (auto-generated if omitted)

**Model:** `gemini-3-flash-preview` (hardcoded default). Max output: 65,536 tokens.

Omit `--context-file` if no context file was generated in Step 0.

### Step 2b: Archive Recording

After transcription, create a trimmed, compressed recording in the vault as a permanent ground truth reference:

1. **Trim + compress** the original recording (remove trailing silence, encode as OGG Vorbis or Opus):
   ```bash
   "${VAULT_PATH}/.claude/skills/meeting-documenter/scripts/compress-audio.sh" \
     "/path/to/original.m4a" \
     --output "${MEETING_RECORDINGS_DIR}/YYYY-MM-DD-HHMM Title.ogg"
   ```
   - If trimming needed, pre-trim with `ffmpeg -i input -t <trim_seconds> /tmp/trimmed.m4a` then compress the trimmed file
   - Naming convention matches the meeting summary filename but with `.ogg` extension
2. **Verify**: Check duration and codec of the archived file
3. **Update transcript** frontmatter `source:` field to wikilink the archived recording
4. **Note**: This step runs after Step 3 (title confirmation) since the filename depends on the confirmed title. Listed here for logical grouping with audio steps.

### Step 3: Detect/Confirm Project

1. Read the transcript
2. Scan for keywords from `references/PROJECT_KEYWORDS.yaml`
3. **Always** confirm with user via AskUserQuestion — even if high confidence
4. Also confirm/adjust the meeting title if it was auto-generated

See `references/WORKFLOW_DETAILS.md` for the no-match fallback flow.

### Step 4: Generate Summary

Read the transcript and generate a structured meeting summary per `references/SUMMARY_FORMAT.md`.

Key features to include:
- **`meeting_outcome`** frontmatter field: `decision | update | planning | blocked | cancelled`
- **Wikilink attendees**: `"[[Name]]"` format in YAML
- **`[assignee:: Name]`** inline field on all action items
- **Parking Lot** section for explicitly tabled items (between Action Items and Topics)
- **`description`** field (~150 chars): capture the meeting's key outcome, not just the topic

Generate a `description` that answers: "What changed as a result of this meeting?"

### Step 5: Update Daily Note

1. Determine daily note path: `${DAILY_NOTES_DIR}/YYYY/MM-MonthName/YYYY-MM-DD.md`
2. If daily note does not exist, create it first (use a daily-note skill if available)
3. Find the `## Meetings` section and its table
4. **Adapt to the existing table column format** — do not assume specific columns
5. Add a new row with wikilinks to transcript and summary
6. **Carryover cross-reference**: Scan the daily note's Carryover section for tasks that were addressed in this meeting. Suggest marking them complete or note them in the meeting row.
7. Verify the edit succeeded by reading the daily note

See `references/WORKFLOW_DETAILS.md` for daily note edge cases (missing section, variant columns, carryover resolution).

### Step 6: Link to Project

When a project was confirmed in Step 3:

1. Ensure `${PROJECTS_DIR}/{Project}/Meeting/` folder exists (`mkdir -p` if needed)
2. Create a reference note: `Meeting/YYYY-MM-DD-HHMM Title.md` containing:
   - Frontmatter with source/transcript wikilinks
   - Quick reference: executive summary, action items, key decisions
3. Update project `Dashboard.md`'s Recent Meetings section (if the section exists)
4. Update the Dashboard's `updated:` frontmatter field to today's date
5. Verify the reference note via Read

**Multi-project meetings**: When a meeting spans multiple projects, create a project-specific reference note in each project's `Meeting/` folder. Each reference note should contain only that project's decisions and action items, linking back to the full summary. See `references/WORKFLOW_DETAILS.md § Multi-Project Meetings`.

### Step 7: Verify All Links

Read all created files and confirm wikilinks resolve correctly. Run the full verification per `references/QUALITY_CHECKLIST.md`.

**Do NOT mark the task complete until all applicable checklist items pass.**

### Step 8: Cleanup

After successful verification, clean up temp files and archive the source recording.

**8a. Remove temp files:**

```bash
rm -f /tmp/meeting_context.txt
```

**Why:** The context file contains speaker names specific to THIS meeting. If left behind, a subsequent transcription picks up wrong speakers.

**8b. Backup source audio and clean vault root:**

```bash
"${VAULT_PATH}/.claude/skills/meeting-documenter/scripts/cleanup-source-audio.sh" \
  "/path/to/Recording YYYYMMDDHHMMSS.m4a" \
  "${MEETING_RECORDINGS_DIR}/YYYY-MM-DD-HHMM Title.ogg" \
  --meeting-name "Meeting Title" \
  --meeting-time "HHMM"
```

**What happens:**
1. Verifies the OGG archive is valid (decodable, vorbis codec, duration > 0)
2. Renames and moves the original M4A to `${MEETING_AUDIO_BACKUP_DIR}/YYYY-MM/`
   - Backup filename: `YYYY-MM-DD-HHMM Title (rec-HHMM).m4a` — includes both meeting start time and recording start time
3. Removes any pipeline-created duplicate OGG from the vault root

**Why backup instead of delete:** The OGG archive is trimmed (trailing silence removed) and re-encoded. The original M4A preserves the untrimmed, lossless source for future re-processing if needed.

## Additional Resources

| File | Purpose | When to Read |
|------|---------|-------------|
| `references/SUMMARY_FORMAT.md` | Full output specification for meeting summaries | Step 4 |
| `references/PROJECT_KEYWORDS.yaml` | Project keyword mappings for auto-detection | Step 3 |
| `references/KNOWN_SPEAKERS.yaml` | Canonical speaker registry with aliases | Step 0b |
| `references/CONTEXT_TEMPLATE.md` | Speaker diarization context file template | Step 0b |
| `references/WORKFLOW_DETAILS.md` | Edge cases, fast-paths, conditional branches | As needed |
| `references/QUALITY_CHECKLIST.md` | Verification checklist for all outputs | Step 7 |
| `scripts/compress-audio.sh` | Any-format → OGG archival with 3-point verify | Step 2b |
| `scripts/transcribe.sh` | Shell bootstrap: ensures uv venv, loads .env, delegates to pipeline | Step 2 |
| `scripts/transcribe_pipeline.py` | Full transcription pipeline: silence trim, codec fix, chunking, truncation detection | Step 1-2 |
| `scripts/split-audio.sh` | Silence-aware audio splitting for long recordings | Step 2 |
| `scripts/cleanup-source-audio.sh` | Verify archive, backup original with meeting name, clean vault root | Step 8 |

## Troubleshooting

**GOOGLE_API_KEY not set**: Check `.env` (must set `GOOGLE_API_KEY=...` or `GEMINI_API_KEY=...`)

**Python venv broken**: Delete `.venv` and re-run `transcribe.sh` (auto-recreates with uv)

**Truncated transcript**: Now auto-detected and retried with smaller chunks. If persistent, check pipeline output for retry logs. Known Gemini 3 Flash issue: model sometimes stops early despite token budget.

**Long audio**: Automatically handled. Audio >60 min is split at silence points into ~50 min chunks. No manual splitting needed.

## Cost Estimate

| Component | Cost |
|-----------|------|
| Gemini transcription (60 min audio, 1 chunk) | ~$0.08 |
| Gemini transcription (120 min audio, 3 chunks w/ retry) | ~$0.25 |
| Claude summarization | Included in subscription |
