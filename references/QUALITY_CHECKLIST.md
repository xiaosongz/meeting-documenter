---
description: "Verification checklist for meeting documentation pipeline. Read this in Step 7 to validate all outputs before marking the task complete."
---

# Quality Checklist

**VERIFY each item before marking the task complete.**

## Audio Preparation (Step 1)

- [ ] `scripts/compress-audio.sh` exited with code 0
- [ ] OGG file exists at expected path (pipeline handles archival to OGG non-destructively; source format can be any supported type)
- [ ] Original file trashed or archived (check script output for confirmation)
- [ ] Savings reported (expected ~90-97% reduction for WAV sources)

## Speaker Name Resolution (Step 0b — MANDATORY)

- [ ] All raw names resolved against `references/KNOWN_SPEAKERS.yaml`
- [ ] No unvalidated aliases used (e.g., a misheard variant should resolve to its canonical name from the registry)
- [ ] User confirmed the resolved name list via AskUserQuestion
- [ ] Context file written with canonical names only
- [ ] Any new speakers added to `KNOWN_SPEAKERS.yaml` after processing

## Recording Archive (Step 2b)

- [ ] Trimmed OGG saved to `${MEETING_RECORDINGS_DIR}/`
- [ ] Filename matches meeting convention: `YYYY-MM-DD-HHMM Title.ogg`
- [ ] Duration matches trimmed audio (no trailing silence)
- [ ] Codec is Vorbis or Opus (OGG container)
- [ ] Transcript `source:` frontmatter updated to wikilink the recording
- [ ] Summary footer includes recording link

## Transcript (Step 2)

- [ ] Saved to `${MEETING_RAW_DIR}/`
- [ ] Filename: `YYYY-MM-DD-HHMM Title-Transcript.md` (SPACE before title)
- [ ] Contains timestamped speaker-attributed content
- [ ] If context file used, speakers identified by **canonical names** from KNOWN_SPEAKERS.yaml (not "Speaker 1" and not raw AI-guessed names)
- [ ] Pipeline reported no truncation retries (or retries succeeded)
- [ ] Coverage check: transcript timestamps span ≥80% of audio duration

## Summary (Step 4)

- [ ] Saved to `${MEETING_NOTES_DIR}/`
- [ ] Filename: `YYYY-MM-DD-HHMM Title.md` (SPACE before title)
- [ ] YAML frontmatter includes: title, date, type, tags, attendees, meeting_type, meeting_outcome
- [ ] `description` field present (~150 chars, adds info beyond title)
- [ ] Attendees use wikilink format: `"[[Name]]"`
- [ ] Sections present: Executive Summary, Key Decisions, Action Items, Parking Lot, Topics, Follow-Up
- [ ] Action items have: checkbox format, `[assignee::]`, `[project::]`, `[due::]` (when explicit)
- [ ] Parking Lot items have NO owner (deferred items only)
- [ ] Footer links to raw transcript

## Daily Note (Step 5 — MANDATORY)

- [ ] Daily note exists at `${DAILY_NOTES_DIR}/YYYY/MM-Month/YYYY-MM-DD.md`
- [ ] Meetings section has table
- [ ] New row added with correct wikilinks to transcript and summary
- [ ] **Read the daily note to confirm the row was added**

## Carryover Cross-Reference (Step 5b)

- [ ] Daily note Carryover section scanned for tasks addressed in meeting
- [ ] Resolved carryover tasks suggested to user (or noted as "none applicable")

## Project Linking (Step 6 — when project identified)

- [ ] Reference note created at `${PROJECTS_DIR}/{Project}/Meeting/`
- [ ] Contains wikilinks to summary and transcript
- [ ] Includes quick reference (executive summary, action items, key decisions)
- [ ] Dashboard updated if it has a Recent Meetings section
- [ ] Dashboard `updated:` frontmatter field bumped to today's date
- [ ] **Read the reference note to confirm wikilinks are correct**
- [ ] **Multi-project**: If meeting spans multiple projects, reference notes exist in each project's `Meeting/` folder with project-scoped content

## Link Validation (Step 7)

- [ ] All wikilinks use Obsidian format: `[[path/to/note|Display Text]]`
- [ ] All created files exist and are non-empty
- [ ] Attendee wikilinks match `wikilink` field from `KNOWN_SPEAKERS.yaml`
- [ ] All `[assignee:: Name]` values match the canonical names used in attendees
- [ ] No placeholder names remain: `[[Unknown]]`, `[[TBD]]`, `[[Person 1]]`

## Cleanup (Step 8)

### 8a: Temp Files
- [ ] `/tmp/meeting_context.txt` removed (prevents stale context contaminating future transcriptions)
- [ ] Confirmed file no longer exists
- [ ] Chunk temp files cleaned (pipeline auto-cleans, but verify `/tmp/meeting_chunks_*` if needed)

### 8b: Source Audio Backup
- [ ] `cleanup-source-audio.sh` ran successfully (exit code 0)
- [ ] OGG archive verified (decodable, vorbis codec, duration > 0)
- [ ] Original moved to `${MEETING_AUDIO_BACKUP_DIR}/YYYY-MM/`
- [ ] Backup filename includes meeting name + both timestamps: `YYYY-MM-DD-HHMM Title (rec-HHMM).ext`
- [ ] Pipeline duplicate OGG removed from vault root (if it existed)
- [ ] No `Recording *.m4a` or `Recording *.ogg` files remain in vault root for this meeting
