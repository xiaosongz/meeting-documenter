---
description: "Verification checklist for meeting documentation pipeline. Read this in Step 7 to validate all outputs before marking the task complete."
---

# Quality Checklist

**VERIFY each item before marking the task complete.**

> **Note on link-form checks:** The `wikilink`-form examples below (`[[Name]]`, `[[raw/...|Full Transcript]]`) are the default. When `LINK_STYLE=markdown`, substitute `[Name](path/Name.md)` form; when `LINK_STYLE=plain`, the check is "name string present" with no link assertion. See `SUMMARY_FORMAT.md § Link Styles` for the substitution table.
>
> **Note on path checks:** The daily-note path uses `DAILY_NOTE_PATH_FORMAT` (default `%Y/%m-%B/%Y-%m-%d.md`). The project reference-note subdir uses `PROJECT_MEETING_SUBDIR` (default `Meeting`; empty string = project root). Substitute these when verifying.

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
- [ ] Transcript `source:` frontmatter updated to link the recording (form per `LINK_STYLE`)
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
- [ ] Attendees use the link form set by `LINK_STYLE` (default `wikilink` → `"[[Name]]"`; see SUMMARY_FORMAT.md § Link Styles)
- [ ] Sections present: Executive Summary, Key Decisions, Action Items, Parking Lot, Topics, Follow-Up
- [ ] Action items have: checkbox format, `[assignee::]`, `[project::]`, `[due::]` (when explicit)
- [ ] Parking Lot items have NO owner (deferred items only)
- [ ] Footer links to raw transcript

## Daily Note (Step 5 — MANDATORY)

- [ ] Daily note exists at `${DAILY_NOTES_DIR}/$(LC_TIME=C date "+${DAILY_NOTE_PATH_FORMAT:-%Y/%m-%B/%Y-%m-%d.md}")` (LC_TIME=C pins month name to English)
- [ ] Meetings section has table
- [ ] New row added with correct transcript and summary links (form per `LINK_STYLE`)
- [ ] **Read the daily note to confirm the row was added**

## Carryover Cross-Reference (Step 5b)

- [ ] Daily note Carryover section scanned for tasks addressed in meeting
- [ ] Resolved carryover tasks suggested to user (or noted as "none applicable")

## Project Linking (Step 6 — when project identified)

- [ ] Reference note created in the per-project reference directory — `${PROJECTS_DIR}/{Project}/${PROJECT_MEETING_SUBDIR-Meeting}/` when `PROJECT_MEETING_SUBDIR` is set, otherwise the project root `${PROJECTS_DIR}/{Project}/`
- [ ] Contains links to summary and transcript (form per `LINK_STYLE`)
- [ ] Includes quick reference (executive summary, action items, key decisions)
- [ ] Dashboard updated if it has a Recent Meetings section
- [ ] Dashboard `updated:` frontmatter field bumped to today's date
- [ ] **Read the reference note to confirm links are correct**
- [ ] **Multi-project**: If meeting spans multiple projects, a reference note exists for each project in its per-project reference directory (`${PROJECT_MEETING_SUBDIR-Meeting}/` subdir if set, otherwise the project root) with project-scoped content

## Link Validation (Step 7)

- [ ] All links use the form set by `LINK_STYLE` (default `wikilink`: `[[path/to/note|Display Text]]`; `markdown`: `[Display Text](path/to/note.md)`; `plain`: bare name with no link)
- [ ] All created files exist and are non-empty
- [ ] Attendee links match the canonical entry from `KNOWN_SPEAKERS.yaml` (wikilink in `wikilink:` field; same canonical_name otherwise)
- [ ] All `[assignee:: Name]` values match the canonical names used in attendees
- [ ] No placeholder names remain as attendees/assignees. Match per `LINK_STYLE`: `wikilink` → `[[Unknown]]`/`[[TBD]]`/`[[Person 1]]`; `markdown` → `[Unknown](`/`[TBD](`/`[Person 1](`; `plain` → bare `Unknown`/`TBD`/`Person 1` appearing as a YAML attendee value or `[assignee:: ...]` value (NOT as substring in topic prose, which may legitimately mention the words)

## Cleanup (Step 8)

### 8a: Temp Files
- [ ] `/tmp/meeting_context.txt` removed (prevents stale context contaminating future transcriptions)
- [ ] Confirmed file no longer exists
- [ ] `/tmp/meeting_chunks_<PID>/` removed (pipeline cleans in its `finally` block; verify for the PID just printed in pipeline output and clean any orphans from older crashes)

### 8b: Source Audio Backup
- [ ] `cleanup-source-audio.sh` ran successfully (exit code 0)
- [ ] OGG archive verified (decodable, vorbis codec, duration > 0)
- [ ] Original moved to `${MEETING_AUDIO_BACKUP_DIR}/YYYY-MM/`
- [ ] Backup filename includes meeting name + both timestamps: `YYYY-MM-DD-HHMM Title (rec-HHMM).ext`
- [ ] Pipeline duplicate OGG removed from vault root (if it existed)
- [ ] No `Recording *.m4a` or `Recording *.ogg` files remain in vault root for this meeting
