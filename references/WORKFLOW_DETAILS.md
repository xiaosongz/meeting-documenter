---
description: "Edge cases, conditional branches, and fast-paths for the meeting documentation pipeline. Read specific sections as needed during execution — not all sections apply to every meeting."
---

# Workflow Details

## Fast-Path: Existing Transcript

When starting from an existing transcript, no transcription service or audio tools are needed.

**N/A:** Steps 1 (audio prep), 2 (transcription), 2.5 (cluster mapping from a new backend run), 2b (archive), and 8b (source backup). Mark their audio-only checklist items N/A.

**Still required:**

1. Gather meeting metadata in Step 0 from the transcript or user; do not confuse its export timestamp with the meeting time.
2. Resolve names found in the transcript in Step 0b, using the private registry and user confirmation for ambiguity. No context/keyterms file is needed.
3. Read the complete transcript, then perform Steps 3–8a. Preserve missing/uncertain timestamps and identities as uncertainty.
4. Derive the summary filename from the transcript when appropriate (remove `-Transcript`). Preserve the supplied source and link its actual path.
5. Omit the Recording footer link when no recording archive exists. Do not claim audio authenticity or backend provenance that was not verified.

## Step 0: Metadata Gathering Details

### When User Provides Full Details

If the user's prompt includes title, project, and attendees, no questions needed. Extract and proceed.

### When User Provides Only an Audio Path

Use AskUserQuestion to gather:
- Meeting title (suggest based on audio filename)
- Meeting date/start time (establish whether a filename timestamp is start, stop, or export time before estimating; label uncertainty)
- Attendees (if known — "Who was in this meeting?")

### Companion Files (Optional)

If a sidecar file with title, timestamp, attendees, or an AI-generated pre-summary lives alongside the recording, read it first and pre-populate the metadata. Treat generated sidecars as hints; confirm material ambiguities and use the transcript/audio as evidence for decisions and tasks.

### Speaker Name Resolution (Step 0b — MANDATORY)

**Why this exists:** Transcription engines and generated attendee lists can misidentify speaker names. These guesses are often close but wrong (misheard or transliterated variants of the same name). A single misspelled name propagates through context file → transcript → summary → daily note → project reference — 5 files with wrong names.

**Procedure:**

1. Read `references/KNOWN_SPEAKERS.yaml`
2. For each raw name from the transcript, metadata, or user input:
   a. Check for case-insensitive exact match on `canonical_name`
   b. Check for case-insensitive match on any entry in `aliases`
   c. If matched → record the `canonical_name`, `role`, and `wikilink`
   d. If unmatched → mark as `[NEW SPEAKER]`
3. Present unresolved or changed mappings via AskUserQuestion; reuse names already confirmed in the conversation:

```
Speaker Name Resolution:
| Raw Name   | → | Resolved Name   | Role     | Status         |
|------------|---|-----------------|----------|----------------|
| Allice     | → | Alice Example   | PI       | ✓ Alias match  |
| Bobby      | → | Bob Sample      | Analyst  | ✓ Alias match  |
| NewPerson  | → | ???             | ???      | [NEW] — needs input |
```

4. For `[NEW]` speakers:
   - Ask user for the correct full name and role
   - Add only confirmed identity/aliases to the private registry when the user wants that registry updated; never populate it from an unsupported transcription guess

**Edge cases:**
- **Ambiguous match** (e.g., a first name matches two registry entries): Present both options and ask user to choose
- **Team-based filtering**: If the detected project is known, prefer speakers whose `teams` field includes that project
- **Name not in summary**: If a sidecar AI summary doesn't mention a known attendee but they spoke in the meeting, the user can add them during confirmation

### Context and Keyterms Files

Follow `references/CONTEXT_TEMPLATE.md`: use a unique private directory for each run, confirmed canonical names, and only the relevant domain terms. AssemblyAI uses newline-separated keyterms; Gemini can use a context prompt. Pass the exact file path and retain it for cleanup.

If attendees are unknown, omit the roster and retain generic labels. Existing transcripts need neither file.

## Step 1: Audio Formats

Pass the source audio to `scripts/transcribe.sh`. AssemblyAI inspects metadata and uploads the original. Gemini prepares working copies and repairs container/codec issues when needed. Neither path changes the original. Use the separate named archive step only after the title is confirmed.

## Step 2: Transcription

Transcription environment is managed automatically by `scripts/transcribe.sh`. No manual setup needed.

### Transcript Frontmatter Update

`transcribe_pipeline.py` emits transcript metadata and backend/model provenance. Review the title/date/source fields against the confirmed meeting; preserve requested versus returned model/language details. After a named archive is verified, rewrite `source:` in the configured `LINK_STYLE`. See `BACKENDS.md` for cluster mapping and provenance.

## Step 3: No Project Detected

If no keywords match any project in PROJECT_KEYWORDS.yaml:

1. Ask user: "No project keywords detected. Which project should this meeting be linked to?"
2. Provide list of active projects as options (from your projects directory)
3. Include "None — skip project linking" option
4. If "None" selected, skip Step 6 entirely

## Step 5: Daily Note Edge Cases

### Daily Note Does Not Exist

Use the meeting date and configured `DAILY_NOTE_PATH_FORMAT`, not today's processing date. Create the note with an available daily-note skill, existing template, or the minimal skeleton in `SKILL.md`, then add the meeting row. Check for an existing row/reference before adding duplicates on reruns.

### Meetings Table Has Different Columns

The daily note meeting table may not match the expected format. Common variations:

| Format | Columns |
|--------|---------|
| Standard | `Time \| Meeting \| Transcript \| Summary` |
| Extended | `Time \| Meeting \| Attendees \| Notes` |

Adapt to whatever column structure exists. Place transcript and summary links (form per `LINK_STYLE`) in the most appropriate column (usually "Notes" or create separate entries).

### No Meetings Section

If the daily note has no `## Meetings` section, add one after the Focus or Carryover section with a new table.

## Step 6: Project Folder Edge Cases

### No Reference-Note Subdir

Let `SUBDIR="${PROJECT_MEETING_SUBDIR-Meeting}"`. Validate it as a single relative segment or empty before creating anything: reject absolute paths, slashes, and `..`, as in `SKILL.md` Step 6. Then create it:
```bash
mkdir -p "${PROJECTS_DIR}/{ProjectName}/${SUBDIR}"
```

When `PROJECT_MEETING_SUBDIR=""`, reference notes go directly into the project root — no subdir to create.

### No Dashboard or No Recent Meetings Section

If the project `Dashboard.md` does not have a Recent Meetings section, skip the dashboard update. The reference note in the per-project reference directory (`${PROJECT_MEETING_SUBDIR-Meeting}/` subdir if set, otherwise the project root) is sufficient for discoverability.

## Step 3b: Transcript Renaming (Optional)

The transcript is created in Step 2 before the meeting title is confirmed in Step 3. This means the transcript filename may not match the final summary filename (e.g., transcript: `Generic-Meeting-Transcript.md`, summary: `ProjectAlpha-Sync.md`).

After the title is confirmed in Step 3, you may optionally rename the transcript to match. If `LINK_STYLE=wikilink` and Obsidian is running, use its native rename (which auto-updates wikilinks); otherwise grep + manual fixup.

**When to rename:**
- When the auto-generated title is generic and the confirmed title is much more descriptive
- When the meeting spans multiple topics and the original title only captures one

**When to skip:**
- When the titles are similar enough that the mismatch is trivial
- When Obsidian is not running (manual link updates needed)

If you skip renaming, the transcript link in the summary will still work because it uses the original path. The mismatch is cosmetic, not functional.

## Step 5b: Carryover Task Cross-Reference

After updating the Meetings table in Step 5, scan the daily note's **Carryover** section for tasks that were addressed during this meeting.

**Procedure:**
1. Read the daily note's Carryover section
2. For each open carryover task, check if the meeting transcript/summary contains resolution:
   - Was a question answered? (e.g., "Ask PI about X" → answered in meeting)
   - Was a decision made that resolves the task?
   - Was the deliverable explicitly completed or accepted? Discussion alone is not completion.
3. If a carryover task was addressed, suggest marking it `[x]` with a note linking to the meeting summary
4. Present any suggested completions to the user via AskUserQuestion before making changes

**Example:**
```
Carryover task resolved by this meeting:
- [x] Ask PI: which event flags should the snapshot include?
  → Addressed in meeting: confirmed list of event flags discussed
```

This step prevents carryover tasks from becoming stale when their resolution is documented in meeting notes but not reflected in the daily note.

## Multi-Project Meetings

When a meeting spans multiple projects (detected by keywords from different projects in Step 3):

### Detection
- Keywords match 2+ projects in `PROJECT_KEYWORDS.yaml`
- Confirm with user: "Link to both projects?" (recommend yes)

### File Structure

One meeting produces these files:

| File | Location | Content |
|------|----------|---------|
| **Full summary** | `${MEETING_NOTES_DIR}/YYYY-MM-DD-HHMM Title.md` | Complete summary with all topics, decisions, action items |
| **Project A reference** | `${PROJECTS_DIR}/ProjectA/${PROJECT_MEETING_SUBDIR-Meeting}/YYYY-MM-DD-HHMM Title.md` | Filtered: only Project A decisions + action items |
| **Project B reference** | `${PROJECTS_DIR}/ProjectB/${PROJECT_MEETING_SUBDIR-Meeting}/YYYY-MM-DD-HHMM Title.md` | Filtered: only Project B decisions + action items |

### Reference Note Format

Each project reference note should:
1. Use the **same filename** as the full summary (for consistency)
2. Include only that project's **decisions** and **action items**
3. Include a brief **executive summary** scoped to that project's topics
4. Link back to the full summary and transcript in the footer
5. Use project-specific tags (one project's tag per reference note)

### Dashboard Updates

Update **both** project Dashboards:
- Add meeting to each Dashboard's Recent Meetings / Meetings section
- Adapt to each Dashboard's format (bullet list vs. table — they may differ)
- Update the `updated:` frontmatter field on each Dashboard

## Multi-Segment Audio

Some recording tools split long meetings into multiple files. If user provides multiple audio files:

1. Confirm segment order and inspect codec/sample-rate/channel compatibility. Use ffmpeg's concat demuxer for compatible inputs, or normalize separate working copies first; do not assume byte-concatenating compressed files produces a valid timeline. Preserve each original and validate combined duration/full decode before transcription.
2. After concatenation, run the combined file through `scripts/transcribe.sh` — chunking, transcription, and combination are handled automatically.
3. Note in the transcript: "Source: combined from N segments"
