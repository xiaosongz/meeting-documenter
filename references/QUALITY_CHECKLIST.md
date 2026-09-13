---
description: "Verify source fidelity, transcript provenance, summaries, links, and cleanup before completing meeting documentation."
---

# Quality Checklist

Verify applicable items before marking the task complete. Audio-only checks are N/A when starting with an existing transcript. Unknown identity may remain explicitly unresolved; it must not be replaced with a fabricated attendee or assignee.

Use configured paths, `DAILY_NOTE_PATH_FORMAT`, and `PROJECT_MEETING_SUBDIR` (empty means the project root). Link examples use default `wikilink` form; substitute `markdown` or `plain` per `SUMMARY_FORMAT.md`. For plain text, verify names/paths, not nonexistent clickable links.

## Metadata and speaker resolution (Steps 0/0b)

- [ ] Date/time/title reflect source evidence or user confirmation; estimates are labeled.
- [ ] Transcript-only input still received metadata and name resolution.
- [ ] Names match confirmed canonical names; ambiguous identities remain explicitly uncertain.
- [ ] Context/keyterms, if used, contain only relevant confirmed information in unique per-run files.
- [ ] Registries remain private/gitignored; fictional template entries were not treated as real attendees.

## Audio preparation (Step 1 — audio only)

- [ ] AssemblyAI audio metadata was inspected; the original was submitted without Gemini preparation. For Gemini, pipeline logs show the trailing-silence and codec/container decisions.
- [ ] Original source is untouched; preparation used working files.
- [ ] Any intended trim retains the meeting's relevant speech and is documented.

## Transcript (Step 2)

- [ ] Transcript exists, is non-empty, and is readable; a new output uses `${MEETING_RAW_DIR}/YYYY-MM-DD-HHMM Title-Transcript.md`.
- [ ] Timestamped content and speaker labels are present where the source provides them; missing timestamps in a supplied transcript were not invented.
- [ ] Backend/model provenance is retained for new transcriptions; requested and returned values are not conflated.
- [ ] Provider failures/retries completed successfully before the output was accepted (audio only).
- [ ] Beginning, middle, and end coverage were reviewed against audio when available; gaps are explained (audio only).
- [ ] A span of at least 80% of audio duration is treated only as a coarse warning heuristic, never proof of completeness or accuracy.

## Speaker-cluster mapping (Step 2.5 — AssemblyAI only)

- [ ] Mappings use explicit dialogue/audio evidence and are checked against the roster.
- [ ] Original clusters and manual mappings/ambiguities are documented in Diarization Notes.
- [ ] Split/merged clusters were considered; the expected count was not treated as identity proof.
- [ ] Unresolved labels remain visible. No person was invented to fill a cluster.

## Transcript authenticity (Step 7 — audio only)

Generative transcription can produce plausible speech in quiet stretches or repeat text while omitting real dialogue. Speech-to-text output can also contain errors and omissions. Check both backends; context names and timestamp span do not establish authenticity.

- [ ] Reviewed cyclic greeting/farewell passages or dialogue that appears to recite the context roster.
- [ ] Reviewed repeated lines (for example, five consecutive copies), frozen timestamps, implausibly dense speech, or unexplained jumps. These are review triggers, not automatic deletion rules.
- [ ] Compared suspicious windows with audio, including interior silence or low-volume passages; plausible timestamps alone did not clear them.
- [ ] Corrupt windows were corrected/retranscribed and rechecked, or clearly excluded with a stated limitation; the source transcript was preserved.
- [ ] Summary decisions/tasks use supported content only. A replacement backend's output was also checked.
- [ ] If audio is unavailable, audio authenticity is explicitly unverified rather than claimed to pass.

## Recording archive (Step 2b — audio only, when requested)

- [ ] Only one archival path was used; built-in archival was not combined with a second redundant archive.
- [ ] Named OGG exists at `${MEETING_RECORDINGS_DIR}/YYYY-MM-DD-HHMM Title.ogg` after title confirmation.
- [ ] Full decode succeeds; codec is Vorbis or Opus; duration matches the intended source interval.
- [ ] Transcript `source:` points to the archive in `LINK_STYLE` form without losing transcription provenance.
- [ ] Recording links appear only when the linked recording exists.

## Summary (Step 4)

- [ ] Saved to `${MEETING_NOTES_DIR}/YYYY-MM-DD-HHMM Title.md`.
- [ ] Valid frontmatter includes title, description, date, type, tags, attendees, meeting_type, and meeting_outcome.
- [ ] Description states the key outcome; decisions reflect actual decisions rather than proposals.
- [ ] Required sections are present: Executive Summary, Key Decisions, Action Items, Parking Lot, Topics, Follow-Up.
- [ ] Every action item is an explicit assignment/commitment, has a supported owner, and remains unresolved at meeting end.
- [ ] Later completion, cancellation, or reassignment was reconciled before finalizing the task list.
- [ ] Original deadline wording is preserved. `[due:: YYYY-MM-DD]` is included only when the stated date resolves unambiguously; absent/ambiguous dates were not invented.
- [ ] Action items use checkbox format and `[assignee::]`; `[project::]` appears when applicable. No fabricated `Team`/`TBD` owners.
- [ ] Unassigned requests remain unresolved discussion/follow-up, and only explicitly deferred topics enter Parking Lot.
- [ ] Footer links to the actual transcript; recording link omitted when no archive exists.

## Daily note and carryover (Steps 5/5b)

- [ ] Daily-note path uses the meeting date and configured format/locale.
- [ ] Entry fits existing Meetings table columns and was not duplicated on a rerun.
- [ ] Transcript/summary links use the configured form; table has a preceding blank line.
- [ ] Carryover completions, if suggested, are supported by actual resolution and approved before edits.
- [ ] Read back the note to verify the saved entry.

## Project references (Step 6 — when confirmed)

- [ ] Reference path uses the confirmed project and validated single-segment/empty subdir.
- [ ] Each applicable project has one reference with scoped decisions/tasks and links to the canonical full summary/transcript.
- [ ] References agree with the canonical summary; no new task or deadline was introduced in a project copy.
- [ ] Dashboard Recent Meetings entry was updated if present, with `updated:` changed only when edited.
- [ ] Read back references and Dashboard edits; reruns did not create duplicate entries.

## Files and links (Step 7)

- [ ] All output files exist and are non-empty; frontmatter is valid.
- [ ] Note and attendee links resolve in the chosen `LINK_STYLE`; Markdown links are relative to the containing note.
- [ ] Assignee names match supported canonical identities; unresolved transcript labels are not false person-note links.
- [ ] Summary, transcript, daily note, and references point to the intended files after any rename.

## Cleanup and original preservation (Step 8)

- [ ] This run's private context/keyterms and working files were removed after use.
- [ ] Pipeline exited; an interrupted run has no unaccounted child process or temporary audio directory.
- [ ] Original recording is still at its source path, or a verified backup exists at the reported destination after an authorized move.
- [ ] Cleanup helper, if used, completed successfully; an existing destination was not assumed to contain this source.
- [ ] No unrelated same-stem audio was removed as a presumed duplicate.
- [ ] Final response names the created outputs and any material unresolved identity/content limitation.

## Testing repository changes

- [ ] Automated tests use synthetic/public fixtures and mocked provider responses; no private recording was uploaded as a smoke test.
- [ ] No private registries, transcripts, recordings, provider identifiers, local paths, or credentials were added to tracked fixtures/logs.
- [ ] Any authorized live test is explicitly reported separately from offline checks, with its actual limits.
