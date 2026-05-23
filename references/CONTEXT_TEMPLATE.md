---
description: "Template for generating speaker diarization context files. Fill in attendee names, roles, and domain terms before passing to the transcription pipeline via --context-file."
---

# Speaker Diarization Context Template

Generate a context file at `/tmp/meeting_context.txt` before running transcription. Fill in the bracketed fields based on information gathered in Step 0.

## Template

```text
## Meeting Context for Transcription

**Date:** [YYYY-MM-DD]
**Expected Speakers:**
- [Name] ([Role — e.g., PI, analyst, coordinator])
- [Name] ([Role])

**Domain Terms:**
[Comma-separated list of project-specific terms from PROJECT_KEYWORDS.yaml]

**Instructions:** Use the speaker names above when attributing dialogue. Do NOT use "Speaker 1/2/3". If a speaker cannot be confidently identified, use "Unknown Speaker".
```

## How to Generate

1. Complete Step 0b (Speaker Name Resolution) FIRST — do NOT write the context file with unvalidated names
2. Use the **resolved canonical names** from `KNOWN_SPEAKERS.yaml`, not the raw extracted names
3. Read `references/PROJECT_KEYWORDS.yaml` for domain terms matching the detected/specified project
4. Write filled template to `/tmp/meeting_context.txt`
5. Pass to transcription: `transcribe.sh audio.ogg --context-file /tmp/meeting_context.txt`

## Name Validation Checklist (before writing context file)

- [ ] Every speaker name resolved against `KNOWN_SPEAKERS.yaml`
- [ ] No raw/unvalidated names remain (e.g., a misheard alias should be resolved to its canonical name)
- [ ] User confirmed the resolved name list via AskUserQuestion
- [ ] New speakers added to `KNOWN_SPEAKERS.yaml` after meeting processing
- [ ] Roles match the speaker's actual role (not a guess from the AI summary)

## When Attendees Are Unknown

If no attendee information is available, skip context file generation. The transcription will use generic "Speaker 1/2/3" labels. Speaker identification can be done manually during Step 4 (summary generation) based on conversational patterns.
