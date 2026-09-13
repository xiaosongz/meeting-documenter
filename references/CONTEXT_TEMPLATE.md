---
description: "Generate minimal private per-run recognition hints for AssemblyAI or context for Gemini after resolving attendee names."
---

# Speaker Context and Keyterms

Complete Step 0b before creating hints. Use confirmed names and relevant terminology only; do not export the full private speaker/project registries.

Create a unique temporary directory for this run and keep its path for cleanup:

```bash
MEETING_RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/meeting-documenter.XXXXXX")
```

Shell state may not persist between agent calls; carry this exact path into subsequent calls. Files under the directory can contain private data. Remove only this run's directory when finished.

## AssemblyAI keyterms (default backend)

Write `${MEETING_RUN_DIR}/keyterms.txt` as UTF-8 with one confirmed name or domain term per line. Fictional example:

```text
Alex Example
Jordan Sample
Project Atlas
release checklist
```

Pass `--keyterms-file "${MEETING_RUN_DIR}/keyterms.txt"`. Add `--speakers-expected` only for a supported expected count. Keyterms help recognition; they do not assign names to diarization clusters. Review mappings in Step 2.5. See `BACKENDS.md` for flag precedence and model selection.

## Gemini context (opt-in backend)

Write `${MEETING_RUN_DIR}/context.txt` using confirmed metadata:

```text
## Meeting Context for Transcription

Date: [YYYY-MM-DD]
Possible speakers:
- [Canonical name] ([confirmed role, if relevant])
- [Canonical name] ([confirmed role, if relevant])

Domain terms: [Relevant names/acronyms only]

Transcribe only audible speech. Use a name only when the dialogue supports
that identity; otherwise retain a consistent generic speaker label. Do not
invent dialogue during silence or turn this roster into spoken content.
```

Pass `--backend gemini --context-file "${MEETING_RUN_DIR}/context.txt"`. Prompt instructions do not replace transcript-authenticity review.

## Before writing hints

- Resolve names against the private registry, or with the user if it is unavailable.
- Keep uncertain people/roles out of assertions; generic speaker labels are acceptable.
- Include only information relevant to this recording and its authorized transcription service.
- If attendees are unknown, omit the files rather than inventing a roster.
- On the existing-transcript path, do not generate context/keyterms or make a transcription request.
