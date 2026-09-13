---
description: "Backend selection, AssemblyAI request flags, transcription provenance, keyterms, and speaker-cluster mapping."
---

# Transcription Backends

| | `assemblyai` (default) | `gemini` (opt-in) |
|---|---|---|
| Default model | `universal-3-5-pro` | `gemini-2.5-flash` |
| Transcription | Speech-to-text API with diarization | Generative audio transcription |
| Speaker output | Anonymous clusters such as `Speaker A` | Prompt-guided labels or inferred names |
| Long recordings | Submitted as one asynchronous transcription job, subject to provider limits | Silence-aware chunks with truncation detection and retries |
| Credential | `ASSEMBLYAI_API_KEY` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Review required | Text accuracy, omissions, and cluster/name mapping | Text accuracy, omissions, generated names, repetition, and silence artifacts |

AssemblyAI uploads the original audio after inspecting metadata. The Gemini path performs silence/container preparation and splitting on working copies.

These defaults describe this repository's configuration. Availability, supported languages, request limits, and pricing can change; check the provider's current documentation when choosing a different model or language. No backend guarantees an accurate or complete transcript.

## Commands

```bash
# Default backend, with known speaker count and an optional private keyterms file
"${SKILL_DIR}/scripts/transcribe.sh" "/path/to/audio.m4a" \
  --speakers-expected 2 \
  --keyterms-file "/path/to/run/keyterms.txt" \
  --output "${MEETING_RAW_DIR}/YYYY-MM-DD-HHMM Team-Sync-Transcript.md"

# Explicit model/language request
"${SKILL_DIR}/scripts/transcribe.sh" "/path/to/audio.m4a" \
  --aai-speech-models universal-3-5-pro \
  --language-code en \
  --output "${MEETING_RAW_DIR}/YYYY-MM-DD-HHMM Team-Sync-Transcript.md"

# Gemini is opt-in; omit --context-file if no context was prepared
"${SKILL_DIR}/scripts/transcribe.sh" "/path/to/audio.m4a" \
  --backend gemini --model gemini-2.5-flash \
  --context-file "/path/to/run/context.txt" \
  --output "${MEETING_RAW_DIR}/YYYY-MM-DD-HHMM Team-Sync-Transcript.md"
```

The backend command does not archive by default. `--no-archive` is retained for compatibility; `--archive-dir <directory>` explicitly enables built-in archival. The meeting workflow prefers `compress-audio.sh --output <confirmed-title>.ogg` after title confirmation. Use one archival path. Built-in archival refuses a different existing destination or symlink and reports failure without discarding the transcript or source.

## AssemblyAI-only flags

| Flag | Meaning |
|------|---------|
| `--keyterms-file <path>` | UTF-8 file, one name/domain term per line, sent as recognition hints |
| `--speakers-expected <N>` | Positive expected speaker-count hint; omit if unknown |
| `--aai-speech-models <model...>` | Ordered request list; default `universal-3-5-pro` |
| `--language-code <code>` | Explicit dominant language; omit for automatic detection |
| `--aai-poll-timeout <seconds>` | Maximum job-polling duration; positive finite number, default `3600` |

Use the exact API identifier `universal-3-5-pro`, not a display name with dots. Supply a fallback model only when its current capabilities match the recording and the requested workflow; do not silently substitute models to satisfy a failed request. Do not assume an explicit language is honored without checking the response/provenance.

## Keyterms and context

Create keyterms from confirmed canonical attendee names and relevant domain terms. Use one term per line, for example:

```text
Alex Example
Jordan Sample
Project Atlas
release checklist
```

These entries are fictional. Do not include private biographies, contact details, credentials, or unrelated registry contents. Only send hints needed for this recording. Hints can improve recognition but do not establish who spoke.

For AssemblyAI, the wrapper gives `--keyterms-file` precedence over context prompt text; it does not send both hint forms together. For Gemini, use `--context-file` per `CONTEXT_TEMPLATE.md`. Confirming a roster helps review; it does not authorize forcing those names onto ambiguous speech.

Both files may contain private data. Create unique files per run, pass their exact paths, and remove them after use. Never store them in tracked test fixtures.

## Provenance

Keep emitted backend/model metadata in the saved transcript. AssemblyAI output includes **Transcript ID**, **Submitted Models**, **Returned Models**, **Model Used**, **Detected Language**, and **Speakers Expected**. A missing returned model is recorded as unavailable rather than guessed. For AssemblyAI, distinguish the submitted model list and requested language from provider-returned model/language information and transcript identifier when available. A requested fallback is not evidence it was selected. For Gemini, preserve the requested model/backend metadata and any available response metadata. Do not invent response values when the provider omits them.

HTTP failures, failed jobs, unexpected states, polling timeout, or no speech stop the run rather than creating a successful empty transcript. Do not accept an older output at the same path as the failed run's result.

Record any later retranscription, excluded corrupt interval, or manual name mapping in the transcript's processing/diarization notes. Do not overwrite provenance to make a different run appear original. Transcript identifiers and local source paths are operational/private output, not public examples or test fixtures.

## Step 2.5: Map clusters to names

1. Read the full transcript and the confirmed attendee roster.
2. Gather explicit self-identification and direct-address cues for each cluster. Use audio review when needed; cluster order, inferred role, or first appearance alone is insufficient.
3. Check for split or merged clusters. An expected count is a hint, not a guarantee; do not discard a short cluster solely because it has few utterances.
4. Map supported identities to canonical names. Preserve original labels in a Diarization Notes mapping, including any segment-specific split.
5. Leave unresolved labels as `Speaker A` or `Unknown speaker` and document uncertainty. Ask the user about material ambiguity instead of choosing a plausible name.

Gemini's generated names require the same evidence check even though a separate cluster-mapping pass may not be needed.

## Transcript quality and recovery

Check for long repeated lines, stalled timestamps, implausible dialogue in quiet stretches, and missing speech. A late final timestamp or normal completion status proves neither accuracy nor completeness. See `QUALITY_CHECKLIST.md`.

When output is corrupted, compare the affected window with audio and retry or choose a different supported backend within existing upload authorization. Preserve the source and note what changed. Recheck recovered text; neither a new backend nor a different cluster count resolves identity automatically.

## Provider references

The wrapper's model-list and language fields correspond to AssemblyAI's [transcript request/response API](https://www.assemblyai.com/docs/pre-recorded-audio/api-reference/transcripts/submit). Review this reference for current limits and response-field semantics.

AssemblyAI explains anonymous speaker labels and count hints in [Speaker Diarization](https://www.assemblyai.com/docs/pre-recorded-audio/label-speakers), and recognition hints in [Prompting and Keyterms](https://www.assemblyai.com/docs/pre-recorded-audio/universal-3-5-pro/prompting).

Gemini's [audio documentation](https://ai.google.dev/gemini-api/docs/audio) describes audio input and transcription prompting. This repository's chunking, retry, and acceptance checks are its own workflow, not provider guarantees.
