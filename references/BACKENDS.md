---
description: "Backend selection, AssemblyAI request flags, transcription provenance, keyterms, local speaker tracking, speaker-cluster mapping, and voiceprint enrollment."
---

# Transcription Backends

| | `assemblyai` (default) | `gemini` (opt-in) |
|---|---|---|
| Default model | `universal-3-5-pro` | `gemini-2.5-flash` |
| Transcription | Speech-to-text API with diarization | Generative audio transcription |
| Speaker output | Anonymous clusters such as `Speaker A`; with `--local-speakers`, names from local voiceprint tracking | Prompt-guided labels or inferred names |
| Long recordings | Submitted as one asynchronous transcription job, subject to provider limits | Silence-aware chunks with truncation detection and retries |
| Credential | `ASSEMBLYAI_API_KEY` | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Review required | Text accuracy, omissions, and cluster/name mapping | Text accuracy, omissions, generated names, repetition, and silence artifacts |

The wrapper loads guarded configuration before bootstrap and installs only `requests` for AssemblyAI or `google-genai` for Gemini. Gemini imports are lazy; the default path does not require its SDK.

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

Use a new transcript destination for each run. The pipeline rejects existing transcript destinations, symlinks/hardlinks to the input, and known archive destination collisions before provider work. Transcript writes use exclusive creation, so a prior transcript is never replaced.

The backend command does not archive by default. `--no-archive` is retained for compatibility; `--archive-dir <directory>` explicitly enables built-in archival. The meeting workflow prefers `compress-audio.sh --output <confirmed-title>.ogg` after title confirmation. Use one archival path. The compressor stages and verifies output before publishing without overwrite. Only the exact same real OGG input/output path may be verified as a no-op; other existing destinations, hardlinks, and symlinks are refused. A later archive failure preserves any completed transcript and the source.

## AssemblyAI-only flags

| Flag | Meaning |
|------|---------|
| `--keyterms-file <path>` | UTF-8 file, one name/domain term per line, sent as recognition hints |
| `--speakers-expected <N>` | Positive expected speaker-count hint; omit if unknown |
| `--aai-speech-models <model...>` | Ordered request list; default `universal-3-5-pro` |
| `--language-code <code>` | Explicit dominant language; omit for automatic detection |
| `--aai-poll-timeout <seconds>` | Maximum job-polling duration; positive finite number, default `3600` |
| `--local-speakers` / `--no-local-speakers` | Local speaker tracking on or off for this run (default from `MEETING_DOCUMENTER_LOCAL_SPEAKERS`, else off); see [Local speaker tracking](#local-speaker-tracking-step-25a-optional-beta) |
| `--roster "A,B"` | Local tracking: confirmed attendees (default: the context file's `Possible speakers:` list, else everyone enrolled) |
| `--channel phone\|zoom\|room` | Local tracking: recording channel (per-channel voiceprints; recorded in the timeline) |
| `--track-backend nemotron3\|seg3\|auto` | Local tracking: boundary model that must be installed (default `nemotron3`) |
| `--track-timeout <seconds>` | Local tracking: maximum wait after AssemblyAI finishes, default `3600` |

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

Keep emitted backend/model metadata in the saved transcript. AssemblyAI output includes **Transcript ID**, **Submitted Models**, **Returned Models**, **Model Used**, **Requested Language**, **Detected Language**, and **Speakers Expected**. The **AAI Requested Language** field records the manual hint or `automatic detection`; **AAI Detected Language** separately records the provider response. A missing returned model is recorded as unavailable rather than guessed. For AssemblyAI, distinguish the submitted model list and requested language from provider-returned model/language information and transcript identifier when available. A requested fallback is not evidence it was selected. For Gemini, preserve the requested model/backend metadata and any available response metadata. Do not invent response values when the provider omits them.

HTTP failures, failed jobs, unexpected states, polling timeout, or no speech stop the run rather than creating a successful empty transcript. Empty or whitespace-only Gemini output also fails without writing a transcript. Do not accept an older output at the same path as the failed run's result.

AssemblyAI runs also write a sidecar next to the transcript: the utterances sidecar (diarized runs; see [Legacy voiceprint check](#legacy-voiceprint-check-cluster-path)) or the words sidecar plus a private speaker timeline (local tracking). A sidecar or timeline write problem is reported but never fails a completed transcript.

Record any later retranscription, excluded corrupt interval, or manual name mapping in the transcript's processing/diarization notes. Do not overwrite provenance to make a different run appear original. Transcript identifiers and local source paths are operational/private output, not public examples or test fixtures.

## Local speaker tracking (Step 2.5a, optional beta)

Local tracking answers "who speaks when" on this machine from enrolled voiceprints. AssemblyAI supplies only text and word timestamps (`speaker_labels=false`), so it creates no speaker embeddings. Nothing about voices leaves the machine.

**Pipeline.** `transcribe.sh --local-speakers` (or `MEETING_DOCUMENTER_LOCAL_SPEAKERS=1`):

1. Before upload, `identify_speakers.py track-check` confirms that local tracking can name people: the voiceprint dependencies and ReDimNet2 model, the nemotron3 boundary model, enrolled voiceprints, the store's own track calibration, and at least one enrolled roster member. It loads no models and reads no audio.
2. Not ready: the run requests AssemblyAI diarization as before, and the transcript header starts with `> [!warning] Local speaker tracking was requested but is unavailable (<reason>)`. Continue with the cluster mapping below.
3. Ready: `identify_speakers.py track` runs in a child process while AssemblyAI transcribes. Nemotron 3 speaker slots → clean-speech units → ReDimNet2 embeddings → calibrated scores against the roster's voiceprints → a per-slot HMM over roster ∪ unknown → `UNCERTAIN(A|B)` where the evidence is weak, `GUEST_n` for unenrolled voices.
4. Each AssemblyAI word takes the timeline speaker it overlaps most, and turns are rebuilt at every speaker change. The transcript renders names as-is, `GUEST_n` as `Guest n`, `UNCERTAIN(A|B)` as `A or B (uncertain)`, `UNKNOWN` as `Unknown speaker`, and adds a folded **Speaker review** callout listing those turns' timestamps.
5. Outputs: the transcript, `<transcript stem>.words.json` next to it (text and timing, no speakers), and `<transcript stem>.timeline.json` in `MEETING_DOCUMENTER_TIMELINE_DIR` (default `~/.local/share/meeting-documenter/timelines`, mode 600). The timeline holds names and times; it is refused inside notes roots, Obsidian vaults, or this clone. `enroll --timeline` and `audit sample` read it.
6. If tracking fails or times out after transcription, the transcript is written unlabelled with a `[!warning]` giving the reason, and the words sidecar is kept. Map speakers from content, or re-run the transcription.

**Setup.**

```bash
echo 'MEETING_DOCUMENTER_VOICEPRINT=1' >> .env            # installs torch + sherpa-onnx on the next run
./scripts/transcribe.sh --help
.venv/bin/python scripts/voiceprint/fetch_models.py        # ReDimNet2, Silero (sha256-pinned)
# nemotron3 needs a transformers build that ships the Nemotron 3 Diarization class (newer than 5.17.0)
.venv/bin/python scripts/voiceprint/fetch_models.py nemotron3
# enroll a few meetings per person (Step 8c), then fit the tracker's calibration on your own store
./scripts/identify-speakers.sh track-calibrate
./scripts/identify-speakers.sh track-check --roster "Alice Example,Bob Example"   # {"ready": true, ...}
```

**Default mode.** Local tracking is off unless `--local-speakers` or `MEETING_DOCUMENTER_LOCAL_SPEAKERS=1` is set, and `MEETING_DOCUMENTER_LOCAL_SPEAKERS=0` or `--no-local-speakers` turns it off. To make it the default for every install, the maintainer sets `LOCAL_SPEAKERS_DEFAULT = True` in `scripts/transcribe_pipeline.py`. The readiness check keeps the fallback safe either way.

**Standalone.** `identify-speakers.sh track --audio X --roster "A,B" [--channel zoom] --out X.timeline.json` writes a timeline for any recording; `--rttm` exports RTTM. It is a no-op (exit 0) without dependencies, models, or voiceprints.

**Limits.** The nemotron3 boundary model tracks at most 8 speakers; use the cluster path for larger meetings. The `seg3` fallback (`--track-backend seg3`) names far more strangers (24 % of unenrolled speech on AMI vs 3 % with nemotron3), so the pipeline requires nemotron3 unless you pass `--track-backend seg3` or `auto` on purpose. Speech right at turn edges is sometimes missed or given to the neighbouring speaker. AssemblyAI's Mandarin tokens can span pauses, so a token that crosses a speaker change is attributed to whoever it overlaps most.

## Step 2.5c: Map clusters to names

1. Read the full transcript and the confirmed attendee roster.
2. Gather explicit self-identification and direct-address cues for each cluster. Use audio review when needed; cluster order, inferred role, or first appearance alone is insufficient.
3. Check for split or merged clusters. An expected count is a hint, not a guarantee; do not discard a short cluster solely because it has few utterances.
4. Map supported identities to canonical names. Preserve original labels in a Diarization Notes mapping, including any segment-specific split.
5. Leave unresolved labels as `Speaker A` or `Unknown speaker` and document uncertainty. Ask the user about material ambiguity instead of choosing a plausible name.

Gemini's generated names require the same evidence check even though a separate cluster-mapping pass may not be needed.

## Legacy voiceprint check (cluster path)

The cluster-level check predates local tracking and is kept as a fallback and for comparison; new work should use local tracking. Here voiceprints are an **advisory** signal layered on AssemblyAI diarization. AssemblyAI still decides the clusters; the voiceprint step only proposes which enrolled person each cluster sounds like. It never edits transcripts, and it never names anyone the roster and content do not support.

**Utterances sidecar.** Each AssemblyAI run writes `<transcript stem>.utterances.json` next to the transcript (created exclusively; an existing sidecar stops the run before upload). It contains the meeting text, so treat it like the transcript: private, never committed.

```json
{"schema": 1, "source": "assemblyai", "transcript_id": "...", "audio_duration_ms": 3200,
 "speaker_map": null,
 "utterances": [{"speaker": "A", "start": 0, "end": 1000, "text": "..."}]}
```

`speaker_map` stays `null` until names are confirmed; Step 8c writes it (`{"A": "Alice Example", "B": "Bob Example"}`), and later label checks read it.

**Run.** `scripts/identify-speakers.sh` loads the same trusted config as `transcribe.sh`, then runs `scripts/identify_speakers.py`:

```bash
"${SKILL_DIR}/scripts/identify-speakers.sh" identify \
  --audio "/path/to/audio.m4a" \
  --utterances "<...>-Transcript.utterances.json" \
  --roster "Alice Example,Bob Example" [--channel phone|zoom|room] \
  --json "/path/to/run/voiceprint-identify.json"
```

- `--roster` is the Step 0b confirmed attendee list; it closes the candidate set. Names resolve through the private registry's aliases.
- Output: a cluster × person score matrix (raw cosine, and AS-norm when the cohort is large enough), a proposed mapping, per-cluster confidence (`high`/`low`/`unknown`), best-vs-second gaps, pairwise swap margins, and a `label_check` against any existing `speaker_map`.
- Exemplars from the same meeting are excluded by default, so a re-run cannot vote for itself.
- **Graceful no-op:** `status` is `no_voiceprints`, `no_deps`, `no_embedder`, or `error`, with exit code 0. Continue with content-only mapping.

**Reconcile, never auto-apply.**

| Voiceprint result | Action |
|---|---|
| `high`, agrees with content mapping | Corroborating evidence; proceed |
| Disagrees with content, or `label_check` is `inversion`/`conflict` | Stop and ask the user; offer a short clip for listening, cut to a temporary directory outside notes |
| `low`/`unknown`, insufficient cluster, or person not enrolled | Keep `Speaker X` unless content and roster settle it |
| `by_elimination` (roster leftover) | Not voice evidence; verify with content |

**Calibration.** Thresholds (τ open-set accept, τ_low closed-roster floor, δ two-speaker swap margin, gap best-vs-second) depend on the embedding model, channel mix, and the people enrolled, so they are calibrated per store. Without `calibration.json` the CLI uses conservative priors and **caps every proposal at `low`** (and disables roster elimination): voice alone never names anyone until `calibrate` has run on the user's own data.

## Voiceprint enrollment (Step 8c)

Enroll only after the user confirmed the names, and only people who consented to biometric enrollment. Ask before enrolling anyone.

**From a speaker timeline (local tracking).** One call per confirmed person. Only timeline turns of at least 3 s are used (`--min-turn-s`), and every other speaker's turns, ± 0.3 s, are cut out as crosstalk:

```bash
TL="$HOME/.local/share/meeting-documenter/timelines/<...>-Transcript.timeline.json"
"${SKILL_DIR}/scripts/identify-speakers.sh" enroll --audio "/path/to/audio.m4a" --timeline "$TL" \
  --name "Alice Example" --channel zoom --source confirmed_meeting        # Alice's own turns
"${SKILL_DIR}/scripts/identify-speakers.sh" enroll --audio "/path/to/audio.m4a" --timeline "$TL" \
  --speaker GUEST_1 --name "Carol Example" --source user_ear              # a guest the user identified
```

`--spans "03:10-03:40,12:05-12:30"` restricts enrollment to spans the user confirmed by ear. It is required for an `UNCERTAIN(A|B)` label; `UNKNOWN` and overlap turns are never enrolled. A named label that is not `--name` is refused (exit 2) unless `--force`.

**From a diarized cluster (cluster path).** Write the confirmed names into the utterances sidecar's `speaker_map`, then one call per confirmed cluster:

```bash
"${SKILL_DIR}/scripts/identify-speakers.sh" enroll \
  --audio "/path/to/audio.m4a" --utterances "<...>-Transcript.utterances.json" \
  --name "Alice Example" --cluster A --channel zoom --source confirmed_meeting
```

- **Guards.** Exit 2: the name is not in the private registry (`--allow-unregistered` to override), or the sidecar `speaker_map` or the timeline label says otherwise. Exit 3: the new clips sound closer to someone already enrolled (likely an inverted or mislabelled cluster). Exit 4: the clip config differs from `calibration.json`. Do not pass `--force` without the user; verify by ear first.
- **Idempotent.** Re-enrolling the same meeting replaces that meeting's clips; each person keeps at most 200 clips, trimming the most-represented meeting first.
- **Cohort (optional).** `cohort-add --audio ... --utterances ... [--clusters C,D]` stores clips of other, non-enrolled voices. They are only used for score normalisation (AS-norm) and as "stranger" trials during calibration.
- **Track calibration.** Local tracking needs `identify-speakers.sh track-calibrate` once a few meetings per person are enrolled, and again after enrolling substantially more. It refits the tracker's duration-aware score calibration leave-one-meeting-out on the stored vectors (no audio) and writes it into `calibration.json`. Without it, `track-check` reports `no_calibration` and the pipeline falls back to cluster diarization.
- **Calibrate (legacy cluster check).** After at least a few meetings per person, run `identify-speakers.sh calibrate`. It rebuilds enrollment leave-one-meeting-out from the stored vectors (no audio needed): τ = 99th and τ_low = 95th percentile of scores for voices that are *not* enrolled (people with a single meeting, leave-one-speaker-out passes, and cohort voices); δ = 5th percentile of true pair margins; gap = 5th percentile of target-minus-best-other. It needs at least 50 stranger scores (otherwise it writes nothing and exits 5), reports a nested leave-one-meeting-out error estimate, and records the enrollment clip config (with an explicit clip cap) so `identify` and `enroll` reproduce the same clips on CPU or MPS. `--dry-run` reports without writing. Re-run it after enrolling substantially more data.
- **Admin.** `list` shows people, clip counts, channels and model ids (no vectors). `forget --name "Alice Example" --yes` deletes that person's exemplars and cohort rows from every model and vacuums the database; also remove their `voiceprint:` registry metadata.
- **Setup.** Set `MEETING_DOCUMENTER_VOICEPRINT=1` and run `transcribe.sh --help` once to install the optional dependencies (`scripts/voiceprint/requirements.txt`), then fetch pinned models with `.venv/bin/python scripts/voiceprint/fetch_models.py` (downloads into `VOICEPRINT_MODEL_DIR`, sha256-verified).

## Transcript quality and recovery

Check for long repeated lines, stalled timestamps, implausible dialogue in quiet stretches, and missing speech. A late final timestamp or normal completion status proves neither accuracy nor completeness. See `QUALITY_CHECKLIST.md`.

When output is corrupted, compare the affected window with audio and retry or choose a different supported backend within existing upload authorization. Preserve the source, choose a new transcript destination, and note what changed. Recheck recovered text; neither a new backend nor a different cluster count resolves identity automatically.

## Provider references

The wrapper's model-list and language fields correspond to AssemblyAI's [transcript request/response API](https://www.assemblyai.com/docs/pre-recorded-audio/api-reference/transcripts/submit). Review this reference for current limits and response-field semantics.

AssemblyAI explains anonymous speaker labels and count hints in [Speaker Diarization](https://www.assemblyai.com/docs/pre-recorded-audio/label-speakers), and recognition hints in [Prompting and Keyterms](https://www.assemblyai.com/docs/pre-recorded-audio/universal-3-5-pro/prompting).

Gemini's [audio documentation](https://ai.google.dev/gemini-api/docs/audio) describes audio input and transcription prompting. This repository's chunking, retry, and acceptance checks are its own workflow, not provider guarantees.
