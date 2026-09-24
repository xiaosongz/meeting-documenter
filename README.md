# meeting-documenter

A [Claude Code](https://claude.com/claude-code) skill that turns a meeting recording or existing transcript into structured, linked notes: a summary, supported action items, a daily-note row, and project references. The next release adds optional, local [known-speaker tracking](#who-speaks-when-local-speaker-tracking-beta): who speaks when, from voiceprints of consenting people, with no speaker embeddings in the cloud. Release **0.8.0-beta.1** added an advisory voiceprint check on top of AssemblyAI diarization; 0.7.0 added AssemblyAI transcription, explicit model/language controls, provenance, and stronger source-fidelity checks. See the [changelog](CHANGELOG.md).

## What it does

1. Resolves metadata and speaker names using confirmed context and your private registries.
2. Transcribes audio with **AssemblyAI by default** (`universal-3-5-pro`), or **Gemini on request** (`gemini-2.5-flash`). Gemini includes audio preparation, silence-aware chunking, and truncation retries.
3. Reviews anonymous speaker clusters or generated names, preserving unresolved identities. Optionally tracks who speaks when on your machine from locally stored voiceprints of consenting people, marking weak evidence as uncertain rather than guessing.
4. Creates a source-linked summary with actual decisions and explicit assignments still open at meeting end; preserves stated deadline wording.
5. Updates the meeting date's daily note and confirmed project references using your link style and folder conventions.
6. Creates a named OGG archive after title confirmation, and can preserve the original in your configured backup directory.

An existing transcript skips transcription and audio handling while keeping metadata, name resolution, summary, and integration checks. No transcription API key is needed for that path.

## Requirements

- Claude Code and a filesystem-based Markdown notes directory, such as an Obsidian vault.
- For audio: Python 3, `ffmpeg`, `ffprobe`, `bc`, and [`uv`](https://github.com/astral-sh/uv) on `$PATH`.
- For the default backend: an `ASSEMBLYAI_API_KEY`.
- For opt-in Gemini: `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

The wrapper validates and loads the selected trusted configuration before bootstrapping a local Python environment. It installs only the selected backend's missing dependency: `requests` for AssemblyAI or `google-genai` for Gemini. Gemini imports are lazy, so the default AssemblyAI path works without `google-genai`. The archive helper uses Python 3 for atomic publication without overwriting existing files.

## Install and setup

```bash
git clone https://github.com/xiaosongz/meeting-documenter
cd meeting-documenter
```

For guided setup, ask your agent:

> Read `references/ONBOARDING_PROMPT.md` in the meeting-documenter repo and follow it to set this skill up for my system. Start in diagnostic mode.

Onboarding inspects your existing layout before adapting configuration/registries or creating a minimal notes structure. Diagnostic mode is read-only.

For manual setup:

```bash
cp .env.example .env
chmod 600 .env
$EDITOR .env  # set VAULT_PATH and the selected backend's API key locally

cp references/KNOWN_SPEAKERS.template.yaml references/KNOWN_SPEAKERS.yaml
cp references/PROJECT_KEYWORDS.template.yaml references/PROJECT_KEYWORDS.yaml
$EDITOR references/KNOWN_SPEAKERS.yaml
$EDITOR references/PROJECT_KEYWORDS.yaml
```

Replace fictional registry entries with your own confirmed data. Runtime registries and `.env` are gitignored; the templates remain public.

**Configuration loading:** `scripts/transcribe.sh` loads this clone's `.env`, or a file selected by the shell export `MEETING_DOCUMENTER_ENV_FILE=/path/to/trusted.env`. Export that selector outside the file it selects. An empty value disables file loading and uses exported variables. The wrapper never implicitly loads `~/.env`; it rejects config files not owned by the current user or writable by group/others. A sourced config is executable shell code, so only use one you trust.

**First-run side effect:** after guarded configuration loading, even `scripts/transcribe.sh --help` can create `.venv/` and install the selected backend's missing dependency. It is not a read-only diagnostic or proof of a successful provider transcription. An external `.env` does not make a read-only clone writable; provision its environment before making an installation read-only.

## Use

Ask Claude Code:

```text
process this recording: /path/to/team-sync.m4a
```

```text
create meeting notes from the existing transcript at /path/to/team-sync-transcript.md
```

For transcription only:

```bash
# AssemblyAI, default backend; optional speaker-count hint
./scripts/transcribe.sh /path/to/audio.m4a \
  --speakers-expected 2 \
  --output /path/to/Team-Sync-Transcript.md

# Explicit model/language and optional one-term-per-line hints
./scripts/transcribe.sh /path/to/audio.m4a \
  --aai-speech-models universal-3-5-pro \
  --language-code en \
  --keyterms-file /path/to/private-keyterms.txt \
  --output /path/to/Team-Sync-Transcript.md

# Gemini is opt-in
./scripts/transcribe.sh /path/to/audio.m4a \
  --backend gemini --model gemini-2.5-flash \
  --output /path/to/Team-Sync-Transcript.md
```

Choose a new transcript output path for each run. Before provider work, the pipeline rejects existing transcript destinations, symlinks/hardlinks to the input, and known archive destination collisions. Transcript files are created exclusively and never replace prior transcripts. Empty or whitespace-only Gemini output fails without writing a transcript.

These CLI commands produce a transcript; the agent follows [SKILL.md](SKILL.md) to create the linked notes. Built-in recording archival is disabled unless `--archive-dir <directory>` is supplied; `--no-archive` remains compatible. The full meeting workflow instead uses `compress-audio.sh --output <confirmed-title>.ogg` after title confirmation. The compressor stages and verifies output before publishing without overwrite; the exact same real OGG input/output path can be verified as a no-op, while other existing destinations, hardlinks, and symlinks are refused.

See [backend guidance](references/BACKENDS.md) for keyterms/context, model/language options, provenance, polling limits, and speaker mapping. The pipeline distinguishes requested model lists and language hints from returned model information and detected language. Speaker clusters are not verified identities, and neither backend's output is automatically complete or accurate.

## Configuration

Set these in the selected trusted config or shell environment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VAULT_PATH` | required for default paths | Notes root |
| `MEETING_NOTES_DIR` | `${VAULT_PATH}/MeetingNotes` | Summaries |
| `MEETING_RAW_DIR` | `${MEETING_NOTES_DIR}/raw` | Transcripts |
| `MEETING_RECORDINGS_DIR` | `${MEETING_NOTES_DIR}/recordings` | Named OGG archives |
| `DAILY_NOTES_DIR` | `${VAULT_PATH}/DailyNotes` | Daily notes |
| `PROJECTS_DIR` | `${VAULT_PATH}/Projects` | Project folders |
| `MEETING_AUDIO_BACKUP_DIR` | `$HOME/audio-backups/meetings` | Original recordings |
| `DAILY_NOTE_PATH_FORMAT` | `%Y/%m-%B/%Y-%m-%d.md` | Relative daily-note path; `%Y-%m-%d.md` for flat notes |
| `PROJECT_MEETING_SUBDIR` | `Meeting` | Single relative subdirectory; empty means project root |
| `LINK_STYLE` | `wikilink` | `wikilink`, `markdown`, or `plain` |
| `MEETING_DOCUMENTER_ENV_FILE` | this clone's `.env` | Exported selector for trusted configuration; empty disables loading |
| `ASSEMBLYAI_API_KEY` | required for AssemblyAI | Default backend credential |
| `GOOGLE_API_KEY` | required for Gemini | Opt-in credential; `GEMINI_API_KEY` also accepted |
| `MEETING_DOCUMENTER_VOICEPRINT` | unset | `1` installs optional voiceprint dependencies |
| `MEETING_DOCUMENTER_LOCAL_SPEAKERS` | unset (off) | Beta: `1` turns on local speaker tracking for AssemblyAI runs, `0` turns it off |
| `MEETING_DOCUMENTER_TIMELINE_DIR` | `~/.local/share/meeting-documenter/timelines` | Private speaker timelines (names and times); never inside notes |
| `VOICEPRINT_DIR` | `~/.local/share/meeting-documenter/voiceprints` | Voiceprint store (biometric) |
| `VOICEPRINT_MODEL_DIR` | `~/.cache/meeting-documenter/models` | Voiceprint model code and weights |
| `VOICEPRINT_FORBIDDEN_ROOTS` | unset | Extra `:`-separated directories the store must never be inside |
| `MEETING_DOCUMENTER_KNOWN_SPEAKERS` | `references/KNOWN_SPEAKERS.yaml` | Private registry path for alias resolution |

The agent resolves directory defaults and conventions during the workflow. `LINK_STYLE` controls agent-authored summary, daily-note, project-reference, and recording links. The Python pipeline initially writes the transcript's `source:` as a bare path; the agent rewrites it after verifying the named archive.

## Who speaks when: local speaker tracking (beta)

AssemblyAI's own diarization gives anonymous clusters (`Speaker A`, `Speaker B`) that can split or merge people. With `--local-speakers`, AssemblyAI transcribes **words only** (`speaker_labels=false`, so it creates no speaker embeddings), and this machine works out who speaks at each moment from the enrolled voiceprints of the meeting's attendees:

```text
**[00:00] Alice Example:** Let's start with the release checklist.
**[00:04] Bob Example:** The build is green since this morning.
**[00:07] Guest 1:** I can take the documentation part.
**[00:09] Alice Example or Bob Example (uncertain):** Sounds good.
```

- **Pipeline.** Nemotron 3 Diarization speaker slots → clean-speech units → ReDimNet2 voice embeddings → calibrated scores against the roster → an HMM per slot over the roster plus "unknown" → words attributed to the resulting timeline. It runs in a child process while AssemblyAI transcribes (about 1 min of compute per hour of audio on an Apple M4 Max).
- **Unknown beats wrong.** Weak or contradictory evidence becomes `A or B (uncertain)`; unenrolled voices become `Guest n`; unmatched speech is `Unknown speaker`. The transcript header lists these turns for a quick spot-check, and the agent asks you when a name conflicts with the content.
- **Safe fallback.** A readiness check runs before anything is uploaded. Without the dependencies, models, enrolled voiceprints, or your own calibration, the run uses AssemblyAI diarization as before and the transcript header says why.
- **Private outputs.** The speaker timeline (names and times) goes to `MEETING_DOCUMENTER_TIMELINE_DIR`, never into your notes; the `.words.json` sidecar next to the transcript holds text and timing only.
- **Default.** Off until you set `MEETING_DOCUMENTER_LOCAL_SPEAKERS=1` or pass `--local-speakers`. Maintainers switch the default with `LOCAL_SPEAKERS_DEFAULT` in `scripts/transcribe_pipeline.py`.

**Public benchmark** (cross-session enrollment: each person is enrolled only from other meetings; identification error rate, IER, at collar 0; lower is better; details in [bench/README.md](bench/README.md)):

| | AMI test (16 meetings, far-field) | AISHELL-4 eval (20 sessions, leave-one-group-out) |
|---|---|---|
| Local tracker (this release) | **20.9 %** | **12.6 %** |
| Diarize, then identify each cluster (B1) | 23.3 % | 20.5 % |
| Sliding windows + argmax (B0) | 37.3 % | 25.3 % |
| Speech ≥ 3 s the tracker names that gets the right name | 97.9 % | 96.4 % |
| Unenrolled speech given an enrolled name (false names) | 3.0 % | 5.8 % |

Most of the remaining AMI error is missed speech at turn edges, not wrong names. Nemotron 3's AMI training data may overlap the AMI test set, so treat the AMI numbers as optimistic; AISHELL-4 is the cleaner evidence.

**Setup.**

```bash
echo 'MEETING_DOCUMENTER_VOICEPRINT=1' >> .env         # opt in; installs torch + sherpa-onnx (large)
./scripts/transcribe.sh --help                         # bootstraps .venv with the optional deps
.venv/bin/python scripts/voiceprint/fetch_models.py    # pinned, sha256-verified models → VOICEPRINT_MODEL_DIR
.venv/bin/python scripts/voiceprint/fetch_models.py nemotron3   # needs a transformers build with Nemotron 3 (> 5.17.0)
```

ReDimNet2 runs on CPU, or on Apple-silicon MPS after a CPU parity self-test; Nemotron 3 Diarization (OpenMDW-1.1) runs on CPU. The optional ERes2NetV2, CAM++, and WeSpeaker ResNet34 ONNX models run through sherpa-onnx, and Silero VAD v5 filters non-speech such as ringtones or hold music. Models are downloaded from their upstream releases, not stored in this repository; check each model's license before use.

**Use.** `scripts/identify-speakers.sh` loads the same trusted config as `transcribe.sh`:

```bash
# Transcribe with local speaker tracking for the confirmed attendees
./scripts/transcribe.sh team-sync.m4a --local-speakers --roster "Alice Example,Bob Example" \
  --output "2026-01-15-1000 Team-Sync-Transcript.md"

# After the names are confirmed, enroll consenting speakers from the timeline (turns ≥ 3 s)
TL=~/.local/share/meeting-documenter/timelines/"2026-01-15-1000 Team-Sync-Transcript.timeline.json"
./scripts/identify-speakers.sh enroll --audio team-sync.m4a --timeline "$TL" --name "Alice Example"
./scripts/identify-speakers.sh enroll --audio team-sync.m4a --timeline "$TL" \
  --speaker GUEST_1 --name "Carol Example"            # a guest you identified

# After a few meetings per person: fit the tracker's calibration on your own data, then check
./scripts/identify-speakers.sh track-calibrate
./scripts/identify-speakers.sh track-check --roster "Alice Example,Bob Example"

./scripts/identify-speakers.sh list
./scripts/identify-speakers.sh forget --name "Alice Example" --yes
```

The first voiceprints come from a diarized run: map AssemblyAI's clusters to names as usual, then `enroll --utterances X.utterances.json --cluster A --name "Alice Example"`. `track-calibrate` needs no audio: it refits the tracker's duration-aware score calibration leave-one-meeting-out on the stored vectors. Until it has run, the readiness check reports `no_calibration` and transcription falls back to diarization.

**Legacy cluster check.** The earlier advisory check (`identify` on a diarized run's `.utterances.json` sidecar, thresholds from `calibrate`) still works and is kept as a fallback and for comparison. It only proposes a name per AssemblyAI cluster and never renames anything. Details: [BACKENDS.md § Legacy voiceprint check](references/BACKENDS.md#legacy-voiceprint-check-cluster-path).

**Listening audit.** To measure accuracy on your own meetings without annotating whole timelines, `audit sample` draws about 300 short windows from speaker timelines (stratified by turn duration, system confidence and channel; short turns, UNCERTAIN regions and switches between a close pair are oversampled), renders each with ±2 s of context and a soft tone before and after the span to judge, and writes an offline answering page. About an hour of listening with the keyboard gives per-stratum accuracy with 95 % confidence intervals:

```bash
./scripts/identify-speakers.sh audit sample --input team-sync.m4a team-sync.timeline.json \
  --channel zoom --out ~/private/speaker-audit        # then open ~/private/speaker-audit/index.html
./scripts/identify-speakers.sh audit score --manifest ~/private/speaker-audit/manifest.json \
  --answers ~/Downloads/answers.json                  # → report.md / report.json next to the manifest
```

The clips, page and reports contain your recordings and names, so the output directory gets the same guard as the voiceprint store (never inside the notes vault or this clone).

**Limits.** The public benchmark above is far-field meeting audio; accuracy on your own meetings is measured with the listening audit. Accuracy drops for turns under about a second, phone or heavily compressed audio, overlapping speech, similar voices, and people with little enrolled speech. The boundary model tracks at most 8 speakers per recording; the `seg3` fallback boundary names far more strangers and is not used unless you ask for it. The words and utterances sidecars and the timeline contain meeting content. Voiceprints from different models are not comparable; changing models means re-enrolling.

## Privacy and consent for voiceprints

Voiceprints are **biometric data**. Speaker embeddings are designed to recognise a person by voice, so treat the store like other sensitive personal data:

- **Consent first.** Enroll only people who agreed to voiceprint enrollment, and record when (for example in the registry's `voiceprint:` metadata). The agent must ask before every enrollment; never enroll someone from a recording they did not know would be used this way.
- **Local only.** The store lives in `VOICEPRINT_DIR` (default `~/.local/share/meeting-documenter/voiceprints`, mode 700; database and calibration mode 600). The CLI refuses a store inside `VAULT_PATH`, `MEETING_NOTES_DIR`, `VOICEPRINT_FORBIDDEN_ROOTS`, any Obsidian vault, or this clone, so vectors are not synced with notes or committed. Nothing is uploaded; AssemblyAI receives only the audio it already transcribes, and with local speaker tracking it is asked for words only, so it creates no speaker embeddings. Speaker timelines stay in `MEETING_DOCUMENTER_TIMELINE_DIR` under the same notes guard.
- **Deletion.** `forget --name ... --yes` deletes a person's vectors from every model and vacuums the database. Delete the whole store directory to remove everything.
- **Law.** Some jurisdictions regulate biometric identifiers specifically, for example the Illinois Biometric Information Privacy Act (BIPA, written notice and consent, retention schedule) and the EU/UK GDPR (Article 9 special-category data when used to identify a person). Workplace recording and consent rules also vary. This is informational, not legal advice; check what applies to you and to the people you record.

## Repository layout

```text
SKILL.md                         Skill entry point
scripts/transcribe.sh            Environment bootstrap and guarded config loading
scripts/transcribe_pipeline.py   Transcription backends and provenance
scripts/compress-audio.sh        Named OGG archive and verification
scripts/split-audio.sh           Silence-aware splitting
scripts/cleanup-source-audio.sh  Original recording backup
scripts/identify-speakers.sh     Optional voiceprint CLI wrapper (guarded config loading)
scripts/identify_speakers.py     Voiceprints: track / track-check / enroll / track-calibrate / list / forget / audit (legacy identify / calibrate)
scripts/voiceprint/              Tracker, boundary layer, timeline, word attribution, embedders, store, calibration, audit, tests
bench/                           Public speaker-tracking benchmark (AMI, AISHELL-4) and baselines
scripts/dev/check_pii.py         PII/private-file guard (CI and pre-commit hook)
.githooks/pre-commit             Local hook running the PII guard on staged changes
references/ONBOARDING_PROMPT.md  Guided setup
references/BACKENDS.md           Backend flags and speaker mapping
references/SUMMARY_FORMAT.md     Summary/task fidelity and link styles
references/QUALITY_CHECKLIST.md  Source, output, and cleanup verification
references/WORKFLOW_DETAILS.md   Existing transcripts and integration edge cases
references/CONTEXT_TEMPLATE.md   Private per-run keyterms/context
references/*.template.yaml      Fictional registry templates
.env.example                    Configuration template
CHANGELOG.md                    Release notes
CONTRIBUTING.md                 Public-first rules, test data policy, hook setup, roadmap
```

## Privacy and verification

Audio and supplied hints are sent to the selected transcription provider. Generated notes can contain private meeting content, speaker names, source paths, and provider identifiers. Keep them outside public repositories. Removing identifiers from a summary does not make an original recording public.

`.gitignore` reduces accidental staging; it cannot prevent force-added files or remove sensitive data already committed. Before publishing a fork or PR, inspect the actual tracked files and staged diff for credentials, private registries, recordings, transcripts, local paths, and identifiers. Rotate any exposed credential.

Tests should use generated/public audio and fictional metadata with mocked provider responses. Do not upload private meeting recordings as a smoke test. A live API test needs authorized test material and incurs provider charges; report it separately from offline test results. With the local environment provisioned, run offline regression checks with `.venv/bin/python3 -B -m unittest discover -s tests -v`, and the voiceprint tests with `.venv/bin/python3 -m pytest scripts/voiceprint/tests` (model-dependent tests skip unless the optional dependencies and models are installed). `scripts/dev/check_pii.py` scans tracked files for local paths, e-mail addresses, private files, and, when your private registry exists, registry names; see [CONTRIBUTING.md](CONTRIBUTING.md). Follow the [quality checklist](references/QUALITY_CHECKLIST.md) before accepting actual meeting notes.

## Customization and cost

Use environment variables for note layout and link form, CLI flags for supported backend/model choices, and [SUMMARY_FORMAT.md](references/SUMMARY_FORMAT.md) for output structure. The agent can omit steps the user explicitly excludes.

Transcription and agent usage are billed under your provider/account terms. This repository does not promise a per-meeting price or processing time; check current provider pricing for your chosen model.

## License

MIT — see [LICENSE](LICENSE).
