# meeting-documenter

A [Claude Code](https://claude.com/claude-code) skill that turns a meeting recording or existing transcript into structured, linked notes: a summary, supported action items, a daily-note row, and project references. Release **0.7.0** adds AssemblyAI transcription, explicit model/language controls, provenance, and stronger source-fidelity checks while preserving configurable note layouts.

## What it does

1. Resolves metadata and speaker names using confirmed context and your private registries.
2. Transcribes audio with **AssemblyAI by default** (`universal-3-5-pro`), or **Gemini on request** (`gemini-2.5-flash`). Gemini includes audio preparation, silence-aware chunking, and truncation retries.
3. Reviews anonymous speaker clusters or generated names, preserving unresolved identities.
4. Creates a source-linked summary with actual decisions and explicit assignments still open at meeting end; preserves stated deadline wording.
5. Updates the meeting date's daily note and confirmed project references using your link style and folder conventions.
6. Creates a named OGG archive after title confirmation, and can preserve the original in your configured backup directory.

An existing transcript skips transcription and audio handling while keeping metadata, name resolution, summary, and integration checks. No transcription API key is needed for that path.

## Requirements

- Claude Code and a filesystem-based Markdown notes directory, such as an Obsidian vault.
- For audio: `ffmpeg`, `ffprobe`, and [`uv`](https://github.com/astral-sh/uv) on `$PATH`.
- For the default backend: an `ASSEMBLYAI_API_KEY`.
- For opt-in Gemini: `GOOGLE_API_KEY` or `GEMINI_API_KEY`.

The wrapper creates a local Python environment and installs `requests` and `google-genai` when missing. Audio helper scripts may also require standard command-line tools such as `bc`.

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

**First-run side effect:** even `scripts/transcribe.sh --help` can create `.venv/` and install missing dependencies before reading configuration. It is not a read-only diagnostic or proof of a successful provider transcription. An external `.env` does not make a read-only clone writable; provision its environment before making an installation read-only.

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

These CLI commands produce a transcript; the agent follows [SKILL.md](SKILL.md) to create the linked notes. Built-in recording archival is disabled unless `--archive-dir <directory>` is supplied; `--no-archive` remains compatible. The full meeting workflow instead uses `compress-audio.sh --output <confirmed-title>.ogg` after title confirmation.

See [backend guidance](references/BACKENDS.md) for keyterms/context, model/language options, provenance, polling limits, and speaker mapping. The pipeline distinguishes requested model lists from returned model information. Speaker clusters are not verified identities, and neither backend's output is automatically complete or accurate.

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

The agent resolves directory defaults and conventions during the workflow. `LINK_STYLE` controls agent-authored summary, daily-note, project-reference, and recording links. The Python pipeline initially writes the transcript's `source:` as a bare path; the agent rewrites it after verifying the named archive.

## Repository layout

```text
SKILL.md                         Skill entry point
scripts/transcribe.sh            Environment bootstrap and guarded config loading
scripts/transcribe_pipeline.py   Transcription backends and provenance
scripts/compress-audio.sh        Named OGG archive and verification
scripts/split-audio.sh           Silence-aware splitting
scripts/cleanup-source-audio.sh  Original recording backup
references/ONBOARDING_PROMPT.md  Guided setup
references/BACKENDS.md           Backend flags and speaker mapping
references/SUMMARY_FORMAT.md     Summary/task fidelity and link styles
references/QUALITY_CHECKLIST.md  Source, output, and cleanup verification
references/WORKFLOW_DETAILS.md   Existing transcripts and integration edge cases
references/CONTEXT_TEMPLATE.md   Private per-run keyterms/context
references/*.template.yaml      Fictional registry templates
.env.example                    Configuration template
```

## Privacy and verification

Audio and supplied hints are sent to the selected transcription provider. Generated notes can contain private meeting content, speaker names, source paths, and provider identifiers. Keep them outside public repositories. Removing identifiers from a summary does not make an original recording public.

`.gitignore` reduces accidental staging; it cannot prevent force-added files or remove sensitive data already committed. Before publishing a fork or PR, inspect the actual tracked files and staged diff for credentials, private registries, recordings, transcripts, local paths, and identifiers. Rotate any exposed credential.

Tests should use generated/public audio and fictional metadata with mocked provider responses. Do not upload private meeting recordings as a smoke test. A live API test needs authorized test material and incurs provider charges; report it separately from offline test results. With the local environment provisioned, run offline regression checks with `.venv/bin/python3 -B -m unittest discover -s tests -v`. Follow the [quality checklist](references/QUALITY_CHECKLIST.md) before accepting actual meeting notes.

## Customization and cost

Use environment variables for note layout and link form, CLI flags for supported backend/model choices, and [SUMMARY_FORMAT.md](references/SUMMARY_FORMAT.md) for output structure. The agent can omit steps the user explicitly excludes.

Transcription and agent usage are billed under your provider/account terms. This repository does not promise a per-meeting price or processing time; check current provider pricing for your chosen model.

## License

MIT — see [LICENSE](LICENSE).
