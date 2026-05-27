# meeting-documenter

A [Claude Code](https://claude.com/claude-code) skill that turns a meeting recording into structured, linked notes in an Obsidian vault — transcript, summary, action items, daily-note row, and per-project reference note — in one pipeline.

## What it does

Given an audio file (MP3 / WAV / OGG / M4A / FLAC / WebM / AAC), the skill:

1. **Prepares audio** — trims trailing silence, fixes codec/container mismatches automatically
2. **Transcribes** with [Gemini](https://ai.google.dev/) (speaker diarization, silence-aware chunking for long meetings, auto-retry on truncation)
3. **Resolves speaker names** against a canonical registry to stop misheard variants from propagating
4. **Detects the project** by keyword matching, then asks the user to confirm
5. **Generates a structured summary** — frontmatter, executive summary, decisions, action items (with assignees), parking lot, topics, follow-ups
6. **Updates the daily note** — adds a row to the `## Meetings` table, cross-references carryover tasks
7. **Links to the project** — creates a per-project reference note + bumps the project Dashboard
8. **Archives** the OGG-compressed recording in the vault + backs up the original lossless source

## Requirements

- [Claude Code](https://claude.com/claude-code) CLI
- An Obsidian vault (default paths use neutral folder names — override via env vars to match any layout)
- `ffmpeg` and `ffprobe` on `$PATH`
- [`uv`](https://github.com/astral-sh/uv) (Python package manager — bootstraps the venv automatically)
- A [Google AI Studio](https://aistudio.google.com/apikey) API key for Gemini

## Install

```bash
# 1. Clone the repo somewhere accessible to Claude Code
git clone https://github.com/xiaosongz/meeting-documenter
cd meeting-documenter
```

### Guided setup (recommended)

The skill ships with an **onboarding prompt** designed to be read by your own Claude Code agent. It inspects your note-taking system, then either adapts the skill to your existing layout or scaffolds a minimum structure if you don't have one yet.

Open Claude Code in this repo (or wherever your notes live) and run:

> Read `references/ONBOARDING_PROMPT.md` in the meeting-documenter repo and follow it to set this skill up for my system. Start in diagnostic mode.

The agent will walk through three modes:

- **Diagnostic** — read-only inspection; reports what's ready, what's missing, what doesn't match
- **Adapt** — fits the skill to your existing notes layout by generating `.env` + populating registries
- **Bootstrap** — scaffolds a minimum folder + template structure if you're starting from scratch

### Manual setup (alternative)

```bash
# 1. Configure the API key + vault path
cp .env.example .env
$EDITOR .env   # set GOOGLE_API_KEY and VAULT_PATH

# 2. Copy the registry templates and fill in your team's data.
#    The .yaml versions are gitignored — only the .template.yaml versions ship publicly.
cp references/KNOWN_SPEAKERS.template.yaml references/KNOWN_SPEAKERS.yaml
cp references/PROJECT_KEYWORDS.template.yaml references/PROJECT_KEYWORDS.yaml
$EDITOR references/KNOWN_SPEAKERS.yaml
$EDITOR references/PROJECT_KEYWORDS.yaml
```

> **First-run side effect:** the first invocation of `scripts/transcribe.sh` (including `--help`) creates a `.venv/` inside this repo via `uv` and installs `google-genai`. The directory is gitignored. Skip the first run if the repo is on a read-only mount.

## Use

In Claude Code, just describe what you want:

```
process this recording: /path/to/Recording 20260119143000.m4a
```

```
document this meeting: /path/to/team-sync.mp3
```

The skill walks through the pipeline and asks for confirmation at decision points (title, speakers, project).

## Configuration

Override defaults via environment variables (e.g., in your shell config or a project-level `.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `VAULT_PATH` | (required) | Vault root |
| `MEETING_NOTES_DIR` | `${VAULT_PATH}/MeetingNotes` | Summaries |
| `MEETING_RAW_DIR` | `${MEETING_NOTES_DIR}/raw` | Transcripts |
| `MEETING_RECORDINGS_DIR` | `${MEETING_NOTES_DIR}/recordings` | OGG archives |
| `DAILY_NOTES_DIR` | `${VAULT_PATH}/DailyNotes` | Daily notes (year/month subdirs) |
| `PROJECTS_DIR` | `${VAULT_PATH}/Projects` | Active projects with `Dashboard.md` |
| `MEETING_AUDIO_BACKUP_DIR` | `$HOME/audio-backups/meetings` | Original recordings (post-archive) |
| `GOOGLE_API_KEY` | (required) | Gemini API key. `GEMINI_API_KEY` also accepted. |

Defaults use neutral folder names. If your vault uses PARA, Johnny.Decimal, or any other layout, point each path env var at the matching directory.

## Repository layout

```
.
├── SKILL.md                          # Skill entry point (Claude reads this)
├── scripts/
│   ├── transcribe.sh                 # Bootstrap venv + load .env + run pipeline
│   ├── transcribe_pipeline.py        # Full transcription pipeline
│   ├── compress-audio.sh             # Any-format → OGG with 3-point verify
│   ├── split-audio.sh                # Silence-aware splitter for long audio
│   └── cleanup-source-audio.sh       # Verify archive, backup original, clean vault root
├── references/
│   ├── ONBOARDING_PROMPT.md               # Prompt your Claude Code agent reads to adapt the skill to your system
│   ├── SUMMARY_FORMAT.md                  # Output spec for summaries
│   ├── QUALITY_CHECKLIST.md               # Verification checklist (Step 7)
│   ├── WORKFLOW_DETAILS.md                # Edge cases + fast-paths
│   ├── CONTEXT_TEMPLATE.md                # Speaker diarization context template
│   ├── KNOWN_SPEAKERS.template.yaml       # Template — copy to KNOWN_SPEAKERS.yaml (gitignored) and edit
│   └── PROJECT_KEYWORDS.template.yaml     # Template — copy to PROJECT_KEYWORDS.yaml (gitignored) and edit
├── .env.example
├── .gitignore
└── LICENSE
```

## Cost

Roughly Gemini's published rate for `gemini-3-flash-preview`. Empirically:

| Audio length | Approx. cost |
|-------------:|-------------:|
| 60 min, 1 chunk | ~$0.08 |
| 120 min, 3 chunks w/ retry | ~$0.25 |

Claude summarization runs inside your existing Claude Code subscription.

## Security

- **`.env`** — already in `.gitignore`. If you accidentally commit it, rotate the key.
- **`references/KNOWN_SPEAKERS.yaml` and `references/PROJECT_KEYWORDS.yaml`** contain real names, emails, and project codenames once you fill them in. Both are already in this repo's `.gitignore` so they cannot be pushed to a fork. Only the `.template.yaml` versions are tracked. If you migrate to a different setup, audit any backup of these files before sharing.

## Customization

The skill is designed to be edited:

- **Different vault layout**: override the path env vars (see Configuration)
- **Different transcription model**: edit `scripts/transcribe_pipeline.py` (look for `gemini-3-flash-preview`)
- **Different summary format**: edit `references/SUMMARY_FORMAT.md` — the skill reads this at Step 4
- **Skip steps**: the pipeline is documented in `SKILL.md` step-by-step; you can ask Claude to skip any step (e.g., "process this recording but skip Step 6")

## License

MIT — see [LICENSE](LICENSE).
