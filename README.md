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
- An Obsidian vault (default paths assume a PARA-style layout, but everything is configurable via env vars)
- `ffmpeg` and `ffprobe` on `$PATH`
- [`uv`](https://github.com/astral-sh/uv) (Python package manager — bootstraps the venv automatically)
- A [Google AI Studio](https://aistudio.google.com/apikey) API key for Gemini

## Install

```bash
# 1. Clone into your vault's skills directory
cd /path/to/your/vault
git clone https://github.com/<you>/meeting-documenter .claude/skills/meeting-documenter

# 2. Configure the API key
cp .claude/skills/meeting-documenter/.env.example .claude/skills/meeting-documenter/.env
$EDITOR .claude/skills/meeting-documenter/.env   # paste GOOGLE_API_KEY

# 3. Make sure .env is gitignored in your vault repo
echo '.env' >> .gitignore

# 4. Fill in your team's data
$EDITOR .claude/skills/meeting-documenter/references/KNOWN_SPEAKERS.yaml
$EDITOR .claude/skills/meeting-documenter/references/PROJECT_KEYWORDS.yaml
```

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
| `MEETING_NOTES_DIR` | `${VAULT_PATH}/50_RESOURCES/Professional/MeetingNotes` | Summaries |
| `MEETING_RAW_DIR` | `${MEETING_NOTES_DIR}/raw` | Transcripts |
| `MEETING_RECORDINGS_DIR` | `${MEETING_NOTES_DIR}/recordings` | OGG archives |
| `DAILY_NOTES_DIR` | `${VAULT_PATH}/20_DAILY` | Daily notes (year/month subdirs) |
| `PROJECTS_DIR` | `${VAULT_PATH}/30_PROJECTS/Active` | Active projects with `Dashboard.md` |
| `MEETING_AUDIO_BACKUP_DIR` | `$HOME/audio-backups/meetings` | Original recordings (post-archive) |
| `GOOGLE_API_KEY` | (required) | Gemini API key |

If your vault doesn't use the PARA layout above, the skill still works — point each path env var wherever you actually keep that content.

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
│   ├── SUMMARY_FORMAT.md             # Output spec for summaries
│   ├── QUALITY_CHECKLIST.md          # Verification checklist (Step 7)
│   ├── WORKFLOW_DETAILS.md           # Edge cases + fast-paths
│   ├── CONTEXT_TEMPLATE.md           # Speaker diarization context template
│   ├── KNOWN_SPEAKERS.yaml           # ← Edit me: your team's speaker registry
│   └── PROJECT_KEYWORDS.yaml         # ← Edit me: your project keyword map
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

- **Never commit `.env`** — it's already in `.gitignore`. If you accidentally do, rotate the key.
- **`references/KNOWN_SPEAKERS.yaml` contains real names and emails for your team.** Do NOT commit it to a public fork of this repo. Consider adding it to your local `.gitignore` and shipping only the template.

## Customization

The skill is designed to be edited:

- **Different vault layout**: override the path env vars (see Configuration)
- **Different transcription model**: edit `scripts/transcribe_pipeline.py` (look for `gemini-3-flash-preview`)
- **Different summary format**: edit `references/SUMMARY_FORMAT.md` — the skill reads this at Step 4
- **Skip steps**: the pipeline is documented in `SKILL.md` step-by-step; you can ask Claude to skip any step (e.g., "process this recording but skip Step 6")

## License

MIT — see [LICENSE](LICENSE).
