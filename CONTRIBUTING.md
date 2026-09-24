# Contributing

Thanks for helping. This repository is **public-first**: it is the upstream for everyone's installation, including the maintainer's. Changes must be generic and must not depend on any one person's meetings, colleagues, or machine.

## Public-first rules

- Develop here, not in a private copy that is later synced. Personal customisation (names, projects, paths, note layout) lives only in gitignored local files: `.env`, `references/KNOWN_SPEAKERS.yaml`, `references/PROJECT_KEYWORDS.yaml`, and the voiceprint store in `VOICEPRINT_DIR`.
- New paths are configuration: add an environment variable with a portable default (XDG-style for data and caches), document it in `.env.example`, `README.md`, and `SKILL.md`, and never hardcode an absolute path.
- Describe failure modes generically ("an inverted two-speaker mapping", "a ringtone attributed to a speaker"). Do not reference specific meetings, dates, incidents, people, or project codenames.
- Accuracy claims must be reproducible by others or stated generically ("validated on the maintainer's private meetings; public benchmark pending"). Aggregate, method-level numbers such as an EER are acceptable only without names, meetings, or anything that could identify a person.

## Test data policy

- Use **synthetic or openly licensed** data only: generated tones, synthetic embeddings, text-to-speech from the operating system, or public corpora whose licence allows redistribution.
- Never commit real recordings, transcripts, utterances sidecars, voiceprint stores (`*.sqlite`), `calibration.json`, embeddings, model weights, or registries with real people.
- Use placeholder people (`Alice Example`, `Bob Example`, `Speaker A`), meetings (`2026-01-01-1000 Example Meeting`), domains (`example.com`, `.invalid`), and paths (`~/...`, `/path/to/...`).
- Mock provider calls. Live API tests need authorised test material, cost money, and are reported separately from offline results.
- Tests that need optional heavy dependencies (torch, sherpa-onnx, model files) must skip cleanly when they are missing; CI installs none of them and downloads no models.

Run everything locally before opening a pull request:

```bash
.venv/bin/python3 -B -m unittest discover -s tests -v
.venv/bin/python3 -m pytest scripts/voiceprint/tests -rs
.venv/bin/python3 scripts/dev/check_pii.py --tracked
.venv/bin/python3 scripts/dev/check_pii.py --log origin/main..HEAD
```

## PII guard and pre-commit hook

`scripts/dev/check_pii.py` fails on absolute home or mounted-volume paths, e-mail addresses outside reserved example domains, and forbidden files (voiceprint databases, `calibration.json`, `*.utterances.json`, embeddings, model weights, audio, `.env`, private registries). CI runs it on every push and pull request.

Locally, when your private `references/KNOWN_SPEAKERS.yaml` exists (or `--roster PATH` / `MEETING_DOCUMENTER_KNOWN_SPEAKERS` points to one), it also blocks every canonical name, alias, and e-mail from that registry. Findings are printed masked. Enable the hook once per clone:

```bash
git config core.hooksPath .githooks
```

The hook scans staged additions and file names. It cannot remove data that was already committed or pushed: if something slips through, rewrite the unpublished history, or rotate/redact and treat a published leak as disclosed. `.gitignore` reduces accidents but does not stop `git add -f`.

## Pull requests

- Keep changes focused; update `CHANGELOG.md` under an "Unreleased" heading.
- Explain any change to identity handling, thresholds, or privacy guards, and add a regression test.
- Voiceprint output stays advisory: AssemblyAI remains the diarizer, below-threshold or unenrolled speakers stay `Speaker X`, voice/content disagreement goes to the user, and the no-dependency path exits 0 without importing torch.

## Roadmap

Contributions are welcome on:

1. **Public benchmark.** Reproducible identification and calibration results on an open multi-speaker corpus such as the AMI Meeting Corpus or VoxConverse: enrollment from some meetings, leave-one-meeting-out evaluation, open-set false-accept rate, wrong-name rate, and nested-calibration error, with scripts that download the corpus rather than committing it.
2. **Channel robustness.** Phone (8 kHz, codec), video-call, and far-field room audio: per-channel sub-centroids, channel-aware calibration, and tests on public or synthetic channel simulations.
3. **Multilingual and code-switching speech.** Whether embeddings and thresholds hold when the same person speaks different languages across or within meetings.
4. **Alternative embedders.** Additional pinned, licence-compatible models behind the existing `Embedder` interface, compared on the public benchmark; score fusion if it helps.
5. **Calibration quality.** Confidence intervals for thresholds on small stores, and guidance on how much enrolled speech is enough.
