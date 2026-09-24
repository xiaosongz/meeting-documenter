# Changelog

## 0.8.0-beta.1

Public beta of optional, advisory voiceprint speaker identification.

### Added

- **Utterances sidecar.** AssemblyAI runs write `<transcript stem>.utterances.json` (schema 1: exact per-cluster millisecond timestamps and text, `speaker_map: null`) next to the transcript. It is created exclusively; an existing sidecar stops the run before upload, and a write failure after transcription is reported without failing the transcript.
- **Voiceprint CLI** (`scripts/identify_speakers.py`, wrapper `scripts/identify-speakers.sh`) with `identify`, `enroll`, `cohort-add`, `calibrate`, `list`, and `forget`:
  - `identify` scores AssemblyAI clusters against locally enrolled voiceprints (ReDimNet2 by default; sherpa-onnx ERes2NetV2, CAM++, WeSpeaker ResNet34), with Silero VAD gating, MAD-pruned duration-weighted cluster embeddings, AS-norm, Hungarian assignment with an Unknown option, two-speaker swap margins, roster elimination guards, and a label check against the sidecar `speaker_map`. It is advisory and never renames anything.
  - `enroll` refuses unregistered names, sidecar conflicts, clips that sound like someone already enrolled, and clip configurations that differ from the calibration.
  - `calibrate` derives τ, τ_low, δ, and gap from the user's own enrolled data (leave-one-meeting-out, leave-one-speaker-out and cohort voices as stranger trials), reports a nested held-out estimate, and records the clip configuration in `calibration.json`.
- `scripts/voiceprint/fetch_models.py` downloads pinned, sha256-verified models into `VOICEPRINT_MODEL_DIR`.
- New configuration: `MEETING_DOCUMENTER_VOICEPRINT`, `VOICEPRINT_DIR`, `VOICEPRINT_MODEL_DIR`, `VOICEPRINT_FORBIDDEN_ROOTS`, `VOICEPRINT_THREADS`, `MEETING_DOCUMENTER_KNOWN_SPEAKERS`.
- SKILL.md Step 2.5a (voiceprint check), a Step 7 reconciliation check, and Step 8c (consented enrollment); registry `voiceprint:` metadata field (metadata only).
- `scripts/dev/check_pii.py`, `.githooks/pre-commit`, and a CI scan for local paths, e-mail addresses, forbidden files, and (locally) private registry names.
- `CONTRIBUTING.md` with public-first rules, test data policy, hook setup, and roadmap; README sections on voiceprints and biometric privacy/consent.

### Changed

- CI also runs on `feat/**` pushes, runs the voiceprint tests with light dependencies (model tests skip), and scans for PII.
- `transcribe.sh` installs the optional voiceprint dependencies only when `MEETING_DOCUMENTER_VOICEPRINT=1`; failures there never block transcription.

### Safety defaults

- Without `calibration.json`, voiceprint proposals are capped at `low` confidence and roster elimination is disabled.
- The store is refused inside `VAULT_PATH`, `MEETING_NOTES_DIR`, `VOICEPRINT_FORBIDDEN_ROOTS`, any Obsidian vault, or this clone; it is created with mode 700 and its files with mode 600.
- The identify path exits 0 without importing torch when dependencies, models, or voiceprints are missing.

## 0.7.0

AssemblyAI transcription by default, explicit model/language controls, provenance, and source-fidelity safeguards.
