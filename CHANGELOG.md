# Changelog

## Unreleased

### Added

- **Speaker timeline and word attribution** (`scripts/voiceprint/timeline.py`, `scripts/voiceprint/words.py`, standard library only): turns plus an exclusive one-speaker-at-a-time view with RTTM/JSON I/O, and max-intersection attribution of ASR words to that view (straddle, gap, UNCERTAIN and overlap flags; turns rebuilt at every speaker change). Groundwork for local known-speaker tracking.
- **Known-speaker tracking** (`scripts/voiceprint/track.py`, CLI `identify_speakers.py track` and `track-calibrate`): who speaks when, from enrolled voiceprints, locally. Boundary tracks (Nemotron 3 global slots, or seg-3.0 chunk speakers linked across chunks) → units of clean speech → ReDimNet2 → owner-aware AS-norm → duration-aware calibrated LLR → per-track HMM over roster ∪ UNKNOWN → exclusive assignment, track-level identities, `UNCERTAIN(A|B)`, `GUEST_n` → a `Timeline`. On the public benchmark (cross-session enrollment) IER is 20.9 % on AMI test and 12.6 % on AISHELL-4 eval, vs 37.3 % / 25.3 % for the window baseline B0 and 23.3 % / 20.5 % for diarize-then-identify B1.
- **Benchmark:** tracker driver (`bench/systems/tracker.py`: prepare, calibrate, tune, run, leave-one-group-out), baseline B1 (`bench/baselines/ahc_id.py`), and AMI enrollment QC (`bench/datasets/enroll_qc.py`, `--enroll-qc`), which explains B0's dev-vs-test gap (permuted IB4001/IB4002 labels, missing headset channels).
- `scripts/voiceprint/footprint.py` (physical footprint incl. MPS buffers, and a guard) and `tools/perf_embed.py`.
- **Local speaker tracking in the transcription pipeline** (`transcribe.sh --local-speakers`, or `MEETING_DOCUMENTER_LOCAL_SPEAKERS=1`; off by default): AssemblyAI returns words only (`speaker_labels=false`, so no cloud speaker embeddings) plus a `<transcript stem>.words.json` sidecar, `identify_speakers.py track` runs in a child process during transcription, and each word takes its timeline speaker. The transcript shows names, `Guest n`, `A or B (uncertain)` and `Unknown speaker`, with a Speaker review list of timestamps to spot-check. The roster comes from `--roster` or the context file's `Possible speakers:` list. The private timeline is written to `MEETING_DOCUMENTER_TIMELINE_DIR` (refused inside notes). New flags: `--roster`, `--channel`, `--track-backend`, `--track-timeout`, `--no-local-speakers`.
- **Fallback to AssemblyAI diarization.** `identify_speakers.py track-check` (no models, no audio) runs before upload. When the voiceprint dependencies, the nemotron3 boundary model, enrolled voiceprints, the store's track calibration, or an enrolled roster member is missing, the run diarizes in the cloud as before and the transcript header carries a `[!warning]` with the reason. A tracker failure after transcription gives an unlabelled transcript with a warning; the words sidecar is kept.
- **`enroll --timeline`**: enroll a confirmed speaker from a Timeline's turns of at least 3 s (`--speaker GUEST_1` for an identified guest; `--spans` for user-confirmed spans, required for `UNCERTAIN` labels), with the same label guard, self-check and vault guard as cluster enrollment.
- `embedders.model_id_for(name)`: the store key of a pinned model without loading it.

### Changed

- Docs: SKILL.md Step 2.5 is now local speaker tracking (2.5a), review of tracked names (2.5b) and cluster mapping (2.5c); Step 8c enrolls from the timeline. The cluster-level `identify` check is documented as legacy/fallback (code unchanged).

### Fixed

- **MPS memory in the shared embedder.** ReDimNet2 on MPS kept per-shape buffers (74–85 GB physical footprint while RSS showed 3–8 GB). Equal-length batches are capped at 48 s of audio, the MPS cache is released every 48 s of audio, and the MPS watermarks default to 0.15/0.1: 30 min of 279 distinct clip lengths now peaks at 4.3 GB.

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
