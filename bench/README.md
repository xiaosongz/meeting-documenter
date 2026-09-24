# Speaker-tracking benchmark (`bench/`)

A public, reproducible harness for **known-speaker tracking**: given enrolled voiceprints,
label who speaks when in a mono recording. It measures identification with no label
permutation, separates wrong names from abstentions ("unknown is acceptable; a wrong name
is not"), and breaks errors down by reference-turn duration and overlap.

Data never enters this repository. Datasets, caches and runs live under
`$SPEAKER_BENCH_DIR` (default `${XDG_DATA_HOME:-~/.local/share}/meeting-documenter/bench-data`).

```bash
uv pip install -r bench/requirements.txt              # scoring only
uv pip install -r scripts/voiceprint/requirements.txt  # + baseline B0
python scripts/voiceprint/fetch_models.py              # ReDimNet2 + Silero (B0)

python -m bench.datasets.ami --splits test dev        # 4.3 GB (test+dev, SDM + headset mix)
python -m bench.datasets.aishell4                     # 5.2 GB archive, ~12 GB on disk
python -m bench.run --dataset ami --split test --sanity reference --out runs/sanity-ref
python -m bench.run --dataset ami --split test --system <rttm-dir> --out runs/mysys
python -m bench.baselines.window_id --dataset ami --split test --tune-split dev --out runs/b0-ami
```

## Datasets and licences

| Dataset | What is used | Licence |
|---|---|---|
| [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | `Array1-01` (single distant mic, SDM) as test input; `Mix-Headset` (close-talk mix) as enrollment source; official mirror | CC BY 4.0 |
| [BUT AMI diarization setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup) | Full-corpus-ASR partition, `only_words` RTTMs and UEMs, pinned commit `2509d8933721` | Apache-2.0 |
| [AISHELL-4](https://www.openslr.org/111/) (OpenSLR 111) | `test.tar.gz` (20 sessions); array channel 0 as mono 16 kHz; the archive's RTTMs | CC BY-SA 4.0 |

Downloads are resumable (`*.part` + HTTP Range), verified against Content-Length and the
sha256 values pinned in `bench/datasets/checksums/`, and recorded in
`$SPEAKER_BENCH_DIR/manifest-<dataset>.json` together with speaker lists and disk usage.

## Protocol: cross-session enrollment (`bench/protocol.py`)

For a test meeting *M* in series *S*, each participant of *S* is enrolled only from the
*other* meetings of *S*, using their reference-labelled non-overlapped regions from the
enrollment stream. *M* itself is never used for enrollment.

- **AMI.** Participant IDs (e.g. `FEE013`) are corpus-global. The test split is four full
  series (ES2004, IS1009, TS3003, EN2002; 16 meetings), each with the same four
  participants in all its meetings. One participant is absent from EN2002c and stays on
  that meeting's roster, which tests false names for absent people. Enrollment uses the
  headset mix and testing uses the far-field SDM, so the channel is mismatched. The dev
  split's unlettered meetings (IB40xx) form a series with the other unlettered meetings
  that have the same participants (IB4001+IB4002, IB4003+IB4004, IB4010+IB4011).
- **AISHELL-4.** Speaker labels (`001-M` … `007-M`) are **local to a session group**, not
  global. Sessions are `<room>_R<group>S<session>C<n>`. The four sessions of a group
  (e.g. `L_R003S01…S04`) share one label set, and a label in another group is a different
  person (the gender suffix of `002` differs between groups). The protocol therefore
  treats a group as a series and enrolls from the group's other three sessions (same
  array channel). `python -m bench.datasets.verify_ids` checks acoustically that each
  label is one voice across its group: 342/342 session-pair matches on the eval set
  (same-label cosine p05 0.85, different-label p95 0.56). A few people do recur across
  groups under different labels (cross-group centroid cosine 0.86–0.91), which matters
  for the cohort below.
- **Rosters.** `closed` holds every series participant with enrollment material.
  `loo:<spk>` is the closed roster minus one present speaker; that speaker's speech is
  unenrolled and measures the open-set false-name rate.
- **Cohort** (AS-norm): speakers of the other series in the same split, never anyone
  from *S*. B0 also drops any cohort speaker whose centroid has cosine ≥ 0.8 to an
  enrolled model, because group-local labels cannot exclude a recurring person by ID.

## Metrics (`bench/metrics.py`)

System labels are enrolled IDs or abstentions: `GUEST_*`, `UNKNOWN`, `UNCERTAIN(A|B)`.

| Metric | Definition |
|---|---|
| **IER** (primary) | pyannote.metrics identification error rate, no permutation. Open-set mapping first: unenrolled reference speakers → `UNKNOWN`, `GUEST_*` → `UNKNOWN`. `UNCERTAIN(...)` always counts as an error. |
| DER, JER | pyannote.metrics, optimal mapping (JER at collar 0 only). |
| Outcome breakdown | Per (10 ms frame, reference speaker): `correct` / `wrong_name` (hypothesis names an enrolled ID that is not speaking) / `abstain` / `missed`. Reported by reference-turn duration (<1, 1–3, 3–10, ≥10 s; a turn merges one speaker's segments across gaps ≤ 0.5 s), overlap vs non-overlap, and wrong-name seconds per (reference, hypothesis) pair. |
| False-name rate | Share of unenrolled speech (leave-one-out rosters) labelled with an enrolled name. |
| Abstain rate | Share of enrolled speech labelled with an abstention. |
| Named false alarm | Seconds of reference non-speech labelled with a name. |

**Collars** are NIST-style ±seconds around every reference boundary (the md-eval / dscore
"0.25 s"). pyannote's `collar` argument is the total width, so the harness passes
2 × collar. Reports use collars 0 and 0.25. Sanity checks: reference vs reference gives
IER 0, and a label derangement gives IER > 0 (`--sanity reference|shuffled`).

## Baseline B0 (`bench/baselines/window_id.py`)

Silero VAD → 1.5 s windows at a 0.5 s hop (≥ 50 % speech) → ReDimNet2 → cosine against
cross-session enrollment centroids → AS-norm (top-100 cohort) → Viterbi over the roster
plus an `UNKNOWN` state with constant score θ (the absolute threshold) and switch penalty
λ. λ = 0 is plain per-window argmax, reported as `b0-argmax`. θ and λ are tuned on AMI dev
to minimise the mean of closed and leave-one-out IER, then frozen for the test sets.
On MPS, clips are embedded in length-sorted batches of at most 48 s of audio, and the MPS
cache is released after each batch. Without this, the caching allocator reached an 83 GB
footprint while RSS showed 3–8 GB.

<!-- RESULTS -->
### Results (2026-09-24; ReDimNet2, θ = 0.5, λ = 1 tuned on AMI dev)

Closed roster, collar 0 (collar 0.25 in brackets). False-name rate is over
leave-one-out rosters.

| Set | IER % | DER % | abstain % | false-name % | IER % `b0-argmax` |
|---|---|---|---|---|---|
| AMI test (16, SDM; headset enrollment) | 37.26 (30.50) | 37.26 | 0.96 | 22.61 | 37.44 |
| AISHELL-4 eval (20, ch 0) | 25.33 (18.58) | 24.77 | 1.03 | 34.89 | 25.67 |

Enrolled speech by reference-turn duration, collar 0 (correct / wrong name / missed, %):

| Bucket | AMI test | AISHELL-4 eval |
|---|---|---|
| < 1 s | 8.0 / 5.8 / 83.7 | 7.6 / 13.0 / 77.3 |
| 1–3 s | 38.9 / 3.6 / 55.4 | 47.2 / 7.7 / 43.2 |
| 3–10 s | 60.9 / 2.1 / 35.6 | 70.7 / 2.8 / 25.1 |
| ≥ 10 s | 74.7 / 0.8 / 24.1 | 80.4 / 0.7 / 18.2 |
| overlap | 37.1 / 2.7 / 57.9 | 41.7 / 4.4 / 51.8 |

Missed speech dominates B0's errors. On AMI SDM, Silero VAD recalls only 77.5 % of the
reference speech time (at 98 % precision), and a single label per frame cannot cover
overlap. Once speech is labelled, 96 % of enrolled speech gets the right name. The
open-set false-name rate (22–35 %) is the other weakness: θ is not calibrated per
dataset.
