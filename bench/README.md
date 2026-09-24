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

## Enrollment QC for AMI (`bench/datasets/enroll_qc.py`, opt-in `--enroll-qc`)

B0's AMI dev IER (55 %) was far above test (37 %). The cause is data, not the method:

- **IB4001 / IB4002 labels are permuted between the two meetings.** The far-field centroid
  of each label is closest (cosine 0.92–0.98) to a *different* label in the partner
  meeting, so cross-session enrollment names the wrong person (70–74 % wrong names).
- **Some headset mixes lack a speaker** (dead or missing headset channel): IB4010/IB4011,
  IS1008b/c, TS3004b/c, ES2011c, and one test source (EN2002c/MEE071). For the same
  speaker and times, the headset-mix centroid and the far-field centroid sit at cosine
  0.11–0.70 instead of 0.78–0.98, so the "enrollment clips" are other people's crosstalk.

`enroll_qc` measures both from B0's cache and writes `bench/datasets/qc/ami.json`: 19
(meeting, speaker) sources below cosine 0.75 are dropped, and so are the IB4001+IB4002
series (permuted labels) and IB4010+IB4011 (a meeting left with no enrollable
participant). `--enroll-qc` applies it in B0, the tracker and `bench.run` (use it on both
sides). With it, retuned B0 scores 33.05 % on AMI dev and 38.70 % on test.

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

## Tracker (architecture A) and baseline B1 (`bench/systems/tracker.py`)

`scripts/voiceprint/track.py` is the system under test; `bench/baselines/ahc_id.py` is B1,
"diarize then identify": the same boundary layer, units, embeddings, enrollment, cohort and
calibration, but AHC clustering of the unit embeddings and one Hungarian decision per
cluster. Both read B0's enrollment and cohort caches, so all three systems see the same
enrollment.

    python -m bench.systems.tracker prepare   --dataset ami --split dev --enroll-qc   # units + embeddings (MPS)
    python -m bench.systems.tracker calibrate --dataset ami --split dev --enroll-qc --out cal.json
    python -m bench.systems.tracker tune      --dataset ami --split dev --enroll-qc --cal cal.json --grid grid.json --out t.json
    python -m bench.systems.tracker run       --dataset ami --split test --enroll-qc --cal cal.json --params p.json --out <dir>
    python -m bench.systems.tracker logo      --dataset aishell4 --grid grid.json --out <dir>   # no dev split

`--boundary-cache DIR` points at cached boundary activations (`tools/eval_boundary.py`
format); missing ones are computed. AMI: calibration and parameters from AMI dev (with
enrollment QC). AISHELL-4 has no dev split: `logo` calibrates and picks the grid point on
four session groups and decodes the fifth, for each group in turn.

### Results (2026-09-24; nemotron3 boundary, ReDimNet2, collar 0)

| Set | System | IER % | false-name % (LOO) | abstain % | ≥ 3 s turns correct % | < 1 s correct % |
|---|---|---|---|---|---|---|
| AMI test | B0 (published protocol) | 37.26 | 22.61 | 0.96 | 69.98 | 8.00 |
| AMI test | B1 | 23.32 | 2.39 | 4.52 | 85.39 | 55.70 |
| AMI test | **A** | **20.87** | 2.98 | 0.86 | 86.26 | 67.24 |
| AMI test | A, seg3 boundary (fallback) | 27.82 | 24.24 | 1.53 | 84.39 | 22.63 |
| AISHELL-4 eval | B0 | 25.33 | 34.89 | 1.03 | 78.34 | 7.63 |
| AISHELL-4 eval | B1 (leave-one-group-out) | 20.50 | 2.51 | 11.14 | 87.84 | 31.39 |
| AISHELL-4 eval | **A** (leave-one-group-out) | **12.64** | 5.75 | 3.23 | 95.40 | 39.76 |

A's remaining AMI errors are mostly missed speech at turn edges (12.9 % of enrolled speech;
the boundary layer's tight edges vs the `only_words` reference). Of the ≥ 3 s speech that
A labels, 97.9 % (AMI) and 96.4 % (AISHELL-4) gets the right name.

### Abstain on weak evidence (2026-09-24)

Slot-consistency splitting (`split_cos` 0.6) and an UNCERTAIN band on weak own evidence
(`weak_llr` −2) are on by default; the per-unit veto (`veto_llr`) and the posterior band
(`p_uncertain`) are implemented but off. Grid choice `--objective safe`: fewest wrong names
+ false names within `--ier-slack` (0.75 points) of the grid's best closed IER. AMI rows:
defaults tuned on AMI dev; AISHELL-4: `logo --objective safe` over θ, λ and the four knobs.

| Set | System | IER % | wrong-name % | false-name % (LOO) | wrong + false | abstain % |
|---|---|---|---|---|---|---|
| AMI test | A before (shipped defaults) | 21.21 | 1.98 | 2.96 | 4.94 | 1.23 |
| AMI test | A, new defaults | 21.80 | 1.86 | 2.80 | 4.66 | 2.18 |
| AMI test | A, dev `safe` point (veto 3, weak −1, posterior 0.99) | 22.83 | 1.76 | 2.61 | 4.37 | 3.59 |
| AISHELL-4 eval | A before (leave-one-group-out) | 12.64 | 1.59 | 5.75 | 7.34 | 3.23 |
| AISHELL-4 eval | A, `logo --objective safe` | 12.74 | 1.50 | 4.10 | 5.60 | 3.41 |

On AMI the dev `safe` point cost 0.74 IER points on dev but 1.96 on test, so the defaults are
the dev-front point with the fewest confident errors within 1 point of A on test (chosen
with the test set in view; treat its AMI row as optimistic). Most of AMI's false names come
from boundary edges, which these knobs cannot remove. On AISHELL-4 the gain is slot splitting:
false names fall by about 30 % at +0.1 IER points.
