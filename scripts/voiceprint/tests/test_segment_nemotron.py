"""Optional nemotron3 boundary backend. The chunking test is pure; the model test needs a
transformers build with Nemotron 3 Diarization, torch and the fetched weights, and skips
otherwise (CI never has them). Audio is synthetic."""

import numpy as np
import pytest

from voiceprint import segment as sg
from voiceprint import segment_nemotron as nm

needs_nemotron = pytest.mark.skipif(not nm.available(), reason="transformers Nemotron3 / weights missing")


class FakeProc:
    """Offline sizes of the released checkpoint: 340 + 40 encoder frames, 8 mel frames each, hop 160."""
    hop, n_fft, win = 160, 512, 400
    num_mel_frames_per_audio_chunk = (340 + 40) * 8
    num_mel_frames_per_step = 340 * 8
    num_samples_first_audio_chunk = (num_mel_frames_per_audio_chunk - 1) * hop + win // 2
    num_samples_per_audio_chunk = num_mel_frames_per_audio_chunk * hop + win

    def audio_chunk_start(self, mel):
        return mel * self.hop - self.n_fft // 2


@pytest.mark.parametrize("seconds", [3.0, 30.0, 31.0, 95.0])
def test_chunks_follow_streaming_contract(seconds):
    p = FakeProc()
    wave = np.arange(int(seconds * 16000), dtype=np.float32)
    pieces = list(nm._chunks(p, wave))
    assert pieces[0][1] and pieces[-1][2]                                   # first / last flags
    assert sum(first for _, first, _ in pieces) == 1 and sum(last for *_, last in pieces) == 1
    assert pieces[-1][0][-1] == wave[-1]                                   # the tail is fed
    for audio, first, last in pieces[1:-1]:
        assert len(audio) == p.num_samples_per_audio_chunk
    if len(pieces) > 1:
        assert len(pieces[0][0]) == p.num_samples_first_audio_chunk
        # consecutive middle pieces advance by exactly one step of mel frames
        starts = [int(a[0]) for a, *_ in pieces[1:]]
        assert all(b - a == p.num_mel_frames_per_step * p.hop for a, b in zip(starts, starts[1:]))


def test_backend_is_listed_and_lazy():
    assert "nemotron3" in sg.BACKENDS
    assert nm.local_dir().name.startswith("nemotron3-diarization@")
    assert nm.url("config.json").endswith(f"{nm.HF_REVISION}/config.json")
    assert "-preview" not in nm.HF_REPO


@needs_nemotron
def test_nemotron_returns_one_global_chunk():
    rng = np.random.default_rng(0)
    x = (0.01 * rng.normal(size=40 * 16000)).astype(np.float32)          # noise only
    seg = sg.segment(x, backend="nemotron3")
    assert seg.n_chunks == 1 and seg.n_local == 8 and seg.frame_step == pytest.approx(0.01)
    assert seg.n_frames == pytest.approx(4000, abs=2)
    assert seg.activations.min() >= 0 and seg.activations.max() <= 1
    assert seg.meta["speech"].shape[0] == seg.n_frames
    assert sg.local_regions(seg, min_clean_s=1.0) == []
