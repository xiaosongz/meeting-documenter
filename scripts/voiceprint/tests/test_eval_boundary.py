"""Boundary-quality metrics (tools/eval_boundary.py). Pure logic on synthetic grids; no audio."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[1] / "tools" / "eval_boundary.py"
_spec = importlib.util.spec_from_file_location("eval_boundary", _P)
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)


def grid(n, spans):
    """(n, n_labels) bool grid at 10 ms from {label: [(start_s, end_s), ...]}."""
    g = np.zeros((n, max(spans) + 1), bool)
    for k, ss in spans.items():
        for a, b in ss:
            g[int(round(a * 100)):int(round(b * 100)), k] = True
    return g


def test_turns_fill_small_gaps_and_drop_short():
    g = grid(1000, {0: [(0, 1), (1.1, 2)], 1: [(5, 5.05)]})
    assert eb.turns_from_activity(g) == [(0.0, 2.0, 0), (5.0, 5.05, 1)]
    assert eb.turns_from_activity(g, min_turn_s=0.1) == [(0.0, 2.0, 0)]


def test_change_points_rule():
    turns = [(0.0, 2.0, 0), (2.4, 4.0, 1),      # short gap → midpoint 2.2
             (6.0, 8.0, 0),                     # 2 s after 1 ended → no change (speech onset)
             (7.0, 7.5, 1)]                     # overlap onset 7.0; offset 7.5 while 0 continues
    assert eb.change_points(turns).tolist() == [2.2, 7.0, 7.5]
    assert eb.change_points([(0.0, 1.0, 0), (1.5, 2.0, 0)]).size == 0   # same speaker: no change


def test_match_points_is_one_to_one():
    ref = np.array([1.0, 2.0, 3.0])
    assert eb.match_points(ref, np.array([1.1, 1.2, 2.6]), 0.25) == 1
    assert eb.match_points(ref, np.array([1.1, 1.2, 2.6]), 0.5) == 2
    assert eb.match_points(ref, ref, 0.0) == 3
    assert eb.prf(2, 3, 4)["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))


def test_window_stats_perfect_and_merged():
    ref = grid(1000, {0: [(0, 4)], 1: [(4.5, 9)]})
    st = eb.summarize_windows(eb.window_stats(ref, ref.copy(), 100))
    assert st["purity"] == 1 and st["coverage"] == 1 and st["missed"] == 0
    merged = ref.any(1, keepdims=True)                    # one local speaker for both → impure
    st = eb.summarize_windows(eb.window_stats(ref, merged, 100))
    assert st["purity"] == pytest.approx(450 / 850) and st["confusion"] == pytest.approx(400 / 850)
    assert st["coverage"] == 1
    split = np.zeros((1000, 2), bool)                     # speaker 0 split across two locals
    split[:200, 0] = True
    split[200:400, 1] = True
    st = eb.summarize_windows(eb.window_stats(ref, split, 100))
    assert st["purity"] == 1 and st["coverage"] == pytest.approx(200 / 850)
    assert st["missed"] == pytest.approx(450 / 850)


def test_score_meeting_reference_as_hypothesis():
    n = 6000
    ref = grid(n, {0: [(1, 10), (20, 30)], 1: [(10.3, 19.8)], 2: [(25, 26), (40, 55)]})
    starts = np.arange(0, 50.01, 2.5)
    cache = {"probs": ref.astype(np.float32), "window_starts": starts}
    r = eb.score_meeting(cache, ref, np.ones(n, bool), "nemotron3")
    assert r["n_ref"] == r["n_hyp"] > 0
    assert all(h == r["n_ref"] for h in r["hits"].values())
    w = eb.summarize_windows(r["windows"])
    assert w["purity"] == 1 and w["coverage"] == 1
    assert eb.summarize_windows(r["global"])["purity"] == 1
