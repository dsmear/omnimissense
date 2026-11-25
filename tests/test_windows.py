import itertools

import numpy as np
import pytest

from omnimissense.windows import Window, combine_windows, plan_windows, window_weights


def test_short_protein_uses_a_single_window() -> None:
    assert plan_windows(1022) == [Window(0, 1022)]
    assert plan_windows(5) == [Window(0, 5)]


@pytest.mark.parametrize("length", [1023, 1500, 3000, 34350])
def test_long_protein_windows_cover_everything_with_full_size_windows(length) -> None:
    windows = plan_windows(length)
    assert windows[0].start == 0
    assert windows[-1].end == length
    assert all(len(w) == 1022 for w in windows)
    covered = np.zeros(length, dtype=bool)
    for w in windows:
        covered[w.start : w.end] = True
    assert covered.all()
    # consecutive windows overlap by at least the requested overlap
    assert all(a.end - b.start >= 511 for a, b in itertools.pairwise(windows))


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        plan_windows(0)
    with pytest.raises(ValueError):
        plan_windows(100, max_len=20, overlap=20)


def test_weights_are_normalised_and_fade_at_artificial_edges() -> None:
    length = 3000
    windows = plan_windows(length)
    weights = window_weights(windows, length)
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    # The true protein ends are fully trusted in the only window that covers them.
    assert weights[0, 0] == pytest.approx(1.0)
    assert weights[-1, -1] == pytest.approx(1.0)
    # At the artificial left edge of window 1, window 0 (where that residue is central)
    # dominates.
    edge = windows[1].start
    assert weights[1, edge] < 0.01
    assert weights[0, edge] > 0.99


def test_combine_windows_is_a_weighted_average() -> None:
    length = 50
    windows = plan_windows(length, max_len=20, overlap=10)
    parts = [np.full((len(w), 2), float(k)) for k, w in enumerate(windows)]
    combined = combine_windows(parts, windows, length, overlap=10)
    weights = window_weights(windows, length, overlap=10)
    expected = (weights * np.arange(len(windows))[:, None]).sum(axis=0)
    np.testing.assert_allclose(combined[:, 0], expected, rtol=1e-6)
    assert combined[0, 0] == 0.0  # only window 0 covers the first residue
    assert combined[-1, 0] == len(windows) - 1
