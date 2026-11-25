import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from stats import auroc, bootstrap_ci, rankdata, spearman


def test_rankdata_averages_ties() -> None:
    np.testing.assert_array_equal(rankdata(np.array([10, 20, 20, 5])), [2, 3.5, 3.5, 1])


def test_spearman() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    # textbook example with a tie: matches scipy.stats.spearmanr
    assert spearman(np.array([1, 2, 2, 3]), np.array([1, 3, 2, 4])) == pytest.approx(
        0.9486833, 1e-6
    )


def test_auroc() -> None:
    labels = np.array([1, 1, 0, 0])
    assert auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0
    assert auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert auroc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert auroc(np.array([1, 0, 0]), np.array([0.5, 0.9, 0.1])) == 0.5
    with pytest.raises(ValueError):
        auroc(np.array([1, 1]), np.array([0.1, 0.2]))


def test_bootstrap_interval_contains_the_estimate() -> None:
    rng = np.random.default_rng(1)
    labels = np.array([1] * 40 + [0] * 60)
    scores = labels + rng.normal(0, 1, 100)
    low, high = bootstrap_ci(labels, scores, n=300)
    assert low < auroc(labels, scores) < high


def test_blosum62_is_the_standard_symmetric_matrix() -> None:
    from blosum import BLOSUM62

    assert len(BLOSUM62) == 400
    assert all(BLOSUM62[a, b] == BLOSUM62[b, a] for a, b in BLOSUM62)
    assert (BLOSUM62["W", "W"], BLOSUM62["C", "C"], BLOSUM62["I", "V"]) == (11, 9, 3)
    assert (BLOSUM62["W", "D"], BLOSUM62["R", "K"], BLOSUM62["F", "Y"]) == (-4, 2, 3)
