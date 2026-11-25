"""Small, dependency-free statistics used by the benchmarks."""

from __future__ import annotations

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1; ties get the average of the ranks they span."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    _, inverse, counts = np.unique(values[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average = ends - (counts - 1) / 2
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = average[inverse]
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve; ``labels`` are 1 for positives, higher score = more positive.

    Equals the probability that a random positive scores higher than a random negative
    (ties count one half), computed with the Mann-Whitney U statistic.
    """
    labels = np.asarray(labels, dtype=bool)
    ranks = rankdata(np.asarray(scores, dtype=np.float64))
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need at least one positive and one negative")
    u = ranks[labels].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def bootstrap_ci(
    labels: np.ndarray, scores: np.ndarray, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap confidence interval of the AUROC (stratified resampling)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values.append(auroc(labels[idx], scores[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_diff_ci(
    labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap interval of AUROC(a) - AUROC(b) on the same variants (paired)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=bool)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    pos, neg = np.flatnonzero(labels), np.flatnonzero(~labels)
    values = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        values.append(auroc(labels[idx], a[idx]) - auroc(labels[idx], b[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)
