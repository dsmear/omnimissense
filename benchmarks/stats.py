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


#: Proteome-wide percentile thresholds reported by the benchmarks.
PERCENTILE_THRESHOLDS = (1, 5, 10, 20, 30, 50)


def percentile_table(
    percentiles: np.ndarray,
    groups: dict[str, np.ndarray],
    thresholds: tuple[float, ...] = PERCENTILE_THRESHOLDS,
) -> list[dict[str, object]]:
    """Share of each group's variants below each proteome-wide percentile threshold.

    ``groups`` maps a name to a boolean mask over ``percentiles``; empty groups are left out.
    ``all_possible_missense`` is the share of the reference distribution (every possible
    missense SNV of the genome's canonical transcripts) below the threshold, i.e. the
    threshold itself, for comparison.
    """
    percentiles = np.asarray(percentiles, dtype=np.float64)
    rows = []
    for threshold in thresholds:
        row: dict[str, object] = {
            "percentile_below": threshold,
            "all_possible_missense": round(threshold / 100, 3),
        }
        for name, mask in groups.items():
            if mask.any():
                row[name] = round(float((percentiles[mask] < threshold).mean()), 3)
        rows.append(row)
    return rows
