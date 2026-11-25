"""Sliding windows for proteins longer than the model context.

ESM-2 was trained on at most 1022 residues (1024 tokens with the start/end tokens). Longer
proteins are split into overlapping windows; every residue is scored in each window that
contains it and the results are averaged with weights that fade out near *artificial* window
edges, where the model lacks context on one side. Edges that coincide with the real ends of
the protein are not down-weighted.

This follows the scheme of Brandes et al., Nat Genet 2023 (``esm-variants``, MIT license):
1022-residue windows, ~511-residue overlap, sigmoid edge weights over the first/last
``overlap / 2`` residues of a window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_RESIDUES = 1022
DEFAULT_OVERLAP = 511


@dataclass(frozen=True)
class Window:
    start: int  # 0-based, inclusive
    end: int  # 0-based, exclusive

    def __len__(self) -> int:
        return self.end - self.start


def plan_windows(
    length: int, max_len: int = MAX_RESIDUES, overlap: int = DEFAULT_OVERLAP
) -> list[Window]:
    """Windows covering ``length`` residues; a single window if the protein fits."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not 0 < overlap < max_len:
        raise ValueError("overlap must be between 0 and max_len")
    if length <= max_len:
        return [Window(0, length)]
    stride = max_len - overlap
    starts = list(range(0, length - max_len + 1, stride))
    if starts[-1] != length - max_len:
        starts.append(length - max_len)
    return [Window(s, s + max_len) for s in starts]


def _edge_ramp(ramp: int) -> np.ndarray:
    """Weights for the ``ramp`` residues next to an artificial cut (index 0 = at the cut)."""
    distance = np.arange(ramp, dtype=np.float64)
    scale = max(ramp / 12.8, 1e-6)  # 256 / 12.8 = 20, as in Brandes et al.
    return 1.0 / (1.0 + np.exp(-(distance - ramp / 2) / scale))


def window_weights(
    windows: list[Window], length: int, overlap: int = DEFAULT_OVERLAP
) -> np.ndarray:
    """Normalised weights, shape ``(len(windows), length)``; each column sums to 1."""
    weights = np.zeros((len(windows), length), dtype=np.float64)
    ramp = max(overlap // 2, 1)
    for k, window in enumerate(windows):
        w = np.ones(len(window), dtype=np.float64)
        n = min(ramp, len(window))
        edge = _edge_ramp(ramp)[:n]
        if window.start > 0:
            w[:n] = np.minimum(w[:n], edge)
        if window.end < length:
            w[len(window) - n :] = np.minimum(w[len(window) - n :], edge[::-1])
        weights[k, window.start : window.end] = w
    return weights / weights.sum(axis=0, keepdims=True)


def combine_windows(
    parts: list[np.ndarray], windows: list[Window], length: int, overlap: int = DEFAULT_OVERLAP
) -> np.ndarray:
    """Weighted average of per-window arrays (``len(window) x K``) into ``length x K``."""
    if len(windows) == 1:
        return parts[0]
    weights = window_weights(windows, length, overlap)
    combined = np.zeros((length, parts[0].shape[1]), dtype=np.float64)
    for k, (window, part) in enumerate(zip(windows, parts, strict=True)):
        combined[window.start : window.end] += weights[k, window.start : window.end, None] * part
    return combined.astype(parts[0].dtype)
