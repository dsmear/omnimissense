"""On-disk score cache (Parquet), keyed by protein sequence hash.

Layout::

    <root>/v1/<model-slug>/wt_marginal/part-*.parquet       seq_hash, length, n_windows, llr
    <root>/v1/<model-slug>/masked_marginal/part-*.parquet   seq_hash, position, llr

``llr`` is a flat list of float32 (``length x 20`` for wild-type marginals, 20 for masked
marginals; columns in ``AMINO_ACIDS`` order). Files are append-only. A lookup reads only the
row groups that hold the requested hashes, found through an in-memory index of the
``seq_hash`` column (built on first use, extended with new files).
The ``v1`` level must be bumped whenever the scoring procedure (e.g. windowing) changes.
A cache directory must not be written by two processes at the same time.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from omnimissense.sequences import AMINO_ACIDS

CACHE_VERSION = "v1"
_N_AA = len(AMINO_ACIDS)


def default_cache_dir() -> Path:
    if env := os.environ.get("OMNIMISSENSE_CACHE"):
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "omnimissense"


def _slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "--", model_name)


class ScoreCache:
    def __init__(self, root: Path, model_name: str) -> None:
        self.directory = Path(root) / CACHE_VERSION / _slug(model_name)
        self._wt_dir = self.directory / "wt_marginal"
        self._masked_dir = self.directory / "masked_marginal"
        # per directory: seq_hash -> {(file, row group)}, and the files already indexed
        self._indexes: dict[Path, tuple[dict[str, set[tuple[Path, int]]], set[Path]]] = {}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _dataset(directory: Path) -> ds.Dataset | None:
        if not directory.exists() or not any(directory.glob("*.parquet")):
            return None
        return ds.dataset(directory, format="parquet")

    @staticmethod
    def _write(directory: Path, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        directory.mkdir(parents=True, exist_ok=True)
        table = table.sort_by("seq_hash")
        final = directory / f"part-{uuid.uuid4().hex}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, row_group_size=128, compression="zstd")
        tmp.rename(final)

    def _locations(self, directory: Path) -> dict[str, set[tuple[Path, int]]]:
        """Where each ``seq_hash`` is stored: ``{seq_hash: {(file, row group), ...}}``.

        Built once per directory from the ``seq_hash`` column only, then extended with the
        files written since (files are append-only, never modified). Row-group min/max
        statistics cannot replace it: hashes are random, so every row group spans almost the
        whole hash range.
        """
        index, indexed = self._indexes.setdefault(directory, ({}, set()))
        for path in sorted(directory.glob("*.parquet")):
            if path in indexed:
                continue
            parquet = pq.ParquetFile(path)
            for group in range(parquet.metadata.num_row_groups):
                hashes = parquet.read_row_group(group, columns=["seq_hash"]).column(0)
                for seq_hash in set(hashes.to_pylist()):
                    index.setdefault(seq_hash, set()).add((path, group))
            indexed.add(path)
        return index

    def _read_matching(self, directory: Path, hashes: set[str]) -> pa.Table | None:
        """Rows whose ``seq_hash`` is in ``hashes``, reading only the row groups that hold
        them, so time and memory follow the result, not the cache size."""
        if not hashes or not directory.exists():
            return None
        index = self._locations(directory)
        groups: dict[Path, set[int]] = {}
        for seq_hash in hashes:
            for path, group in index.get(seq_hash, ()):
                groups.setdefault(path, set()).add(group)
        if not groups:
            return None
        wanted = pa.array(sorted(hashes), pa.string())
        tables = []
        for path in sorted(groups):
            table = pq.ParquetFile(path).read_row_groups(sorted(groups[path]))
            tables.append(table.filter(pc.is_in(table["seq_hash"], wanted)))
        return pa.concat_tables(tables)

    @staticmethod
    def _split_rows(table: pa.Table) -> list[np.ndarray]:
        llr = table.column("llr").combine_chunks()
        offsets = llr.offsets.to_numpy()
        values = llr.values.to_numpy(zero_copy_only=False)
        return [values[offsets[i] : offsets[i + 1]] for i in range(len(llr))]

    # ------------------------------------------------------------------ wild-type marginals

    def wt_hashes(self) -> set[str]:
        dataset = self._dataset(self._wt_dir)
        if dataset is None:
            return set()
        return set(dataset.to_table(columns=["seq_hash"]).column("seq_hash").to_pylist())

    def get_wt(self, hashes: Iterable[str]) -> dict[str, tuple[np.ndarray, int]]:
        table = self._read_matching(self._wt_dir, set(hashes))
        if table is None:
            return {}
        found = {}
        rows = self._split_rows(table)
        for seq_hash, n_windows, flat in zip(
            table.column("seq_hash").to_pylist(),
            table.column("n_windows").to_pylist(),
            rows,
            strict=True,
        ):
            found[seq_hash] = (flat.reshape(-1, _N_AA), n_windows)
        return found

    def put_wt(self, scores: Mapping[str, tuple[np.ndarray, int]]) -> None:
        hashes = list(scores)
        matrices = [np.asarray(scores[h][0], dtype=np.float32) for h in hashes]
        table = pa.table(
            {
                "seq_hash": pa.array(hashes, pa.string()),
                "length": pa.array([m.shape[0] for m in matrices], pa.int32()),
                "n_windows": pa.array([scores[h][1] for h in hashes], pa.int16()),
                "llr": pa.array([m.ravel() for m in matrices], pa.list_(pa.float32())),
            }
        )
        self._write(self._wt_dir, table)

    # ------------------------------------------------------------------ masked marginals

    def get_masked(self, keys: Iterable[tuple[str, int]]) -> dict[tuple[str, int], np.ndarray]:
        keys = set(keys)
        table = self._read_matching(self._masked_dir, {h for h, _ in keys})
        if table is None:
            return {}
        found = {}
        for seq_hash, position, flat in zip(
            table.column("seq_hash").to_pylist(),
            table.column("position").to_pylist(),
            self._split_rows(table),
            strict=True,
        ):
            if (seq_hash, position) in keys:
                found[(seq_hash, position)] = flat
        return found

    def put_masked(self, scores: Mapping[tuple[str, int], np.ndarray]) -> None:
        keys = list(scores)
        table = pa.table(
            {
                "seq_hash": pa.array([h for h, _ in keys], pa.string()),
                "position": pa.array([p for _, p in keys], pa.int32()),
                "llr": pa.array(
                    [np.asarray(scores[k], dtype=np.float32) for k in keys],
                    pa.list_(pa.float32()),
                ),
            }
        )
        self._write(self._masked_dir, table)
