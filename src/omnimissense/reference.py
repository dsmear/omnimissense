"""Proteome-wide percentiles of the scores.

The reference distribution of a genome annotation is the wild-type-marginal score of every
possible missense SNV of its canonical protein-coding transcripts, one value per SNV and
transcript (the rows that ``export-variants`` writes by default). The percentile of a score is the share of that
distribution with a lower (more damaging) score, in percent: 1.0 means "among the 1% most
damaging missense changes possible in this genome". It is a monotone transform of the
score: it changes no ranking, it only puts the score on a scale that reads the same for
every gene, species and model.

The distribution is stored in the score cache as a table of quantiles, finer in the
damaging tail (steps of 0.001 percentage points below the 1st percentile, 0.01 above), under
a key derived from the GFF3 file, the genome's sequence names and lengths and the genetic
codes. ``omnimissense precompute`` builds it; ``annotate`` uses it when it exists.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam

from omnimissense.genetic_code import CodeAssignment
from omnimissense.gff import Transcript
from omnimissense.proteome import Protein, ProteinStatus
from omnimissense.snvs import transcript_snvs

log = logging.getLogger(__name__)

#: Bump when the reference set or the quantile table changes.
REFERENCE_VERSION = "1"
#: Quantile levels (fractions): 1e-5 steps up to 0.01, then 1e-4 steps.
LEVELS = np.concatenate([np.linspace(0.0, 0.01, 1001), np.linspace(0.01, 1.0, 9901)[1:]])


def reference_key(gff3: Path, fasta: Path, codes: CodeAssignment) -> str:
    """Identifies the reference set: GFF3 bytes, genome sequence names and lengths, codes."""
    digest = hashlib.sha256(f"omnimissense-reference-v{REFERENCE_VERSION}\n".encode())
    with gff3.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    with pysam.FastaFile(str(fasta)) as genome:
        digest.update(
            json.dumps(list(zip(genome.references, genome.lengths, strict=True))).encode()
        )
    digest.update(json.dumps(codes.describe(), sort_keys=True).encode())
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class Reference:
    quantiles: np.ndarray  # score at each of ``levels``
    levels: np.ndarray  # fractions in [0, 1]
    n_snvs: int

    def __post_init__(self) -> None:
        # Equal quantiles (ties) would make the interpolation ambiguous: average their levels.
        values, inverse = np.unique(self.quantiles, return_inverse=True)
        levels = np.bincount(inverse, weights=self.levels) / np.bincount(inverse)
        object.__setattr__(self, "_xp", values)
        object.__setattr__(self, "_fp", levels * 100.0)

    def percentile(self, scores: np.ndarray | float) -> np.ndarray:
        """Percent of the reference SNVs with a lower (more damaging) score."""
        return np.interp(np.asarray(scores, dtype=np.float64), self._xp, self._fp)

    @classmethod
    def from_scores(cls, scores: np.ndarray) -> Reference:
        if len(scores) == 0:
            raise ValueError("no SNV scores to build a reference from")
        # float32 in, float64 quantiles out: no float64 copy of the (large) score array
        quantiles = np.quantile(np.asarray(scores, dtype=np.float32), LEVELS).astype(np.float64)
        return cls(quantiles, LEVELS.copy(), len(scores))


def _path(cache_directory: Path, key: str) -> Path:
    return cache_directory / "reference" / f"{key}.npz"


def load_reference(cache_directory: Path, key: str) -> Reference | None:
    path = _path(cache_directory, key)
    if not path.exists():
        return None
    with np.load(path) as data:
        return Reference(data["quantiles"], data["levels"], int(data["n_snvs"]))


def save_reference(cache_directory: Path, key: str, reference: Reference) -> Path:
    path = _path(cache_directory, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    np.savez(tmp, quantiles=reference.quantiles, levels=reference.levels, n_snvs=reference.n_snvs)
    tmp.rename(path)
    return path


def residue_batches(
    groups: Iterable[tuple[str, list[str]]],
    proteins: Mapping[str, Protein],
    limit: int = 2_000_000,
) -> Iterator[list[tuple[str, list[str]]]]:
    """Consecutive ``(seqid, transcript IDs)`` groups packed into batches of at most
    ``limit`` residues (a larger group is a batch of its own): one cache read per batch
    instead of one per sequence (genomes can have thousands of small scaffolds)."""
    batch: list[tuple[str, list[str]]] = []
    residues = 0
    for seqid, tids in groups:
        size = sum(len(proteins[t].sequence) for t in tids)
        if batch and residues + size > limit:
            yield batch
            batch, residues = [], 0
        batch.append((seqid, tids))
        residues += size
    if batch:
        yield batch


def build_reference(
    fasta: Path,
    transcripts: Mapping[str, Transcript],
    proteins: Mapping[str, Protein],
    wt_scores: Callable[[dict[str, str]], Mapping[str, tuple[np.ndarray, int]]],
) -> Reference:
    """Reference distribution of the canonical transcripts with a scored protein.

    ``wt_scores`` maps ``{seq_hash: sequence}`` to wild-type-marginal matrices (from the cache,
    computing what is missing); it is called once per batch of sequences to bound memory.
    The reference holds one value per SNV and canonical transcript: an SNV in two overlapping
    canonical transcripts counts once for each.
    """
    by_seqid: dict[str, list[str]] = defaultdict(list)
    for tid, tx in transcripts.items():
        protein = proteins.get(tid)
        if tx.canonical and protein is not None and protein.status is ProteinStatus.OK:
            by_seqid[tx.seqid].append(tid)
    values = []
    with pysam.FastaFile(str(fasta)) as genome:
        for batch in residue_batches(by_seqid.items(), proteins):
            tids = [t for _seqid, group in batch for t in group]
            matrices = wt_scores({proteins[t].seq_hash: proteins[t].sequence for t in tids})
            for tid in tids:
                llr = matrices[proteins[tid].seq_hash][0]
                values.append(transcript_snvs(genome, transcripts[tid], proteins[tid], llr).score)
    reference = Reference.from_scores(np.concatenate(values))
    log.info(
        "Percentile reference: %d missense SNV x transcript values, %d canonical transcripts",
        reference.n_snvs,
        sum(len(t) for t in by_seqid.values()),
    )
    return reference
