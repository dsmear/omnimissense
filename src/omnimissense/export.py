"""Export a proteome's wild-type-marginal scores to one self-describing Parquet file.

One row per coding transcript. ``llr`` is the flattened ``protein_length x 20`` matrix of
log-likelihood ratios ``log P(alt) - log P(ref)`` (row = residue, column = amino acid in
``AMINO_ACIDS`` order; 0 for the reference residue, NaN for non-standard reference residues).
Rows are grouped by sequence hash, not sorted by genome position.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from omnimissense import __version__
from omnimissense.cache import ScoreCache
from omnimissense.fasta import prepare_fasta
from omnimissense.gff import read_transcripts, scan_genetic_codes
from omnimissense.pipeline import LazyScorer, ScorerFactory, cached_wt_scores
from omnimissense.proteome import ProteinStatus, build_proteome
from omnimissense.sequences import AMINO_ACIDS

log = logging.getLogger(__name__)

SCHEMA = pa.schema(
    [
        ("transcript_id", pa.string()),
        ("gene_id", pa.string()),
        ("gene_name", pa.string()),
        ("canonical", pa.bool_()),
        ("biotype", pa.string()),
        ("seqid", pa.string()),
        ("strand", pa.string()),
        ("protein_length", pa.int32()),
        ("sequence", pa.string()),
        ("sequence_sha256", pa.string()),
        ("n_windows", pa.int16()),
        ("llr", pa.list_(pa.float32())),
    ]
)


@dataclass
class ExportResult:
    n_transcripts: int
    n_unique_proteins: int
    n_residues: int


def _chunks(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def export_scores(
    fasta: Path,
    gff3: Path,
    output: Path,
    *,
    cache: ScoreCache,
    scorer_factory: ScorerFactory,
    model_name: str,
    canonical_only: bool = False,
    chunk_size: int = 2000,
    genetic_code: int | None = None,
    mito_genetic_code: int | None = None,
) -> ExportResult:
    """Write wild-type-marginal scores of every coding transcript; missing scores are computed."""
    fasta = prepare_fasta(fasta)
    codes = scan_genetic_codes(gff3, genetic_code, mito_genetic_code)
    transcripts = read_transcripts(gff3, fasta, codes=codes)
    proteins = build_proteome(fasta, transcripts)
    rows = [
        (tid, protein)
        for tid, protein in proteins.items()
        if protein.status is ProteinStatus.OK and (transcripts[tid].canonical or not canonical_only)
    ]
    rows.sort(key=lambda r: r[1].seq_hash)  # contiguous hash ranges -> few cache row groups
    metadata = {
        "omnimissense_version": __version__,
        "model": model_name,
        "score": "wild-type marginal log-likelihood ratio log P(alt) - log P(ref)",
        "amino_acid_order": AMINO_ACIDS,
        "llr_layout": "row-major protein_length x 20",
        "genome": Path(fasta).name,
        "annotation": Path(gff3).name,
        "genetic_codes": codes.describe(),
        "created": datetime.date.today().isoformat(),
    }
    schema = SCHEMA.with_metadata({"omnimissense": json.dumps(metadata)})
    scorer = LazyScorer(scorer_factory)
    unique, residues = set(), 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(output, schema, compression="zstd") as writer:
        for chunk in _chunks(rows, chunk_size):
            sequences = {p.seq_hash: p.sequence for _, p in chunk}
            scores = cached_wt_scores(sequences, cache, scorer)
            columns: dict[str, list] = {name: [] for name in SCHEMA.names}
            for tid, protein in chunk:
                tx = transcripts[tid]
                matrix, n_windows = scores[protein.seq_hash]
                columns["transcript_id"].append(tid)
                columns["gene_id"].append(tx.gene_id)
                columns["gene_name"].append(tx.gene_name)
                columns["canonical"].append(tx.canonical)
                columns["biotype"].append(tx.biotype)
                columns["seqid"].append(tx.seqid)
                columns["strand"].append(tx.strand)
                columns["protein_length"].append(len(protein.sequence))
                columns["sequence"].append(protein.sequence)
                columns["sequence_sha256"].append(protein.seq_hash)
                columns["n_windows"].append(n_windows)
                columns["llr"].append(np.asarray(matrix, dtype=np.float32).ravel())
                if protein.seq_hash not in unique:
                    unique.add(protein.seq_hash)
                    residues += len(protein.sequence)
            writer.write_table(pa.table(columns, schema=schema))
            log.info("Exported %d transcripts", len(columns["transcript_id"]))
    return ExportResult(n_transcripts=len(rows), n_unique_proteins=len(unique), n_residues=residues)


def read_export_metadata(path: Path) -> dict[str, str]:
    """The ``omnimissense`` metadata block of an exported file."""
    return json.loads(pq.read_schema(path).metadata[b"omnimissense"])
