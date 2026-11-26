"""Every missense SNV of a translated transcript, at genome coordinates, with its score.

Each single-nucleotide change of every codon is translated with the transcript's genetic
code. Kept: changes of the amino acid (missense). Left out, as bcftools csq does not call
them missense: changes of the initiation codon at a complete 5' end (start lost), changes to
a stop codon (stop gained), and the stop codon itself (not part of the protein).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pysam

from omnimissense.genetic_code import GENETIC_CODES
from omnimissense.gff import Transcript
from omnimissense.proteome import Protein, cds_positions, coding_sequence
from omnimissense.sequences import AA_INDEX

_BASES = "ACGT"
_BASE_INDEX = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(_BASES.encode()):
    _BASE_INDEX[_b] = _i
_COMPLEMENT = np.array([3, 2, 1, 0], dtype=np.int8)  # A<->T, C<->G


@dataclass(frozen=True)
class _CodeArrays:
    amino_acid: np.ndarray  # codon index (16*b0 + 4*b1 + b2) -> AMINO_ACIDS index, -1 if none
    stop: np.ndarray  # codon index -> stop codon of the table
    start: np.ndarray  # codon index -> initiation codon of the table


def _code_arrays(table_id: int, _cache: dict[int, _CodeArrays] = {}) -> _CodeArrays:  # noqa: B006
    if table_id not in _cache:
        code = GENETIC_CODES[table_id]
        codons = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
        _cache[table_id] = _CodeArrays(
            amino_acid=np.array([AA_INDEX.get(code.codons[c], -1) for c in codons]),
            stop=np.array([c in code.stops for c in codons]),
            start=np.array([c in code.starts for c in codons]),
        )
    return _cache[table_id]


@dataclass
class TranscriptSnvs:
    """Missense SNVs of one transcript, as parallel arrays (genome orientation)."""

    pos: np.ndarray  # 1-based
    ref: np.ndarray  # base index in "ACGT", + strand
    alt: np.ndarray
    protein_pos: np.ndarray  # 1-based
    ref_aa: np.ndarray  # AMINO_ACIDS index
    alt_aa: np.ndarray
    score: np.ndarray


def transcript_snvs(
    fasta: pysam.FastaFile, transcript: Transcript, protein: Protein, llr: np.ndarray
) -> TranscriptSnvs:
    """All missense SNVs of a translated transcript with their wild-type-marginal scores."""
    arrays = _code_arrays(transcript.genetic_code)
    n_codons = len(protein.sequence)
    if n_codons == 0:
        empty = np.zeros(0, dtype=np.int64)
        return TranscriptSnvs(empty, empty, empty, empty, empty, empty, np.zeros(0, np.float32))
    cds = coding_sequence(fasta, transcript)[: 3 * n_codons]
    positions = cds_positions(transcript)[: 3 * n_codons].reshape(n_codons, 3)
    bases = _BASE_INDEX[np.frombuffer(cds.encode(), dtype=np.uint8)].reshape(n_codons, 3)
    valid = (bases < 4).all(axis=1)
    safe = np.where(valid[:, None], bases, 0).astype(np.int64)
    codon = 16 * safe[:, 0] + 4 * safe[:, 1] + safe[:, 2]
    ref_aa = np.array([AA_INDEX.get(aa, -1) for aa in protein.sequence])
    # Codons read as written (not the initiation codon turned into M, not ambiguous bases).
    usable = valid & (ref_aa >= 0) & (arrays.amino_acid[codon] == ref_aa)
    if transcript.complete_5prime and arrays.start[codon[0]]:
        usable[0] = False  # changes of the start codon are "start lost" for csq
    parts = defaultdict(list)
    weights = (16, 4, 1)
    for j in range(3):
        for alt in range(4):
            new = codon + (alt - safe[:, j]) * weights[j]
            new_aa = arrays.amino_acid[new]
            missense = (
                usable
                & (safe[:, j] != alt)
                & (new_aa >= 0)
                & ~arrays.stop[new]
                & (new_aa != ref_aa)
            )
            idx = np.flatnonzero(missense)
            parts["pos"].append(positions[idx, j])
            parts["ref"].append(bases[idx, j])
            parts["alt"].append(np.full(len(idx), alt, dtype=np.int8))
            parts["protein_pos"].append(idx + 1)
            parts["ref_aa"].append(ref_aa[idx])
            parts["alt_aa"].append(new_aa[idx])
            parts["score"].append(llr[idx, new_aa[idx]])
    result = {key: np.concatenate(values) for key, values in parts.items()}
    if transcript.strand == "-":
        result["ref"] = _COMPLEMENT[result["ref"]]
        result["alt"] = _COMPLEMENT[result["alt"]]
    return TranscriptSnvs(**result)
