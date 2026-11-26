"""DNA and protein sequence helpers."""

from __future__ import annotations

import hashlib

from omnimissense.genetic_code import GENETIC_CODES, STANDARD

#: The 20 standard amino acids, in the column order used for every score matrix.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

#: Standard genetic code (NCBI table 1), as used by bcftools csq.
CODON_TABLE = GENETIC_CODES[STANDARD].codons


def reverse_complement(seq: str) -> str:
    """Reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def translate(cds: str, table: int = STANDARD) -> str:
    """Translate a coding sequence codon by codon with an NCBI genetic code.

    Codons with ambiguous bases become ``X``; a trailing partial codon is ignored.
    Stop codons are kept as ``*``.
    """
    codons = GENETIC_CODES[table].codons
    cds = cds.upper()
    return "".join(codons.get(cds[i : i + 3], "X") for i in range(0, len(cds) - 2, 3))


def sequence_hash(protein: str) -> str:
    """Stable identifier of a protein sequence, used as the score-cache key."""
    return hashlib.sha256(protein.encode("ascii")).hexdigest()
