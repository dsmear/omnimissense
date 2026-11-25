"""DNA and protein sequence helpers."""

from __future__ import annotations

import hashlib

#: The 20 standard amino acids, in the column order used for every score matrix.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# Standard genetic code (NCBI table 1), codons ordered AAA, AAC, AAG, AAT, ACA, ..., TTT.
# Same table as bcftools csq (csq.c, gencode_tables, id=1).
_CODE = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
_BASES = "ACGT"
CODON_TABLE = {
    a + b + c: _CODE[16 * i + 4 * j + k]
    for i, a in enumerate(_BASES)
    for j, b in enumerate(_BASES)
    for k, c in enumerate(_BASES)
}


def reverse_complement(seq: str) -> str:
    """Reverse complement of a DNA sequence."""
    return seq.translate(_COMPLEMENT)[::-1]


def translate(cds: str) -> str:
    """Translate a coding sequence codon by codon (standard code).

    Codons with ambiguous bases become ``X``; a trailing partial codon is ignored.
    Stop codons are kept as ``*``.
    """
    cds = cds.upper()
    return "".join(CODON_TABLE.get(cds[i : i + 3], "X") for i in range(0, len(cds) - 2, 3))


def sequence_hash(protein: str) -> str:
    """Stable identifier of a protein sequence, used as the score-cache key."""
    return hashlib.sha256(protein.encode("ascii")).hexdigest()
