"""Genetic codes (NCBI translation tables), exactly as bcftools csq uses them.

The tables are copied from bcftools 1.24 ``csq.c`` (``gencode_tables``, generated there from
NCBI's ``gc.prt`` by ``misc/gencode-tables``), so csq's consequences and omnimissense's protein
translation use the same code by construction; a test checks them against Biopython's copy of
the NCBI tables. For each table: name, the amino acid of every codon in the order AAA, AAC,
AAG, AAT, ACA, ..., TTT, and a mask marking initiation codons (``M``) and stop codons (``*``).

bcftools csq has no tables 7, 8 (retired by NCBI) and 32; its extra table 0 ("Standard
simplified", its default) has table 1's codons with ATG as the only start. In tables 27, 28
and 31 some codons are stops only at the end of a CDS (otherwise Q, Q or E): they are
translated as amino acids and listed in ``stops``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

STANDARD = 1

_BASES = "ACGT"
_CODONS = tuple(a + b + c for a in _BASES for b in _BASES for c in _BASES)

# fmt: off
_TABLES: dict[int, tuple[str, str, str]] = {
    1: (
        "Standard",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF",
        "--------------M---------------M-----------------*-*-----*-----M-",
    ),
    2: (
        "Vertebrate Mitochondrial",
        "KNKNTTTT*S*SMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "--------*-*-MMMM------------------------------M-*-*-------------",
    ),
    3: (
        "Yeast Mitochondrial",
        "KNKNTTTTRSRSMIMIQHQHPPPPRRRRTTTTEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "------------M-M-------------------------------M-*-*-------------",
    ),
    4: (
        "Mold Mitochondrial; Protozoan Mitochondrial; "
        "Coelenterate Mitochondrial; Mycoplasma; Spiroplasma",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "------------MMMM--------------M---------------M-*-*---------M-M-",
    ),
    5: (
        "Invertebrate Mitochondrial",
        "KNKNTTTTSSSSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "------------MMMM------------------------------M-*-*-----------M-",
    ),
    6: (
        "Ciliate Nuclear; Dasycladacean Nuclear; Hexamita Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVQYQYSSSS*CWCLFLF",
        "--------------M-----------------------------------------*-------",
    ),
    9: (
        "Echinoderm Mitochondrial; Flatworm Mitochondrial",
        "NNKNTTTTSSSSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "--------------M-------------------------------M-*-*-------------",
    ),
    10: (
        "Euplotid Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSCCWCLFLF",
        "--------------M---------------------------------*-*-------------",
    ),
    11: (
        "Bacterial, Archaeal and Plant Plastid",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF",
        "------------MMMM--------------M---------------M-*-*-----*-----M-",
    ),
    12: (
        "Alternative Yeast Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLSLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF",
        "--------------M---------------M-----------------*-*-----*-------",
    ),
    13: (
        "Ascidian Mitochondrial",
        "KNKNTTTTGSGSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "------------M-M-------------------------------M-*-*-----------M-",
    ),
    14: (
        "Alternative Flatworm Mitochondrial",
        "NNKNTTTTSSSSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVYY*YSSSSWCWCLFLF",
        "--------------M-----------------------------------*-------------",
    ),
    15: (
        "Blepharisma Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*YQYSSSS*CWCLFLF",
        "--------------M---------------------------------*-------*-------",
    ),
    16: (
        "Chlorophycean Mitochondrial",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*YLYSSSS*CWCLFLF",
        "--------------M---------------------------------*-------*-------",
    ),
    21: (
        "Trematode Mitochondrial",
        "NNKNTTTTSSSSMIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "--------------M-------------------------------M-*-*-------------",
    ),
    22: (
        "Scenedesmus obliquus Mitochondrial Code",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*YLY*SSS*CWCLFLF",
        "--------------M---------------------------------*---*---*-------",
    ),
    23: (
        "Thraustochytrium mitochondrial code",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWC*FLF",
        "--------------MM------------------------------M-*-*-----*---*---",
    ),
    24: (
        "Pterobranchia Mitochondrial",
        "KNKNTTTTSSKSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSWCWCLFLF",
        "--------------M---------------M---------------M-*-*-----------M-",
    ),
    25: (
        "Candidate Division SR1 and Gracilibacteria",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSSGCWCLFLF",
        "--------------M-------------------------------M-*-*-----------M-",
    ),
    26: (
        "Pachysolen tannophilus Nuclear Code",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLALEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF",
        "--------------M---------------M-----------------*-*-----*-------",
    ),
    27: (
        "Karyorelict Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVQYQYSSSSWCWCLFLF",
        "--------------M-----------------------------------------*-------",
    ),
    28: (
        "Condylostoma Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVQYQYSSSSWCWCLFLF",
        "--------------M---------------------------------*-*-----*-------",
    ),
    29: (
        "Mesodinium Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVYYYYSSSS*CWCLFLF",
        "--------------M-----------------------------------------*-------",
    ),
    30: (
        "Peritrich Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVEYEYSSSS*CWCLFLF",
        "--------------M-----------------------------------------*-------",
    ),
    31: (
        "Blastocrithidia Nuclear",
        "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVEYEYSSSSWCWCLFLF",
        "--------------M---------------------------------*-*-------------",
    ),
    33: (
        "Cephalodiscidae Mitochondrial UAA-Tyr",
        "KNKNTTTTSSKSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVVYY*YSSSSWCWCLFLF",
        "--------------M---------------M---------------M---*-----------M-",
    ),
}
# fmt: on


@dataclass(frozen=True)
class GeneticCode:
    id: int
    name: str
    codons: Mapping[str, str]  # all 64 codons -> amino acid, "*" for stop
    starts: frozenset[str]  # initiation codons: read as methionine at a complete 5' end
    stops: frozenset[str]  # termination codons (a few also code for an amino acid)


GENETIC_CODES: dict[int, GeneticCode] = {
    table_id: GeneticCode(
        id=table_id,
        name=name,
        codons=dict(zip(_CODONS, code, strict=True)),
        starts=frozenset(c for c, mark in zip(_CODONS, marks, strict=True) if mark == "M"),
        stops=frozenset(c for c, mark in zip(_CODONS, marks, strict=True) if mark == "*"),
    )
    for table_id, (name, code, marks) in _TABLES.items()
}


def genetic_code(table_id: int) -> GeneticCode:
    """The NCBI translation table ``table_id``; ``ValueError`` if bcftools csq lacks it.

    Never pass an unchecked ID to csq: in bcftools 1.24 an unknown ``--genetic-code`` run
    in-process (as pysam does) silently keeps the table of the previous run.
    """
    try:
        return GENETIC_CODES[table_id]
    except KeyError:
        raise ValueError(
            f"unknown genetic code {table_id}; available NCBI tables: "
            + ", ".join(str(i) for i in sorted(GENETIC_CODES))
        ) from None


#: Sequence names commonly used for mitochondrial DNA (Ensembl, UCSC, ...). NCBI names
#: sequences by accession; its GFF3 marks them with ``genome=mitochondrion`` instead.
MITOCHONDRIAL_SEQIDS = frozenset({"MT", "Mt", "M", "chrM", "chrMT", "mitochondrion"})


@dataclass(frozen=True)
class CodeAssignment:
    """Which genetic code each sequence uses, for csq and for the protein translation.

    * ``nuclear`` (``--genetic-code``): every non-mitochondrial sequence. Default: the
      sequence's ``transl_table`` in the GFF3, else the standard code.
    * ``mitochondrial`` (``--mito-genetic-code``): every mitochondrial sequence. Default: the
      sequence's ``transl_table`` in the GFF3, else none, and its transcripts are skipped.
    * ``annotated``: the ``transl_table`` of each sequence's CDS features in the GFF3 (NCBI
      writes it whenever the code is not the standard one; Ensembl never does).
    * ``mito_seqids``: sequences the GFF3 marks as mitochondrial.
    """

    nuclear: int | None = None
    mitochondrial: int | None = None
    annotated: Mapping[str, int] = field(default_factory=dict)
    mito_seqids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for table_id in (self.nuclear, self.mitochondrial):
            if table_id is not None:
                genetic_code(table_id)
        for seqid in self.annotated:
            if self.code(seqid) is not None and self.code(seqid) not in GENETIC_CODES:
                option = "--mito-genetic-code" if self.is_mitochondrial(seqid) else "--genetic-code"
                raise ValueError(
                    f"the GFF3 gives sequence {seqid} the genetic code {self.annotated[seqid]}, "
                    f"which bcftools csq does not have; choose one with {option}"
                )

    def is_mitochondrial(self, seqid: str) -> bool:
        return seqid in self.mito_seqids or seqid in MITOCHONDRIAL_SEQIDS

    def code(self, seqid: str) -> int | None:
        """The genetic code of ``seqid``; ``None`` for mitochondrial DNA of unknown code."""
        if self.is_mitochondrial(seqid):
            return (
                self.mitochondrial if self.mitochondrial is not None else self.annotated.get(seqid)
            )
        return self.nuclear if self.nuclear is not None else self.annotated.get(seqid, STANDARD)

    def csq_code(self, seqid: str) -> int:
        """The code to run csq with for variants on ``seqid`` (standard when unknown: those
        transcripts are not translated, so their variants are reported, not scored)."""
        return self.code(seqid) or STANDARD

    def describe(self) -> dict[str, int | None]:
        """Codes in use, for logs and file metadata: ``{"default": table}`` for sequences
        without their own entry, plus one entry per annotated or mitochondrial sequence."""
        used: dict[str, int | None] = {
            "default": self.nuclear if self.nuclear is not None else STANDARD
        }
        for seqid in sorted(set(self.annotated) | self.mito_seqids):
            used[seqid] = self.code(seqid)
        return used
