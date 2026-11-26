"""A tiny hand-designed genome for coordinate tests.

The genome is built from *designed* coding sequences, so the expected protein and the
expected effect of every variant are known independently of the code under test.

Layout of chromosome "1" (1-based, inclusive):

    1..40     filler
    41..47    gene A, exon 1 (+ strand)  CDS "ATGAAAC"
    48..77    gene A, intron (30 nt, GT...AG)
    78..97    gene A, exon 2              CDS "GTTGGCTGGAACATTGCTAA"
    98..137   filler
    138..153  gene B, exon 2 (- strand)   reverse complement of "TGGCAAATGGCGATAA"
    154..183  gene B, intron
    184..194  gene B, exon 1 (- strand)   reverse complement of "ATGGATCCATT"
    195..234  filler

Gene A protein: M K R W L E H C (codon 3 "CGT" is split C | GT across the intron).
Gene B protein: M D P F G K W R (codon 4 "TTT" is split TT | T across the intron).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CHROM = "1"

_FILLER_1 = "ACGTTGCAAGGCTTACCGATAGCTTAGGCATCGATCGGAT"  # 40 nt
_FILLER_2 = "TTAGCCAGTACGGATCCATGCTAGGCTTCAAGCTTGCATC"  # 40 nt
_FILLER_3 = "GGATCCTTAGCAAGTCTGCATGCGATTACGGCTAGCATTG"  # 40 nt
_INTRON = "GTAAGTCTATTCTTATCTTCTTTTCCTTAG"  # 30 nt, canonical GT...AG

GENE_A_EXON1 = "ATGAAAC"
GENE_A_EXON2 = "GTTGGCTGGAACATTGCTAA"
GENE_B_EXON1 = "ATGGATCCATT"  # transcript (sense) orientation
GENE_B_EXON2 = "TGGCAAATGGCGATAA"

PROTEIN_A = "MKRWLEHC"
PROTEIN_B = "MDPFGKWR"

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


GENOME = (
    _FILLER_1
    + GENE_A_EXON1
    + _INTRON
    + GENE_A_EXON2
    + _FILLER_2
    + _revcomp(GENE_B_EXON1 + _INTRON + GENE_B_EXON2)
    + _FILLER_3
)
assert len(GENOME) == 234

# (start, end, phase) of CDS segments, 1-based inclusive, genomic order.
GENE_A_CDS = [(41, 47, 0), (78, 97, 2)]
GENE_B_CDS = [(138, 153, 1), (184, 194, 0)]


def _cds_to_genome(gene: str, cds_index: int) -> int:
    """Map a 0-based position in the spliced CDS to a 1-based genomic coordinate."""
    if gene == "A":
        if cds_index < len(GENE_A_EXON1):
            return 41 + cds_index
        return 78 + cds_index - len(GENE_A_EXON1)
    if cds_index < len(GENE_B_EXON1):
        return 194 - cds_index
    return 153 - (cds_index - len(GENE_B_EXON1))


@dataclass(frozen=True)
class DesignedVariant:
    """A variant defined in transcript coordinates, with its known effect."""

    gene: str  # "A" or "B"
    cds_index: int  # 0-based position in the spliced CDS
    alt_base_sense: str  # new base, in the transcript (sense) orientation
    expected: str | None  # e.g. "K2R" for missense, None when not missense
    label: str

    @property
    def pos(self) -> int:
        return _cds_to_genome(self.gene, self.cds_index)

    @property
    def ref(self) -> str:
        return GENOME[self.pos - 1]

    @property
    def alt(self) -> str:
        return self.alt_base_sense if self.gene == "A" else _revcomp(self.alt_base_sense)


DESIGNED_VARIANTS = [
    DesignedVariant("A", 4, "G", "K2R", "plus strand, simple codon"),
    DesignedVariant("A", 6, "T", "R3C", "plus strand, codon split by intron (exon 1 side)"),
    DesignedVariant("A", 7, "A", "R3H", "plus strand, codon split by intron (exon 2 side)"),
    DesignedVariant("A", 11, "T", "W4C", "plus strand, third codon base"),
    DesignedVariant("A", 14, "A", None, "plus strand, synonymous CTG>CTA"),
    DesignedVariant("A", 23, "A", None, "plus strand, stop gained TGC>TGA"),
    DesignedVariant("B", 4, "G", "D2G", "minus strand, simple codon"),
    DesignedVariant("B", 10, "C", "F4S", "minus strand, codon split by intron (exon 1 side)"),
    DesignedVariant("B", 11, "A", "F4L", "minus strand, codon split by intron (exon 2 side)"),
    DesignedVariant("B", 15, "G", "K6E", "minus strand, exon 2"),
]

# A multi-allelic site in gene A codon 6 (GAA): ALT C gives GAC (E6D, missense),
# ALT G gives GAG (synonymous). Position = CDS index 17.
MULTIALLELIC_POS = _cds_to_genome("A", 17)
MULTIALLELIC_ALTS = ("C", "G")
INTERGENIC_POS = 20


def _gff3_lines() -> list[str]:
    def row(ftype: str, start: int, end: int, strand: str, phase: str, attrs: str) -> str:
        return "\t".join([CHROM, "test", ftype, str(start), str(end), ".", strand, phase, attrs])

    lines = ["##gff-version 3", f"##sequence-region {CHROM} 1 {len(GENOME)}"]
    lines.append(row("gene", 41, 97, "+", ".", "ID=gene:GA;Name=GENEA;biotype=protein_coding"))
    lines.append(
        row(
            "mRNA",
            41,
            97,
            "+",
            ".",
            "ID=transcript:TXA;Parent=gene:GA;biotype=protein_coding;tag=basic,Ensembl_canonical",
        )
    )
    for start, end, phase in GENE_A_CDS:
        lines.append(row("exon", start, end, "+", ".", "Parent=transcript:TXA"))
        lines.append(row("CDS", start, end, "+", str(phase), "ID=CDS:PA;Parent=transcript:TXA"))
    lines.append(row("gene", 138, 194, "-", ".", "ID=gene:GB;Name=GENEB;biotype=protein_coding"))
    lines.append(
        row(
            "mRNA",
            138,
            194,
            "-",
            ".",
            "ID=transcript:TXB;Parent=gene:GB;biotype=protein_coding;tag=basic",
        )
    )
    for start, end, phase in GENE_B_CDS:
        lines.append(row("exon", start, end, "-", ".", "Parent=transcript:TXB"))
        lines.append(row("CDS", start, end, "-", str(phase), "ID=CDS:PB;Parent=transcript:TXB"))
    return lines


def ncbi_gff3_lines(*, mitochondrial: bool = False, select: tuple[str, ...] = ()) -> list[str]:
    """The same two genes in the NCBI RefSeq layout (``gene-``/``rna-``/``cds-`` IDs,
    ``gene_biotype`` on genes, no biotype on mRNAs). Transcripts are ``rna-XM_GENEA.1`` and
    ``rna-XM_GENEB.1``; ``select`` lists genes whose mRNA is tagged ``RefSeq Select``."""

    def row(source: str, ftype: str, start: int, end: int, strand: str, phase: str, attrs: str):
        return "\t".join([CHROM, source, ftype, str(start), str(end), ".", strand, phase, attrs])

    genome = "mitochondrion" if mitochondrial else "chromosome"
    lines = [
        "##gff-version 3",
        f"##sequence-region {CHROM} 1 {len(GENOME)}",
        row("RefSeq", "region", 1, len(GENOME), "+", ".", f"ID={CHROM}:1..234;genome={genome}"),
    ]
    for gene, strand, cds, (start, end) in (
        ("GENEA", "+", GENE_A_CDS, (41, 97)),
        ("GENEB", "-", GENE_B_CDS, (138, 194)),
    ):
        rna, protein = f"XM_{gene}.1", f"XP_{gene}.1"
        tag = ";tag=RefSeq Select" if gene in select else ""
        lines.append(
            row(
                "Gnomon",
                "gene",
                start,
                end,
                strand,
                ".",
                f"ID=gene-{gene};Name={gene};gbkey=Gene;gene={gene};gene_biotype=protein_coding",
            )
        )
        lines.append(
            row(
                "Gnomon",
                "mRNA",
                start,
                end,
                strand,
                ".",
                f"ID=rna-{rna};Parent=gene-{gene};Name={rna};gbkey=mRNA;transcript_id={rna}{tag}",
            )
        )
        for i, (s, e, _phase) in enumerate(cds, start=1):
            lines.append(
                row("Gnomon", "exon", s, e, strand, ".", f"ID=exon-{rna}-{i};Parent=rna-{rna}")
            )
        for s, e, phase in cds:
            lines.append(
                row(
                    "Gnomon",
                    "CDS",
                    s,
                    e,
                    strand,
                    str(phase),
                    f"ID=cds-{protein};Parent=rna-{rna};gbkey=CDS;protein_id={protein}",
                )
            )
    return lines


def _vcf_records() -> list[tuple[int, str, str]]:
    records = [(v.pos, v.ref, v.alt) for v in DESIGNED_VARIANTS]
    records.append((MULTIALLELIC_POS, GENOME[MULTIALLELIC_POS - 1], ",".join(MULTIALLELIC_ALTS)))
    intergenic_ref = GENOME[INTERGENIC_POS - 1]
    records.append((INTERGENIC_POS, intergenic_ref, "A" if intergenic_ref != "A" else "C"))
    return sorted(records)


@dataclass(frozen=True)
class SyntheticFiles:
    fasta: Path
    gff3: Path
    vcf: Path


def write_synthetic(directory: Path) -> SyntheticFiles:
    """Write FASTA, GFF3 and VCF (with one sample) for the synthetic genome."""
    directory.mkdir(parents=True, exist_ok=True)
    fasta = directory / "genome.fa"
    fasta.write_text(
        f">{CHROM}\n" + "\n".join(GENOME[i : i + 60] for i in range(0, 234, 60)) + "\n"
    )

    gff3 = directory / "annotation.gff3"
    gff3.write_text("\n".join(_gff3_lines()) + "\n")

    vcf = directory / "variants.vcf"
    header = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={CHROM},length={len(GENOME)}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "\t".join(
            ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT", "DOG1"]
        ),
    ]
    body = [
        "\t".join([CHROM, str(pos), ".", ref, alt, ".", "PASS", ".", "GT", "0/1"])
        for pos, ref, alt in _vcf_records()
    ]
    vcf.write_text("\n".join(header + body) + "\n")
    return SyntheticFiles(fasta=fasta, gff3=gff3, vcf=vcf)


# A mitochondrial sequence with one gene that only makes sense in the vertebrate
# mitochondrial code (NCBI table 2): ATA = M (start), TGA = W, AGA = stop.
MT_CHROM = "MT"
MT_GENOME = "GCGCGCGCGC" + "ATAAAATGACTTTGGAGA" + "GCGCGCGCGC"  # CDS 11..28
MT_PROTEIN = "MKWLW"  # table 2; the standard code gives I K * L W R
#: (position, ref, alt, expected change in table 2)
MT_VARIANTS = [(15, "A", "C", "K2T"), (17, "T", "C", "W3R"), (20, "C", "A", "L4I")]


def write_synthetic_with_mitochondrion(directory: Path) -> SyntheticFiles:
    """The synthetic genome plus an ``MT`` sequence carrying gene GM (transcript TXM)."""
    files = write_synthetic(directory)
    with files.fasta.open("a") as fasta:
        fasta.write(f">{MT_CHROM}\n{MT_GENOME}\n")
    rows = [
        [MT_CHROM, "test", "gene", "11", "28", ".", "+", ".",
         "ID=gene:GM;Name=GENEM;biotype=protein_coding"],
        [MT_CHROM, "test", "mRNA", "11", "28", ".", "+", ".",
         "ID=transcript:TXM;Parent=gene:GM;biotype=protein_coding;tag=Ensembl_canonical"],
        [MT_CHROM, "test", "exon", "11", "28", ".", "+", ".", "Parent=transcript:TXM"],
        [MT_CHROM, "test", "CDS", "11", "28", ".", "+", "0", "ID=CDS:PM;Parent=transcript:TXM"],
    ]  # fmt: skip
    with files.gff3.open("a") as gff3:
        gff3.write("".join("\t".join(row) + "\n" for row in rows))
    lines = files.vcf.read_text().splitlines()
    contig = f"##contig=<ID={MT_CHROM},length={len(MT_GENOME)}>"
    lines.insert(next(i for i, line in enumerate(lines) if line.startswith("#CHROM")), contig)
    lines += [
        "\t".join([MT_CHROM, str(pos), ".", ref, alt, ".", "PASS", ".", "GT", "1"])
        for pos, ref, alt, _change in MT_VARIANTS
    ]
    files.vcf.write_text("\n".join(lines) + "\n")
    return files
