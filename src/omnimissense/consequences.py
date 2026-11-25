"""Protein-level consequences of VCF variants, computed with bcftools csq.

bcftools is used through the copy bundled in pysam, so no external installation is needed.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pysam
import pysam.bcftools
import pysam.utils

_AA_CHANGE = re.compile(r"^(?P<pos>\d+)(?P<ref>[A-Z*]+)>(?P<alt_pos>\d+)(?P<alt>[A-Z*]+)$")
_DNA_CHANGE = re.compile(r"^(?P<pos>\d+)(?P<ref>[A-Za-z]+)>(?P<alt>[A-Za-z]+)$")

_UNUSABLE_GFF_HINT = (
    "bcftools csq found no usable transcripts in the GFF3. It reads Ensembl and NCBI RefSeq "
    "GFF3 (gene -> mRNA/transcript -> CDS). Other layouts must be converted first: GTF to "
    "GFF3 with e.g. AGAT or gffread; prokaryotic GFF3 whose CDS hang directly from genes "
    "needs a transcript level (bcftools' misc/gff2gff.py, or gbk2bcfgff from a GenBank file); "
    "bcftools' misc/gff2gff fills in IDs and biotypes from GENCODE-style attributes."
)


class CsqError(RuntimeError):
    """bcftools csq failed."""


@dataclass(frozen=True)
class Consequence:
    """One protein-changing entry of INFO/BCSQ."""

    consequence: str  # e.g. "missense", or combined with "&"
    gene: str
    transcript_id: str
    biotype: str
    strand: str
    protein_pos: int  # 1-based
    ref_aa: str
    alt_aa: str
    dna_pos: int
    dna_ref: str
    dna_alt: str

    @property
    def is_missense(self) -> bool:
        return "missense" in self.consequence.split("&")


def parse_bcsq(entry: str) -> Consequence | None:
    """Parse one INFO/BCSQ entry.

    Format: ``Consequence|gene|transcript|biotype|strand|amino_acid_change|dna_change``,
    e.g. ``missense|GENEA|TXA|protein_coding|+|2K>2R|45A>G``. Returns ``None`` for entries
    without a parsable amino-acid change (UTR, intron, splice region, synonymous, ``@pos``
    back-references, combined haplotype changes).
    """
    fields = entry.split("|")
    if len(fields) < 7:
        return None
    consequence, gene, transcript_id, biotype, strand, aa_change, dna_change = fields[:7]
    aa = _AA_CHANGE.match(aa_change)
    dna = _DNA_CHANGE.match(dna_change)
    if aa is None or dna is None or aa["pos"] != aa["alt_pos"]:
        return None
    return Consequence(
        consequence=consequence,
        gene=gene,
        transcript_id=transcript_id,
        biotype=biotype,
        strand=strand,
        protein_pos=int(aa["pos"]),
        ref_aa=aa["ref"],
        alt_aa=aa["alt"],
        dna_pos=int(dna["pos"]),
        dna_ref=dna["ref"].upper(),
        dna_alt=dna["alt"].upper(),
    )


def run_csq(
    vcf: Path,
    fasta: Path,
    gff3: Path,
    output: Path,
    threads: int = 0,
    dump_gff: Path | None = None,
) -> str:
    """Annotate ``vcf`` with INFO/BCSQ; writes uncompressed BCF if ``output`` ends in .bcf,
    otherwise BGZF-compressed VCF.

    Runs in local mode (``-l``: one record at a time, no phasing needed) on all ALT alleles
    regardless of genotypes (``-s -``); sample columns are kept unchanged. With ``dump_gff``,
    the annotation as csq parsed it is also written there (gzip GFF3, ``--dump-gff``).
    Returns bcftools' log messages.
    """
    output_type = "-Ou" if output.suffix == ".bcf" else "-Oz"
    args = ["-f", str(fasta), "-g", str(gff3), "-l", "-s", "-", output_type, "-o", str(output)]
    if threads:
        args += ["--threads", str(threads)]
    if dump_gff is not None:
        args += ["--dump-gff", str(dump_gff)]
    try:
        pysam.bcftools.csq(*args, str(vcf), catch_stdout=False)
    except pysam.utils.SamtoolsError as exc:
        if "No usable transcripts" in str(exc):
            raise CsqError(f"{_UNUSABLE_GFF_HINT}\nbcftools csq said: {exc}") from exc
        raise CsqError(f"bcftools csq failed: {exc}") from exc
    return pysam.bcftools.csq.get_messages()


def dump_annotation(fasta: Path, gff3: Path, output: Path) -> None:
    """Write the annotation as bcftools csq parses it (``--dump-gff``) to ``output`` (gzip).

    csq is run on an empty VCF, so nothing is called. The VCF header lists the FASTA's
    sequences: bcftools 1.24 crashes on an empty VCF without ``##contig`` lines.
    """
    with pysam.FastaFile(str(fasta)) as genome:
        contigs = [f"##contig=<ID={name}>" for name in genome.references]
    with tempfile.TemporaryDirectory(prefix="omnimissense-") as tmp:
        vcf = Path(tmp) / "empty.vcf"
        header = ["##fileformat=VCFv4.2", *contigs, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"]
        vcf.write_text("\n".join(header) + "\n")
        run_csq(vcf, fasta, gff3, Path(tmp) / "empty.bcf", dump_gff=output)


def missense_consequences(record: pysam.VariantRecord) -> list[tuple[int, Consequence]]:
    """Missense consequences of a csq-annotated record, as (ALT index, consequence) pairs."""
    if "BCSQ" not in record.info:
        return []
    alts = record.alts or ()
    found = []
    for entry in record.info["BCSQ"]:
        consequence = parse_bcsq(entry)
        if consequence is None or not consequence.is_missense:
            continue
        if consequence.dna_pos != record.pos or consequence.dna_ref != record.ref.upper():
            continue
        upper_alts = [alt.upper() for alt in alts]
        if consequence.dna_alt not in upper_alts:
            continue
        found.append((upper_alts.index(consequence.dna_alt), consequence))
    return found
