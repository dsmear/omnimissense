"""Protein-level consequences of VCF variants, computed with bcftools csq.

bcftools is used through the copy bundled in pysam, so no external installation is needed.
Each csq call runs in its own Python process: pysam runs bcftools inside the calling
process, and csq keeps state from one call to the next (the genetic-code table, and the last
position read, which makes a later call fail with "the file is not sorted" when it starts
before that position on the same sequence).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pysam

from omnimissense.genetic_code import STANDARD
from omnimissense.genetic_code import genetic_code as check_genetic_code

_AA_CHANGE = re.compile(r"^(?P<pos>\d+)(?P<ref>[A-Z*]+)>(?P<alt_pos>\d+)(?P<alt>[A-Z*]+)$")
_DNA_CHANGE = re.compile(r"^(?P<pos>\d+)(?P<ref>[A-Za-z]+)>(?P<alt>[A-Za-z]+)$")

_UNUSABLE_GFF_HINT = (
    "bcftools csq found no usable transcripts in the annotation. It needs genes, transcripts "
    "and CDS (omnimissense reads Ensembl, NCBI RefSeq, Prokka and Bakta GFF3, and GTF; it adds "
    "a transcript to CDS directly under genes). Other layouts must be converted first, e.g. "
    "with AGAT; bcftools' misc/gff2gff fills in IDs and biotypes from GENCODE-style "
    "attributes."
)


class CsqError(RuntimeError):
    """bcftools csq failed."""


#: Runs ``bcftools csq`` (pysam) with the given arguments in a fresh interpreter; csq's log
#: goes to stderr, a failure exits with status 1.
_CSQ_RUNNER = """
import sys
import pysam.bcftools
import pysam.utils
try:
    pysam.bcftools.csq(*sys.argv[1:], catch_stdout=False)
except pysam.utils.SamtoolsError as exc:
    sys.stderr.write(str(exc))
    sys.exit(1)
sys.stderr.write(pysam.bcftools.csq.get_messages())
"""


def _bcftools_csq(*args: str) -> str:
    """Run ``bcftools csq`` in its own process; return its log messages."""
    process = subprocess.run(
        [sys.executable, "-c", _CSQ_RUNNER, *args], capture_output=True, text=True
    )
    if process.returncode != 0:
        if "No usable transcripts" in process.stderr:
            raise CsqError(f"{_UNUSABLE_GFF_HINT}\nbcftools csq said: {process.stderr}")
        raise CsqError(f"bcftools csq failed: {process.stderr}")
    return process.stderr


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

    @property
    def substitutions(self) -> list[tuple[int, str, str]]:
        """``(protein position, ref, alt)`` of every changed residue; an MNV over two codons
        (``4WL>4CM``) gives two. Empty if the change alters the protein length."""
        if len(self.ref_aa) != len(self.alt_aa):
            return []
        return [
            (self.protein_pos + i, ref, alt)
            for i, (ref, alt) in enumerate(zip(self.ref_aa, self.alt_aa, strict=True))
            if ref != alt
        ]


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
    genetic_code: int = STANDARD,
) -> str:
    """Annotate ``vcf`` with INFO/BCSQ; writes uncompressed BCF if ``output`` ends in .bcf,
    otherwise BGZF-compressed VCF.

    Runs in local mode (``-l``: one record at a time, no phasing needed) on all ALT alleles
    regardless of genotypes (``-s -``); sample columns are kept unchanged. With ``dump_gff``,
    the annotation as csq parsed it is also written there (gzip GFF3, ``--dump-gff``).
    ``genetic_code`` is an NCBI table (``-C``); for the standard code csq keeps its default
    (table 0: the standard codons, ATG as the only start). Returns bcftools' log messages.
    """
    output_type = "-Ou" if output.suffix == ".bcf" else "-Oz"
    args = ["-f", str(fasta), "-g", str(gff3), "-l", "-s", "-", output_type, "-o", str(output)]
    if threads:
        args += ["--threads", str(threads)]
    if dump_gff is not None:
        args += ["--dump-gff", str(dump_gff)]
    if genetic_code != STANDARD:
        check_genetic_code(genetic_code)  # a clear error instead of csq's
        args += ["-C", str(genetic_code)]
    return _bcftools_csq(*args, str(vcf))


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
