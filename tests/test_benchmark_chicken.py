import sys
from pathlib import Path

import numpy as np
import pysam
from synthetic import CHROM

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from chicken_omia import (
    auroc_row,
    extract_common_snvs,
    in_protein_percentile,
    population_source,
    write_cds_bed,
)
from dog_omia import read_omia

from omnimissense.gff import CdsSegment, Transcript
from omnimissense.sequences import AA_INDEX


def _population_vcf(path: Path) -> Path:
    """96 birds summarised in INFO (AC/AN), as in the EVA file; genotypes omitted."""
    rows = [
        ("10", "A", "G", "PASS", 20),  # AF 0.104: common
        ("12", "C", "T", "PASS", 4),  # AF 0.021: rare
        ("15", "G", "A", "LowQual", 50),  # filtered
        ("20", "T", "C,G", "PASS", 30),  # multi-allelic
        ("25", "AT", "A", "PASS", 40),  # indel
        ("30", "C", "A", "PASS", 10),  # AF 0.052: common
        ("150", "G", "T", "PASS", 60),  # outside the BED
    ]
    lines = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=LowQual,Description="Low quality">',
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count">',
        '##INFO=<ID=AN,Number=1,Type=Integer,Description="Allele number">',
        f"##contig=<ID={CHROM},length=1000>",
        "##contig=<ID=2,length=1000>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
    for pos, ref, alt, filt, ac in rows:
        counts = ",".join(str(ac) for _ in alt.split(","))
        lines.append(f"{CHROM}\t{pos}\t.\t{ref}\t{alt}\t.\t{filt}\tAC={counts};AN=192")
    plain = path.with_suffix("")
    plain.write_text("\n".join(lines) + "\n")
    pysam.tabix_index(str(plain), preset="vcf", force=True, csi=True)
    return Path(f"{plain}.gz")


def test_extract_common_snvs_keeps_common_pass_biallelic_snvs(synthetic, tmp_path) -> None:
    vcf = _population_vcf(tmp_path / "population.vcf.gz")
    bed = tmp_path / "cds.bed"
    # two overlapping regions (a variant must be written once) and a sequence the VCF lacks
    bed.write_text(f"{CHROM}\t5\t28\n{CHROM}\t25\t40\nMT\t0\t100\n")
    output = tmp_path / "common.vcf"
    assert extract_common_snvs(str(vcf), bed, synthetic.fasta, output, 0.05) == 2
    with pysam.VariantFile(str(output)) as result:
        assert [(r.pos, r.ref, r.alts) for r in result] == [(10, "A", ("G",)), (30, "C", ("A",))]
        assert list(result.header.contigs) == [CHROM]  # the FASTA's sequences


def test_population_source_keeps_local_paths(tmp_path) -> None:
    assert population_source("data/x.vcf.gz", None, tmp_path) == "data/x.vcf.gz"


def test_write_cds_bed(tmp_path) -> None:
    tx = Transcript(
        transcript_id="T1",
        gene_id="G1",
        seqid=CHROM,
        strand="+",
        biotype="protein_coding",
        cds=[CdsSegment(41, 47, 0), CdsSegment(11, 20, 0)],
    )
    bed = tmp_path / "cds.bed"
    write_cds_bed([tx], bed)
    assert bed.read_text() == f"{CHROM}\t10\t20\n{CHROM}\t40\t47\n"


def test_in_protein_percentile() -> None:
    sequence = "MK"
    matrix = np.tile(np.arange(20, dtype=float), (2, 1))  # score = column index
    matrix[0, AA_INDEX["M"]] = matrix[1, AA_INDEX["K"]] = 0.0  # wild type
    # 38 substitutions; scores below 1.0: column 0 (A) at both positions
    assert in_protein_percentile(matrix, sequence, 1.0) == 2 / 38


def test_auroc_row_scores_unscored_variants_last() -> None:
    labels = np.array([1, 1, 0, 0])
    table = {
        "good": np.array([3.0, 2.0, 1.0, 0.0]),
        "partial": np.array([3.0, -np.inf, 1.0, 0.0]),  # one causal variant not scored
    }
    row = auroc_row(labels, table, ("good", "partial"), (("good", "partial"),))
    assert (row["n_causal"], row["n_negative"]) == (2, 2)
    assert row["good"] == 1.0
    assert row["partial"] == 0.5
    assert row["diff_good_vs_partial"] == 0.5


def test_read_omia_takes_the_species_and_its_traits(tmp_path) -> None:
    csv_file = tmp_path / "omia.csv"
    csv_file.write_text(
        "OMIA Variant ID,Species Name,Gene,Variant Phenotype,Variant Effect,"
        "Pathogenicity Classification,p.\n"
        "1,chicken,MC1R,Extended black,missense,Not currently evaluated,p.(E92K)\n"
        "2,chicken,ABCA1,Wilah,missense,Not currently evaluated,p.(R10W)\n"
        "3,dog,SOD1,Degenerative myelopathy,missense,P,p.(E40K)\n"
    )
    records = read_omia(csv_file, species="chicken", traits={"Extended black"}, excluded=set())
    assert [(r["id"], r["category"]) for r in records] == [
        ("1", "trait"),
        ("2", "disease_unclassified"),
    ]
