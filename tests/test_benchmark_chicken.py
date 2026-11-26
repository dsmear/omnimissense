import sys
from pathlib import Path

import numpy as np
import pysam
import pytest
from synthetic import CHROM

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from chicken_omia import (
    POPULATION_INDEX,
    POPULATION_VCF,
    auroc_row,
    extract_common_snvs,
    fetch_refseq_proteins,
    in_protein_percentile,
    index_url,
    map_omia_via_refseq,
    merge_spans,
    population_source,
    protein_accession,
    write_cds_bed,
)
from dog_omia import read_omia

from omnimissense.gff import CdsSegment, Transcript
from omnimissense.proteome import Protein, ProteinStatus
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
        ("50", "G", "C", "PASS", 30),  # common, in the merged span but outside the regions
        ("65", "A", "T", "PASS", 30),  # common, in the third region
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
    # two overlapping regions (a variant must be written once), a third one read in the same
    # span, and a sequence the VCF lacks
    bed.write_text(f"{CHROM}\t5\t28\n{CHROM}\t25\t40\n{CHROM}\t60\t70\nMT\t0\t100\n")
    output = tmp_path / "common.vcf"
    assert extract_common_snvs(str(vcf), bed, synthetic.fasta, output, 0.05, workers=2) == 3
    assert (tmp_path / "common.parts" / f"{CHROM}.spans.bed").read_text() == f"{CHROM}\t5\t70\n"
    with pysam.VariantFile(str(output)) as result:
        assert [(r.pos, r.ref, r.alts) for r in result] == [
            (10, "A", ("G",)),
            (30, "C", ("A",)),
            (65, "A", ("T",)),
        ]
        assert list(result.header.contigs) == [CHROM]  # the FASTA's sequences


def test_merge_spans() -> None:
    assert merge_spans([(100, 200), (0, 50), (150, 300), (40_000, 40_100)], max_gap=10) == [
        (0, 50),
        (100, 300),
        (40_000, 40_100),
    ]
    assert merge_spans([(0, 50), (100, 200)], max_gap=1000) == [(0, 200)]


def test_population_source_keeps_local_paths(tmp_path) -> None:
    assert population_source("data/x.vcf.gz", None, tmp_path) == "data/x.vcf.gz"


def test_index_url_of_the_default_eva_vcf() -> None:
    assert index_url(POPULATION_VCF, None) == POPULATION_INDEX  # EVA names it .vcf.csi
    assert POPULATION_INDEX.endswith(".vcf.csi")
    assert index_url("https://x/y.vcf.gz", None) == "https://x/y.vcf.gz.csi"
    assert index_url("https://x/y.vcf.gz", "https://x/y.tbi") == "https://x/y.tbi"


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


def test_protein_accession() -> None:
    assert protein_accession("NP_989476.1:p.(E89K)") == "NP_989476.1"
    assert protein_accession("XP_015155859.1:p.(V206_S210del)") == "XP_015155859.1"
    assert protein_accession("p.(H215P)") is None
    assert protein_accession("") is None


def test_fetch_refseq_proteins_reads_the_cache_without_network(tmp_path) -> None:
    cache = tmp_path / "refseq.fa"
    cache.write_text(">NP_1.1 some protein [Gallus gallus]\nMKT\nAY*\n>NP_2.1 other\nMA\n")
    assert fetch_refseq_proteins({"NP_1.1"}, cache) == {"NP_1.1": "MKTAY"}


def _gene(transcripts: dict[str, str]) -> tuple[dict, dict]:
    txs = [
        Transcript(
            transcript_id=tid,
            gene_id="G1",
            seqid=CHROM,
            strand="+",
            biotype="protein_coding",
            canonical=(i == 0),
            gene_name="gene1",
            cds=[CdsSegment(1, 3 * len(seq) + 3, 0)],
        )
        for i, (tid, seq) in enumerate(transcripts.items())
    ]
    proteins = {tid: Protein(tid, seq, ProteinStatus.OK) for tid, seq in transcripts.items()}
    return {"GENE1": txs}, proteins


def _record(protein: str, category: str = "trait") -> dict[str, str]:
    return {"id": "1", "gene": "GENE1", "protein": protein, "category": category}


def test_map_omia_via_refseq_carries_positions_through_the_alignment() -> None:
    pytest.importorskip("Bio", reason="benchmark dependencies: uv sync --all-groups")
    # Ensembl's protein has 3 extra N-terminal residues: RefSeq Y5 is Ensembl Y8
    genes, proteins = _gene({"T1": "MGGMKTAYIAKQRQISFVK"})
    refseq = {"NP_1.1": "MKTAYIAKQRQISFVK"}
    variants, restored, counts = map_omia_via_refseq(
        [_record("NP_1.1:p.(Y5D)"), _record("NP_1.1:p.(Y5D)"), _record("NP_1.1:p.(W5D)")],
        genes, proteins, refseq,
    )  # fmt: skip
    assert [(v.transcript_id, v.position, v.ref, v.alt) for v in variants] == [("T1", 8, "Y", "D")]
    assert restored == {}
    assert counts["mapped"] == 2 and counts["mapped_unique"] == 1
    assert counts["refseq_residue_mismatch"] == 1  # RefSeq has neither W nor D at 5


def test_map_omia_via_refseq_restores_the_reference_residue() -> None:
    pytest.importorskip("Bio", reason="benchmark dependencies: uv sync --all-groups")
    # the genome (and the RefSeq protein) carry the variant allele K at RefSeq position 5
    genes, proteins = _gene({"T1": "MGGMKTAKIAKQRQISFVK"})
    refseq = {"NP_1.1": "MKTAKIAKQRQISFVK"}
    variants, restored, counts = map_omia_via_refseq(
        [_record("NP_1.1:p.(E5K)")], genes, proteins, refseq
    )
    assert [(v.transcript_id, v.position, v.ref, v.alt) for v in variants] == [
        ("T1#E8", 8, "E", "K")
    ]
    assert restored == {"T1#E8": "MGGMKTAEIAKQRQISFVK"}
    assert counts["reference_genome_carries_alt"] == 1


def test_map_omia_via_refseq_without_accession_uses_ensembl_numbering() -> None:
    pytest.importorskip("Bio", reason="benchmark dependencies: uv sync --all-groups")
    genes, proteins = _gene({"T1": "MKTAYIAK"})
    variants, _, counts = map_omia_via_refseq(
        [_record("p.(Y5D)"), _record("p.(K5D)"), _record("NP_9.1:p.(Y5D)")], genes, proteins, {}
    )
    assert [(v.position, v.ref, v.alt) for v in variants] == [(5, "Y", "D")]
    assert counts["ensembl_residue_mismatch"] == 1
    assert counts["refseq_protein_not_found"] == 1
