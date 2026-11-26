"""Codons recoded by ``transl_except``: selenocysteine (U) and pyrrolysine (O)."""

import pytest

from omnimissense.cache import ScoreCache
from omnimissense.fasta import prepare_fasta
from omnimissense.gff import read_transcripts
from omnimissense.pipeline import annotate
from omnimissense.proteome import ProteinStatus, build_proteome

CDS = "ATGAAATGAAAACTGTAA"  # M K (TGA = Sec) K L stop
GENOME_PLUS = "CCCCC" + CDS + "CCCCC"  # CDS at 6..23, Sec codon at 12..14
COMPLEMENT = str.maketrans("ACGT", "TGCA")
GENOME_MINUS = GENOME_PLUS.translate(COMPLEMENT)[::-1]  # same CDS on the - strand, Sec 15..17


def _files(tmp_path, strand, transl_except):
    genome = GENOME_PLUS if strand == "+" else GENOME_MINUS
    fasta = tmp_path / "genome.fa"
    fasta.write_text(f">1\n{genome}\n")
    attrs = "ID=cds-WP_S.1;Parent=gene-S;protein_id=WP_S.1;transl_table=11"
    if transl_except:
        pos = "12..14" if strand == "+" else "complement(15..17)"
        attrs += f";transl_except=(pos:{pos}%2Caa:Sec)"
    gff3 = tmp_path / "annotation.gff3"
    gff3.write_text(
        "##gff-version 3\n"
        f"1\tt\tgene\t6\t23\t.\t{strand}\t.\tID=gene-S;Name=selA;gene_biotype=protein_coding\n"
        f"1\tt\tCDS\t6\t23\t.\t{strand}\t0\t{attrs}\n"
    )
    return prepare_fasta(fasta), gff3


@pytest.mark.parametrize("strand", ["+", "-"])
def test_selenocysteine_codon_is_read_as_u(tmp_path, strand) -> None:
    fasta, gff3 = _files(tmp_path, strand, transl_except=True)
    transcripts = read_transcripts(gff3, fasta)
    sec = (12, 14, "U") if strand == "+" else (15, 17, "U")
    assert transcripts["tx-WP_S.1"].recoded == [sec]
    protein = build_proteome(fasta, transcripts)["tx-WP_S.1"]
    assert (protein.sequence, protein.status) == ("MKUKL", ProteinStatus.OK)


def test_without_transl_except_the_tga_is_an_internal_stop(tmp_path) -> None:
    fasta, gff3 = _files(tmp_path, "+", transl_except=False)
    protein = build_proteome(fasta, read_transcripts(gff3, fasta))["tx-WP_S.1"]
    assert (protein.sequence, protein.status) == ("MK*KL", ProteinStatus.INTERNAL_STOP)


def test_variants_after_the_selenocysteine_are_scored(tmp_path, tiny_scorer) -> None:
    fasta, gff3 = _files(tmp_path, "+", transl_except=True)
    vcf = tmp_path / "variants.vcf"
    records = [(10, "A", "C"), (13, "G", "C"), (16, "A", "C")]  # K2T, Sec codon, K4T
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        + "".join(f"1\t{p}\t.\t{r}\t{a}\t.\t.\t.\n" for p, r, a in records)
    )
    result = annotate(
        vcf,
        fasta,
        gff3,
        tmp_path / "out.vcf.gz",
        tmp_path / "out.tsv",
        cache=ScoreCache(tmp_path / "cache", "tiny"),
        scorer_factory=lambda: tiny_scorer,
        model_name="tiny",
    )
    calls = {
        f"{v.consequence.ref_aa}{v.consequence.protein_pos}{v.consequence.alt_aa}": v.status.value
        for v in result.variants
    }
    # csq calls the change of the Sec codon "stop_lost", not missense: never scored.
    assert calls == {"K2T": "ok", "K4T": "ok"}
