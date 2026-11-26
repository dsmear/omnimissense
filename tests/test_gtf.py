import pytest
from synthetic import DESIGNED_VARIANTS, PROTEIN_A, PROTEIN_B, gtf_lines

from omnimissense.cache import ScoreCache
from omnimissense.gff import CdsSegment, _merge_cds, is_gtf, read_transcripts
from omnimissense.pipeline import annotate
from omnimissense.proteome import build_proteome

N_MISSENSE = sum(1 for v in DESIGNED_VARIANTS if v.expected) + 1


def _write(tmp_path, lines, name="annotation.gtf"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def test_gtf_is_detected(synthetic, tmp_path) -> None:
    assert is_gtf(_write(tmp_path, gtf_lines()))
    assert not is_gtf(synthetic.gff3)


@pytest.mark.parametrize("with_parents", [True, False])
def test_gtf_gives_the_same_transcripts_and_proteins(synthetic, tmp_path, with_parents) -> None:
    gtf = _write(tmp_path, gtf_lines(with_parents=with_parents))
    transcripts = read_transcripts(gtf, synthetic.fasta)
    assert set(transcripts) == {"TXA", "TXB"}
    # The stop codon is merged back into the CDS, as in the GFF3 of the same genes.
    assert transcripts["TXA"].cds_in_transcript_order() == [
        CdsSegment(41, 47, 0),
        CdsSegment(78, 97, 2),
    ]
    assert transcripts["TXB"].cds_in_transcript_order() == [
        CdsSegment(184, 194, 0),
        CdsSegment(138, 153, 1),
    ]
    assert (transcripts["TXA"].gene_id, transcripts["TXA"].gene_name) == ("GA", "GENEA")
    assert transcripts["TXA"].biotype == "protein_coding"
    proteins = build_proteome(synthetic.fasta, transcripts)
    assert (proteins["TXA"].sequence, proteins["TXB"].sequence) == (PROTEIN_A, PROTEIN_B)
    if with_parents:  # tag "Ensembl_canonical" kept; gene B has none: longest CDS
        assert transcripts["TXA"].canonical and transcripts["TXB"].canonical


def test_annotate_with_gtf_matches_gff3(synthetic, tmp_path, tiny_scorer) -> None:
    results = {}
    for name, annotation in (("gtf", _write(tmp_path, gtf_lines())), ("gff3", synthetic.gff3)):
        result = annotate(
            synthetic.vcf,
            synthetic.fasta,
            annotation,
            tmp_path / f"{name}.vcf.gz",
            tmp_path / f"{name}.tsv",
            cache=ScoreCache(tmp_path / "cache", "tiny"),
            scorer_factory=lambda: tiny_scorer,
            model_name="tiny",
        )
        results[name] = {
            (v.pos, v.alt, v.consequence.transcript_id, v.consequence.protein_pos, v.wt_marginal)
            for v in result.variants
            if v.status.value == "ok"
        }
    assert len(results["gtf"]) == N_MISSENSE
    assert results["gtf"] == results["gff3"]


def test_merge_cds_with_a_split_stop_codon_and_phases() -> None:
    # + strand: stop codon split over two exons (2 bases + 1 base)
    merged = _merge_cds([(1, 10, 0), (21, 28, 2), (29, 30, None), (41, 41, None)], "+")
    assert merged == [(1, 10, 0), (21, 30, 2), (41, 41, 1)]  # 20 bases before: skip 1
    # - strand: transcript order is right to left; phases follow from the 5' segment
    merged = _merge_cds([(41, 50, 1), (21, 27, 0), (18, 20, None)], "-")
    assert merged == [(41, 50, 1), (18, 27, 0)]
