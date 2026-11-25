import csv
import logging
import math

import pysam
import pytest
from synthetic import DESIGNED_VARIANTS, INTERGENIC_POS, MULTIALLELIC_POS

from omnimissense import pipeline
from omnimissense.cache import ScoreCache
from omnimissense.consequences import Consequence
from omnimissense.pipeline import (
    ReferenceMismatchError,
    VariantStatus,
    annotate,
    classify,
    precompute,
)
from omnimissense.proteome import Protein, ProteinStatus

N_MISSENSE = sum(1 for v in DESIGNED_VARIANTS if v.expected) + 1  # + E6D at the multi-allelic site


def _consequence(ref_aa="K", alt_aa="R", pos=2) -> Consequence:
    return Consequence(
        "missense", "G", "T", "protein_coding", "+", pos, ref_aa, alt_aa, 1, "A", "G"
    )


@pytest.mark.parametrize(
    ("consequence", "protein", "status"),
    [
        (_consequence(), Protein("T", "MKR", ProteinStatus.OK), VariantStatus.OK),
        (_consequence(), Protein("T", "MAR", ProteinStatus.OK), VariantStatus.REF_AA_MISMATCH),
        (_consequence(pos=9), Protein("T", "MKR", ProteinStatus.OK), VariantStatus.REF_AA_MISMATCH),
        (_consequence("KR", "WW"), Protein("T", "MKR", ProteinStatus.OK), VariantStatus.MULTI_AA),
        (
            _consequence("X", "R"),
            Protein("T", "MXR", ProteinStatus.OK),
            VariantStatus.NONSTANDARD_AA,
        ),
        (
            _consequence(),
            Protein("T", "", ProteinStatus.MITOCHONDRIAL),
            VariantStatus.MITOCHONDRIAL,
        ),
        (
            _consequence(),
            Protein("T", "MK*R", ProteinStatus.INTERNAL_STOP),
            VariantStatus.PROTEIN_UNAVAILABLE,
        ),
        (_consequence(), None, VariantStatus.PROTEIN_UNAVAILABLE),
    ],
)
def test_classify(consequence, protein, status) -> None:
    assert classify(consequence, protein) is status


def _run(synthetic, tmp_path, scorer, *, masked=True, vcf=None):
    cache = ScoreCache(tmp_path / "cache", "tiny")
    result = annotate(
        vcf or synthetic.vcf,
        synthetic.fasta,
        synthetic.gff3,
        tmp_path / "out.vcf.gz",
        tmp_path / "out.tsv",
        cache=cache,
        scorer_factory=lambda: scorer,
        model_name="tiny",
        masked=masked,
    )
    return result, cache


def test_annotate_end_to_end(synthetic, tmp_path, tiny_scorer) -> None:
    result, _ = _run(synthetic, tmp_path, tiny_scorer)
    assert result.n_records == len(DESIGNED_VARIANTS) + 2
    assert result.status_counts == {"ok": N_MISSENSE}
    assert all(
        math.isfinite(v.wt_marginal) and math.isfinite(v.masked_marginal) for v in result.variants
    )

    with pysam.VariantFile(str(tmp_path / "out.vcf.gz")) as reader:
        for tag in ("OMS", "OMS_WT", "OMS_MM", "OMS_REF_MISMATCH", "BCSQ"):
            assert tag in reader.header.info
        assert list(reader.header.samples) == ["DOG1"]
        records = {r.pos: r for r in reader}

    k2r = records[45]
    assert k2r.samples["DOG1"]["GT"] == (0, 1)
    [entry] = k2r.info["OMS"]
    fields = entry.split("|")
    assert fields[:7] == ["G", "GENEA", "TXA", "1", "2", "K", "R"]
    assert fields[-1] == "ok"
    assert k2r.info["OMS_WT"][0] == pytest.approx(float(fields[7]), abs=1e-3)

    multi = records[MULTIALLELIC_POS]
    assert multi.info["OMS_MM"][0] is not None
    assert multi.info["OMS_MM"][1] is None  # second ALT is synonymous
    assert "OMS" not in records[85].info  # synonymous
    assert "OMS" not in records[INTERGENIC_POS].info

    with (tmp_path / "out.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == N_MISSENSE
    assert {r["status"] for r in rows} == {"ok"}
    assert {(r["transcript"], r["ref_aa"] + r["protein_pos"] + r["alt_aa"]) for r in rows} >= {
        ("TXB", "D2G"),
        ("TXB", "F4L"),
    }


def test_second_run_reads_scores_from_cache(synthetic, tmp_path, tiny_scorer) -> None:
    first, _ = _run(synthetic, tmp_path, tiny_scorer)

    def no_model():
        raise AssertionError("the model should not be loaded when everything is cached")

    second = annotate(
        synthetic.vcf,
        synthetic.fasta,
        synthetic.gff3,
        tmp_path / "again.vcf",
        tmp_path / "again.tsv",
        cache=ScoreCache(tmp_path / "cache", "tiny"),
        scorer_factory=no_model,
        model_name="tiny",
        masked=True,
    )
    assert [v.masked_marginal for v in second.variants] == [
        v.masked_marginal for v in first.variants
    ]


def test_no_masked_option(synthetic, tmp_path, tiny_scorer) -> None:
    result, _ = _run(synthetic, tmp_path, tiny_scorer, masked=False)
    assert all(v.masked_marginal is None and v.wt_marginal is not None for v in result.variants)


def test_ref_allele_mismatch_is_flagged_not_ignored(
    synthetic, tmp_path, tiny_scorer, caplog
) -> None:
    lines = synthetic.vcf.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("1\t45\t"):
            lines[i] = line.replace("\tA\tG\t", "\tC\tG\t")  # genome has A
    bad_vcf = tmp_path / "bad.vcf"
    bad_vcf.write_text("\n".join(lines) + "\n")

    with caplog.at_level(logging.WARNING):
        result, _ = _run(synthetic, tmp_path, tiny_scorer, vcf=bad_vcf)
    assert result.n_ref_mismatches == 1
    assert "REF allele mismatch at 1:45" in caplog.text
    with pysam.VariantFile(str(tmp_path / "out.vcf.gz")) as reader:
        records = {r.pos: r for r in reader}
    assert records[45].info["OMS_REF_MISMATCH"] is True
    assert "OMS" not in records[45].info
    assert "OMS" in records[82].info  # the rest is still annotated


def test_wrong_assembly_is_an_error(synthetic, tmp_path, tiny_scorer) -> None:
    renamed = tmp_path / "chr.vcf"
    renamed.write_text(
        synthetic.vcf.read_text().replace("\n1\t", "\nchr1\t").replace("ID=1,", "ID=chr1,")
    )
    with pytest.raises(ReferenceMismatchError, match="same assembly"):
        _run(synthetic, tmp_path, tiny_scorer, vcf=renamed)


def test_ref_amino_acid_mismatch_is_reported(synthetic, tmp_path, tiny_scorer, monkeypatch, caplog):
    real_build = pipeline.build_proteome

    def corrupted(*args, **kwargs):
        proteins = real_build(*args, **kwargs)
        proteins["TXA"] = Protein("TXA", "MAAWLEHC", ProteinStatus.OK)  # K2 and R3 changed
        return proteins

    monkeypatch.setattr(pipeline, "build_proteome", corrupted)
    with caplog.at_level(logging.WARNING):
        result, _ = _run(synthetic, tmp_path, tiny_scorer)
    mismatched = [v for v in result.variants if v.status is VariantStatus.REF_AA_MISMATCH]
    assert {v.consequence.protein_pos for v in mismatched} == {2, 3}
    assert all(v.wt_marginal is None for v in mismatched)
    assert "Reference amino acid mismatch" in caplog.text


def test_precompute_fills_the_cache(synthetic, tmp_path, tiny_scorer) -> None:
    cache = ScoreCache(tmp_path / "cache", "tiny")
    result = precompute(
        synthetic.fasta,
        synthetic.gff3,
        cache=cache,
        scorer_factory=lambda: tiny_scorer,
        progress=False,
    )
    assert (result.n_unique_proteins, result.n_scored) == (2, 2)
    assert result.residues_scored == 16
    again = precompute(
        synthetic.fasta, synthetic.gff3, cache=cache, scorer_factory=lambda: None, progress=False
    )
    assert (again.n_already_cached, again.n_scored) == (2, 0)


def test_annotating_an_annotated_vcf_replaces_old_values(synthetic, tmp_path, tiny_scorer) -> None:
    first, _ = _run(synthetic, tmp_path, tiny_scorer)
    again = tmp_path / "again"
    again.mkdir()
    second, _ = _run(synthetic, again, tiny_scorer, vcf=tmp_path / "out.vcf.gz")
    assert second.status_counts == first.status_counts
    with pysam.VariantFile(str(again / "out.vcf.gz")) as reader:
        assert sum(1 for line in str(reader.header).splitlines() if "ID=OMS," in line) == 1
        k2r = next(r for r in reader if r.pos == 45)
    assert len(k2r.info["OMS"]) == 1
