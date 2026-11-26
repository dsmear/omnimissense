import csv
import subprocess
import sys

import numpy as np
import pytest
from synthetic import DESIGNED_VARIANTS, PROTEIN_A

import omnimissense as om
from omnimissense.cache import ScoreCache
from omnimissense.pipeline import annotate as pipeline_annotate

N_MISSENSE = sum(1 for v in DESIGNED_VARIANTS if v.expected) + 1  # + E6D at the multi-allelic site


@pytest.fixture
def model(tiny_scorer, tmp_path) -> om.Model:
    return om.Model.from_scorer(tiny_scorer, name="tiny", cache_dir=tmp_path / "cache")


def test_import_is_light_and_exports_the_api() -> None:
    code = (
        "import sys, omnimissense; "
        "assert 'torch' not in sys.modules and 'omnimissense.api' not in sys.modules; "
        "omnimissense.Model; assert 'omnimissense.api' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert set(om.__all__) >= {"Model", "annotate", "precompute", "ProteinScores", "VariantStatus"}
    with pytest.raises(AttributeError):
        om.not_a_name  # noqa: B018


def test_score_protein_wild_type_marginal(model, tiny_scorer) -> None:
    scores = model.score_protein(PROTEIN_A.lower() + "*")  # case and terminal stop ignored
    assert scores.sequence == PROTEIN_A
    assert (len(scores), scores.method, scores.model, scores.n_windows) == (
        8,
        "wt_marginal",
        "tiny",
        1,
    )
    [(expected, _)] = tiny_scorer.wt_marginals([PROTEIN_A])
    np.testing.assert_allclose(scores.llr, expected, atol=1e-6)
    k = om.AMINO_ACIDS.index("R")
    assert scores["K2R"] == pytest.approx(float(expected[1, k]), abs=1e-6)
    assert scores["p.(Lys2Arg)"] == scores["p.K2R"] == scores.score(2, "R")
    assert scores["K2K"] == 0.0  # the reference residue
    with pytest.raises(ValueError, match="residue 2 of the protein is K, not R"):
        scores["R2K"]
    with pytest.raises(IndexError):
        scores["K99R"]
    with pytest.raises(ValueError, match="single amino-acid substitution"):
        scores["K2*"]


def test_score_protein_is_cached(model, tmp_path) -> None:
    first = model.score_protein(PROTEIN_A)

    def fail():
        raise AssertionError("the model must not be loaded: the scores are cached")

    cached = om.Model.from_scorer(None, name="tiny", cache_dir=tmp_path / "cache")
    cached._scorer._factory = fail
    np.testing.assert_allclose(cached.score_protein(PROTEIN_A).llr, first.llr, atol=1e-3)


def test_score_protein_masked_marginal_at_chosen_positions(model, tiny_scorer) -> None:
    scores = model.score_protein(PROTEIN_A, masked=True, positions=[2, 5])
    assert scores.method == "masked_marginal"
    [expected] = tiny_scorer.masked_marginals([(PROTEIN_A, 2)])
    assert scores["K2R"] == pytest.approx(float(expected[om.AMINO_ACIDS.index("R")]), abs=1e-5)
    assert np.isnan(scores.llr[0]).all() and np.isfinite(scores.llr[4]).all()
    with pytest.raises(ValueError, match="not requested"):
        scores["M1K"]
    with pytest.raises(ValueError, match="only used with masked=True"):
        model.score_protein(PROTEIN_A, positions=[2])


def test_score_proteins_batches_in_order(model) -> None:
    a, b, a2 = model.score_proteins([PROTEIN_A, "MDPFGKWR", PROTEIN_A])
    assert (a.sequence, b.sequence) == (PROTEIN_A, "MDPFGKWR")
    np.testing.assert_array_equal(a.llr, a2.llr)


def test_protein_scores_tables(model) -> None:
    scores = model.score_protein(PROTEIN_A)
    table = scores.to_arrow()
    assert table.column_names == ["position", "ref_aa", "alt_aa", "change", "wt_marginal"]
    assert table.num_rows == 19 * len(PROTEIN_A)
    row = table.to_pylist()[0]
    assert row["change"] == f"{row['ref_aa']}{row['position']}{row['alt_aa']}"
    assert row["wt_marginal"] == pytest.approx(scores[row["change"]], abs=1e-6)
    frame = scores.to_pandas()
    assert len(frame) == table.num_rows


def test_annotate_matches_the_pipeline(synthetic, tmp_path, model, tiny_scorer) -> None:
    result = om.annotate(
        synthetic.vcf,
        fasta=synthetic.fasta,
        gff3=synthetic.gff3,
        model=model,
        output_tsv=tmp_path / "api.tsv",
    )
    assert len(result) == N_MISSENSE
    assert result.status_counts() == {"ok": N_MISSENSE}
    assert all(v.is_scored and v.status == "ok" for v in result)
    assert result.model == "tiny" and result.n_records == len(DESIGNED_VARIANTS) + 2
    assert "missense consequences" in repr(result)

    reference = pipeline_annotate(
        synthetic.vcf,
        synthetic.fasta,
        synthetic.gff3,
        tmp_path / "cli.vcf.gz",
        tmp_path / "cli.tsv",
        cache=ScoreCache(tmp_path / "cache", "tiny"),
        scorer_factory=lambda: tiny_scorer,
        model_name="tiny",
    )
    assert (tmp_path / "api.tsv").read_text() == (tmp_path / "cli.tsv").read_text()
    for public, internal in zip(result, reference.variants, strict=True):
        c = internal.consequence
        assert (public.gene, public.transcript, public.protein_pos) == (
            c.gene,
            c.transcript_id,
            c.protein_pos,
        )
        assert public.protein_change == f"{c.ref_aa}{c.protein_pos}{c.alt_aa}"
        assert public.wt_marginal == internal.wt_marginal
        assert public.masked_marginal is None  # not requested
    k2r = next(v for v in result if v.pos == 45)
    assert (k2r.gene, k2r.transcript, k2r.protein_change, k2r.canonical) == (
        "GENEA",
        "TXA",
        "K2R",
        True,
    )
    assert k2r.wt_percentile is None  # no precompute for this genome


def test_annotate_tables_and_files(synthetic, tmp_path, model) -> None:
    result = om.annotate(
        synthetic.vcf,
        fasta=synthetic.fasta,
        gff3=synthetic.gff3,
        model=model,
        masked=True,
        output_vcf=tmp_path / "out.vcf.gz",
    )
    assert (tmp_path / "out.vcf.gz").exists()
    table = result.to_arrow()
    assert table.column_names[:18] == [
        "chrom", "pos", "ref", "alt", "gene", "transcript", "canonical", "biotype", "strand",
        "protein_pos", "ref_aa", "alt_aa", "protein_length", "n_windows", "wt_marginal",
        "masked_marginal", "status", "wt_percentile",
    ]  # fmt: skip
    assert table.column_names[18:] == ["protein_change", "record_index"]
    assert table.num_rows == len(result)
    assert set(table.column("status").to_pylist()) == {"ok"}
    assert all(v is not None for v in table.column("masked_marginal").to_pylist())
    frame = result.to_pandas()
    assert list(frame["protein_change"]) == [v.protein_change for v in result]
    result.write_tsv(tmp_path / "again.tsv")
    with (tmp_path / "again.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == len(result)


def test_precompute_then_annotate_reports_percentiles(synthetic, model) -> None:
    done = om.precompute(synthetic.fasta, synthetic.gff3, model=model)
    assert done.n_scored == done.n_unique_proteins and done.reference_snvs > 0
    result = om.annotate(synthetic.vcf, fasta=synthetic.fasta, gff3=synthetic.gff3, model=model)
    assert all(0 <= v.wt_percentile <= 100 for v in result)


def test_to_pandas_without_pandas_explains_how_to_install(model, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pandas", None)  # import pandas -> ImportError
    with pytest.raises(ImportError, match=r"omnimissense\[pandas\]"):
        model.score_protein(PROTEIN_A).to_pandas()


def test_model_accepts_a_name(tmp_path) -> None:
    m = om.Model("facebook/esm2_t33_650M_UR50D", cache_dir=tmp_path)  # nothing is loaded yet
    assert m.name == "facebook/esm2_t33_650M_UR50D"
    assert "esm2" in repr(m)


@pytest.mark.parametrize(
    ("ref_aa", "alt_aa", "expected"),
    [("K", "R", "K4R"), ("WL", "CM", "W4C+L5M"), ("WL", "WM", "L5M"), ("WL", "C", "4WL>4C")],
)
def test_protein_change_notation(ref_aa, alt_aa, expected) -> None:
    from omnimissense.api import _public
    from omnimissense.consequences import Consequence
    from omnimissense.pipeline import ScoredVariant

    consequence = Consequence(
        "missense", "G", "T", "protein_coding", "+", 4, ref_aa, alt_aa, 10, "A", "G"
    )
    variant = ScoredVariant(0, "1", 10, "A", "G", 0, consequence)
    assert _public(variant).protein_change == expected
