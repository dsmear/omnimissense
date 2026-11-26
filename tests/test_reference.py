import itertools
import logging

import numpy as np
import pysam
import pytest

from omnimissense.cache import ScoreCache
from omnimissense.fasta import prepare_fasta
from omnimissense.genetic_code import CodeAssignment
from omnimissense.pipeline import annotate, precompute
from omnimissense.reference import (
    Reference,
    load_reference,
    reference_key,
    save_reference,
)


def _exact_percentile(reference_values: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference_values)
    below = np.searchsorted(ordered, scores, side="left")
    equal = np.searchsorted(ordered, scores, side="right") - below
    return 100.0 * (below + 0.5 * equal) / len(ordered)


def test_percentiles_match_the_exact_ranks() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(-3.0, 2.5, size=200_000)
    reference = Reference.from_scores(values)
    probe = np.quantile(values, [0.0001, 0.001, 0.005, 0.01, 0.1, 0.5, 0.9, 0.999])
    exact = _exact_percentile(values, probe)
    # finer steps in the damaging tail: 0.001 percentage points below the 1st percentile
    np.testing.assert_allclose(reference.percentile(probe)[:4], exact[:4], atol=0.002)
    np.testing.assert_allclose(reference.percentile(probe), exact, atol=0.02)


def test_percentile_is_monotone_and_bounded() -> None:
    reference = Reference.from_scores(np.linspace(-10, 5, 1000))
    scores = np.linspace(-20, 20, 500)
    percentiles = reference.percentile(scores)
    assert np.all(np.diff(percentiles) >= 0)
    assert percentiles[0] == 0.0 and percentiles[-1] == 100.0


def test_ties_do_not_break_the_interpolation() -> None:
    reference = Reference.from_scores(np.array([-1.0] * 900 + [0.5] * 100))
    assert np.all(np.isfinite(reference.percentile([-2.0, -1.0, 0.0, 0.5, 1.0])))
    assert reference.percentile(-1.0) < reference.percentile(0.5)


def test_save_and_load(tmp_path) -> None:
    reference = Reference.from_scores(np.arange(100.0))
    assert load_reference(tmp_path, "abc") is None
    save_reference(tmp_path, "abc", reference)
    loaded = load_reference(tmp_path, "abc")
    assert loaded.n_snvs == 100
    np.testing.assert_array_equal(loaded.quantiles, reference.quantiles)


def test_reference_key_follows_annotation_genome_and_codes(synthetic, tmp_path) -> None:
    fasta = prepare_fasta(synthetic.fasta)
    key = reference_key(synthetic.gff3, fasta, CodeAssignment())
    assert key == reference_key(synthetic.gff3, fasta, CodeAssignment())
    assert key != reference_key(synthetic.gff3, fasta, CodeAssignment(nuclear=11))
    edited = tmp_path / "edited.gff3"
    edited.write_text(synthetic.gff3.read_text().replace("Name=GENEA", "Name=GENEX"))
    assert key != reference_key(edited, fasta, CodeAssignment())


def _annotate(synthetic, tmp_path, scorer, cache):
    return annotate(
        synthetic.vcf,
        synthetic.fasta,
        synthetic.gff3,
        tmp_path / "out.vcf.gz",
        tmp_path / "out.tsv",
        cache=cache,
        scorer_factory=lambda: scorer,
        model_name="tiny",
    )


def test_annotate_without_reference_leaves_percentiles_empty(
    synthetic, tmp_path, tiny_scorer, caplog
) -> None:
    cache = ScoreCache(tmp_path / "cache", "tiny")
    with caplog.at_level(logging.WARNING):
        result = _annotate(synthetic, tmp_path, tiny_scorer, cache)
    assert all(v.wt_percentile is None for v in result.variants)
    assert "omnimissense precompute" in caplog.text
    with pysam.VariantFile(str(tmp_path / "out.vcf.gz")) as reader:
        assert not any("OMS_WT_PCT" in record.info for record in reader)


def test_precompute_builds_the_reference_used_by_annotate(synthetic, tmp_path, tiny_scorer):
    cache = ScoreCache(tmp_path / "cache", "tiny")
    first = precompute(
        synthetic.fasta, synthetic.gff3, cache=cache, scorer_factory=lambda: tiny_scorer
    )
    assert first.reference_snvs > 0
    again = precompute(
        synthetic.fasta, synthetic.gff3, cache=cache, scorer_factory=lambda: tiny_scorer
    )
    assert again.reference_snvs == 0  # already built

    result = _annotate(synthetic, tmp_path, tiny_scorer, cache)
    scored = sorted((v.wt_marginal, v.wt_percentile) for v in result.variants)
    assert all(pct is not None and 0 <= pct <= 100 for _, pct in scored)
    assert all(a[1] <= b[1] for a, b in itertools.pairwise(scored))  # monotone
    with pysam.VariantFile(str(tmp_path / "out.vcf.gz")) as reader:
        tagged = [r for r in reader if "OMS_WT" in r.info]
    assert tagged and all("OMS_WT_PCT" in r.info for r in tagged)
    header = (tmp_path / "out.tsv").read_text().splitlines()[0].split("\t")
    assert header[-1] == "wt_percentile"


def test_reference_requires_scores() -> None:
    with pytest.raises(ValueError, match="no SNV scores"):
        Reference.from_scores(np.array([]))
