import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from proteingym_all import (
    AVERAGE,
    FIELDS,
    append_result,
    group_score,
    load_all_assays,
    paired_protein_ci,
    parse_mutant,
    read_results,
    score_assay,
)

SEQUENCE = "MKTAYIAKQRQISFVKSHFSRQ"


def test_parse_mutant_handles_multiple_mutants_and_checks_wild_type() -> None:
    assert parse_mutant("K2R", SEQUENCE) == [(2, "R")]
    assert parse_mutant("K2R:Y5W", SEQUENCE) == [(2, "R"), (5, "W")]
    with pytest.raises(ValueError, match="wild type mismatch"):
        parse_mutant("K2R:A5W", SEQUENCE)


def test_multiple_mutant_scores_are_sums_of_single_scores(tiny_scorer) -> None:
    mutants = ["K2R", "Y5W", "K2R:Y5W"]
    scores = score_assay(tiny_scorer, SEQUENCE, mutants, masked=True)
    for kind in ("wt", "masked"):
        single_k2r, single_y5w, double = scores[kind]
        assert double == pytest.approx(single_k2r + single_y5w)
    [(matrix, _)] = tiny_scorer.wt_marginals([SEQUENCE])
    assert scores["wt"][0] == pytest.approx(matrix[1, "ACDEFGHIKLMNPQRSTVWY".index("R")])


def test_load_all_assays_groups_rows_by_assay(tmp_path) -> None:
    rows = {
        "DMS_id": ["B", "A", "B", "A"],
        "target_seq": ["MKT", "MAA", "MKT", "MAA"],
        "mutant": ["K2R", "A2C", "K2A:T3S", "A3D"],
        "DMS_score": [0.5, 1.0, -0.5, 2.0],
    }
    paths = [tmp_path / "part0.parquet", tmp_path / "part1.parquet"]
    table = pa.table(rows)
    pq.write_table(table.slice(0, 2), paths[0])
    pq.write_table(table.slice(2), paths[1])
    assays = load_all_assays(paths)
    assert set(assays) == {"A", "B"}
    sequence, mutants, scores = assays["B"]
    assert (sequence, mutants) == ("MKT", ["K2R", "K2A:T3S"])
    np.testing.assert_array_equal(scores, [0.5, -0.5])


def test_group_score_averages_per_protein_then_per_group() -> None:
    rows = [
        # protein P1 has two assays (mean 0.4), P2 one (0.8): taxon mean 0.6, not 0.533
        {"UniProt_ID": "P1", "taxon": "Human", "function": "Stability", "v": 0.2},
        {"UniProt_ID": "P1", "taxon": "Human", "function": "Stability", "v": 0.6},
        {"UniProt_ID": "P2", "taxon": "Human", "function": "Activity", "v": 0.8},
        {"UniProt_ID": "P3", "taxon": "Virus", "function": "Activity", "v": 0.0},
    ]
    value = lambda r: r["v"]  # noqa: E731
    assert group_score(rows, value, "Human") == pytest.approx(0.6)
    assert group_score(rows, value, "Virus") == pytest.approx(0.0)
    # by function: Stability = 0.4 (P1); Activity = mean(0.8, 0.0) = 0.4 -> 0.4
    assert group_score(rows, value, AVERAGE) == pytest.approx(0.4)
    assert np.isnan(group_score(rows, value, "Prokaryote"))


def test_paired_protein_ci() -> None:
    rows = [{"UniProt_ID": f"P{i}", "a": i / 10, "b": i / 10 - 0.05} for i in range(10)]
    diff, low, high = paired_protein_ci(rows, lambda r: r["a"], lambda r: r["b"], n=200)
    assert (diff, low, high) == pytest.approx((0.05, 0.05, 0.05))
    same = paired_protein_ci(rows, lambda r: r["a"], lambda r: r["a"], n=200)
    assert same == (0.0, 0.0, 0.0)


def test_results_file_round_trip_and_resume(tmp_path) -> None:
    path = tmp_path / "results.tsv"
    assert read_results(path) == []
    row = dict.fromkeys(FIELDS, "x") | {"DMS_id": "A", "spearman_masked": ""}
    append_result(path, row)
    append_result(path, row | {"DMS_id": "B"})
    assert [r["DMS_id"] for r in read_results(path)] == ["A", "B"]
    path.write_text("other\tcolumns\n")
    with pytest.raises(ValueError, match="other columns"):
        read_results(path)


def test_run_benchmark_scores_resumes_and_summarises(tmp_path, tiny_scorer) -> None:
    from proteingym_all import COMPARE, run_benchmark, summarise

    rng = np.random.default_rng(0)
    taxa = {"A": "Human", "B": "Human", "C": "Prokaryote", "D": "Virus"}
    reference, assays, published = {}, {}, {}
    for dms_id, taxon in taxa.items():
        mutants = ["K2R", "Y5W", "Q9E", "K2R:Y5W", "S13A", "F14L"]
        assays[dms_id] = (SEQUENCE, mutants, rng.normal(size=len(mutants)))
        reference[dms_id] = {
            "UniProt_ID": "P_" + dms_id if dms_id != "B" else "P_A",  # A and B: same protein
            "taxon": taxon,
            "coarse_selection_type": "Stability",
            "seq_len": str(len(SEQUENCE)),
        }
        published[dms_id] = {model: "0.3" for model in COMPARE}
    output = tmp_path / "results.tsv"
    calls = []

    def factory():
        calls.append(1)
        return tiny_scorer

    first = run_benchmark(assays, reference, ["A", "C"], factory, output, masked=False)
    assert [r["DMS_id"] for r in first] == ["A", "C"] and len(calls) == 1
    rows = run_benchmark(assays, reference, list(taxa), factory, output, masked=False)
    assert sorted(r["DMS_id"] for r in rows) == ["A", "B", "C", "D"] and len(calls) == 2
    assert rows[-1]["n_multiple"] == "1" and rows[-1]["spearman_masked"] == ""
    again = run_benchmark(assays, reference, list(taxa), factory, output, masked=False)
    assert len(again) == 4 and len(calls) == 2  # all done: the model is not loaded

    summary = {row["group"]: row for row in summarise(rows, published)}
    assert set(summary) == {"Average (by function)", "Human", "Prokaryote", "Virus"}
    assert (summary["Human"]["n_assays"], summary["Human"]["n_proteins"]) == (2, 1)
    assert summary["Virus"]["ESM2 (650M)"] == 0.3
    ours = float(next(r["spearman_wt"] for r in rows if r["DMS_id"] == "D"))
    assert summary["Virus"]["ours_wt"] == pytest.approx(ours, abs=1e-3)
    assert summary["Virus"]["ours_wt - ESM2 (650M)"] == pytest.approx(ours - 0.3, abs=1e-3)
