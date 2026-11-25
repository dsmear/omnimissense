import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from dog_realistic import (
    Candidate,
    alignment_map,
    load_uniprot,
    make_aligner,
    map_to_human,
    rank_in,
    write_canonical_gff3,
)
from stats import bootstrap_diff_ci

from omnimissense.gff import read_transcripts


def test_alignment_map_handles_insertions() -> None:
    mapping = alignment_map("MKTAYIAK", "MKTWAYIAK", make_aligner())  # human has an extra W
    assert list(mapping[:3]) == [0, 1, 2]
    assert list(mapping[3:]) == [4, 5, 6, 7, 8]


def test_map_to_human_requires_an_identical_reference_residue(tmp_path) -> None:
    uniprot_tsv = tmp_path / "uniprot.tsv"
    uniprot_tsv.write_text(
        "Entry\tGene Names (primary)\tGene Names (synonym)\tSequence\n"
        "P1\tGENE1\tALIAS1\tMKTWAYIAK\n"
    )
    uniprot = load_uniprot(uniprot_tsv)
    assert uniprot["ALIAS1"] == uniprot["GENE1"] == ("P1", "MKTWAYIAK")
    dog = {"T1": "MKTAYIAK"}
    variants = [
        Candidate("gene1", "T1", 5, "Y", "D"),  # dog Y5 aligns to human Y6
        Candidate("GENE1", "T1", 4, "A", "V"),  # dog A4 aligns to human A5
        Candidate("OTHER", "T1", 1, "M", "V"),  # no human gene
    ]
    mapped = map_to_human(variants, dog, uniprot)
    assert mapped == {("T1", 5, "D"): ("P1", 6, "Y", "D"), ("T1", 4, "V"): ("P1", 5, "A", "V")}


def test_rank_in_averages_ties() -> None:
    candidates = np.array([5.0, 3.0, 3.0, 1.0])
    assert rank_in(candidates, 6.0) == 1
    assert rank_in(candidates, 3.0) == 3  # one better, two ties
    assert rank_in(candidates, -1e9) == 5


def test_write_canonical_gff3_keeps_only_selected_transcripts(synthetic, tmp_path) -> None:
    output = tmp_path / "canonical.gff3"
    write_canonical_gff3(synthetic.gff3, {"TXA"}, output)
    assert set(read_transcripts(output)) == {"TXA"}
    assert len(read_transcripts(output)["TXA"].cds) == 2


def test_bootstrap_difference_of_identical_scores_is_zero() -> None:
    labels = np.array([1] * 20 + [0] * 30)
    scores = np.random.default_rng(0).normal(size=50) + labels
    assert bootstrap_diff_ci(labels, scores, scores, n=200) == (0.0, 0.0)


def test_alignment_map_accepts_selenocysteine() -> None:
    mapping = alignment_map("MKTAYIAK", "MKTUAYIAK", make_aligner())  # U = selenocysteine
    assert list(mapping[3:]) == [4, 5, 6, 7, 8]
