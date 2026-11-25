import numpy as np
import pyarrow.parquet as pq
from synthetic import PROTEIN_A, PROTEIN_B

from omnimissense.cache import ScoreCache
from omnimissense.export import export_scores, read_export_metadata
from omnimissense.sequences import AA_INDEX


def test_export_writes_one_row_per_coding_transcript(synthetic, tmp_path, tiny_scorer) -> None:
    output = tmp_path / "scores.parquet"
    result = export_scores(
        synthetic.fasta,
        synthetic.gff3,
        output,
        cache=ScoreCache(tmp_path / "cache", "tiny"),
        scorer_factory=lambda: tiny_scorer,  # nothing cached yet: scores are computed
        model_name="tiny",
    )
    assert (result.n_transcripts, result.n_unique_proteins, result.n_residues) == (2, 2, 16)

    table = pq.read_table(output).to_pylist()
    rows = {r["transcript_id"]: r for r in table}
    assert rows["TXA"]["sequence"] == PROTEIN_A and rows["TXB"]["sequence"] == PROTEIN_B
    assert rows["TXA"]["canonical"] is True and rows["TXB"]["canonical"] is False
    assert (rows["TXA"]["gene_name"], rows["TXB"]["strand"]) == ("GENEA", "-")
    matrix = np.array(rows["TXA"]["llr"]).reshape(-1, 20)
    assert matrix.shape == (len(PROTEIN_A), 20)
    np.testing.assert_array_equal(matrix[np.arange(8), [AA_INDEX[a] for a in PROTEIN_A]], 0)
    expected, _ = tiny_scorer.wt_marginals([PROTEIN_A])[0]
    np.testing.assert_allclose(matrix, expected, atol=1e-6)

    metadata = read_export_metadata(output)
    assert metadata["model"] == "tiny"
    assert metadata["amino_acid_order"] == "ACDEFGHIKLMNPQRSTVWY"


def test_export_canonical_only(synthetic, tmp_path, tiny_scorer) -> None:
    output = tmp_path / "canonical.parquet"
    export_scores(
        synthetic.fasta,
        synthetic.gff3,
        output,
        cache=ScoreCache(tmp_path / "cache", "tiny"),
        scorer_factory=lambda: tiny_scorer,
        model_name="tiny",
        canonical_only=True,
    )
    assert pq.read_table(output, columns=["transcript_id"]).column(0).to_pylist() == ["TXA"]
