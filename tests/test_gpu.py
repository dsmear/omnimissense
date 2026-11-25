"""Checks with the real ESM-2 650M model. Run with: uv run pytest -m gpu"""

import numpy as np
import pytest
import torch

from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

# Human ubiquitin (UniProt P0CG48, residues 1-76): extremely conserved.
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


@pytest.fixture(scope="module")
def esm2() -> MaskedLMScorer:
    return MaskedLMScorer.from_pretrained()


def test_real_model_scores_are_sane(esm2) -> None:
    [(llr, n_windows)] = esm2.wt_marginals([UBIQUITIN])
    assert n_windows == 1
    assert llr.shape == (76, 20)
    assert np.isfinite(llr).all()
    wt = [AA_INDEX[aa] for aa in UBIQUITIN]
    np.testing.assert_allclose(llr[np.arange(76), wt], 0, atol=1e-6)
    # Replacing the core residue I44 (hydrophobic patch) with a charged residue must look
    # much worse to the model than a conservative I44V.
    [masked] = esm2.masked_marginals([(UBIQUITIN, 44)])
    assert masked[AA_INDEX["D"]] < masked[AA_INDEX["V"]] - 2
