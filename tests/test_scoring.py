import numpy as np
import pytest
import torch
from conftest import VOCAB

from omnimissense.scoring import log_probs_to_llr
from omnimissense.sequences import AA_INDEX, AMINO_ACIDS
from omnimissense.windows import plan_windows, window_weights

PROTEIN = "MKTAYIAKQRQISFVKSHFSRQ"
AA_COLUMNS = [VOCAB[aa] for aa in AMINO_ACIDS]


def _manual_log_probs(model, sequence: str, masked: int | None = None) -> np.ndarray:
    ids = [VOCAB["<cls>"], *(VOCAB[aa] for aa in sequence), VOCAB["<eos>"]]
    if masked is not None:
        ids[masked + 1] = VOCAB["<mask>"]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]
    return torch.log_softmax(logits, dim=-1)[1:-1, AA_COLUMNS].numpy()


def test_llr_is_zero_for_the_wild_type_and_nan_for_nonstandard_residues() -> None:
    log_probs = np.log(np.full((3, 20), 1 / 20))
    log_probs[0, AA_INDEX["M"]] = np.log(0.5)
    llr = log_probs_to_llr(log_probs, "MKX")
    assert llr[0, AA_INDEX["M"]] == 0
    assert llr[0, AA_INDEX["A"]] == pytest.approx(np.log(0.05 / 0.5), rel=1e-5)
    assert llr[1, AA_INDEX["K"]] == 0
    assert np.isnan(llr[2]).all()


def test_wt_marginals_match_a_direct_forward_pass(tiny_model, tiny_scorer) -> None:
    [(llr, n_windows)] = tiny_scorer.wt_marginals([PROTEIN])
    expected = log_probs_to_llr(_manual_log_probs(tiny_model, PROTEIN), PROTEIN)
    assert n_windows == 1
    assert llr.shape == (len(PROTEIN), 20)
    np.testing.assert_allclose(llr, expected, atol=1e-5)


def test_masked_marginals_match_a_direct_forward_pass(tiny_model, tiny_scorer) -> None:
    position = 5  # 1-based
    [llr] = tiny_scorer.masked_marginals([(PROTEIN, position)])
    log_probs = _manual_log_probs(tiny_model, PROTEIN, masked=position - 1)[position - 1]
    expected = log_probs - log_probs[AA_INDEX[PROTEIN[position - 1]]]
    np.testing.assert_allclose(llr, expected, atol=1e-5)
    assert llr[AA_INDEX[PROTEIN[position - 1]]] == 0


def test_scores_do_not_depend_on_batch_composition(tiny_scorer) -> None:
    alone = tiny_scorer.wt_marginals([PROTEIN])[0][0]
    others = ["MK", "ACDEFGHIKLMNPQRSTVWYACDEFGHIK", PROTEIN[:10]]
    batched = tiny_scorer.wt_marginals([*others, PROTEIN])[-1][0]
    np.testing.assert_allclose(alone, batched, atol=1e-5)

    tiny_scorer.max_tokens = 40  # forces one sequence per batch
    unbatched = tiny_scorer.wt_marginals([*others, PROTEIN])[-1][0]
    np.testing.assert_allclose(alone, unbatched, atol=1e-5)


def test_long_protein_wt_marginals_combine_windows(small_window_scorer) -> None:
    protein = (PROTEIN * 3)[:50]
    windows = plan_windows(50, max_len=20, overlap=10)
    [(llr, n_windows)] = small_window_scorer.wt_marginals([protein])
    assert n_windows == len(windows) > 1
    assert llr.shape == (50, 20)
    # the first residue is only in window 0: equal to scoring that window alone
    [(first_window, _)] = small_window_scorer.wt_marginals([protein[:20]])
    np.testing.assert_allclose(llr[0], first_window[0], atol=1e-5)


def test_long_protein_masked_marginal_is_weighted_over_windows(small_window_scorer) -> None:
    protein = (PROTEIN * 3)[:50]
    windows = plan_windows(50, max_len=20, overlap=10)
    weights = window_weights(windows, 50, overlap=10)
    index = 15  # 0-based, inside the overlap of windows 0 and 1
    containing = [k for k, w in enumerate(windows) if w.start <= index < w.end]
    assert len(containing) == 2
    expected = np.zeros(20)
    for k in containing:
        w = windows[k]
        log_probs = small_window_scorer.aa_log_probs([(protein[w.start : w.end], index - w.start)])
        expected += weights[k, index] * log_probs[0]
    expected -= expected[AA_INDEX[protein[index]]]
    [llr] = small_window_scorer.masked_marginals([(protein, index + 1)])
    np.testing.assert_allclose(llr, expected, atol=1e-5)


def test_sequences_longer_than_the_context_must_use_windows(small_window_scorer) -> None:
    with pytest.raises(ValueError, match="windows"):
        small_window_scorer.aa_log_probs([("A" * 21, None)])
