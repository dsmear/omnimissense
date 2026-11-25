from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import EsmConfig, EsmForMaskedLM

sys.path.insert(0, str(Path(__file__).parent))

from synthetic import SyntheticFiles, write_synthetic

from omnimissense.scoring import ESM_ALPHABET, MaskedLMScorer

VOCAB = {token: index for index, token in enumerate(ESM_ALPHABET)}


def make_tiny_model(seed: int = 0) -> EsmForMaskedLM:
    """A randomly initialised 2-layer ESM: same interface as ESM-2, no download needed."""
    torch.manual_seed(seed)
    config = EsmConfig(
        vocab_size=len(ESM_ALPHABET),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=1026,
        position_embedding_type="rotary",
        pad_token_id=VOCAB["<pad>"],
        mask_token_id=VOCAB["<mask>"],
        token_dropout=True,
        emb_layer_norm_before=False,
    )
    return EsmForMaskedLM(config).eval()


@pytest.fixture(scope="session")
def tiny_model() -> EsmForMaskedLM:
    return make_tiny_model()


@pytest.fixture
def tiny_scorer(tiny_model) -> MaskedLMScorer:
    return MaskedLMScorer(tiny_model, VOCAB, device="cpu", max_tokens=4096, name="tiny")


@pytest.fixture
def small_window_scorer(tiny_model) -> MaskedLMScorer:
    """Scorer with 20-residue windows, to exercise the long-protein path cheaply."""
    return MaskedLMScorer(
        tiny_model, VOCAB, device="cpu", max_tokens=4096, max_residues=20, overlap=10
    )


@pytest.fixture
def synthetic(tmp_path) -> SyntheticFiles:
    return write_synthetic(tmp_path / "synthetic")
