"""Variant effect scores from a masked protein language model (ESM-2 family).

Two scores, both log-likelihood ratios ``log P(alt) - log P(ref)`` at the mutated position
(more negative = the substitution looks less "natural" to the model):

* **wild-type marginal**: one forward pass on the unmodified protein gives, at every position,
  the probabilities of all 20 amino acids, hence the scores of all 19 substitutions at once.
  Cheap enough to precompute a whole proteome.
* **masked marginal**: the mutated position is replaced by ``<mask>`` and the model predicts
  it from the rest of the protein. One forward pass per position; usually more accurate
  (Meier et al. 2021), so it is used only for the variants actually present in a VCF.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
import torch

from omnimissense.defaults import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from omnimissense.sequences import AA_INDEX, AMINO_ACIDS
from omnimissense.windows import (
    DEFAULT_OVERLAP,
    MAX_RESIDUES,
    combine_windows,
    plan_windows,
    window_weights,
)

log = logging.getLogger(__name__)

#: ESM-1b / ESM-2 token alphabet, in vocabulary order.
ESM_ALPHABET = (
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y",
    "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>",
)  # fmt: skip


#: VESM (Dinh, Jang, Zaitlen & Ntranos 2025; code and weights MIT): ESM-2 models whose last
#: layers and output head were re-trained for variant effect prediction. The ``.pth`` files
#: hold only the re-trained tensors and are applied on top of the base ESM-2 checkpoint.
#: VESM3 (built on ESM3-open, non-commercial license) is deliberately not supported.
VESM_REPO = "ntranoslab/vesm"
VESM_REVISION = "946749efe0c5bda4693884092df3a73c542752f3"
VESM_BASES = {
    "VESM_35M": "facebook/esm2_t12_35M_UR50D",
    "VESM_150M": "facebook/esm2_t30_150M_UR50D",
    "VESM_650M": "facebook/esm2_t33_650M_UR50D",
    "VESM_3B": "facebook/esm2_t36_3B_UR50D",
}


def vesm_base(variant: str) -> str:
    """Base ESM-2 checkpoint of a VESM variant."""
    if variant not in VESM_BASES:
        raise ValueError(
            f"unsupported VESM model {variant!r}; choose one of {sorted(VESM_BASES)} "
            "(VESM3 is not supported: it is built on ESM3-open, non-commercial license)"
        )
    return VESM_BASES[variant]


def load_vesm_state(variant: str) -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download

    vesm_base(variant)  # validate the name before downloading
    path = hf_hub_download(VESM_REPO, f"{variant}.pth", revision=VESM_REVISION)
    return torch.load(path, map_location="cpu", weights_only=True)


def apply_partial_state(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> int:
    """Overwrite a subset of the model's tensors; every key must exist in the model."""
    if not state:
        raise ValueError("empty state dict")
    result = model.load_state_dict(dict(state), strict=False)
    if result.unexpected_keys:
        raise ValueError(f"tensors not found in the model: {result.unexpected_keys[:5]}")
    log.info("Replaced %d of %d tensors of the base model", len(state), len(model.state_dict()))
    return len(state)


def log_probs_to_llr(log_probs: np.ndarray, sequence: str) -> np.ndarray:
    """Turn per-position log-probabilities (``L x 20``) into LLRs vs the wild-type residue.

    Rows whose wild-type residue is not a standard amino acid are NaN.
    """
    wt = np.array([AA_INDEX.get(aa, -1) for aa in sequence])
    llr = log_probs - log_probs[np.arange(len(sequence)), np.maximum(wt, 0)][:, None]
    llr[wt < 0] = np.nan
    return llr.astype(np.float32)


class MaskedLMScorer:
    """Scores substitutions with a masked language model that returns token logits."""

    def __init__(
        self,
        model: torch.nn.Module,
        vocab: Mapping[str, int],
        *,
        device: str | torch.device = "cpu",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_residues: int = MAX_RESIDUES,
        overlap: int = DEFAULT_OVERLAP,
        name: str = "custom",
    ) -> None:
        self.model = model.eval()
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self.max_residues = max_residues
        self.overlap = overlap
        self.name = name
        self._vocab = dict(vocab)
        self._cls = vocab["<cls>"]
        self._eos = vocab["<eos>"]
        self._pad = vocab["<pad>"]
        self._mask = vocab["<mask>"]
        self._unk = vocab["<unk>"]
        self._aa_columns = torch.tensor([vocab[aa] for aa in AMINO_ACIDS], device=self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        half: bool | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> MaskedLMScorer:
        """Load an ESM-style checkpoint from Hugging Face (fp16 on GPU, fp32 on CPU).

        ``model_name`` is a Hugging Face model ID (``facebook/esm2_t33_650M_UR50D``) or a VESM
        model as ``ntranoslab/vesm/VESM_650M`` (see ``VESM_BASES``).
        """
        from transformers import AutoTokenizer, EsmForMaskedLM

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if half is None:
            half = device.startswith("cuda")
        dtype = torch.float16 if half else torch.float32
        log.info("Loading %s on %s (%s)", model_name, device, dtype)
        if model_name.startswith(VESM_REPO + "/"):
            variant = model_name.removeprefix(VESM_REPO + "/")
            base = vesm_base(variant)
            tokenizer = AutoTokenizer.from_pretrained(base)
            model = EsmForMaskedLM.from_pretrained(base, attn_implementation="sdpa")
            apply_partial_state(model, load_vesm_state(variant))
            model = model.to(dtype)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = EsmForMaskedLM.from_pretrained(
                model_name, dtype=dtype, attn_implementation="sdpa"
            )
        return cls(
            model.to(device),
            tokenizer.get_vocab(),
            device=device,
            max_tokens=max_tokens,
            name=model_name,
        )

    # ------------------------------------------------------------------ forward passes

    def _encode(self, sequence: str) -> list[int]:
        body = [self._vocab.get(aa, self._unk) for aa in sequence]
        return [self._cls, *body, self._eos]

    def _batches(self, lengths: Sequence[int]) -> Iterator[list[int]]:
        """Group item indices into batches of at most ``max_tokens`` (padded) tokens."""
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        batch: list[int] = []
        for i in order:
            width = lengths[batch[0]] + 2 if batch else lengths[i] + 2
            if batch and (len(batch) + 1) * width > self.max_tokens:
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    def aa_log_probs(self, items: Sequence[tuple[str, int | None]]) -> list[np.ndarray]:
        """Log-probabilities of the 20 amino acids.

        ``items`` are ``(sequence, masked_index)`` pairs, with sequences of at most
        ``max_residues`` residues. With ``masked_index=None`` the result is ``len(seq) x 20``
        (all positions, unmasked); otherwise that residue is masked and the result is the
        20-vector at that position.
        """
        results: list[np.ndarray | None] = [None] * len(items)
        lengths = [len(seq) for seq, _ in items]
        if max(lengths, default=0) > self.max_residues:
            raise ValueError(f"sequence longer than {self.max_residues} residues; use windows")
        for batch in self._batches(lengths):
            width = lengths[batch[0]] + 2
            input_ids = torch.full((len(batch), width), self._pad, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, i in enumerate(batch):
                sequence, masked = items[i]
                ids = self._encode(sequence)
                if masked is not None:
                    ids[masked + 1] = self._mask
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention_mask[row, : len(ids)] = 1
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                ).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)[..., self._aa_columns]
            log_probs_np = log_probs.cpu().numpy()
            for row, i in enumerate(batch):
                sequence, masked = items[i]
                if masked is None:
                    results[i] = log_probs_np[row, 1 : len(sequence) + 1]
                else:
                    results[i] = log_probs_np[row, masked + 1]
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------ scores

    def wt_marginals(self, sequences: Sequence[str]) -> list[tuple[np.ndarray, int]]:
        """Wild-type-marginal LLR matrix (``L x 20``) and window count for each protein."""
        items: list[tuple[str, int | None]] = []
        plans = []
        for sequence in sequences:
            windows = plan_windows(len(sequence), self.max_residues, self.overlap)
            plans.append(windows)
            items.extend((sequence[w.start : w.end], None) for w in windows)
        log_probs = self.aa_log_probs(items)
        results = []
        cursor = 0
        for sequence, windows in zip(sequences, plans, strict=True):
            parts = log_probs[cursor : cursor + len(windows)]
            cursor += len(windows)
            combined = combine_windows(parts, windows, len(sequence), self.overlap)
            results.append((log_probs_to_llr(combined, sequence), len(windows)))
        return results

    def masked_marginals(self, queries: Sequence[tuple[str, int]]) -> list[np.ndarray]:
        """Masked-marginal LLRs (20-vector) for ``(sequence, 1-based position)`` queries."""
        items: list[tuple[str, int | None]] = []
        owners: list[int] = []
        weights: list[float] = []
        for q, (sequence, position) in enumerate(queries):
            index = position - 1
            windows = plan_windows(len(sequence), self.max_residues, self.overlap)
            column = (
                window_weights(windows, len(sequence), self.overlap)[:, index]
                if len(windows) > 1
                else np.ones(1)
            )
            for k, window in enumerate(windows):
                if window.start <= index < window.end:
                    items.append((sequence[window.start : window.end], index - window.start))
                    owners.append(q)
                    weights.append(float(column[k]))
        log_probs = self.aa_log_probs(items)
        combined = np.zeros((len(queries), len(AMINO_ACIDS)), dtype=np.float64)
        for vector, owner, weight in zip(log_probs, owners, weights, strict=True):
            combined[owner] += weight * vector
        results = []
        for (sequence, position), vector in zip(queries, combined, strict=True):
            wt = AA_INDEX.get(sequence[position - 1])
            llr = np.full(len(AMINO_ACIDS), np.nan) if wt is None else vector - vector[wt]
            results.append(llr.astype(np.float32))
        return results
