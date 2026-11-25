"""Short tags for model IDs, used in result file names."""

from __future__ import annotations

import re

MODEL_TAGS = {
    "facebook/esm2_t33_650M_UR50D": "esm2_650m",
    "ntranoslab/vesm/VESM_650M": "vesm_650m",
}


def model_tag(model: str) -> str:
    return MODEL_TAGS.get(model) or re.sub(r"[^a-z0-9]+", "_", model.lower().split("/")[-1])
