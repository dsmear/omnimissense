"""Default settings, kept free of heavy imports so the CLI starts fast."""

DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"

#: Maximum padded tokens per forward batch. 8192 fits ESM-2 650M in fp16 on an 8 GB GPU.
DEFAULT_MAX_TOKENS = 8192
