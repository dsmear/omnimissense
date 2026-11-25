"""Default settings, kept free of heavy imports so the CLI starts fast."""

#: VESM 650M (an ESM-2 650M re-trained for variant effects, MIT license), scored with the
#: wild-type marginal: the best model in our benchmarks (see README, Validation).
DEFAULT_MODEL = "ntranoslab/vesm/VESM_650M"
#: The reference ESM-2 checkpoint (ProteinGym publishes its scores).
ESM2_650M = "facebook/esm2_t33_650M_UR50D"

#: Maximum padded tokens per forward batch. 8192 fits ESM-2 650M in fp16 on an 8 GB GPU.
DEFAULT_MAX_TOKENS = 8192
