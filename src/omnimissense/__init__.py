"""omnimissense: score missense variants in any species with protein language models.

The Python API (``Model``, ``annotate``, ``precompute`` ...) is documented in
``omnimissense.api``; it is imported on first use, so ``import omnimissense`` stays light.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

try:
    __version__ = version("omnimissense")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0+unknown"

__all__ = [
    "AMINO_ACIDS",
    "AnnotatedVariant",
    "AnnotatedVariants",
    "Model",
    "PrecomputeResult",
    "ProteinScores",
    "ReferenceMismatchError",
    "VariantStatus",
    "__version__",
    "annotate",
    "precompute",
]
_API = frozenset(__all__) - {"__version__"}

if TYPE_CHECKING:
    from omnimissense.api import (
        AMINO_ACIDS,
        AnnotatedVariant,
        AnnotatedVariants,
        Model,
        PrecomputeResult,
        ProteinScores,
        ReferenceMismatchError,
        VariantStatus,
        annotate,
        precompute,
    )


def __getattr__(name: str) -> Any:
    # torch and transformers load only when the API is used, not for the command line
    if name in _API:
        return getattr(import_module("omnimissense.api"), name)
    raise AttributeError(f"module 'omnimissense' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
