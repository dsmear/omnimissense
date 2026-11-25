"""Inspect the runtime environment: bundled bcftools and the ML stack."""

from __future__ import annotations

import importlib
import importlib.util
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    """Result of a single environment check."""

    name: str
    ok: bool
    detail: str


def check_bcftools() -> Check:
    """bcftools is used through the copy bundled in pysam."""
    try:
        version = importlib.import_module("pysam.version")
    except ImportError:
        return Check("bcftools", ok=False, detail="pysam not installed")
    return Check(
        "bcftools",
        ok=True,
        detail=f"{version.__bcftools_version__} (bundled in pysam {version.__version__})",
    )


def check_torch() -> list[Check]:
    """Check whether PyTorch is installed and whether it can see a CUDA GPU."""
    if importlib.util.find_spec("torch") is None:
        return [Check("torch", ok=False, detail="not installed")]
    torch = importlib.import_module("torch")
    checks = [Check("torch", ok=True, detail=torch.__version__)]
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gib = props.total_memory / 1024**3
        checks.append(Check("cuda", ok=True, detail=f"{props.name}, {vram_gib:.1f} GiB"))
    else:
        checks.append(Check("cuda", ok=False, detail="no CUDA device visible (CPU only, slow)"))
    return checks


def run_checks() -> list[Check]:
    """Run all environment checks, in display order."""
    return [
        Check("python", ok=True, detail=platform.python_version()),
        check_bcftools(),
        *check_torch(),
    ]
