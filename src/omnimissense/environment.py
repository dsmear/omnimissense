"""Inspect the runtime environment: external tools and optional ML dependencies."""

from __future__ import annotations

import importlib
import importlib.util
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    """Result of a single environment check."""

    name: str
    ok: bool
    detail: str


def _tool_version(executable: str) -> str:
    """Return the first line of ``<executable> --version``, or a short error note."""
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"found but `--version` failed ({exc.__class__.__name__})"
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0] if lines else "found (version unknown)"


def check_tool(name: str) -> Check:
    """Check that an external command-line tool is on PATH."""
    path = shutil.which(name)
    if path is None:
        return Check(name, ok=False, detail="not found on PATH")
    return Check(name, ok=True, detail=f"{_tool_version(path)} ({path})")


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
        checks.append(Check("cuda", ok=False, detail="no CUDA device visible (CPU only)"))
    return checks


def run_checks() -> list[Check]:
    """Run all environment checks, in display order."""
    return [
        Check("python", ok=True, detail=platform.python_version()),
        check_tool("bcftools"),
        check_tool("tabix"),
        *check_torch(),
    ]
