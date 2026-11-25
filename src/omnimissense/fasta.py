"""Reference genome access."""

from __future__ import annotations

import gzip
import logging
import shutil
from pathlib import Path

import pysam

log = logging.getLogger(__name__)


def compression(path: Path) -> str:
    """Return ``"bgzf"``, ``"gzip"`` or ``"none"``."""
    with path.open("rb") as handle:
        header = handle.read(18)
    if header[:2] != b"\x1f\x8b":
        return "none"
    # BGZF = gzip with the FEXTRA flag and a "BC" extra subfield.
    if len(header) >= 14 and header[3] & 4 and header[12:14] == b"BC":
        return "bgzf"
    return "gzip"


def prepare_fasta(path: Path) -> Path:
    """Return a random-access FASTA (plain or BGZF) with its index, creating them if needed.

    htslib cannot index plain-gzip FASTA, which is how Ensembl distributes genomes. Such a
    file is recompressed once to BGZF next to the original (``<name>.bgz``).
    """
    kind = compression(path)
    if kind == "gzip":
        bgzf_path = path.with_name(path.name.removesuffix(".gz") + ".bgz")
        if not bgzf_path.exists():
            log.info("Recompressing %s to BGZF (one-off): %s", path.name, bgzf_path.name)
            tmp_path = bgzf_path.with_name(bgzf_path.name + ".tmp")
            with gzip.open(path, "rb") as source, pysam.BGZFile(str(tmp_path), "wb") as target:
                shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
            tmp_path.rename(bgzf_path)
        path = bgzf_path
    index = path.with_name(path.name + ".fai")
    if not index.exists():
        log.info("Indexing %s", path.name)
        pysam.faidx(str(path))
    return path
