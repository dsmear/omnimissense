"""Minimal GFF3 reader for Ensembl-style annotations.

bcftools csq only accepts the Ensembl GFF3 layout (``ID=gene:...``, ``ID=transcript:...``,
``Parent=transcript:...``), so this reader follows the same conventions: both tools then see
the same transcripts under the same IDs.
"""

from __future__ import annotations

import gzip
import io
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO
from urllib.parse import unquote


class GffFormatError(ValueError):
    """The GFF3 file is not in the layout required by bcftools csq."""


@dataclass(frozen=True)
class CdsSegment:
    """One CDS feature: 1-based inclusive coordinates and GFF3 phase."""

    start: int
    end: int
    phase: int | None


@dataclass
class Transcript:
    transcript_id: str
    gene_id: str
    seqid: str
    strand: str
    biotype: str
    canonical: bool = False
    gene_name: str = ""
    cds: list[CdsSegment] = field(default_factory=list)

    def cds_in_transcript_order(self) -> list[CdsSegment]:
        """CDS segments in the order they are read (5' to 3' of the transcript)."""
        return sorted(self.cds, key=lambda s: s.start, reverse=self.strand == "-")


def open_text(path: Path) -> TextIO:
    """Open a plain or gzip/BGZF-compressed text file."""
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open(encoding="utf-8")


def parse_attributes(text: str) -> dict[str, str]:
    """Parse a GFF3 column-9 attribute string."""
    attributes = {}
    for item in text.strip().split(";"):
        if not item:
            continue
        key, _, value = item.partition("=")
        attributes[key.strip()] = unquote(value)
    return attributes


def read_transcripts(path: Path) -> dict[str, Transcript]:
    """Read all transcripts and their CDS segments, keyed by transcript ID (without prefix)."""
    gene_names: dict[str, str] = {}
    transcripts: dict[str, Transcript] = {}
    cds: dict[str, list[CdsSegment]] = defaultdict(list)

    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise GffFormatError(f"{path}:{line_number}: expected 9 tab-separated columns")
            seqid, _source, feature, start, end, _score, strand, phase, attr_text = columns
            attributes = parse_attributes(attr_text)
            feature_id = attributes.get("ID", "")

            if feature_id.startswith("gene:"):
                gene_names[feature_id.removeprefix("gene:")] = attributes.get("Name", "")
            elif feature_id.startswith("transcript:"):
                parent = attributes.get("Parent", "")
                if not parent.startswith("gene:"):
                    continue
                transcript_id = feature_id.removeprefix("transcript:")
                transcripts[transcript_id] = Transcript(
                    transcript_id=transcript_id,
                    gene_id=parent.removeprefix("gene:"),
                    seqid=seqid,
                    strand=strand,
                    biotype=attributes.get("biotype", ""),
                    canonical="Ensembl_canonical" in attributes.get("tag", "").split(","),
                )
            elif feature == "CDS":
                segment = CdsSegment(int(start), int(end), None if phase == "." else int(phase))
                for parent in attributes.get("Parent", "").split(","):
                    if parent.startswith("transcript:"):
                        cds[parent.removeprefix("transcript:")].append(segment)

    if not transcripts:
        raise GffFormatError(
            f"{path}: no 'ID=transcript:...' features found. bcftools csq (and omnimissense) "
            "require Ensembl-style GFF3; files from other sources must be converted first "
            "(bcftools ships a 'gff2gff' helper script for this)."
        )
    for transcript_id, segments in cds.items():
        transcript = transcripts.get(transcript_id)
        if transcript is not None:
            transcript.cds = segments
    for transcript in transcripts.values():
        transcript.gene_name = gene_names.get(transcript.gene_id, "")
    return transcripts
