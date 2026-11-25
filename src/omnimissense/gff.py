"""Transcript models from a GFF3 annotation, exactly as bcftools csq reads them.

bcftools csq parses several GFF3 layouts (Ensembl, NCBI RefSeq, ...) into one internal model
and can write that model back (``--dump-gff``). omnimissense reads the dump, so csq and the
protein translation see the same transcripts, IDs and coding segments by construction, for
every layout csq accepts. Attributes the dump does not keep are read from the GFF3 itself:
the original transcript biotype, gene names, canonical-transcript tags and which sequences
are mitochondrial.
"""

from __future__ import annotations

import gzip
import io
import logging
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO
from urllib.parse import unquote

from omnimissense.consequences import dump_annotation

log = logging.getLogger(__name__)

#: Tags marking a gene's representative transcript: Ensembl, and NCBI RefSeq (MANE Select
#: exists for human only, RefSeq Select for some other species).
CANONICAL_TAGS = frozenset({"Ensembl_canonical", "MANE Select", "RefSeq Select"})

#: Values of the ``genome`` attribute of NCBI ``region`` features for mitochondrial DNA.
#: NCBI names sequences by accession (e.g. ``NC_001224.1``), so the name alone is not enough.
_MITOCHONDRIAL_GENOMES = frozenset({"mitochondrion"})

#: ID prefixes of the Ensembl layout, which csq strips.
_ID_PREFIXES = ("gene:", "transcript:")


class GffFormatError(ValueError):
    """The GFF3 file cannot be read."""


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
    mitochondrial: bool = False
    cds: list[CdsSegment] = field(default_factory=list)

    def cds_in_transcript_order(self) -> list[CdsSegment]:
        """CDS segments in the order they are read (5' to 3' of the transcript)."""
        return sorted(self.cds, key=lambda s: s.start, reverse=self.strand == "-")

    @property
    def cds_length(self) -> int:
        return sum(s.end - s.start + 1 for s in self.cds)


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


def csq_id(feature_id: str) -> str:
    """The ID bcftools csq reports for a feature (Ensembl ``gene:``/``transcript:`` removed)."""
    for prefix in _ID_PREFIXES:
        if feature_id.startswith(prefix):
            return feature_id.removeprefix(prefix)
    return feature_id


def _features(path: Path) -> Iterator[list[str]]:
    """The 9 columns of every feature line, up to an embedded ``##FASTA`` section."""
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##FASTA"):
                break
            if not line.strip() or line.startswith("#"):
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                raise GffFormatError(f"{path}:{line_number}: expected 9 tab-separated columns")
            yield columns


def read_csq_dump(path: Path) -> dict[str, Transcript]:
    """Transcripts and CDS segments from a csq ``--dump-gff`` file.

    In the dump, genes are ``gene`` lines, transcripts are lines with ``ID``, ``Parent`` and
    ``biotype`` (feature type = csq's biotype class), and CDS lines point to their transcript.
    """
    transcripts: dict[str, Transcript] = {}
    cds: dict[str, list[CdsSegment]] = defaultdict(list)
    for seqid, _source, feature, start, end, _score, strand, phase, attr_text in _features(path):
        attributes = parse_attributes(attr_text)
        if feature == "CDS":
            segment = CdsSegment(int(start), int(end), None if phase == "." else int(phase))
            cds[attributes["Parent"]].append(segment)
        elif feature != "gene" and {"ID", "Parent", "biotype"} <= attributes.keys():
            transcripts[attributes["ID"]] = Transcript(
                transcript_id=attributes["ID"],
                gene_id=attributes["Parent"],
                seqid=seqid,
                strand=strand,
                biotype=attributes["biotype"],
            )
    for transcript_id, segments in cds.items():
        if transcript_id in transcripts:
            transcripts[transcript_id].cds = segments
    return transcripts


def _add_source_attributes(path: Path, transcripts: dict[str, Transcript]) -> dict[str, bool]:
    """Set biotype, gene name, canonical tag and mitochondrial flag from the original GFF3.

    Returns ``{gene_id: True}`` for genes with a canonical-tagged transcript (coding or not).
    """
    gene_ids = {tx.gene_id for tx in transcripts.values()}
    gene_names: dict[str, str] = {}
    tagged_genes: dict[str, bool] = {}
    mitochondrial_seqids = set()
    for seqid, _source, feature, *_rest, attr_text in _features(path):
        attributes = parse_attributes(attr_text)
        if feature == "region" and attributes.get("genome") in _MITOCHONDRIAL_GENOMES:
            mitochondrial_seqids.add(seqid)
        feature_id = attributes.get("ID")
        if not feature_id:
            continue
        key = csq_id(feature_id)
        if key in transcripts:
            transcript = transcripts[key]
            biotype = attributes.get("biotype") or attributes.get("transcript_biotype")
            if biotype:
                transcript.biotype = biotype
            if CANONICAL_TAGS & set(attributes.get("tag", "").split(",")):
                transcript.canonical = True
                tagged_genes[transcript.gene_id] = True
        elif key in gene_ids:
            gene_names[key] = attributes.get("Name", "")
    for transcript in transcripts.values():
        transcript.gene_name = gene_names.get(transcript.gene_id, "")
        transcript.mitochondrial = transcript.seqid in mitochondrial_seqids
    return tagged_genes


def _choose_untagged_canonicals(
    transcripts: dict[str, Transcript], tagged_genes: dict[str, bool]
) -> int:
    """For genes without a canonical tag, mark the coding transcript with the longest CDS
    (ties: smallest ID). Returns the number of genes handled this way."""
    best: dict[str, Transcript] = {}
    for tx in transcripts.values():
        if not tx.cds or tx.gene_id in tagged_genes:
            continue
        current = best.get(tx.gene_id)
        if current is None or (-tx.cds_length, tx.transcript_id) < (
            -current.cds_length,
            current.transcript_id,
        ):
            best[tx.gene_id] = tx
    for tx in best.values():
        tx.canonical = True
    return len(best)


def read_transcripts(
    gff3: Path, fasta: Path, *, csq_dump: Path | None = None
) -> dict[str, Transcript]:
    """All transcripts of ``gff3`` as bcftools csq reads them, keyed by csq's transcript ID.

    ``fasta`` must be indexable by htslib (plain or BGZF). ``csq_dump`` is the ``--dump-gff``
    output of an earlier csq run on the same files; without it csq is run here.

    Canonical transcripts come from the GFF3's tags (``CANONICAL_TAGS``); genes without any
    tagged transcript get the coding transcript with the longest CDS.
    """
    if csq_dump is None:
        with tempfile.TemporaryDirectory(prefix="omnimissense-") as tmp:
            dump = Path(tmp) / "annotation.gff.gz"
            dump_annotation(fasta, gff3, dump)
            transcripts = read_csq_dump(dump)
    else:
        transcripts = read_csq_dump(csq_dump)
    tagged_genes = _add_source_attributes(gff3, transcripts)
    n_untagged = _choose_untagged_canonicals(transcripts, tagged_genes)
    n_genes = len({tx.gene_id for tx in transcripts.values() if tx.cds})
    if n_untagged:
        log.info(
            "%d of %d coding genes have no canonical-transcript tag (%s): their longest CDS "
            "is used as canonical",
            n_untagged,
            n_genes,
            ", ".join(sorted(CANONICAL_TAGS)),
        )
    return transcripts
