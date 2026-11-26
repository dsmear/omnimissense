"""Transcript models from a GFF3 annotation, exactly as bcftools csq reads them.

bcftools csq parses several GFF3 layouts (Ensembl, NCBI RefSeq, ...) into one internal model
and can write that model back (``--dump-gff``). omnimissense reads the dump, so csq and the
protein translation see the same transcripts, IDs and coding segments by construction, for
every layout csq accepts. Attributes the dump does not keep are read from the GFF3 itself:
the original transcript biotype, gene names, canonical-transcript tags, and which sequences
are mitochondrial or use a non-standard genetic code (NCBI ``transl_table``).

Two common layouts csq cannot read are fixed first, in a normalized copy of the file
(``prepare_annotation``): CDS features that hang directly from a gene or have no parent
(prokaryotes from NCBI, Prokka or Bakta; NCBI vertebrate mitochondria) get a transcript
(and if needed a gene) feature, and an embedded ``##FASTA`` section is dropped.
"""

from __future__ import annotations

import gzip
import io
import logging
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO
from urllib.parse import unquote

from omnimissense.consequences import dump_annotation
from omnimissense.genetic_code import STANDARD, CodeAssignment

log = logging.getLogger(__name__)

#: Tags marking a gene's representative transcript: Ensembl, and NCBI RefSeq (MANE Select
#: exists for human only, RefSeq Select for some other species).
CANONICAL_TAGS = frozenset({"Ensembl_canonical", "MANE Select", "RefSeq Select"})

#: Values of the ``genome`` attribute of NCBI ``region`` features for mitochondrial DNA.
#: NCBI names sequences by accession (e.g. ``NC_001224.1``), so the name alone is not enough.
_MITOCHONDRIAL_GENOMES = frozenset({"mitochondrion"})

#: ``transl_except`` amino acids that omnimissense reads (others, e.g. ``TERM``, are ignored).
_RECODED = {"Sec": "U", "Pyl": "O"}
_TRANSL_EXCEPT = re.compile(r"\(pos:(?:complement\()?(\d+)\.\.(\d+)\)?,aa:(\w+)\)")

#: ID prefixes of the Ensembl layout, which csq strips.
_ID_PREFIXES = ("gene:", "transcript:")

#: Feature types that can parent a transcript; a CDS directly under one lacks a transcript.
_GENE_TYPES = frozenset({"gene", "pseudogene"})


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
    #: NCBI genetic code; None = mitochondrial DNA of unknown code (not translated).
    genetic_code: int | None = STANDARD
    #: The CDS is flagged ``pseudo=true`` (NCBI): a disrupted gene, not a protein.
    pseudo: bool = False
    #: The GFF3 gives the 5'-most CDS a non-zero phase (the CDS starts inside a codon).
    #: csq's dump trims that partial codon and resets the phase, so it is kept here.
    incomplete_5prime: bool = False
    #: Codons read as a non-standard amino acid (GFF3 ``transl_except``): genome start, end
    #: and letter, e.g. selenocysteine ``U`` at a TGA, pyrrolysine ``O`` at a TAG.
    recoded: list[tuple[int, int, str]] = field(default_factory=list)
    cds: list[CdsSegment] = field(default_factory=list)

    def cds_in_transcript_order(self) -> list[CdsSegment]:
        """CDS segments in the order they are read (5' to 3' of the transcript)."""
        return sorted(self.cds, key=lambda s: s.start, reverse=self.strand == "-")

    @property
    def complete_5prime(self) -> bool:
        """The CDS starts at its first codon (a real initiation codon, not a fragment)."""
        return not self.incomplete_5prime and (self.cds_in_transcript_order()[0].phase or 0) == 0

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


def _attribute(text: str, key: str) -> str | None:
    """One raw attribute value from a column-9 string, without parsing the others."""
    for item in text.split(";"):
        name, sep, value = item.partition("=")
        if sep and name.strip() == key:
            return unquote(value.strip())
    return None


def _escape(value: str) -> str:
    """Escape a GFF3 attribute value."""
    for char, code in (("%", "%25"), (";", "%3B"), ("=", "%3D"), ("&", "%26"), (",", "%2C")):
        value = value.replace(char, code)
    return value


def _set_parent(attr_text: str, parent: str) -> str:
    text = attr_text.rstrip(";")
    if re.search(r"(^|;)\s*Parent=", text):
        return re.sub(r"((?:^|;)\s*Parent=)[^;]*", lambda m: m.group(1) + _escape(parent), text)
    return f"{text};Parent={_escape(parent)}"


@dataclass
class _OrphanCds:
    """A CDS (one or more CDS lines with the same ID) that has no transcript feature."""

    seqid: str
    strand: str
    start: int
    end: int
    gene_id: str | None  # the parent gene; None if the CDS has no parent at all
    name: str  # for a gene that has to be created
    transcript_id: str = ""
    new_gene_id: str = ""


def _cds_key(columns: list[str], cds_id: str | None, gene_id: str | None) -> tuple[str, ...]:
    if cds_id:
        return (gene_id or "", cds_id)
    return (gene_id or "", columns[0], columns[3], columns[4], columns[6])


@dataclass(frozen=True)
class Annotation:
    """A GFF3 annotation ready for bcftools csq, with each sequence's genetic code.

    ``gff3`` is the file csq and omnimissense read: ``source`` itself, or a normalized copy
    when ``source`` has CDS without a transcript or an embedded ``##FASTA`` section, which
    csq cannot read.
    """

    source: Path
    gff3: Path
    codes: CodeAssignment
    n_added_transcripts: int = 0


_GTF_ATTRIBUTE = re.compile(r'(\w+)\s+(?:"([^"]*)"|([^;\s]+))')
#: CDS attributes carried over from a GTF file (used downstream: genetic code, recoding).
_GTF_CDS_KEYS = ("transl_table", "transl_except", "pseudo", "protein_id", "locus_tag")


def is_gtf(path: Path) -> bool:
    """True if the first feature line has GTF attributes (``key "value";``)."""
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            columns = line.rstrip("\n").split("\t")
            return len(columns) == 9 and bool(re.match(r'\s*\w+ "', columns[8]))
    return False


def _gtf_attributes(text: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = defaultdict(list)
    for key, quoted, bare in _GTF_ATTRIBUTE.findall(text):
        values[key].append(quoted if quoted or not bare else bare)
    return values


@dataclass
class _GtfFeature:
    seqid: str
    strand: str
    start: int
    end: int
    attributes: dict[str, str] = field(default_factory=dict)

    def extend(self, start: int, end: int) -> None:
        self.start, self.end = min(self.start, start), max(self.end, end)


def _merge_cds(
    segments: list[tuple[int, int, int | None]], strand: str
) -> list[tuple[int, int, int]]:
    """CDS plus stop-codon segments merged (adjacent or overlapping), with phases
    recomputed from the 5'-most segment's phase."""
    ordered = sorted(segments)
    merged: list[list[int]] = []
    for start, end, _phase in ordered:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    five_prime = ordered[0] if strand != "-" else max(ordered, key=lambda s: s[1])
    first_phase = five_prime[2] or 0
    in_order = merged if strand != "-" else merged[::-1]
    result, length = [], 0
    for i, (start, end) in enumerate(in_order):
        phase = first_phase if i == 0 else (3 - (length - first_phase) % 3) % 3
        result.append((start, end, phase))
        length += end - start + 1
    return result


def gtf_to_gff3(gtf: Path, output: Path) -> int:
    """Write a GTF annotation (Ensembl, GENCODE, NCBI, AUGUSTUS/BRAKER) as GFF3 in the layout
    csq reads: gene -> mRNA/transcript -> exon/CDS, IDs from ``gene_id``/``transcript_id``.

    GTF CDS features leave out the stop codon (GTF 2.2); the ``stop_codon`` features are
    merged back into the CDS, as in GFF3. Kept: gene name, gene and transcript biotypes,
    transcript tags (canonical transcripts), and the CDS attributes ``transl_table``,
    ``transl_except``, ``pseudo`` and ``protein_id``. Returns the number of transcripts.
    """
    genes: dict[str, _GtfFeature] = {}
    transcripts: dict[str, _GtfFeature] = {}
    gene_of: dict[str, str] = {}
    exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    cds: dict[str, list[tuple[int, int, int | None]]] = defaultdict(list)
    cds_attributes: dict[str, dict[str, str]] = {}
    for seqid, _source, feature, start, end, _score, strand, frame, attr_text in _features(gtf):
        values = _gtf_attributes(attr_text)
        gene_id = (values.get("gene_id") or [""])[0]
        transcript_id = (values.get("transcript_id") or [""])[0]
        begin, finish = int(start), int(end)
        if gene_id:
            gene = genes.setdefault(gene_id, _GtfFeature(seqid, strand, begin, finish))
            gene.extend(begin, finish)
            for key in ("gene_name", "gene", "gene_biotype", "gene_type"):
                if key in values and key not in gene.attributes:
                    gene.attributes[key] = values[key][0]
        if feature == "gene" or not transcript_id:
            continue
        gene_of.setdefault(transcript_id, gene_id or f"gene-{transcript_id}")
        transcript = transcripts.setdefault(
            transcript_id, _GtfFeature(seqid, strand, begin, finish)
        )
        transcript.extend(begin, finish)
        for key in ("transcript_biotype", "transcript_type"):
            if key in values:
                transcript.attributes["biotype"] = values[key][0]
        if "tag" in values:
            tags = set(transcript.attributes.get("tag", "").split(",")) | set(values["tag"])
            transcript.attributes["tag"] = ",".join(sorted(t for t in tags if t))
        phase = int(frame) if frame.isdigit() else None
        if feature == "exon":
            exons[transcript_id].append((begin, finish))
        elif feature in ("CDS", "stop_codon"):
            cds[transcript_id].append((begin, finish, phase if feature == "CDS" else None))
            if feature == "CDS":
                kept = cds_attributes.setdefault(transcript_id, {})
                for key in _GTF_CDS_KEYS:
                    if key in values and values[key][0]:
                        kept[key] = ",".join(values[key])

    def row(feature: _GtfFeature, ftype: str, attrs: str, start=None, end=None, phase="."):
        fields = [feature.seqid, "omnimissense", ftype, str(start or feature.start)]
        return "\t".join([*fields, str(end or feature.end), ".", feature.strand, phase, attrs])

    by_gene: dict[str, list[str]] = defaultdict(list)
    for transcript_id, gene_id in gene_of.items():
        by_gene[gene_id].append(transcript_id)
    lines = ["##gff-version 3"]
    for gene_id, transcript_ids in by_gene.items():
        gene = genes.get(gene_id) or transcripts[transcript_ids[0]]
        name = gene.attributes.get("gene_name") or gene.attributes.get("gene") or gene_id
        gene_biotype = gene.attributes.get("gene_biotype") or gene.attributes.get("gene_type")
        attrs = f"ID={_escape(gene_id)};Name={_escape(name)}"
        if gene_biotype:
            attrs += f";biotype={_escape(gene_biotype)}"
        lines.append(row(gene, "gene", attrs))
        for transcript_id in transcript_ids:
            transcript = transcripts[transcript_id]
            biotype = transcript.attributes.get("biotype") or gene_biotype
            if not biotype or (biotype == "mRNA" and cds.get(transcript_id)):
                biotype = "protein_coding" if cds.get(transcript_id) else "transcript"
            attrs = (
                f"ID={_escape(transcript_id)};Parent={_escape(gene_id)};biotype={_escape(biotype)}"
            )
            if transcript.attributes.get("tag"):
                attrs += f";tag={_escape(transcript.attributes['tag']).replace('%2C', ',')}"
            lines.append(row(transcript, "mRNA" if cds.get(transcript_id) else "transcript", attrs))
            parent = f"Parent={_escape(transcript_id)}"
            for start, end in sorted(set(exons.get(transcript_id, []))):
                lines.append(row(transcript, "exon", parent, start, end))
            if cds.get(transcript_id):
                extra = "".join(
                    f";{key}={_escape(value)}"
                    for key, value in cds_attributes.get(transcript_id, {}).items()
                )
                for start, end, phase in _merge_cds(cds[transcript_id], transcript.strand):
                    lines.append(row(transcript, "CDS", parent + extra, start, end, str(phase)))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(transcripts)


def prepare_annotation(
    gff3: Path,
    directory: Path,
    genetic_code: int | None = None,
    mito_genetic_code: int | None = None,
) -> Annotation:
    """Scan ``gff3`` once; write a normalized copy into ``directory`` if csq needs one.

    A GTF file is first converted to GFF3 (``gtf_to_gff3``) in ``directory``.

    The scan finds mitochondrial sequences (NCBI ``region`` with ``genome=mitochondrion``),
    each sequence's ``transl_table`` (NCBI writes it on CDS features when the code is not the
    standard one) and CDS features without a transcript. ``genetic_code`` and
    ``mito_genetic_code`` override the GFF3's codes (see ``CodeAssignment``). Parent
    features are assumed to come before their children, as in Ensembl, NCBI, Prokka and
    Bakta files.
    """
    source = gff3
    if is_gtf(gff3):
        converted = directory / "annotation.from_gtf.gff3"
        n_transcripts = gtf_to_gff3(gff3, converted)
        log.info("%s: GTF converted to GFF3 (%d transcripts)", gff3.name, n_transcripts)
        gff3 = converted
    mito_seqids = set()
    tables: dict[str, Counter[int]] = defaultdict(Counter)
    genes: dict[str, str] = {}  # gene-type feature ID -> its biotype ("" if none)
    orphans: dict[tuple[str, ...], _OrphanCds] = {}
    has_fasta = False
    with open_text(gff3) as handle:
        for line in handle:
            if line.startswith("#"):
                if line.startswith("##FASTA"):
                    has_fasta = True
                    break
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                continue
            feature, attr_text = columns[2], columns[8]
            if feature in _GENE_TYPES:
                gene_id = _attribute(attr_text, "ID")
                if gene_id:
                    biotype = _attribute(attr_text, "gene_biotype") or _attribute(
                        attr_text, "biotype"
                    )
                    genes[gene_id] = biotype or ""
            elif feature == "region":
                if _attribute(attr_text, "genome") in _MITOCHONDRIAL_GENOMES:
                    mito_seqids.add(columns[0])
            elif feature == "CDS":
                table = _attribute(attr_text, "transl_table")
                if table and table.isdigit():
                    tables[columns[0]][int(table)] += 1
                parents = [p for p in (_attribute(attr_text, "Parent") or "").split(",") if p]
                gene_parent = next((p for p in parents if p in genes), None)
                if parents and gene_parent is None:
                    continue  # under a transcript: csq reads it as is
                cds_id = _attribute(attr_text, "ID")
                key = _cds_key(columns, cds_id, gene_parent)
                start, end = int(columns[3]), int(columns[4])
                orphan = orphans.get(key)
                if orphan is None:
                    name = next(
                        (
                            v
                            for k in ("gene", "Name", "locus_tag")
                            if (v := _attribute(attr_text, k))
                        ),
                        cds_id or f"{columns[0]}:{start}-{end}",
                    )
                    orphans[key] = _OrphanCds(columns[0], columns[6], start, end, gene_parent, name)
                else:
                    orphan.start, orphan.end = min(orphan.start, start), max(orphan.end, end)

    annotated = {}
    for seqid, counts in tables.items():
        (table_id, _n), *others = counts.most_common()
        if others:
            log.warning(
                "%s: CDS with several transl_table values %s; using %d",
                seqid,
                dict(counts),
                table_id,
            )
        annotated[seqid] = table_id
    codes = CodeAssignment(genetic_code, mito_genetic_code, annotated, frozenset(mito_seqids))
    if not orphans and not has_fasta:
        return Annotation(source, gff3, codes)

    normalized = directory / "annotation.normalized.gff3"
    _write_normalized(gff3, normalized, genes, orphans)
    if orphans:
        log.info(
            "%s: %d CDS without a transcript feature (prokaryote-style); transcripts added "
            "for bcftools csq",
            gff3.name,
            len(orphans),
        )
    return Annotation(source, normalized, codes, len(orphans))


def _write_normalized(
    source: Path, output: Path, genes: dict[str, str], orphans: dict[tuple[str, ...], _OrphanCds]
) -> None:
    """Copy ``source`` without its ``##FASTA`` section, giving each orphan CDS a transcript
    (``tx-<CDS ID without "cds-">``) and, when it has no parent, a gene (``gene-<same>``)."""
    used = set(genes)

    def unique(candidate: str) -> str:
        identifier, n = candidate, 1
        while identifier in used:
            n += 1
            identifier = f"{candidate}-{n}"
        used.add(identifier)
        return identifier

    for key, orphan in orphans.items():
        base = key[1].removeprefix("cds-") if len(key) == 2 else (orphan.gene_id or orphan.name)
        orphan.transcript_id = unique(f"tx-{base}")
        if orphan.gene_id is None:
            orphan.new_gene_id = unique(f"gene-{base}")

    def row(orphan: _OrphanCds, feature: str, attrs: str) -> str:
        fields = [orphan.seqid, "omnimissense", feature, str(orphan.start), str(orphan.end)]
        return "\t".join([*fields, ".", orphan.strand, ".", attrs]) + "\n"

    emitted = set()
    with open_text(source) as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith("##FASTA"):
                break
            columns = line.rstrip("\n").split("\t")
            if line.startswith("#") or len(columns) != 9 or columns[2] != "CDS":
                dst.write(line)
                continue
            parents = [p for p in (_attribute(columns[8], "Parent") or "").split(",") if p]
            gene_parent = next((p for p in parents if p in genes), None)
            key = _cds_key(columns, _attribute(columns[8], "ID"), gene_parent)
            orphan = orphans.get(key) if not parents or gene_parent else None
            if orphan is None:
                dst.write(line)
                continue
            if key not in emitted:
                emitted.add(key)
                gene_id = orphan.gene_id or orphan.new_gene_id
                if orphan.gene_id is None:
                    name = _escape(orphan.name)
                    attrs = f"ID={_escape(gene_id)};Name={name};biotype=protein_coding"
                    dst.write(row(orphan, "gene", attrs))
                biotype = genes.get(orphan.gene_id or "", "") or "protein_coding"
                attrs = (
                    f"ID={_escape(orphan.transcript_id)};Parent={_escape(gene_id)};"
                    f"biotype={_escape(biotype)}"
                )
                dst.write(row(orphan, "mRNA", attrs))
            columns[8] = _set_parent(columns[8], orphan.transcript_id)
            dst.write("\t".join(columns) + "\n")


def _add_source_attributes(path: Path, transcripts: dict[str, Transcript]) -> dict[str, bool]:
    """Set biotype, gene name, canonical tag, pseudo flag, 5' completeness and recoded
    codons (``transl_except``) from the GFF3.

    Returns ``{gene_id: True}`` for genes with a canonical-tagged transcript (coding or not).
    """
    gene_ids = {tx.gene_id for tx in transcripts.values()}
    gene_names: dict[str, str] = {}
    tagged_genes: dict[str, bool] = {}
    first_cds: dict[str, tuple[int, int]] = {}  # transcript -> (5' coordinate, phase)
    for row in _features(path):
        feature, start, end, strand, phase, attr_text = (row[i] for i in (2, 3, 4, 6, 7, 8))
        attributes = parse_attributes(attr_text)
        if feature == "CDS":
            five_prime = int(start) if strand != "-" else -int(end)
            for parent in attributes.get("Parent", "").split(","):
                transcript = transcripts.get(csq_id(parent))
                if transcript is None:
                    continue
                if attributes.get("pseudo") == "true":
                    transcript.pseudo = True
                for begin, finish, amino_acid in _TRANSL_EXCEPT.findall(
                    attributes.get("transl_except", "")
                ):
                    letter = _RECODED.get(amino_acid)
                    codon = (int(begin), int(finish), letter)
                    if letter and codon not in transcript.recoded:
                        transcript.recoded.append(codon)
                current = first_cds.get(transcript.transcript_id)
                if current is None or five_prime < current[0]:
                    first_cds[transcript.transcript_id] = (
                        five_prime,
                        int(phase or 0) if phase.isdigit() else 0,
                    )
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
        transcript.incomplete_5prime = first_cds.get(transcript.transcript_id, (0, 0))[1] != 0
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


def _log_genetic_codes(transcripts: dict[str, Transcript]) -> None:
    coding = [tx for tx in transcripts.values() if tx.cds]
    in_use = Counter(tx.genetic_code for tx in coding)
    if set(in_use) - {STANDARD, None}:
        log.info(
            "Genetic codes (NCBI table: coding transcripts): %s",
            ", ".join(f"{k}: {n}" for k, n in sorted(i for i in in_use.items() if i[0])),
        )
    if in_use[None]:
        seqids = sorted({tx.seqid for tx in coding if tx.genetic_code is None})
        log.warning(
            "%d mitochondrial coding transcripts (%s) are skipped: their genetic code is not "
            "known. Set it with --mito-genetic-code (e.g. 2 vertebrates, 3 yeast, "
            "5 invertebrates)",
            in_use[None],
            ", ".join(seqids),
        )


def read_transcripts(
    annotation: Annotation | Path, fasta: Path, *, csq_dump: Path | None = None
) -> dict[str, Transcript]:
    """All transcripts as bcftools csq reads them, keyed by csq's transcript ID.

    ``annotation`` is a GFF3 path or a ``prepare_annotation`` result (needed to choose the
    genetic codes, and when ``csq_dump`` is given). ``fasta`` must be indexable by htslib
    (plain or BGZF). ``csq_dump`` is the ``--dump-gff`` output of an earlier csq run on
    ``annotation.gff3``; without it csq is run here.

    Canonical transcripts come from the GFF3's tags (``CANONICAL_TAGS``); genes without any
    tagged transcript get the coding transcript with the longest CDS.
    """
    if isinstance(annotation, Annotation):
        return _read_prepared(annotation, fasta, csq_dump)
    if csq_dump is not None:
        raise ValueError("csq_dump requires a prepared Annotation (the file csq read)")
    with tempfile.TemporaryDirectory(prefix="omnimissense-") as tmp:
        return _read_prepared(prepare_annotation(annotation, Path(tmp)), fasta, None)


def _read_prepared(
    annotation: Annotation, fasta: Path, csq_dump: Path | None
) -> dict[str, Transcript]:
    if csq_dump is None:
        with tempfile.TemporaryDirectory(prefix="omnimissense-") as tmp:
            dump = Path(tmp) / "annotation.gff.gz"
            dump_annotation(fasta, annotation.gff3, dump)
            transcripts = read_csq_dump(dump)
    else:
        transcripts = read_csq_dump(csq_dump)
    codes = annotation.codes
    for transcript in transcripts.values():
        transcript.mitochondrial = codes.is_mitochondrial(transcript.seqid)
        transcript.genetic_code = codes.code(transcript.seqid)
    _log_genetic_codes(transcripts)
    tagged_genes = _add_source_attributes(annotation.gff3, transcripts)
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
