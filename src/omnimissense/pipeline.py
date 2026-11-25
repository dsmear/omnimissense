"""End-to-end workflows: annotate a VCF, precompute a proteome."""

from __future__ import annotations

import csv
import logging
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

import numpy as np
import pysam
from tqdm import tqdm

from omnimissense import __version__
from omnimissense.cache import ScoreCache
from omnimissense.consequences import Consequence, missense_consequences, run_csq
from omnimissense.fasta import prepare_fasta
from omnimissense.gff import Transcript, read_transcripts
from omnimissense.proteome import Protein, ProteinStatus, build_proteome
from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX

log = logging.getLogger(__name__)

ScorerFactory = Callable[[], MaskedLMScorer]
T = TypeVar("T")

OMS_FIELDS = (
    "Allele|Gene|Transcript|Canonical|ProteinPos|RefAA|AltAA|WTMarginal|MaskedMarginal|Status"
)
TSV_COLUMNS = (
    "chrom",
    "pos",
    "ref",
    "alt",
    "gene",
    "transcript",
    "canonical",
    "biotype",
    "strand",
    "protein_pos",
    "ref_aa",
    "alt_aa",
    "protein_length",
    "n_windows",
    "wt_marginal",
    "masked_marginal",
    "status",
)


class VariantStatus(StrEnum):
    OK = "ok"
    REF_AA_MISMATCH = "ref_aa_mismatch"
    MULTI_AA = "multi_aa_change"
    NONSTANDARD_AA = "nonstandard_aa"
    PROTEIN_UNAVAILABLE = "protein_unavailable"
    MITOCHONDRIAL = "mitochondrial_skipped"


@dataclass
class ScoredVariant:
    record_index: int
    chrom: str
    pos: int
    ref: str
    alt: str
    alt_index: int
    consequence: Consequence
    canonical: bool = False
    status: VariantStatus = VariantStatus.OK
    seq_hash: str = ""
    protein_length: int | None = None
    n_windows: int | None = None
    wt_marginal: float | None = None
    masked_marginal: float | None = None


@dataclass
class AnnotationResult:
    n_records: int
    variants: list[ScoredVariant]
    n_ref_mismatches: int = 0

    @property
    def status_counts(self) -> Counter[str]:
        return Counter(v.status.value for v in self.variants)


def classify(consequence: Consequence, protein: Protein | None) -> VariantStatus:
    """Decide whether a missense consequence can be scored.

    The reference amino acid reported by bcftools csq (derived from the VCF and the
    genome) must match our translation of the same transcript: a mismatch means the inputs
    disagree (e.g. different genome or annotation versions) and is reported, never ignored.
    """
    if protein is None:
        return VariantStatus.PROTEIN_UNAVAILABLE
    if protein.status is ProteinStatus.MITOCHONDRIAL:
        return VariantStatus.MITOCHONDRIAL
    if protein.status is not ProteinStatus.OK:
        return VariantStatus.PROTEIN_UNAVAILABLE
    if len(consequence.ref_aa) != 1 or len(consequence.alt_aa) != 1:
        return VariantStatus.MULTI_AA
    position = consequence.protein_pos
    if position > len(protein.sequence) or protein.sequence[position - 1] != consequence.ref_aa:
        return VariantStatus.REF_AA_MISMATCH
    if consequence.ref_aa not in AA_INDEX or consequence.alt_aa not in AA_INDEX:
        return VariantStatus.NONSTANDARD_AA
    return VariantStatus.OK


def _chunks(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def open_vcf(path: Path, mode: str = "r", **kwargs) -> pysam.VariantFile:
    """Open a VCF/BCF without htslib's "could not retrieve index" noise for unindexed files."""
    previous = pysam.set_verbosity(0)
    try:
        return pysam.VariantFile(str(path), mode, **kwargs)
    finally:
        pysam.set_verbosity(previous)


class LazyScorer:
    """Load the model only if something actually needs to be computed."""

    def __init__(self, factory: ScorerFactory) -> None:
        self._factory = factory
        self._scorer: MaskedLMScorer | None = None

    def get(self) -> MaskedLMScorer:
        if self._scorer is None:
            self._scorer = self._factory()
        return self._scorer


# ---------------------------------------------------------------------------- scoring


def cached_wt_scores(
    sequences: dict[str, str],
    cache: ScoreCache,
    scorer: LazyScorer,
    chunk_size: int = 256,
    progress: bool = False,
) -> dict[str, tuple[np.ndarray, int]]:
    """Wild-type-marginal matrices for ``{seq_hash: sequence}``, from cache or computed."""
    found = cache.get_wt(sequences)
    missing = sorted((h for h in sequences if h not in found), key=lambda h: len(sequences[h]))
    if not missing:
        return found
    log.info("Computing wild-type marginals for %d proteins", len(missing))
    residues = sum(len(sequences[h]) for h in missing)
    with tqdm(total=residues, unit="aa", unit_scale=True, disable=not progress) as bar:
        for chunk in _chunks(missing, chunk_size):
            computed = dict(
                zip(chunk, scorer.get().wt_marginals([sequences[h] for h in chunk]), strict=True)
            )
            cache.put_wt(computed)
            found.update(computed)
            bar.update(sum(len(sequences[h]) for h in chunk))
    return found


def cached_masked_scores(
    queries: dict[tuple[str, int], str],
    cache: ScoreCache,
    scorer: LazyScorer,
    chunk_size: int = 2048,
) -> dict[tuple[str, int], np.ndarray]:
    """Masked-marginal vectors for ``{(seq_hash, position): sequence}``."""
    found = cache.get_masked(queries)
    missing = [key for key in queries if key not in found]
    if not missing:
        return found
    log.info("Computing masked marginals for %d positions", len(missing))
    for chunk in _chunks(missing, chunk_size):
        vectors = scorer.get().masked_marginals([(queries[key], key[1]) for key in chunk])
        computed = dict(zip(chunk, vectors, strict=True))
        cache.put_masked(computed)
        found.update(computed)
    return found


def score_variants(
    variants: list[ScoredVariant],
    proteins: dict[str, Protein],
    cache: ScoreCache,
    scorer_factory: ScorerFactory,
    masked: bool = False,
) -> None:
    """Fill the score fields of variants whose status is OK (in place)."""
    scorer = LazyScorer(scorer_factory)
    ok = [v for v in variants if v.status is VariantStatus.OK]
    sequences = {}
    for variant in ok:
        protein = proteins[variant.consequence.transcript_id]
        variant.seq_hash = protein.seq_hash
        variant.protein_length = len(protein.sequence)
        sequences[protein.seq_hash] = protein.sequence

    wt = cached_wt_scores(sequences, cache, scorer)
    for variant in ok:
        matrix, n_windows = wt[variant.seq_hash]
        alt = AA_INDEX[variant.consequence.alt_aa]
        variant.wt_marginal = float(matrix[variant.consequence.protein_pos - 1, alt])
        variant.n_windows = n_windows

    if not masked:
        return
    queries = {(v.seq_hash, v.consequence.protein_pos): sequences[v.seq_hash] for v in ok}
    mm = cached_masked_scores(queries, cache, scorer)
    for variant in ok:
        vector = mm[(variant.seq_hash, variant.consequence.protein_pos)]
        variant.masked_marginal = float(vector[AA_INDEX[variant.consequence.alt_aa]])


# ---------------------------------------------------------------------------- annotate


def collect_missense(csq_vcf: Path) -> tuple[int, list[ScoredVariant]]:
    """Read a csq-annotated VCF; return the record count and its missense consequences."""
    variants = []
    n_records = 0
    with open_vcf(csq_vcf) as reader:
        for index, record in enumerate(reader):
            n_records += 1
            for alt_index, consequence in missense_consequences(record):
                variants.append(
                    ScoredVariant(
                        record_index=index,
                        chrom=record.chrom,
                        pos=record.pos,
                        ref=record.ref,
                        alt=record.alts[alt_index],
                        alt_index=alt_index,
                        consequence=consequence,
                    )
                )
    return n_records, variants


def classify_variants(
    variants: Iterable[ScoredVariant],
    transcripts: dict[str, Transcript],
    proteins: dict[str, Protein],
) -> None:
    n_mismatches = 0
    for variant in variants:
        transcript_id = variant.consequence.transcript_id
        transcript = transcripts.get(transcript_id)
        variant.canonical = bool(transcript and transcript.canonical)
        variant.status = classify(variant.consequence, proteins.get(transcript_id))
        if variant.status is VariantStatus.REF_AA_MISMATCH:
            n_mismatches += 1
            if n_mismatches > 10:
                continue
            protein = proteins[transcript_id].sequence
            position = variant.consequence.protein_pos
            observed = protein[position - 1] if position <= len(protein) else "(beyond end)"
            log.warning(
                "Reference amino acid mismatch at %s:%d %s>%s, transcript %s position %d: "
                "csq reports %s, the translated protein has %s",
                variant.chrom,
                variant.pos,
                variant.ref,
                variant.alt,
                transcript_id,
                position,
                variant.consequence.ref_aa,
                observed,
            )
    if n_mismatches > 10:
        log.warning("... %d reference amino acid mismatches in total", n_mismatches)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _sanitize(text: str) -> str:
    return "".join("_" if c in ";=,| \t" else c for c in text)


def _oms_entry(variant: ScoredVariant) -> str:
    c = variant.consequence
    fields = [
        variant.alt,
        _sanitize(c.gene),
        _sanitize(c.transcript_id),
        "1" if variant.canonical else "0",
        str(c.protein_pos),
        c.ref_aa,
        c.alt_aa,
        _fmt(variant.wt_marginal, 3),
        _fmt(variant.masked_marginal, 3),
        variant.status.value,
    ]
    return "|".join(fields)


def _per_allele_min(
    variants: list[ScoredVariant], n_alts: int, attr: str
) -> tuple[float | None, ...] | None:
    values: list[float | None] = [None] * n_alts
    for variant in variants:
        score = getattr(variant, attr)
        if score is None:
            continue
        current = values[variant.alt_index]
        values[variant.alt_index] = score if current is None else min(current, score)
    if all(v is None for v in values):
        return None
    return tuple(values)


def _output_mode(path: Path) -> str:
    if path.name.endswith(".gz"):
        return "wz"
    if path.suffix == ".bcf":
        return "wb"
    return "w"


ANNOTATION_TAGS = ("BCSQ", "OMS", "OMS_WT", "OMS_MM", "OMS_REF_MISMATCH")

_INFO_DEFINITIONS = {
    "OMS": (
        ".",
        "String",
        "omnimissense scores for each missense consequence (ALT allele x transcript). "
        f"Format: {OMS_FIELDS}. Scores are protein language model log-likelihood "
        "ratios log P(alt) - log P(ref); more negative = more likely damaging",
    ),
    "OMS_WT": (
        "A",
        "Float",
        "Lowest (most damaging) wild-type-marginal score across transcripts, per ALT allele",
    ),
    "OMS_MM": (
        "A",
        "Float",
        "Lowest (most damaging) masked-marginal score across transcripts, per ALT allele",
    ),
    "OMS_REF_MISMATCH": (
        0,
        "Flag",
        "REF allele does not match the reference genome: record not annotated",
    ),
}


def _output_header(
    input_header: pysam.VariantHeader, csq_header: pysam.VariantHeader, model_name: str
) -> pysam.VariantHeader:
    """Input header + bcftools csq header lines + omnimissense lines (each added once)."""
    header = input_header.copy()
    for record in csq_header.records:
        if record.key.startswith("bcftools_csq"):
            header.add_record(record)
    if "BCSQ" not in header.info:
        header.add_record(csq_header.info["BCSQ"].record)
    header.add_line(f"##omnimissenseVersion={__version__}")
    header.add_line(f"##omnimissenseModel={model_name}")
    for tag, (number, kind, description) in _INFO_DEFINITIONS.items():
        if tag not in header.info:
            header.info.add(tag, number, kind, description)
    return header


def write_vcf(
    input_vcf: Path,
    csq_vcf: Path,
    output: Path,
    variants: list[ScoredVariant],
    ref_mismatches: set[int],
    model_name: str,
) -> None:
    """Write the input VCF (all records, all samples) with BCSQ and omnimissense INFO tags.

    ``csq_vcf`` holds the records of ``input_vcf`` minus those in ``ref_mismatches``, in the
    same order; ``variants[i].record_index`` is a record ordinal in ``csq_vcf``.
    """
    by_record: dict[int, list[ScoredVariant]] = defaultdict(list)
    for variant in variants:
        by_record[variant.record_index].append(variant)
    with open_vcf(input_vcf) as reader, open_vcf(csq_vcf) as csq_reader:
        header = _output_header(reader.header, csq_reader.header, model_name)
        csq_records = iter(csq_reader)
        csq_index = -1
        with open_vcf(output, _output_mode(output), header=header) as writer:
            for index, record in enumerate(reader):
                record.translate(writer.header)
                for tag in ANNOTATION_TAGS:  # drop stale values when re-annotating
                    if tag in record.info:
                        del record.info[tag]
                if index in ref_mismatches:
                    record.info["OMS_REF_MISMATCH"] = True
                    writer.write(record)
                    continue
                csq_record = next(csq_records)
                csq_index += 1
                if (csq_record.chrom, csq_record.pos, csq_record.ref) != (
                    record.chrom,
                    record.pos,
                    record.ref,
                ):
                    raise RuntimeError(f"csq output out of sync at {record.chrom}:{record.pos}")
                if "BCSQ" in csq_record.info:
                    record.info["BCSQ"] = csq_record.info["BCSQ"]
                record_variants = by_record.get(csq_index)
                if record_variants:
                    n_alts = len(record.alts or ())
                    record.info["OMS"] = tuple(_oms_entry(v) for v in record_variants)
                    for tag, attr in (("OMS_WT", "wt_marginal"), ("OMS_MM", "masked_marginal")):
                        values = _per_allele_min(record_variants, n_alts, attr)
                        if values is not None:
                            record.info[tag] = values
                writer.write(record)


def write_tsv(path: Path, variants: Iterable[ScoredVariant]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for v in variants:
            c = v.consequence
            writer.writerow(
                [
                    v.chrom,
                    v.pos,
                    v.ref,
                    v.alt,
                    c.gene,
                    c.transcript_id,
                    int(v.canonical),
                    c.biotype,
                    c.strand,
                    c.protein_pos,
                    c.ref_aa,
                    c.alt_aa,
                    "" if v.protein_length is None else v.protein_length,
                    "" if v.n_windows is None else v.n_windows,
                    _fmt(v.wt_marginal),
                    _fmt(v.masked_marginal),
                    v.status.value,
                ]
            )


class ReferenceMismatchError(ValueError):
    """Too many VCF records disagree with the reference genome."""


@dataclass(frozen=True)
class RefMismatch:
    index: int  # 0-based record ordinal in the input VCF
    chrom: str
    pos: int
    vcf_ref: str
    genome_ref: str  # empty if the contig is not in the FASTA


def split_ref_mismatches(
    vcf: Path, fasta: Path, sites_out: Path, max_fraction: float = 0.1
) -> tuple[int, list[RefMismatch]]:
    """Copy ``vcf`` to a sites-only file, leaving out records whose REF is not the genome's.

    bcftools csq aborts on the first such record, so they are set aside and reported.
    Raises ``ReferenceMismatchError`` when more than ``max_fraction`` of the records disagree,
    which almost always means the VCF was called on a different assembly or uses different
    chromosome names.
    """
    mismatches = []
    n_records = 0
    with pysam.FastaFile(str(fasta)) as genome, open_vcf(vcf) as reader:
        contigs = set(genome.references)
        reader.subset_samples([])
        with open_vcf(sites_out, "wb", header=reader.header) as writer:
            for index, record in enumerate(reader):
                n_records += 1
                ref = record.ref.upper()
                observed = (
                    genome.fetch(record.chrom, record.start, record.start + len(ref)).upper()
                    if record.chrom in contigs
                    else ""
                )
                if observed != ref:
                    mismatches.append(RefMismatch(index, record.chrom, record.pos, ref, observed))
                    continue
                writer.write(record)
    if n_records and len(mismatches) > max_fraction * n_records:
        example = mismatches[0]
        raise ReferenceMismatchError(
            f"{len(mismatches)} of {n_records} VCF records have a REF allele that does not "
            f"match the reference genome (e.g. {example.chrom}:{example.pos} VCF="
            f"{example.vcf_ref} genome={example.genome_ref or 'contig not in FASTA'}). "
            "Are the VCF and the FASTA from the same assembly, with the same chromosome names?"
        )
    for mismatch in mismatches[:10]:
        log.warning(
            "REF allele mismatch at %s:%d (VCF %s, genome %s): record left unannotated",
            mismatch.chrom,
            mismatch.pos,
            mismatch.vcf_ref,
            mismatch.genome_ref or "contig not in FASTA",
        )
    if len(mismatches) > 10:
        log.warning("... %d REF allele mismatches in total", len(mismatches))
    return n_records, mismatches


def annotate(
    vcf: Path,
    fasta: Path,
    gff3: Path,
    output_vcf: Path,
    output_tsv: Path,
    *,
    cache: ScoreCache,
    scorer_factory: ScorerFactory,
    model_name: str,
    masked: bool = False,
    threads: int = 0,
) -> AnnotationResult:
    """Annotate missense variants in ``vcf`` with protein language model scores."""
    fasta = prepare_fasta(fasta)
    with tempfile.TemporaryDirectory(prefix="omnimissense-") as tmp:
        sites = Path(tmp) / "sites.bcf"
        n_records, ref_mismatches = split_ref_mismatches(vcf, fasta, sites)
        csq_vcf = Path(tmp) / "csq.bcf"
        log.info("Calling consequences with bcftools csq")
        messages = run_csq(sites, fasta, gff3, csq_vcf, threads=threads)
        log.debug("bcftools csq messages:\n%s", messages)
        _, variants = collect_missense(csq_vcf)
        log.info("%d records, %d missense consequences", n_records, len(variants))

        transcripts = read_transcripts(gff3)
        needed = {v.consequence.transcript_id for v in variants}
        proteins = build_proteome(fasta, transcripts, needed)
        classify_variants(variants, transcripts, proteins)
        score_variants(variants, proteins, cache, scorer_factory, masked=masked)

        output_vcf.parent.mkdir(parents=True, exist_ok=True)
        mismatch_indices = {m.index for m in ref_mismatches}
        write_vcf(vcf, csq_vcf, output_vcf, variants, mismatch_indices, model_name)
    write_tsv(output_tsv, variants)
    return AnnotationResult(
        n_records=n_records, variants=variants, n_ref_mismatches=len(ref_mismatches)
    )


# ---------------------------------------------------------------------------- precompute


@dataclass
class PrecomputeResult:
    n_transcripts: int
    n_unique_proteins: int
    n_already_cached: int
    n_scored: int
    residues_scored: int
    seconds: float
    status_counts: Counter[str]


def precompute(
    fasta: Path,
    gff3: Path,
    *,
    cache: ScoreCache,
    scorer_factory: ScorerFactory,
    chunk_size: int = 256,
    progress: bool = True,
) -> PrecomputeResult:
    """Compute and cache wild-type marginals for every protein of an annotation."""
    fasta = prepare_fasta(fasta)
    transcripts = read_transcripts(gff3)
    proteins = build_proteome(fasta, transcripts)
    status_counts = Counter(p.status.value for p in proteins.values())
    sequences = {p.seq_hash: p.sequence for p in proteins.values() if p.status is ProteinStatus.OK}
    cached = cache.wt_hashes() & sequences.keys()
    todo = {h: s for h, s in sequences.items() if h not in cached}
    log.info(
        "%d coding transcripts, %d unique proteins, %d already cached, %d to score",
        len(proteins),
        len(sequences),
        len(cached),
        len(todo),
    )
    start = time.perf_counter()
    cached_wt_scores(
        todo, cache, LazyScorer(scorer_factory), chunk_size=chunk_size, progress=progress
    )
    return PrecomputeResult(
        n_transcripts=len(proteins),
        n_unique_proteins=len(sequences),
        n_already_cached=len(cached),
        n_scored=len(todo),
        residues_scored=sum(len(s) for s in todo.values()),
        seconds=time.perf_counter() - start,
        status_counts=status_counts,
    )
