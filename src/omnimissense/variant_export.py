"""Every possible missense SNV of the coding transcripts, at genome coordinates.

For each transcript with a scored protein, every single-nucleotide change of its codons is
translated with the transcript's genetic code, and each missense change is written with the
wild-type-marginal score of its amino-acid substitution: the score ``omnimissense annotate``
gives that variant (checked in the tests), available without a GPU or csq at annotation
time. Changes that bcftools csq does not call missense are left out: of the initiation codon
at a complete 5' end (start lost), to a stop codon (stop gained), of the stop codon.

The output format follows the file name:

* ``.tsv.gz``: one row per SNV and transcript: ``#CHROM POS REF ALT gene transcript
  canonical protein_variant oms_wt``;
* ``.vcf.gz``: one record per SNV, with ``omnimissense annotate``'s INFO tags (``OMS`` per
  transcript, ``OMS_WT`` = lowest score), ready for ``bcftools annotate -a`` or VEP
  ``--custom``.

Both are BGZF-compressed and tabix-indexed (CSI when a sequence is longer than 2^29 bp).
"""

from __future__ import annotations

import logging
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam

from omnimissense import __version__
from omnimissense.cache import ScoreCache
from omnimissense.fasta import prepare_fasta
from omnimissense.genetic_code import GENETIC_CODES
from omnimissense.gff import Transcript, prepare_annotation, read_transcripts
from omnimissense.pipeline import (
    _INFO_DEFINITIONS,
    LazyScorer,
    ScorerFactory,
    _sanitize,
    cached_wt_scores,
)
from omnimissense.proteome import Protein, ProteinStatus, build_proteome, coding_sequence
from omnimissense.sequences import AA_INDEX, AMINO_ACIDS

log = logging.getLogger(__name__)

TSV_HEADER = "#CHROM\tPOS\tREF\tALT\tgene\ttranscript\tcanonical\tprotein_variant\toms_wt\n"
_BASES = "ACGT"
_BASE_INDEX = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(_BASES.encode()):
    _BASE_INDEX[_b] = _i
_COMPLEMENT = np.array([3, 2, 1, 0], dtype=np.int8)  # A<->T, C<->G
_TBI_MAX = 2**29 - 1  # longest sequence a .tbi index can hold


@dataclass(frozen=True)
class _CodeArrays:
    amino_acid: np.ndarray  # codon index (16*b0 + 4*b1 + b2) -> AMINO_ACIDS index, -1 if none
    stop: np.ndarray  # codon index -> stop codon of the table
    start: np.ndarray  # codon index -> initiation codon of the table


def _code_arrays(table_id: int, _cache: dict[int, _CodeArrays] = {}) -> _CodeArrays:  # noqa: B006
    if table_id not in _cache:
        code = GENETIC_CODES[table_id]
        codons = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
        _cache[table_id] = _CodeArrays(
            amino_acid=np.array([AA_INDEX.get(code.codons[c], -1) for c in codons]),
            stop=np.array([c in code.stops for c in codons]),
            start=np.array([c in code.starts for c in codons]),
        )
    return _cache[table_id]


def cds_positions(transcript: Transcript) -> np.ndarray:
    """1-based genome position of every base of ``coding_sequence(transcript)``."""
    parts = [
        np.arange(s.start, s.end + 1)
        if transcript.strand == "+"
        else np.arange(s.end, s.start - 1, -1)
        for s in transcript.cds_in_transcript_order()
    ]
    first_phase = transcript.cds_in_transcript_order()[0].phase or 0
    return np.concatenate(parts)[first_phase:]


@dataclass
class TranscriptSnvs:
    """Missense SNVs of one transcript, as parallel arrays (genome orientation)."""

    pos: np.ndarray  # 1-based
    ref: np.ndarray  # base index in "ACGT", + strand
    alt: np.ndarray
    protein_pos: np.ndarray  # 1-based
    ref_aa: np.ndarray  # AMINO_ACIDS index
    alt_aa: np.ndarray
    score: np.ndarray


def transcript_snvs(
    fasta: pysam.FastaFile, transcript: Transcript, protein: Protein, llr: np.ndarray
) -> TranscriptSnvs:
    """All missense SNVs of a translated transcript with their wild-type-marginal scores."""
    arrays = _code_arrays(transcript.genetic_code)
    n_codons = len(protein.sequence)
    if n_codons == 0:
        empty = np.zeros(0, dtype=np.int64)
        return TranscriptSnvs(empty, empty, empty, empty, empty, empty, np.zeros(0, np.float32))
    cds = coding_sequence(fasta, transcript)[: 3 * n_codons]
    positions = cds_positions(transcript)[: 3 * n_codons].reshape(n_codons, 3)
    bases = _BASE_INDEX[np.frombuffer(cds.encode(), dtype=np.uint8)].reshape(n_codons, 3)
    valid = (bases < 4).all(axis=1)
    safe = np.where(valid[:, None], bases, 0).astype(np.int64)
    codon = 16 * safe[:, 0] + 4 * safe[:, 1] + safe[:, 2]
    ref_aa = np.array([AA_INDEX.get(aa, -1) for aa in protein.sequence])
    # Codons read as written (not the initiation codon turned into M, not ambiguous bases).
    usable = valid & (ref_aa >= 0) & (arrays.amino_acid[codon] == ref_aa)
    if transcript.complete_5prime and arrays.start[codon[0]]:
        usable[0] = False  # changes of the start codon are "start lost" for csq
    parts = defaultdict(list)
    weights = (16, 4, 1)
    for j in range(3):
        for alt in range(4):
            new = codon + (alt - safe[:, j]) * weights[j]
            new_aa = arrays.amino_acid[new]
            missense = (
                usable
                & (safe[:, j] != alt)
                & (new_aa >= 0)
                & ~arrays.stop[new]
                & (new_aa != ref_aa)
            )
            idx = np.flatnonzero(missense)
            parts["pos"].append(positions[idx, j])
            parts["ref"].append(bases[idx, j])
            parts["alt"].append(np.full(len(idx), alt, dtype=np.int8))
            parts["protein_pos"].append(idx + 1)
            parts["ref_aa"].append(ref_aa[idx])
            parts["alt_aa"].append(new_aa[idx])
            parts["score"].append(llr[idx, new_aa[idx]])
    result = {key: np.concatenate(values) for key, values in parts.items()}
    if transcript.strand == "-":
        result["ref"] = _COMPLEMENT[result["ref"]]
        result["alt"] = _COMPLEMENT[result["alt"]]
    return TranscriptSnvs(**result)


@dataclass
class VariantExportResult:
    n_transcripts: int
    n_rows: int  # SNV x transcript
    n_sites: int  # distinct SNVs
    output: Path


def _format(path: Path) -> str:
    if path.name.endswith(".tsv.gz"):
        return "tsv"
    if path.name.endswith(".vcf.gz"):
        return "vcf"
    raise ValueError(f"{path}: the output must end in .tsv.gz or .vcf.gz")


def _vcf_header(contigs: list[tuple[str, int]], model_name: str) -> str:
    lines = ["##fileformat=VCFv4.2"]
    lines += [f"##contig=<ID={name},length={length}>" for name, length in contigs]
    lines += [f"##omnimissenseVersion={__version__}", f"##omnimissenseModel={model_name}"]
    for tag in ("OMS", "OMS_WT"):
        number, kind, description = _INFO_DEFINITIONS[tag]
        lines.append(f'##INFO=<ID={tag},Number={number},Type={kind},Description="{description}">')
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    return "\n".join(lines) + "\n"


def export_variants(
    fasta: Path,
    gff3: Path,
    output: Path,
    *,
    cache: ScoreCache,
    scorer_factory: ScorerFactory,
    model_name: str,
    all_transcripts: bool = False,
    genetic_code: int | None = None,
    mito_genetic_code: int | None = None,
) -> VariantExportResult:
    """Write every missense SNV of the canonical (or all) coding transcripts with its score.

    Missing scores are computed with the model, like ``export``.
    """
    kind = _format(output)
    fasta = prepare_fasta(fasta)
    with tempfile.TemporaryDirectory(prefix="omnimissense-") as tmp:
        annotation = prepare_annotation(gff3, Path(tmp), genetic_code, mito_genetic_code)
        transcripts = read_transcripts(annotation, fasta)
    selected = {
        tid: tx for tid, tx in transcripts.items() if tx.cds and (all_transcripts or tx.canonical)
    }
    proteins = build_proteome(fasta, selected)
    usable = sorted(t for t, p in proteins.items() if p.status is ProteinStatus.OK)
    order = {tid: i for i, tid in enumerate(usable)}
    by_seqid: dict[str, list[str]] = defaultdict(list)
    for tid in usable:
        by_seqid[selected[tid].seqid].append(tid)
    log.info(
        "%d %s coding transcripts with a scored protein (%s skipped: %s)",
        len(usable),
        "" if all_transcripts else "canonical",
        len(proteins) - len(usable),
        dict(_counts(proteins)),
    )

    scorer = LazyScorer(scorer_factory)
    n_rows = n_sites = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with pysam.FastaFile(str(fasta)) as genome:
        contigs = list(zip(genome.references, genome.lengths, strict=True))
        with pysam.BGZFile(str(output), "wb") as out:
            out.write((TSV_HEADER if kind == "tsv" else _vcf_header(contigs, model_name)).encode())
            for seqid, _length in contigs:
                tids = by_seqid.get(seqid)
                if not tids:
                    continue
                sequences = {proteins[t].seq_hash: proteins[t].sequence for t in tids}
                scores = cached_wt_scores(sequences, cache, scorer)
                snvs = {
                    t: transcript_snvs(
                        genome, selected[t], proteins[t], scores[proteins[t].seq_hash][0]
                    )
                    for t in tids
                }
                rows, sites = _write_sequence(out, kind, seqid, snvs, selected, order)
                n_rows += rows
                n_sites += sites
                log.info("%s: %d transcripts, %d SNVs", seqid, len(tids), sites)
        needs_csi = any(length > _TBI_MAX for _name, length in contigs)
    if kind == "vcf":
        pysam.tabix_index(str(output), preset="vcf", force=True, csi=needs_csi)
    else:
        pysam.tabix_index(
            str(output), seq_col=0, start_col=1, end_col=1, meta_char="#", force=True, csi=needs_csi
        )
    return VariantExportResult(len(usable), n_rows, n_sites, output)


def _counts(proteins: dict[str, Protein]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for protein in proteins.values():
        if protein.status is not ProteinStatus.OK:
            counts[protein.status.value] += 1
    return counts


def _write_sequence(
    out: pysam.BGZFile,
    kind: str,
    seqid: str,
    snvs: dict[str, TranscriptSnvs],
    transcripts: dict[str, Transcript],
    order: dict[str, int],
) -> tuple[int, int]:
    """Write the SNVs of one sequence sorted by position and ALT; return (rows, sites)."""
    tids = list(snvs)
    columns = {
        name: np.concatenate([getattr(snvs[t], name) for t in tids])
        for name in ("pos", "ref", "alt", "protein_pos", "ref_aa", "alt_aa", "score")
    }
    owner = np.concatenate([np.full(len(snvs[t].pos), order[t]) for t in tids])
    by_order = {order[t]: t for t in tids}
    sort = np.lexsort((owner, columns["alt"], columns["pos"]))
    pos, ref, alt = columns["pos"][sort], columns["ref"][sort], columns["alt"][sort]
    protein_pos, ref_aa, alt_aa = (columns[k][sort] for k in ("protein_pos", "ref_aa", "alt_aa"))
    score, owner = columns["score"][sort], owner[sort]

    labels = {}
    for i, t in by_order.items():
        tx = transcripts[t]
        labels[i] = (
            _sanitize(tx.gene_name or tx.gene_id),
            _sanitize(t),
            "1" if tx.canonical else "0",
        )
    lines: list[str] = []

    def flush(force: bool = False) -> None:
        if lines and (force or len(lines) >= 100_000):
            out.write("".join(lines).encode())
            lines.clear()

    if kind == "tsv":
        for k in range(len(pos)):
            gene, transcript, canonical = labels[owner[k]]
            variant = f"{AMINO_ACIDS[ref_aa[k]]}{protein_pos[k]}{AMINO_ACIDS[alt_aa[k]]}"
            lines.append(
                f"{seqid}\t{pos[k]}\t{_BASES[ref[k]]}\t{_BASES[alt[k]]}\t{gene}\t{transcript}"
                f"\t{canonical}\t{variant}\t{score[k]:.4f}\n"
            )
            flush()
        n_sites = int(np.count_nonzero(np.diff(pos * 4 + alt, prepend=-1)))
    else:
        start = 0
        n_sites = 0
        while start < len(pos):
            end = start + 1
            while end < len(pos) and pos[end] == pos[start] and alt[end] == alt[start]:
                end += 1
            entries = []
            for k in range(start, end):
                gene, transcript, canonical = labels[owner[k]]
                entries.append(
                    f"{_BASES[alt[k]]}|{gene}|{transcript}|{canonical}|{protein_pos[k]}"
                    f"|{AMINO_ACIDS[ref_aa[k]]}|{AMINO_ACIDS[alt_aa[k]]}|{score[k]:.3f}||ok"
                )
            lowest = float(score[start:end].min())
            lines.append(
                f"{seqid}\t{pos[start]}\t.\t{_BASES[ref[start]]}\t{_BASES[alt[start]]}\t.\t.\t"
                f"OMS={','.join(entries)};OMS_WT={lowest:g}\n"
            )
            n_sites += 1
            start = end
            flush()
    flush(force=True)
    return len(pos), n_sites
