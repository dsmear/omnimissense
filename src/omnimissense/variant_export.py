"""Every possible missense SNV of the coding transcripts, at genome coordinates.

For each transcript with a scored protein, every single-nucleotide change of its codons is
translated with the transcript's genetic code, and each missense change is written with the
wild-type-marginal score of its amino-acid substitution: the score ``omnimissense annotate``
gives that variant (checked in the tests), available without a GPU or csq at annotation
time. Changes that bcftools csq does not call missense are left out: of the initiation codon
at a complete 5' end (start lost), to a stop codon (stop gained), of the stop codon.

The output format follows the file name:

* ``.tsv.gz``: one row per SNV and transcript: ``#CHROM POS REF ALT gene transcript
  canonical protein_variant oms_wt oms_wt_percentile``;
* ``.vcf.gz``: one record per SNV, with ``omnimissense annotate``'s INFO tags (``OMS`` per
  transcript, ``OMS_WT`` = lowest score, ``OMS_WT_PCT`` its proteome-wide percentile),
  ready for ``bcftools annotate -a`` or VEP ``--custom``.

The percentile reference (``omnimissense.reference``) is built first if the cache lacks it.

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
from omnimissense.gff import Transcript, prepare_annotation, read_transcripts
from omnimissense.pipeline import (
    _INFO_DEFINITIONS,
    LazyScorer,
    ScorerFactory,
    _sanitize,
    cached_wt_scores,
)
from omnimissense.proteome import Protein, ProteinStatus, build_proteome
from omnimissense.reference import (
    Reference,
    build_reference,
    load_reference,
    reference_key,
    residue_batches,
    save_reference,
)
from omnimissense.sequences import AMINO_ACIDS
from omnimissense.snvs import _BASES, TranscriptSnvs, transcript_snvs

log = logging.getLogger(__name__)

TSV_HEADER = (
    "#CHROM\tPOS\tREF\tALT\tgene\ttranscript\tcanonical\tprotein_variant\toms_wt"
    "\toms_wt_percentile\n"
)
_TBI_MAX = 2**29 - 1  # longest sequence a .tbi index can hold


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
    for tag in ("OMS", "OMS_WT", "OMS_WT_PCT"):
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
    key = reference_key(gff3, fasta, annotation.codes)
    reference = load_reference(cache.directory, key)
    if reference is None:
        reference = build_reference(
            fasta, selected, proteins, lambda seqs: cached_wt_scores(seqs, cache, scorer)
        )
        save_reference(cache.directory, key, reference)
    n_rows = n_sites = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with pysam.FastaFile(str(fasta)) as genome:
        contigs = list(zip(genome.references, genome.lengths, strict=True))
        with pysam.BGZFile(str(output), "wb") as out:
            out.write((TSV_HEADER if kind == "tsv" else _vcf_header(contigs, model_name)).encode())
            groups = [(seqid, by_seqid[seqid]) for seqid, _ in contigs if seqid in by_seqid]
            for batch in residue_batches(groups, proteins):
                sequences = {
                    proteins[t].seq_hash: proteins[t].sequence for _s, tids in batch for t in tids
                }
                scores = cached_wt_scores(sequences, cache, scorer)
                for seqid, tids in batch:
                    snvs = {
                        t: transcript_snvs(
                            genome, selected[t], proteins[t], scores[proteins[t].seq_hash][0]
                        )
                        for t in tids
                    }
                    rows, sites = _write_sequence(
                        out, kind, seqid, snvs, selected, order, reference
                    )
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
    reference: Reference,
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
    percentile = reference.percentile(score)

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
                f"\t{canonical}\t{variant}\t{score[k]:.4f}\t{percentile[k]:.3f}\n"
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
            lowest_pct = float(np.float32(percentile[start:end].min()))
            lines.append(
                f"{seqid}\t{pos[start]}\t.\t{_BASES[ref[start]]}\t{_BASES[alt[start]]}\t.\t.\t"
                f"OMS={','.join(entries)};OMS_WT={lowest:g};OMS_WT_PCT={lowest_pct:g}\n"
            )
            n_sites += 1
            start = end
            flush()
    flush(force=True)
    return len(pos), n_sites
