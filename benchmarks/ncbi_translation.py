"""Check the proteome translated from an NCBI RefSeq genome against NCBI's own translation.

NCBI distributes, next to every annotated assembly, ``*_translated_cds.faa.gz``: the CDS
features of the GFF3 translated from the genome. omnimissense translates the same CDS (read
through bcftools csq, see ``omnimissense.gff``); every protein should be identical. Proteins
are matched through the ``protein_id`` of the CDS features (``Parent`` = the transcript).

Files (download them by hand from the assembly's NCBI FTP folder)::

    uv run python benchmarks/ncbi_translation.py \\
        --fasta GCF_000146045.2_R64_genomic.fna.gz \\
        --gff3 GCF_000146045.2_R64_genomic.gff.gz \\
        --translated-cds GCF_000146045.2_R64_translated_cds.faa.gz
"""

from __future__ import annotations

import argparse
import logging
import re
import tempfile
from collections import Counter
from pathlib import Path

from omnimissense.fasta import prepare_fasta
from omnimissense.gff import (
    csq_id,
    open_text,
    parse_attributes,
    prepare_annotation,
    read_transcripts,
)
from omnimissense.proteome import ProteinStatus, build_proteome

log = logging.getLogger("ncbi_translation")

_PROTEIN_ID = re.compile(r"\[protein_id=([^\]]+)\]")
_LOCUS_TAG = re.compile(r"\[locus_tag=([^\]]+)\]")


def _cds_key(protein_id: str | None, locus_tag: str | None) -> str | None:
    """Match key: the protein ID; pseudo CDS have none, so their locus tag instead."""
    if protein_id:
        return protein_id
    return f"locus_tag:{locus_tag}" if locus_tag else None


def read_translated_cds(path: Path) -> dict[str, str]:
    """``{protein_id or locus_tag key: sequence}`` from an NCBI ``translated_cds`` FASTA
    (stop removed)."""
    proteins: dict[str, list[str]] = {}
    current = None
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                protein_id, locus_tag = _PROTEIN_ID.search(line), _LOCUS_TAG.search(line)
                current = _cds_key(
                    protein_id.group(1) if protein_id else None,
                    locus_tag.group(1) if locus_tag else None,
                )
                if current is not None:
                    proteins[current] = []
            elif current is not None:
                proteins[current].append(line.strip())
    return {pid: "".join(parts).removesuffix("*") for pid, parts in proteins.items()}


def cds_protein_ids(gff3: Path) -> dict[str, str]:
    """``{transcript ID as csq reports it: protein_id or locus_tag key}`` from the CDS
    features of a GFF3."""
    mapping = {}
    with open_text(gff3) as handle:
        for line in handle:
            if line.startswith("#"):
                if line.startswith("##FASTA"):
                    break
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9 or columns[2] != "CDS":
                continue
            attributes = parse_attributes(columns[8])
            key = _cds_key(attributes.get("protein_id"), attributes.get("locus_tag"))
            if key:
                for parent in attributes.get("Parent", "").split(","):
                    mapping[csq_id(parent)] = key
    return mapping


def first_difference(a: str, b: str) -> str:
    for i, (x, y) in enumerate(zip(a, b, strict=False), start=1):
        if x != y:
            return f"{x}{i}{y}"
    return f"length {len(a)} vs {len(b)}"


def compare(fasta: Path, gff3: Path, translated_cds: Path, show: int = 10) -> Counter[str]:
    fasta = prepare_fasta(fasta)
    with tempfile.TemporaryDirectory(prefix="ncbi-translation-") as tmp:
        # CDS without a transcript (prokaryotes) get one in a normalized copy: read the
        # protein IDs from the file csq actually read.
        annotation = prepare_annotation(gff3, Path(tmp))
        transcripts = read_transcripts(annotation, fasta)
        protein_ids = cds_protein_ids(annotation.gff3)
    proteins = build_proteome(fasta, transcripts)
    reference = read_translated_cds(translated_cds)

    outcome: Counter[str] = Counter()
    shown = 0
    for tid, protein in sorted(proteins.items()):
        protein_id = protein_ids.get(tid)
        expected = reference.get(protein_id) if protein_id else None
        if protein.status is not ProteinStatus.OK:
            outcome[f"omnimissense status {protein.status.value}"] += 1
            continue
        if expected is None:
            outcome["no NCBI translation for the transcript"] += 1
            continue
        if protein.sequence == expected:
            outcome["identical"] += 1
            continue
        outcome["different"] += 1
        if shown < show:
            shown += 1
            log.warning(
                "%s (%s, %s): %s",
                tid,
                protein_id,
                transcripts[tid].seqid,
                first_difference(protein.sequence, expected),
            )
    matched = {protein_ids[t] for t in proteins if t in protein_ids}
    outcome["NCBI proteins without an omnimissense transcript"] = len(set(reference) - matched)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--gff3", type=Path, required=True)
    parser.add_argument("--translated-cds", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for key, count in sorted(compare(args.fasta, args.gff3, args.translated_cds).items()):
        print(f"{count:>8}  {key}")


if __name__ == "__main__":
    main()
