"""Translate annotated transcripts into protein sequences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pysam

from omnimissense.gff import Transcript
from omnimissense.sequences import reverse_complement, sequence_hash, translate

#: Sequence names used for mitochondrial DNA. Mitochondria use a different genetic code
#: and bcftools csq applies a single code genome-wide, so these transcripts are skipped.
MITOCHONDRIAL_SEQIDS = frozenset({"MT", "Mt", "M", "chrM", "chrMT", "mitochondrion"})

#: Non-ATG initiation codons of the standard code (NCBI table 1). When they start a complete
#: CDS the ribosome still inserts methionine, as Ensembl's own translations do.
ALTERNATIVE_STARTS = frozenset({"CTG", "TTG"})


class ProteinStatus(StrEnum):
    OK = "ok"
    NO_CDS = "no_cds"
    INTERNAL_STOP = "internal_stop"
    MITOCHONDRIAL = "mitochondrial"


@dataclass(frozen=True)
class Protein:
    transcript_id: str
    sequence: str  # without the terminal stop
    status: ProteinStatus

    @property
    def seq_hash(self) -> str:
        return sequence_hash(self.sequence)


def coding_sequence(fasta: pysam.FastaFile, transcript: Transcript) -> str:
    """Spliced CDS of a transcript, in transcript orientation, trimmed by the first phase."""
    segments = transcript.cds_in_transcript_order()
    parts = []
    for segment in segments:
        dna = fasta.fetch(transcript.seqid, segment.start - 1, segment.end).upper()
        parts.append(reverse_complement(dna) if transcript.strand == "-" else dna)
    first_phase = segments[0].phase or 0
    return "".join(parts)[first_phase:]


def translate_transcript(fasta: pysam.FastaFile, transcript: Transcript) -> Protein:
    if not transcript.cds:
        return Protein(transcript.transcript_id, "", ProteinStatus.NO_CDS)
    if transcript.seqid in MITOCHONDRIAL_SEQIDS:
        return Protein(transcript.transcript_id, "", ProteinStatus.MITOCHONDRIAL)
    cds = coding_sequence(fasta, transcript)
    protein = translate(cds).removesuffix("*")
    complete_5prime = (transcript.cds_in_transcript_order()[0].phase or 0) == 0
    if protein and complete_5prime and cds[:3] in ALTERNATIVE_STARTS:
        protein = "M" + protein[1:]
    status = ProteinStatus.INTERNAL_STOP if "*" in protein else ProteinStatus.OK
    return Protein(transcript.transcript_id, protein, status)


def build_proteome(
    fasta_path: Path,
    transcripts: Mapping[str, Transcript],
    transcript_ids: Iterable[str] | None = None,
) -> dict[str, Protein]:
    """Translate the given transcripts (default: all with a CDS)."""
    if transcript_ids is None:
        transcript_ids = [tid for tid, tx in transcripts.items() if tx.cds]
    with pysam.FastaFile(str(fasta_path)) as fasta:
        return {
            tid: translate_transcript(fasta, transcripts[tid])
            for tid in transcript_ids
            if tid in transcripts
        }


def write_fasta(proteins: Iterable[Protein], path: Path, width: int = 60) -> int:
    """Write proteins with status OK to a FASTA file; return how many were written."""
    count = 0
    with path.open("w") as handle:
        for protein in proteins:
            if protein.status is not ProteinStatus.OK:
                continue
            handle.write(f">{protein.transcript_id}\n")
            seq = protein.sequence
            handle.writelines(seq[i : i + width] + "\n" for i in range(0, len(seq), width))
            count += 1
    return count
