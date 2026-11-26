"""Translate annotated transcripts into protein sequences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pysam

from omnimissense.genetic_code import GENETIC_CODES
from omnimissense.gff import Transcript
from omnimissense.sequences import reverse_complement, sequence_hash, translate


class ProteinStatus(StrEnum):
    OK = "ok"
    NO_CDS = "no_cds"
    INTERNAL_STOP = "internal_stop"
    #: Mitochondrial transcript without a known genetic code (``--mito-genetic-code``).
    MITOCHONDRIAL = "mitochondrial"
    #: CDS flagged ``pseudo=true`` (NCBI): a disrupted gene, not translated or scored.
    PSEUDOGENE = "pseudogene"


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
    """Translate with the transcript's genetic code.

    An initiation codon of that code (e.g. CTG, TTG in the standard code) at a complete 5'
    end is read as methionine, as the ribosome does and as Ensembl and NCBI translate.
    """
    if not transcript.cds:
        return Protein(transcript.transcript_id, "", ProteinStatus.NO_CDS)
    if transcript.genetic_code is None:
        return Protein(transcript.transcript_id, "", ProteinStatus.MITOCHONDRIAL)
    if transcript.pseudo:
        return Protein(transcript.transcript_id, "", ProteinStatus.PSEUDOGENE)
    code = GENETIC_CODES[transcript.genetic_code]
    cds = coding_sequence(fasta, transcript)
    protein = translate(cds, code.id)
    if protein and cds[3 * len(protein) - 3 : 3 * len(protein)] in code.stops:
        protein = protein[:-1]  # terminal stop, also where it codes for Q/E inside a CDS
    complete_5prime = (transcript.cds_in_transcript_order()[0].phase or 0) == 0
    if protein and complete_5prime and cds[:3] in code.starts:
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
