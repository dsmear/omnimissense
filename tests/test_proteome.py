import gzip

import pysam
from synthetic import PROTEIN_A, PROTEIN_B

from omnimissense.fasta import compression, prepare_fasta
from omnimissense.gff import CdsSegment, Transcript, read_transcripts
from omnimissense.proteome import (
    ProteinStatus,
    build_proteome,
    translate_transcript,
    write_fasta,
)


def _transcript(strand: str, cds: list[tuple[int, int, int]], seqid: str = "1") -> Transcript:
    return Transcript(
        transcript_id="T",
        gene_id="G",
        seqid=seqid,
        strand=strand,
        biotype="protein_coding",
        cds=[CdsSegment(*c) for c in cds],
    )


def test_both_strands_and_intron_split_codons_translate_to_designed_proteins(synthetic) -> None:
    fasta = prepare_fasta(synthetic.fasta)
    proteins = build_proteome(fasta, read_transcripts(synthetic.gff3, fasta))
    assert proteins["TXA"].sequence == PROTEIN_A
    assert proteins["TXB"].sequence == PROTEIN_B
    assert {p.status for p in proteins.values()} == {ProteinStatus.OK}


def test_phase_of_first_cds_trims_incomplete_5prime_codon(synthetic) -> None:
    fasta = prepare_fasta(synthetic.fasta)
    with pysam.FastaFile(str(fasta)) as genome:
        # Two extra bases before the start codon, declared by phase 2.
        plus = _transcript("+", [(39, 47, 2), (78, 97, 2)])
        assert translate_transcript(genome, plus).sequence == PROTEIN_A
        # Minus strand: the extra bases sit after position 194 in genome coordinates.
        minus = _transcript("-", [(138, 153, 1), (184, 196, 2)])
        assert translate_transcript(genome, minus).sequence == PROTEIN_B


def test_alternative_start_codon_is_translated_as_methionine(tmp_path) -> None:
    genome = tmp_path / "alt_start.fa"
    genome.write_text(">1\nGGCTGAAACTGTAAGG\n")  # CDS 3..14 = CTG AAA CTG TAA
    with pysam.FastaFile(str(prepare_fasta(genome))) as fasta:
        assert translate_transcript(fasta, _transcript("+", [(3, 14, 0)])).sequence == "MKL"
        # A 5'-incomplete CDS (phase != 0) does not start at a real start codon.
        incomplete = _transcript("+", [(2, 14, 1)])
        assert translate_transcript(fasta, incomplete).sequence == "LKL"


def test_wrong_frame_is_reported_as_internal_stop(synthetic) -> None:
    fasta = prepare_fasta(synthetic.fasta)
    with pysam.FastaFile(str(fasta)) as genome:
        shifted = _transcript("+", [(39, 47, 0), (78, 97, 2)])
        protein = translate_transcript(genome, shifted)
    assert protein.status is ProteinStatus.INTERNAL_STOP


def test_mitochondrial_and_noncoding_transcripts_are_skipped(synthetic) -> None:
    fasta = prepare_fasta(synthetic.fasta)
    with pysam.FastaFile(str(fasta)) as genome:
        mito = _transcript("+", [(41, 47, 0)], seqid="MT")
        assert translate_transcript(genome, mito).status is ProteinStatus.MITOCHONDRIAL
        assert translate_transcript(genome, _transcript("+", [])).status is ProteinStatus.NO_CDS


def test_plain_gzip_fasta_is_converted_to_bgzf(synthetic, tmp_path) -> None:
    gzipped = tmp_path / "genome.fa.gz"
    gzipped.write_bytes(gzip.compress(synthetic.fasta.read_bytes()))
    assert compression(gzipped) == "gzip"
    prepared = prepare_fasta(gzipped)
    assert prepared.name == "genome.fa.bgz"
    assert compression(prepared) == "bgzf"
    with pysam.FastaFile(str(prepared)) as genome:
        assert genome.fetch("1", 40, 47) == "ATGAAAC"


def test_write_fasta_skips_invalid_proteins(synthetic, tmp_path) -> None:
    fasta = prepare_fasta(synthetic.fasta)
    proteins = build_proteome(fasta, read_transcripts(synthetic.gff3, fasta))
    output = tmp_path / "out.fa"
    assert write_fasta(proteins.values(), output) == 2
    assert output.read_text().startswith(">TXA\nMKRWLEHC\n")
