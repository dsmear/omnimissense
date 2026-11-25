import gzip

import pytest

from omnimissense.gff import CdsSegment, GffFormatError, parse_attributes, read_transcripts


def test_parse_attributes_decodes_escapes() -> None:
    attrs = parse_attributes("ID=gene:G1;Name=ABC%3B1;tag=basic,Ensembl_canonical;")
    assert attrs == {"ID": "gene:G1", "Name": "ABC;1", "tag": "basic,Ensembl_canonical"}


def test_read_transcripts_from_synthetic_annotation(synthetic) -> None:
    transcripts = read_transcripts(synthetic.gff3)
    assert set(transcripts) == {"TXA", "TXB"}

    txa = transcripts["TXA"]
    assert (txa.gene_id, txa.gene_name, txa.strand, txa.biotype) == (
        "GA",
        "GENEA",
        "+",
        "protein_coding",
    )
    assert txa.canonical is True
    assert txa.cds_in_transcript_order() == [CdsSegment(41, 47, 0), CdsSegment(78, 97, 2)]

    txb = transcripts["TXB"]
    assert txb.canonical is False
    assert txb.strand == "-"
    # On the minus strand the transcript starts at the right-most CDS.
    assert txb.cds_in_transcript_order() == [CdsSegment(184, 194, 0), CdsSegment(138, 153, 1)]


def test_read_transcripts_accepts_gzip(synthetic, tmp_path) -> None:
    compressed = tmp_path / "annotation.gff3.gz"
    compressed.write_bytes(gzip.compress(synthetic.gff3.read_bytes()))
    assert set(read_transcripts(compressed)) == {"TXA", "TXB"}


def test_non_ensembl_gff3_is_rejected_with_a_hint(tmp_path) -> None:
    gff = tmp_path / "ncbi.gff3"
    gff.write_text(
        "##gff-version 3\n"
        "1\tRefSeq\tgene\t1\t100\t.\t+\t.\tID=gene-ABC;Name=ABC\n"
        "1\tRefSeq\tmRNA\t1\t100\t.\t+\t.\tID=rna-XM_1;Parent=gene-ABC\n"
    )
    with pytest.raises(GffFormatError, match="Ensembl-style"):
        read_transcripts(gff)
