import gzip

import pytest
from synthetic import (
    CHROM,
    GENOME,
    PROTEIN_A,
    PROTEIN_B,
    _gff3_lines,
    ncbi_gff3_lines,
    prokaryote_gff3_lines,
)

from omnimissense.consequences import CsqError
from omnimissense.gff import (
    CdsSegment,
    csq_id,
    parse_attributes,
    prepare_annotation,
    read_transcripts,
)
from omnimissense.proteome import ProteinStatus, build_proteome


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


def _row(ftype, start, end, strand, phase, attrs):
    return "\t".join([CHROM, "test", ftype, str(start), str(end), ".", strand, phase, attrs])


def test_parse_attributes_decodes_escapes() -> None:
    attrs = parse_attributes("ID=gene:G1;Name=ABC%3B1;tag=basic,Ensembl_canonical;")
    assert attrs == {"ID": "gene:G1", "Name": "ABC;1", "tag": "basic,Ensembl_canonical"}


def test_csq_id_strips_only_ensembl_prefixes() -> None:
    assert csq_id("transcript:ENST1") == "ENST1"
    assert csq_id("gene:ENSG1") == "ENSG1"
    assert csq_id("rna-XM_1.1") == "rna-XM_1.1"


def test_read_transcripts_from_synthetic_annotation(synthetic) -> None:
    transcripts = read_transcripts(synthetic.gff3, synthetic.fasta)
    assert set(transcripts) == {"TXA", "TXB"}

    txa = transcripts["TXA"]
    assert (txa.gene_id, txa.gene_name, txa.strand, txa.biotype) == (
        "GA",
        "GENEA",
        "+",
        "protein_coding",
    )
    assert txa.canonical is True  # Ensembl_canonical tag
    assert txa.mitochondrial is False
    assert txa.cds_in_transcript_order() == [CdsSegment(41, 47, 0), CdsSegment(78, 97, 2)]

    txb = transcripts["TXB"]
    assert txb.canonical is True  # gene GB has no tagged transcript: longest CDS
    assert txb.strand == "-"
    # On the minus strand the transcript starts at the right-most CDS.
    assert txb.cds_in_transcript_order() == [CdsSegment(184, 194, 0), CdsSegment(138, 153, 1)]


def test_read_transcripts_accepts_gzip(synthetic, tmp_path) -> None:
    compressed = tmp_path / "annotation.gff3.gz"
    compressed.write_bytes(gzip.compress(synthetic.gff3.read_bytes()))
    assert set(read_transcripts(compressed, synthetic.fasta)) == {"TXA", "TXB"}


def test_ncbi_refseq_layout_gives_the_same_proteins(synthetic, tmp_path) -> None:
    gff = _write(tmp_path / "ncbi.gff3", ncbi_gff3_lines())
    transcripts = read_transcripts(gff, synthetic.fasta)
    assert set(transcripts) == {"rna-XM_GENEA.1", "rna-XM_GENEB.1"}
    txa = transcripts["rna-XM_GENEA.1"]
    # NCBI mRNAs carry no biotype: csq derives it from the gene's gene_biotype.
    assert (txa.gene_id, txa.gene_name, txa.biotype) == ("gene-GENEA", "GENEA", "protein_coding")
    assert all(tx.canonical for tx in transcripts.values())  # no tags: longest CDS per gene

    proteins = build_proteome(synthetic.fasta, transcripts)
    assert proteins["rna-XM_GENEA.1"].sequence == PROTEIN_A
    assert proteins["rna-XM_GENEB.1"].sequence == PROTEIN_B


def test_ncbi_mitochondrial_region_is_detected(synthetic, tmp_path) -> None:
    # NCBI names sequences by accession, so only the region's genome= attribute says "MT".
    gff = _write(tmp_path / "ncbi_mt.gff3", ncbi_gff3_lines(mitochondrial=True))
    transcripts = read_transcripts(gff, synthetic.fasta)
    assert all(tx.mitochondrial for tx in transcripts.values())
    proteins = build_proteome(synthetic.fasta, transcripts)
    assert {p.status for p in proteins.values()} == {ProteinStatus.MITOCHONDRIAL}


def _gene(gid, start, end, strand):
    return _row("gene", start, end, strand, ".", f"ID=gene:{gid};biotype=protein_coding")


def _mrna(tid, gid, start, end, strand, extra=""):
    attrs = f"ID=transcript:{tid};Parent=gene:{gid};biotype=protein_coding{extra}"
    return _row("mRNA", start, end, strand, ".", attrs)


def _cds(tid, start, end, strand, phase):
    return _row("CDS", start, end, strand, str(phase), f"Parent=transcript:{tid}")


def test_canonical_tags_take_precedence_over_cds_length(synthetic, tmp_path) -> None:
    tag = ";tag=basic,Ensembl_canonical"
    lines = [
        "##gff-version 3",
        _gene("GA", 41, 97, "+"),
        _mrna("A_TAG", "GA", 41, 97, "+", tag),  # tagged, shorter CDS
        _cds("A_TAG", 78, 97, "+", 0),
        _mrna("A_LONG", "GA", 41, 97, "+"),  # untagged, longer CDS
        _cds("A_LONG", 41, 47, "+", 0),
        _cds("A_LONG", 78, 97, "+", 2),
        _gene("GB", 138, 194, "-"),  # no tags in this gene: the longest CDS wins
        _mrna("B_SHORT", "GB", 138, 194, "-"),
        _cds("B_SHORT", 184, 194, "-", 0),
        _mrna("B_LONG", "GB", 138, 194, "-"),
        _cds("B_LONG", 138, 153, "-", 1),
        _cds("B_LONG", 184, 194, "-", 0),
    ]
    transcripts = read_transcripts(_write(tmp_path / "tags.gff3", lines), synthetic.fasta)
    canonical = {tid for tid, tx in transcripts.items() if tx.canonical}
    assert canonical == {"A_TAG", "B_LONG"}


def test_tagged_non_coding_transcript_keeps_coding_ones_non_canonical(synthetic, tmp_path) -> None:
    attrs = "ID=transcript:TXB_NC;Parent=gene:GB;biotype=retained_intron;tag=Ensembl_canonical"
    lines = [
        *_gff3_lines(),
        _row("lnc_RNA", 138, 194, "-", ".", attrs),
        _row("exon", 138, 194, "-", ".", "Parent=transcript:TXB_NC"),
    ]
    transcripts = read_transcripts(_write(tmp_path / "nc.gff3", lines), synthetic.fasta)
    assert transcripts["TXB"].canonical is False
    assert transcripts["TXB_NC"].canonical is True
    assert transcripts["TXB_NC"].cds == []


def test_original_biotype_is_kept(synthetic, tmp_path) -> None:
    # csq would report "NMD"; the GFF3's own biotype is kept (as before, and as exported).
    txa = "ID=transcript:TXA;Parent=gene:GA;biotype="
    lines = [
        line.replace(txa + "protein_coding", txa + "nonsense_mediated_decay")
        for line in _gff3_lines()
    ]
    transcripts = read_transcripts(_write(tmp_path / "nmd.gff3", lines), synthetic.fasta)
    assert transcripts["TXA"].biotype == "nonsense_mediated_decay"


@pytest.mark.parametrize(
    ("style", "expected"),
    [
        ("ncbi", {"tx-WP_A.1": ("gene-A", "geneA"), "tx-WP_B.1": ("gene-B", "geneB")}),
        (
            "prokka",
            {
                "tx-PROKKA_0000A": ("gene-PROKKA_0000A", "geneA"),
                "tx-PROKKA_0000B": ("gene-PROKKA_0000B", "geneB"),
            },
        ),
    ],
)
def test_cds_without_a_transcript_get_one(synthetic, tmp_path, style, expected) -> None:
    gff = _write(tmp_path / f"{style}.gff3", prokaryote_gff3_lines(style))
    transcripts = read_transcripts(gff, synthetic.fasta)
    assert {t: (tx.gene_id, tx.gene_name) for t, tx in transcripts.items()} == expected
    assert all(tx.canonical and tx.biotype == "protein_coding" for tx in transcripts.values())
    proteins = build_proteome(synthetic.fasta, transcripts)
    assert sorted(p.sequence for p in proteins.values()) == sorted([PROTEIN_A, PROTEIN_B])


def test_pseudo_cds_are_flagged_and_not_translated(synthetic, tmp_path) -> None:
    # NCBI marks disrupted genes with pseudo=true on the CDS (E. coli: 18, yeast: 6).
    lines = [
        line + ";pseudo=true" if "\tCDS\t" in line and "Parent=gene-B" in line else line
        for line in prokaryote_gff3_lines("ncbi")
    ]
    transcripts = read_transcripts(_write(tmp_path / "pseudo.gff3", lines), synthetic.fasta)
    assert {t: tx.pseudo for t, tx in transcripts.items()} == {
        "tx-WP_A.1": False,
        "tx-WP_B.1": True,
    }
    proteins = build_proteome(synthetic.fasta, transcripts)
    assert proteins["tx-WP_A.1"].status is ProteinStatus.OK
    assert proteins["tx-WP_B.1"].status is ProteinStatus.PSEUDOGENE


def test_prokaryotic_genetic_code_comes_from_the_gff3(synthetic, tmp_path) -> None:
    gff = _write(tmp_path / "ncbi.gff3", prokaryote_gff3_lines("ncbi"))
    annotation = prepare_annotation(gff, tmp_path)
    assert annotation.codes.code(CHROM) == 11
    assert annotation.n_added_transcripts == 2
    assert annotation.gff3 != gff  # csq reads the normalized copy
    assert {tx.genetic_code for tx in read_transcripts(annotation, synthetic.fasta).values()} == {
        11
    }


def test_only_cds_without_a_transcript_are_changed(synthetic, tmp_path) -> None:
    # NCBI vertebrate mitochondria: CDS under the gene, in a file whose other genes have mRNAs.
    lines = ncbi_gff3_lines()
    gene_b = [line for line in lines if "GENEB" in line]
    lines = [line for line in lines if "GENEB" not in line]
    for line in gene_b:
        columns = line.split("\t")
        if columns[2] == "gene":
            lines.append(line)
        elif columns[2] == "CDS":
            lines.append(line.replace("Parent=rna-XM_GENEB.1", "Parent=gene-GENEB"))
    transcripts = read_transcripts(_write(tmp_path / "mixed.gff3", lines), synthetic.fasta)
    assert set(transcripts) == {"rna-XM_GENEA.1", "tx-XP_GENEB.1"}
    assert transcripts["tx-XP_GENEB.1"].gene_id == "gene-GENEB"
    proteins = build_proteome(synthetic.fasta, transcripts)
    assert proteins["tx-XP_GENEB.1"].sequence == PROTEIN_B


def test_embedded_fasta_section_is_dropped(synthetic, tmp_path) -> None:
    lines = [*_gff3_lines(), "##FASTA", f">{CHROM}", GENOME]
    annotation = prepare_annotation(_write(tmp_path / "with_fasta.gff3", lines), tmp_path)
    assert annotation.n_added_transcripts == 0
    assert "##FASTA" not in annotation.gff3.read_text()
    assert set(read_transcripts(annotation, synthetic.fasta)) == {"TXA", "TXB"}


def test_unusable_gff3_is_rejected_with_a_hint(synthetic, tmp_path) -> None:
    gff = _write(
        tmp_path / "genes_only.gff3",
        ["##gff-version 3", _row("gene", 41, 97, "+", ".", "ID=gene-A;Name=geneA")],
    )
    with pytest.raises(CsqError, match="GTF to GFF3"):
        read_transcripts(gff, synthetic.fasta)
