"""export-variants must give every SNV exactly the score ``annotate`` gives it."""

import gzip

import pysam
import pytest
from synthetic import CHROM, GENE_A_CDS, GENE_B_CDS, MT_CHROM, write_synthetic_with_mitochondrion

from omnimissense.cache import ScoreCache
from omnimissense.pipeline import annotate
from omnimissense.snvs import cds_positions
from omnimissense.variant_export import export_variants

BASES = "ACGT"
#: Every CDS base of the synthetic genes (genes A and B on chromosome 1, GM on MT), plus two
#: bases in front of gene A's CDS, used by a 5'-incomplete copy of transcript TXA.
CDS_RANGES = {CHROM: [*GENE_A_CDS, (39, 40, 0), *GENE_B_CDS], MT_CHROM: [(11, 28, 0)]}


def _all_snvs_vcf(files, path):
    genome = {name: seq for name, seq in _read_fasta(files.fasta)}
    lines = [line for line in files.vcf.read_text().splitlines() if line.startswith("##")]
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    for chrom, ranges in CDS_RANGES.items():
        positions = sorted({p for start, end, _ in ranges for p in range(start, end + 1)})
        for pos in positions:
            ref = genome[chrom][pos - 1]
            alts = ",".join(b for b in BASES if b != ref)
            lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alts}\t.\t.\t.")
    path.write_text("\n".join(lines) + "\n")
    return path


def _read_fasta(path):
    name, parts = None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name:
                yield name, "".join(parts)
            name, parts = line[1:], []
        else:
            parts.append(line)
    yield name, "".join(parts)


@pytest.fixture
def files(tmp_path):
    files = write_synthetic_with_mitochondrion(tmp_path / "genome")
    # A 5'-incomplete copy of TXA: two extra bases in front, declared by phase 2.
    extra = [
        [CHROM, "t", "mRNA", "39", "97", ".", "+", ".",
         "ID=transcript:TXA5;Parent=gene:GA;biotype=protein_coding"],
        [CHROM, "t", "CDS", "39", "47", ".", "+", "2", "Parent=transcript:TXA5"],
        [CHROM, "t", "CDS", "78", "97", ".", "+", "2", "Parent=transcript:TXA5"],
    ]  # fmt: skip
    with files.gff3.open("a") as gff3:
        gff3.write("".join("\t".join(row) + "\n" for row in extra))
    return files


def _annotated(files, tmp_path, scorer, cache):
    vcf = _all_snvs_vcf(files, tmp_path / "all_snvs.vcf")
    result = annotate(
        vcf,
        files.fasta,
        files.gff3,
        tmp_path / "annotated.vcf.gz",
        tmp_path / "annotated.tsv",
        cache=cache,
        scorer_factory=lambda: scorer,
        model_name="tiny",
        mito_genetic_code=2,
    )
    rows = {
        (
            v.chrom,
            v.pos,
            v.ref,
            v.alt,
            v.consequence.transcript_id,
            f"{v.consequence.ref_aa}{v.consequence.protein_pos}{v.consequence.alt_aa}",
            round(v.wt_marginal, 4),
            round(v.wt_percentile, 3),
        )
        for v in result.variants
        if v.status.value == "ok"
    }
    return result, rows


def test_tsv_export_equals_annotate_on_every_possible_snv(files, tmp_path, tiny_scorer) -> None:
    cache = ScoreCache(tmp_path / "cache", "tiny")
    output = tmp_path / "snvs.tsv.gz"
    result = export_variants(
        files.fasta,
        files.gff3,
        output,
        cache=cache,
        scorer_factory=lambda: tiny_scorer,
        model_name="tiny",
        all_transcripts=True,
        mito_genetic_code=2,
    )
    exported = set()
    with gzip.open(output, "rt") as handle:
        assert handle.readline().startswith("#CHROM\tPOS\tREF\tALT")
        for line in handle:
            chrom, pos, ref, alt, _gene, tid, _canonical, change, score, pct = line.split("\t")
            exported.add(
                (chrom, int(pos), ref, alt, tid, change, round(float(score), 4), float(pct))
            )
    annotation, annotated = _annotated(files, tmp_path, tiny_scorer, cache)
    assert annotation.status_counts.keys() == {"ok"}
    assert exported == annotated
    assert result.n_rows == len(exported)
    assert {row[4] for row in exported} == {"TXA", "TXA5", "TXB", "TXM"}
    # The start codon (complete 5' end) is excluded; TXA5's first codon is not a start.
    assert not any(row[4] in {"TXA", "TXB", "TXM"} and row[5][1:-1] == "1" for row in exported)
    assert any(row[4] == "TXA5" and row[5][1:-1] == "1" for row in exported)
    with pysam.TabixFile(str(output)) as tabix:  # indexed and queryable
        assert len(list(tabix.fetch(MT_CHROM, 0, 40))) == len(
            [r for r in exported if r[0] == MT_CHROM]
        )


def test_vcf_export_reproduces_annotate_info_tags(files, tmp_path, tiny_scorer) -> None:
    cache = ScoreCache(tmp_path / "cache", "tiny")
    output = tmp_path / "snvs.vcf.gz"
    export_variants(
        files.fasta,
        files.gff3,
        output,
        cache=cache,
        scorer_factory=lambda: tiny_scorer,
        model_name="tiny",
        all_transcripts=True,
        mito_genetic_code=2,
    )
    _annotation, _rows = _annotated(files, tmp_path, tiny_scorer, cache)
    expected = {}
    with pysam.VariantFile(str(tmp_path / "annotated.vcf.gz")) as reader:
        for record in reader:
            for i, alt in enumerate(record.alts):
                value = record.info.get("OMS_WT", (None,) * len(record.alts))[i]
                if value is not None:
                    entries = sorted(e for e in record.info["OMS"] if e.startswith(f"{alt}|"))
                    pct = round(record.info["OMS_WT_PCT"][i], 3)
                    expected[(record.chrom, record.pos, alt)] = (round(value, 4), pct, entries)
    observed = {}
    with pysam.VariantFile(str(output)) as reader:
        assert reader.index is not None or (output.with_suffix(".gz.tbi")).exists()
        for record in reader:
            key = (record.chrom, record.pos, record.alts[0])
            observed[key] = (
                round(record.info["OMS_WT"][0], 4),
                round(record.info["OMS_WT_PCT"][0], 3),
                sorted(record.info["OMS"]),
            )
    assert observed == expected


def test_cds_positions_follow_strand_and_phase(files, tmp_path) -> None:
    from omnimissense.gff import CdsSegment, Transcript

    minus = Transcript("T", "G", CHROM, "-", "protein_coding", cds=[CdsSegment(10, 12, 1)])
    assert list(cds_positions(minus)) == [11, 10]
    plus = Transcript(
        "T", "G", CHROM, "+", "protein_coding", cds=[CdsSegment(1, 3, 2), CdsSegment(7, 8, 0)]
    )
    assert list(cds_positions(plus)) == [3, 7, 8]


def test_output_name_selects_the_format(files, tmp_path, tiny_scorer) -> None:
    with pytest.raises(ValueError, match=r"\.tsv\.gz or \.vcf\.gz"):
        export_variants(
            files.fasta,
            files.gff3,
            tmp_path / "snvs.txt",
            cache=ScoreCache(tmp_path / "cache", "tiny"),
            scorer_factory=lambda: tiny_scorer,
            model_name="tiny",
        )
