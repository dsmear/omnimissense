"""bcftools csq and omnimissense must translate with the same genetic code tables.

For every table, a gene made of all its sense codons is mutated at every third codon base;
the amino acids csq reports must be those of ``omnimissense.genetic_code``.
"""

import pysam
import pytest

from omnimissense.consequences import parse_bcsq, run_csq
from omnimissense.genetic_code import GENETIC_CODES

FILLER = "GCGCGCGCGCGCGCGCGCGC"


def _write_gene(directory, table_id):
    code = GENETIC_CODES[table_id]
    sense = [c for c in sorted(code.codons) if c not in code.stops]
    stop = sorted(code.stops)[0]
    cds = "ATG" + "".join(sense) + stop
    genome = FILLER + cds + FILLER
    start, end = len(FILLER) + 1, len(FILLER) + len(cds)
    (directory / "genome.fa").write_text(f">1\n{genome}\n")
    rows = [
        ("gene", ".", "ID=gene:G;Name=G;biotype=protein_coding"),
        ("mRNA", ".", "ID=transcript:T;Parent=gene:G;biotype=protein_coding"),
        ("exon", ".", "Parent=transcript:T"),
        ("CDS", "0", "Parent=transcript:T"),
    ]
    gff = [
        ["1", "t", ftype, str(start), str(end), ".", "+", phase, attrs]
        for ftype, phase, attrs in rows
    ]
    (directory / "a.gff3").write_text("".join("\t".join(row) + "\n" for row in gff))
    records, expected = [], {}
    for i, codon in enumerate(sense, start=2):  # protein position (1 = the ATG)
        pos = start + 3 * (i - 1) + 2  # third base of the codon
        alts = [b for b in "ACGT" if b != codon[2] and codon[:2] + b not in code.stops]
        if not alts:
            continue
        records.append(f"1\t{pos}\t.\t{codon[2]}\t{','.join(alts)}\t.\t.\t.")
        for alt in alts:
            expected[(i, alt)] = (code.codons[codon], code.codons[codon[:2] + alt])
    header = (
        "##fileformat=VCFv4.2\n##contig=<ID=1>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    (directory / "v.vcf").write_text(header + "\n".join(records) + "\n")
    return expected


@pytest.mark.parametrize("table_id", sorted(GENETIC_CODES))
def test_csq_reports_the_amino_acids_of_our_tables(tmp_path, table_id) -> None:
    expected = _write_gene(tmp_path, table_id)
    output = tmp_path / "out.vcf.gz"
    run_csq(
        tmp_path / "v.vcf",
        tmp_path / "genome.fa",
        tmp_path / "a.gff3",
        output,
        genetic_code=table_id,
    )
    observed = {}
    with pysam.VariantFile(str(output)) as reader:
        for record in reader:
            for entry in record.info.get("BCSQ", ()):
                fields = entry.split("|")
                if fields[0] == "synonymous":
                    # e.g. synonymous|G|T|protein_coding|+|5L|20A>G
                    aa = fields[5].lstrip("0123456789")
                    position = int(fields[5][: -len(aa)])
                    alt = fields[6][-1]
                    observed[(position, alt)] = (aa, aa)
                    continue
                consequence = parse_bcsq(entry)
                if consequence is not None and consequence.is_missense:
                    observed[(consequence.protein_pos, consequence.dna_alt)] = (
                        consequence.ref_aa,
                        consequence.alt_aa,
                    )
    assert observed == expected
