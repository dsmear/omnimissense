import pysam
import pytest
from synthetic import DESIGNED_VARIANTS, MULTIALLELIC_ALTS, MULTIALLELIC_POS

from omnimissense.consequences import CsqError, missense_consequences, parse_bcsq, run_csq
from omnimissense.fasta import prepare_fasta


def test_parse_missense_entry() -> None:
    c = parse_bcsq("missense|GENEA|TXA|protein_coding|+|2K>2R|45A>G")
    assert c is not None
    assert c.is_missense
    assert (c.gene, c.transcript_id, c.biotype, c.strand) == ("GENEA", "TXA", "protein_coding", "+")
    assert (c.protein_pos, c.ref_aa, c.alt_aa) == (2, "K", "R")
    assert (c.dna_pos, c.dna_ref, c.dna_alt) == (45, "A", "G")


@pytest.mark.parametrize(
    "entry",
    [
        "synonymous|GENEA|TXA|protein_coding|+|5L|85G>A",  # no amino-acid change
        "splice_region|GENEA|TXA|protein_coding",  # no protein fields
        "@122106101",  # back-reference to another record
        "missense|CLASP1|ENST1|protein_coding|-|1174P>1174L|122106101G>A+122106102G>A",  # haplotype
    ],
)
def test_entries_without_a_single_record_aa_change_are_ignored(entry) -> None:
    assert parse_bcsq(entry) is None


def test_consequence_types() -> None:
    combined = parse_bcsq("missense&splice_region|G|T|protein_coding|+|3R>3H|78G>A")
    assert combined is not None and combined.is_missense
    stop = parse_bcsq("stop_gained|G|T|protein_coding|+|8C>8*|94C>A")
    assert stop is not None and not stop.is_missense
    downstream = parse_bcsq("*missense|G|T|protein_coding|+|9M>9T|99T>C")
    assert downstream is not None and not downstream.is_missense


def _csq_missense(synthetic, tmp_path) -> dict[tuple[int, str], str]:
    fasta = prepare_fasta(synthetic.fasta)
    output = tmp_path / "csq.vcf.gz"
    run_csq(synthetic.vcf, fasta, synthetic.gff3, output)
    found = {}
    with pysam.VariantFile(str(output)) as reader:
        assert list(reader.header.samples) == ["DOG1"]  # samples survive `-s -`
        for record in reader:
            for alt_index, c in missense_consequences(record):
                key = (record.pos, record.alts[alt_index])
                found[key] = f"{c.transcript_id}:{c.ref_aa}{c.protein_pos}{c.alt_aa}"
    return found


def test_csq_recovers_every_designed_missense_on_both_strands(synthetic, tmp_path) -> None:
    """Plus/minus strand, both sides of intron-split codons, multi-allelic sites."""
    expected = {(v.pos, v.alt): f"TX{v.gene}:{v.expected}" for v in DESIGNED_VARIANTS if v.expected}
    expected[(MULTIALLELIC_POS, MULTIALLELIC_ALTS[0])] = "TXA:E6D"  # second ALT is synonymous
    assert _csq_missense(synthetic, tmp_path) == expected


def test_csq_error_is_wrapped(synthetic, tmp_path) -> None:
    with pytest.raises(CsqError):
        run_csq(synthetic.vcf, tmp_path / "missing.fa", synthetic.gff3, tmp_path / "out.vcf.gz")


def test_padded_alleles_and_mixed_multiallelic_sites_are_matched(synthetic, tmp_path) -> None:
    """csq keeps alleles as written (e.g. ``45AA>GA``); they must map to the right ALT."""
    from synthetic import GENOME

    r45, r82 = GENOME[44:46], GENOME[81:83]
    text = synthetic.vcf.read_text()
    text = text.replace("1\t45\t.\tA\tG", f"1\t45\t.\t{r45}\tG{r45[1]}")
    text = text.replace("1\t82\t.\tG\tT", f"1\t82\t.\t{r82}\t{r82[0]},T{r82[1]}")  # del + SNV
    synthetic.vcf.write_text(text)
    found = _csq_missense(synthetic, tmp_path)
    assert found[(45, "G" + r45[1])] == "TXA:K2R"
    assert found[(82, "T" + r82[1])] == "TXA:W4C"
    assert (82, r82[0]) not in found  # the deletion is a frameshift, not missense
