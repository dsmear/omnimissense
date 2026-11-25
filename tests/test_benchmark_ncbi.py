import sys
from pathlib import Path

from synthetic import PROTEIN_A, PROTEIN_B, ncbi_gff3_lines

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from ncbi_translation import compare, first_difference


def test_compare_with_ncbi_translated_cds(synthetic, tmp_path) -> None:
    gff3 = tmp_path / "ncbi.gff3"
    gff3.write_text("\n".join(ncbi_gff3_lines()) + "\n")
    translated = tmp_path / "translated_cds.faa"
    translated.write_text(
        f">lcl|1_prot_XP_GENEA.1_1 [gene=GENEA] [protein_id=XP_GENEA.1]\n{PROTEIN_A}*\n"
        f">lcl|1_prot_XP_GENEB.1_2 [gene=GENEB] [protein_id=XP_GENEB.1]\n{PROTEIN_B[:-1]}W\n"
        ">lcl|1_prot_XP_OTHER.1_3 [gene=OTHER] [protein_id=XP_OTHER.1]\nMK\n"
    )
    outcome = compare(synthetic.fasta, gff3, translated)
    assert outcome == {
        "identical": 1,
        "different": 1,
        "NCBI proteins without an omnimissense transcript": 1,
    }


def test_first_difference() -> None:
    assert first_difference("MKR", "MKW") == "R3W"
    assert first_difference("MK", "MKR") == "length 2 vs 3"
