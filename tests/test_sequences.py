from omnimissense.sequences import (
    AMINO_ACIDS,
    CODON_TABLE,
    reverse_complement,
    sequence_hash,
    translate,
)


def test_reverse_complement() -> None:
    assert reverse_complement("ATGCN") == "NGCAT"
    assert reverse_complement("aacg") == "cgtt"


def test_codon_table_is_complete_standard_code() -> None:
    assert len(CODON_TABLE) == 64
    assert set(CODON_TABLE.values()) == set(AMINO_ACIDS) | {"*"}
    assert CODON_TABLE["ATG"] == "M"
    assert {c for c, aa in CODON_TABLE.items() if aa == "*"} == {"TAA", "TAG", "TGA"}
    assert CODON_TABLE["TGG"] == "W"


def test_translate_keeps_stops_marks_ambiguous_and_drops_partial_codon() -> None:
    assert translate("ATGAAATAA") == "MK*"
    assert translate("ATGNNNTGG") == "MXW"
    assert translate("ATGAAATG") == "MK"
    assert translate("atgaaa") == "MK"


def test_translate_with_another_genetic_code() -> None:
    assert translate("ATGTGAAGA", table=1) == "M*R"
    assert translate("ATGTGAAGA", table=2) == "MW*"  # vertebrate mitochondrial
    assert translate("CTGTAA", table=12) == "S*"  # alternative yeast nuclear (CTG clade)
    assert translate("TAATAG", table=6) == "QQ"  # ciliate nuclear


def test_sequence_hash_is_stable_and_distinct() -> None:
    assert sequence_hash("MKT") == sequence_hash("MKT")
    assert sequence_hash("MKT") != sequence_hash("MKS")
    assert len(sequence_hash("MKT")) == 64
