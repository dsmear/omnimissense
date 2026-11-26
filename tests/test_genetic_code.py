import pytest

from omnimissense.genetic_code import (
    GENETIC_CODES,
    MITOCHONDRIAL_SEQIDS,
    CodeAssignment,
    genetic_code,
)

CSQ_TABLES = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16}
CSQ_TABLES |= {21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33}


def test_tables_are_those_of_bcftools_csq() -> None:
    assert set(GENETIC_CODES) == CSQ_TABLES
    for code in GENETIC_CODES.values():
        assert len(code.codons) == 64
        assert "ATG" in code.starts


@pytest.mark.parametrize("table_id", sorted(CSQ_TABLES))
def test_tables_match_biopython_ncbi_tables(table_id) -> None:
    codon_table = pytest.importorskip("Bio.Data.CodonTable")
    ncbi = codon_table.unambiguous_dna_by_id[table_id]
    ours = GENETIC_CODES[table_id]
    expected = {codon: ncbi.forward_table.get(codon, "*") for codon in ours.codons}
    assert dict(ours.codons) == expected
    assert ours.stops == set(ncbi.stop_codons)
    assert ours.starts == set(ncbi.start_codons)
    # A stop is translated as "*" unless the code also gives it an amino acid (27, 28, 31).
    assert {c for c, aa in ours.codons.items() if aa == "*"} <= ours.stops


def test_unknown_table_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown genetic code 7"):
        genetic_code(7)
    with pytest.raises(ValueError, match="unknown genetic code 32"):
        CodeAssignment(nuclear=32)


def test_code_assignment_defaults() -> None:
    codes = CodeAssignment()
    assert codes.code("1") == 1
    assert all(codes.code(name) is None for name in MITOCHONDRIAL_SEQIDS)  # unknown: skipped
    assert codes.csq_code("MT") == 1


def test_code_assignment_from_annotation_and_overrides() -> None:
    annotated = {"NC_MT": 3, "NC_PLASTID": 11, "NC_CHR1": 12}
    codes = CodeAssignment(annotated=annotated, mito_seqids=frozenset({"NC_MT"}))
    assert codes.code("NC_MT") == 3
    assert codes.code("NC_PLASTID") == 11
    assert codes.code("NC_CHR1") == 12
    assert codes.code("NC_CHR2") == 1
    overridden = CodeAssignment(
        nuclear=6, mitochondrial=4, annotated=annotated, mito_seqids=frozenset({"NC_MT"})
    )
    assert overridden.code("NC_MT") == 4
    assert overridden.code("MT") == 4
    assert overridden.code("NC_PLASTID") == 6  # the option overrides the GFF3
    assert overridden.describe() == {"default": 6, "NC_CHR1": 6, "NC_MT": 4, "NC_PLASTID": 6}


def test_unsupported_annotated_table_needs_an_override() -> None:
    with pytest.raises(ValueError, match="--genetic-code"):
        CodeAssignment(annotated={"NC_PLASTID": 32})
    assert CodeAssignment(nuclear=11, annotated={"NC_PLASTID": 32}).code("NC_PLASTID") == 11
