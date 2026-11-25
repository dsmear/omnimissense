import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))

from dog_omia import parse_protein_changes, read_omia


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("p.Arg106Trp", [("R", 106, "W")]),
        ("p.(Arg106Trp)", [("R", 106, "W")]),
        ("p.R106W", [("R", 106, "W")]),
        ("R106W", [("R", 106, "W")]),
        ("p.Glu40Lys", [("E", 40, "K")]),
        ("p.Arg106*", []),  # nonsense, not missense
        ("p.Arg106=", []),  # synonymous
        ("", []),
    ],
)
def test_parse_protein_changes(text, expected) -> None:
    assert parse_protein_changes(text) == expected


def test_read_omia_categorises_dog_missense(tmp_path) -> None:
    csv_file = tmp_path / "omia.csv"
    header = "OMIA Variant ID,Species Name,Gene,Variant Phenotype,Variant Effect,"
    header += "Pathogenicity Classification,p.\n"
    csv_file.write_text(
        header
        + "1,dog,SOD1,Degenerative myelopathy,missense,P,XP_1.1:p.(E40K)\n"
        + "2,dog,TYRP1,Brown,missense,Not currently evaluated,XP_2.1:p.(Q331*)\n"
        + "3,dog,ABC,Some disease,missense,Not currently evaluated,XP_3.1:p.(R10W)\n"
        + "4,dog,PRCD,Retinal atrophy,nonsense (stop-gain),P,XP_4.1:p.(C2*)\n"
        + "5,domestic cat,TYR,Colourpoint,missense,P,XP_5.1:p.(G301R)\n"
        + "6,raccoon dog,XYZ,Disease,missense,P,XP_6.1:p.(A1V)\n"
        + "7,dog,ATP7B,Wilson disease,missense,Out of scope,XP_7.1:p.(A2V)\n"
    )
    records = read_omia(csv_file)
    assert [(r["id"], r["category"]) for r in records] == [
        ("1", "pathogenic"),
        ("2", "trait"),
        ("3", "disease_unclassified"),
    ]
    assert parse_protein_changes(records[0]["protein"]) == [("E", 40, "K")]
