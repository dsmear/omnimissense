"""Implementation sanity check on ProteinGym deep mutational scanning (DMS) assays.

Scores every single mutant of a few DMS assays with omnimissense and compares the Spearman
correlation with the experimental fitness to the value published by ProteinGym for the same
model. ProteinGym scores ESM-2 with masked marginals
(``scripts/scoring_DMS_zero_shot/scoring_ESM2_substitutions.sh``), so that is the number to
compare; the wild-type-marginal correlation is reported too.

Data are downloaded at pinned revisions (about 110 MB) into ``data/proteingym``:

    uv run python benchmarks/proteingym.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from models import model_tag
from stats import spearman

from omnimissense.defaults import DEFAULT_MODEL
from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX

log = logging.getLogger("proteingym")

HF_REPO = "OATML-Markslab/ProteinGym_v1"
HF_REVISION = "1ea2aa10a328cc085b64328e790c54755fe32b10"
HF_FILES = [f"DMS_substitutions/train-0000{i}-of-00005.parquet" for i in range(5)]
GITHUB_COMMIT = "144fe22b07dfaeec2b366f2346203a9838a55b4c"
GITHUB_RAW = f"https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/{GITHUB_COMMIT}"
REFERENCE_FILE = "reference_files/DMS_substitutions.csv"
PUBLISHED_FILE = (
    "benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv"
)
PUBLISHED_COLUMN = {DEFAULT_MODEL: "ESM2 (650M)"}

#: One prokaryote, one non-human mammal, one human assay; all single mutants, all <= 1022 aa.
DEFAULT_ASSAYS = [
    "BLAT_ECOLX_Stiffler_2015",
    "KCNJ2_MOUSE_Coyote-Maestas_2022_function",
    "SC6A4_HUMAN_Young_2021",
]
#: Proteins longer than 1022 residues. ProteinGym scores them with one 1022-residue window
#: centred on each mutation ("optimal" window); omnimissense averages overlapping windows, so
#: these assays compare the two strategies and are informational, not pass/fail.
LONG_ASSAYS = [
    "UBE4B_MOUSE_Starita_2013",
    "NPC1_HUMAN_Erwood_2022_HEK293T",
    "BRCA1_HUMAN_Findlay_2018",
]
#: |ours - published| above this (single-window assays) is treated as a bug, not as noise.
TOLERANCE = 0.03


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, path)
    return path


def load_assay(parquet_files: list[Path], dms_id: str) -> tuple[str, list[str], np.ndarray]:
    sequences, mutants, scores = set(), [], []
    for path in parquet_files:
        table = pq.read_table(
            path, columns=["target_seq", "mutant", "DMS_score"], filters=[("DMS_id", "=", dms_id)]
        )
        sequences.update(table.column("target_seq").to_pylist())
        mutants.extend(table.column("mutant").to_pylist())
        scores.extend(table.column("DMS_score").to_pylist())
    if len(sequences) != 1:
        raise ValueError(f"{dms_id}: expected one target sequence, found {len(sequences)}")
    return sequences.pop(), mutants, np.asarray(scores, dtype=np.float64)


def parse_single(mutant: str, sequence: str) -> tuple[int, str]:
    """``"A123C"`` -> (123, "C"), checking that residue 123 of ``sequence`` is A."""
    wt, position, mt = mutant[0], int(mutant[1:-1]), mutant[-1]
    if sequence[position - 1] != wt:
        raise ValueError(f"wild type mismatch for {mutant}: sequence has {sequence[position - 1]}")
    return position, mt


def score_assay(
    scorer: MaskedLMScorer, sequence: str, mutants: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    parsed = [parse_single(m, sequence) for m in mutants]
    [(wt_matrix, _)] = scorer.wt_marginals([sequence])
    positions = sorted({p for p, _ in parsed})
    masked = dict(
        zip(positions, scorer.masked_marginals([(sequence, p) for p in positions]), strict=True)
    )
    wt_scores = np.array([wt_matrix[p - 1, AA_INDEX[mt]] for p, mt in parsed])
    masked_scores = np.array([masked[p][AA_INDEX[mt]] for p, mt in parsed])
    return wt_scores, masked_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--assays", nargs="+", default=DEFAULT_ASSAYS + LONG_ASSAYS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=Path("data/proteingym"))
    parser.add_argument(
        "--output", type=Path, help="default: benchmarks/results/proteingym_<model>.tsv"
    )
    args = parser.parse_args()
    args.output = args.output or Path(f"benchmarks/results/proteingym_{model_tag(args.model)}.tsv")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("proteingym", "omnimissense"):
        logging.getLogger(name).setLevel(logging.INFO)

    reference = {
        row["DMS_id"]: row
        for row in csv.DictReader(
            fetch(f"{GITHUB_RAW}/{REFERENCE_FILE}", args.data_dir / "DMS_substitutions.csv").open()
        )
    }
    published = {
        row["DMS ID"]: row
        for row in csv.DictReader(
            fetch(f"{GITHUB_RAW}/{PUBLISHED_FILE}", args.data_dir / "spearman_dms_level.csv").open()
        )
    }
    parquet_files = [
        Path(
            hf_hub_download(
                HF_REPO,
                name,
                repo_type="dataset",
                revision=HF_REVISION,
                local_dir=args.data_dir / "hf",
            )
        )
        for name in HF_FILES
    ]

    scorer = MaskedLMScorer.from_pretrained(args.model)
    column = PUBLISHED_COLUMN.get(args.model)
    rows = []
    for dms_id in args.assays:
        sequence, mutants, dms_scores = load_assay(parquet_files, dms_id)
        info = reference[dms_id]
        expected = int(info["DMS_number_single_mutants"])
        if len(mutants) != expected:
            raise ValueError(f"{dms_id}: {len(mutants)} mutants, reference file says {expected}")
        start = time.perf_counter()
        wt_scores, masked_scores = score_assay(scorer, sequence, mutants)
        seconds = time.perf_counter() - start
        rho_masked = spearman(masked_scores, dms_scores)
        rho_wt = spearman(wt_scores, dms_scores)
        reference_rho = float(published[dms_id][column]) if column else float("nan")
        rows.append(
            {
                "DMS_id": dms_id,
                "organism": info["source_organism"],
                "taxon": info["taxon"],
                "seq_len": len(sequence),
                "windows": "multiple" if len(sequence) > 1022 else "single",
                "n_mutants": len(mutants),
                "spearman_masked": round(rho_masked, 4),
                "spearman_wt": round(rho_wt, 4),
                "published_masked": reference_rho,
                "difference": round(rho_masked - reference_rho, 4),
                "seconds": round(seconds, 1),
            }
        )
        log.info(
            "%s: masked %.3f (published %.3f), wt %.3f", dms_id, rho_masked, reference_rho, rho_wt
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nModel: {args.model} (ProteinGym column: {column})")
    print(
        "| DMS assay | taxon | length | mutants | Spearman masked (ours) | published "
        "| difference | Spearman wt (ours) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['DMS_id']} | {r['taxon']} | {r['seq_len']} | {r['n_mutants']} "
            f"| {r['spearman_masked']:.3f} | {r['published_masked']:.3f} "
            f"| {r['difference']:+.3f} | {r['spearman_wt']:.3f} |"
        )
    if column is None:
        print("\nProteinGym publishes no reference for this model: no pass/fail check.")
        return 0
    strict = [r for r in rows if r["windows"] == "single"]
    failures = [r for r in strict if not abs(r["difference"]) <= TOLERANCE]
    if failures:
        print(f"\nDIVERGENCE above {TOLERANCE}: {[r['DMS_id'] for r in failures]} - investigate.")
        return 1
    print(f"\nAll {len(strict)} single-window assays within {TOLERANCE} of the published values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
