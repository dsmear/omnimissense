"""Full ProteinGym substitution benchmark, summarised by taxon.

Scores all 217 deep mutational scanning (DMS) assays of ProteinGym (2.47 M mutants; a
multiple mutant gets the sum of its single-substitution scores, as ProteinGym does for
single-sequence models) and aggregates the Spearman correlations the way ProteinGym does:
mean per protein (UniProt ID), then mean over the proteins of each taxon (Human, other
Eukaryote, Prokaryote, Virus), and for the overall average, mean over the five function
types. Published ProteinGym models are summarised on exactly the same assays, so all
numbers are comparable; the difference from ESM-2 650M (the base of VESM 650M) gets a 95%
bootstrap interval over proteins.

    uv run python benchmarks/proteingym_all.py                      # VESM 650M, wild-type
    uv run python benchmarks/proteingym_all.py --model facebook/esm2_t33_650M_UR50D --masked

Wild-type marginals need one forward pass per protein (187 distinct sequences, 73 k
residues: minutes on a GPU). ``--masked`` adds masked marginals, one pass per mutated
position (hours). Results: ``benchmarks/results/proteingym_all_<model>.tsv`` (one row per
assay, appended as it goes: an interrupted run resumes) and ``….summary.tsv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
from models import model_tag
from proteingym import (
    GITHUB_RAW,
    HF_FILES,
    HF_REPO,
    HF_REVISION,
    PUBLISHED_FILE,
    REFERENCE_FILE,
    TOLERANCE,
    fetch,
    parse_single,
)
from stats import spearman

from omnimissense.defaults import DEFAULT_MODEL, ESM2_650M
from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX

log = logging.getLogger("proteingym_all")

TAXA = ("Human", "Eukaryote", "Prokaryote", "Virus")
#: Published models shown next to ours: the same base model, other single-sequence models,
#: alignment-based models, and one of the best models overall.
COMPARE = (
    "ESM2 (650M)",
    "ESM-1v (ensemble)",
    "ESM-C (600M)",
    "GEMME",
    "TranceptEVE L",
    "VenusREM",
)
BASELINE = "ESM2 (650M)"
AVERAGE = "Average (by function)"


# ---------------------------------------------------------------------------- scoring


def parse_mutant(mutant: str, sequence: str) -> list[tuple[int, str]]:
    """``"A12C:D45E"`` -> ``[(12, "C"), (45, "E")]``, checking every wild-type residue."""
    return [parse_single(single, sequence) for single in mutant.split(":")]


def additive(parsed: list[list[tuple[int, str]]], lookup: Callable[[int, str], float]):
    """Score of each mutant = sum of the scores of its substitutions."""
    return np.array([sum(lookup(p, aa) for p, aa in subs) for subs in parsed], dtype=np.float64)


def score_assay(
    scorer: MaskedLMScorer, sequence: str, mutants: list[str], masked: bool
) -> dict[str, np.ndarray]:
    parsed = [parse_mutant(m, sequence) for m in mutants]
    [(wt_matrix, _)] = scorer.wt_marginals([sequence])
    scores = {"wt": additive(parsed, lambda p, aa: wt_matrix[p - 1, AA_INDEX[aa]])}
    if masked:
        positions = sorted({p for subs in parsed for p, _ in subs})
        vectors = scorer.masked_marginals([(sequence, p) for p in positions])
        by_position = dict(zip(positions, vectors, strict=True))
        scores["masked"] = additive(parsed, lambda p, aa: by_position[p][AA_INDEX[aa]])
    return scores


def load_all_assays(parquet_files: Iterable[Path]) -> dict[str, tuple[str, list[str], np.ndarray]]:
    """``{DMS_id: (target sequence, mutants, DMS scores)}`` from the ProteinGym Parquet files."""
    columns = ["DMS_id", "target_seq", "mutant", "DMS_score"]
    table = pa.concat_tables(pq.read_table(path, columns=columns) for path in parquet_files)
    ids = np.asarray(table.column("DMS_id").to_pylist(), dtype=object)
    order = np.argsort(ids, kind="stable")
    unique, starts = np.unique(ids[order], return_index=True)
    ends = [*starts[1:], len(order)]
    assays = {}
    for dms_id, start, end in zip(unique, starts, ends, strict=True):
        rows = pa.array(order[start:end])
        sequences = set(table.column("target_seq").take(rows).to_pylist())
        if len(sequences) != 1:
            raise ValueError(f"{dms_id}: expected one target sequence, found {len(sequences)}")
        mutants = table.column("mutant").take(rows).to_pylist()
        scores = np.asarray(table.column("DMS_score").take(rows).to_pylist(), dtype=np.float64)
        assays[str(dms_id)] = (sequences.pop(), mutants, scores)
    return assays


# ---------------------------------------------------------------------------- aggregation


def protein_mean(rows: Iterable[Mapping], value: Callable[[Mapping], float]) -> dict[str, float]:
    """Mean value per protein (``UniProt_ID``), as ProteinGym averages assays."""
    by_protein: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_protein[row["UniProt_ID"]].append(value(row))
    return {protein: float(np.mean(values)) for protein, values in by_protein.items()}


def group_score(rows: list[Mapping], value: Callable[[Mapping], float], group: str) -> float:
    """ProteinGym's aggregate: mean over proteins of a taxon; for ``AVERAGE``, mean over the
    function types of that (verified to reproduce the published summary to 0.001)."""
    if group == AVERAGE:
        functions = sorted({r["function"] for r in rows})
        return float(
            np.mean(
                [
                    np.mean(
                        list(protein_mean([r for r in rows if r["function"] == f], value).values())
                    )
                    for f in functions
                ]
            )
        )
    selected = [r for r in rows if r["taxon"] == group]
    return float(np.mean(list(protein_mean(selected, value).values()))) if selected else np.nan


def paired_protein_ci(
    rows: list[Mapping],
    ours: Callable[[Mapping], float],
    theirs: Callable[[Mapping], float],
    n: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean difference ours - theirs over proteins, with a 95% bootstrap interval (proteins
    resampled)."""
    diffs = np.array(list(protein_mean(rows, lambda r: ours(r) - theirs(r)).values()))
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(n, len(diffs)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(diffs.mean()), float(low), float(high)


def summarise(rows: list[dict], published: Mapping[str, Mapping[str, str]]) -> list[dict]:
    """One summary row per group (overall average, then each taxon)."""
    ours = [key for key in ("spearman_wt", "spearman_masked") if rows and rows[0].get(key) != ""]
    summary = []
    for group in (AVERAGE, *TAXA):
        selected = rows if group == AVERAGE else [r for r in rows if r["taxon"] == group]
        if not selected:
            continue
        row: dict[str, object] = {
            "group": group,
            "n_assays": len(selected),
            "n_proteins": len({r["UniProt_ID"] for r in selected}),
        }
        for key in ours:
            row[f"ours_{key.removeprefix('spearman_')}"] = round(
                group_score(rows, lambda r, k=key: float(r[k]), group), 3
            )
        for model in COMPARE:
            row[model] = round(
                group_score(rows, lambda r, m=model: float(published[r["DMS_id"]][m]), group), 3
            )
        key = ours[0]
        diff, low, high = paired_protein_ci(
            selected,
            lambda r, k=key: float(r[k]),
            lambda r: float(published[r["DMS_id"]][BASELINE]),
        )
        row[f"ours_{key.removeprefix('spearman_')} - {BASELINE}"] = round(diff, 3)
        row["95% CI"] = f"{low:+.3f} to {high:+.3f}"
        summary.append(row)
    return summary


# ---------------------------------------------------------------------------- results file

FIELDS = [
    "DMS_id",
    "UniProt_ID",
    "taxon",
    "function",
    "seq_len",
    "n_mutants",
    "n_multiple",
    "spearman_wt",
    "spearman_masked",
    "seconds",
]


def read_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{path} has other columns: delete it or choose another --output")
        return list(reader)


def append_result(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path: Path, summary: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)


def print_summary(model: str, summary: list[dict]) -> None:
    columns = list(summary[0])
    print(f"\nModel: {model}. Spearman, ProteinGym aggregation (mean per protein, then group).")
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    for row in summary:
        print("| " + " | ".join(str(row[c]) for c in columns) + " |")


def implementation_check(rows: list[dict], published: Mapping[str, Mapping[str, str]]) -> None:
    """For ESM-2 650M masked marginals: agreement with ProteinGym's own ESM-2 scores."""
    single = [r for r in rows if int(r["seq_len"]) <= 1022]
    diffs = np.array(
        [float(r["spearman_masked"]) - float(published[r["DMS_id"]][BASELINE]) for r in single]
    )
    print(
        f"\nImplementation check, ESM-2 masked vs ProteinGym ({len(single)} single-window "
        f"assays): max |difference| {np.abs(diffs).max():.3f}, "
        f"{int((np.abs(diffs) > TOLERANCE).sum())} above {TOLERANCE}"
    )


def run_benchmark(
    assays: Mapping[str, tuple[str, list[str], np.ndarray]],
    reference: Mapping[str, Mapping[str, str]],
    wanted: list[str],
    scorer_factory: Callable[[], MaskedLMScorer],
    output: Path,
    masked: bool,
) -> list[dict]:
    """Score the ``wanted`` assays not yet in ``output`` (appending one row each); return
    the rows of all ``wanted`` assays. The model is loaded only if something is left."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_results(output)
    done = {r["DMS_id"] for r in rows}
    todo = [d for d in wanted if d not in done]
    log.info("%d assays: %d already in %s, %d to score", len(wanted), len(done), output, len(todo))
    scorer = scorer_factory() if todo else None
    # shortest proteins first: quick feedback, and long ones last if memory runs out
    for i, dms_id in enumerate(sorted(todo, key=lambda d: int(reference[d]["seq_len"])), 1):
        sequence, mutants, dms_scores = assays[dms_id]
        info = reference[dms_id]
        start = time.perf_counter()
        scores = score_assay(scorer, sequence, mutants, masked)
        row = {
            "DMS_id": dms_id,
            "UniProt_ID": info["UniProt_ID"],
            "taxon": info["taxon"],
            "function": info["coarse_selection_type"],
            "seq_len": len(sequence),
            "n_mutants": len(mutants),
            "n_multiple": sum(":" in m for m in mutants),
            "spearman_wt": round(spearman(scores["wt"], dms_scores), 4),
            "spearman_masked": round(spearman(scores["masked"], dms_scores), 4) if masked else "",
            "seconds": round(time.perf_counter() - start, 1),
        }
        append_result(output, row)
        rows.append({k: str(v) for k, v in row.items()})
        log.info(
            "[%d/%d] %s (%s, %d aa): wt %.3f%s",
            i,
            len(todo),
            dms_id,
            info["taxon"],
            len(sequence),
            row["spearman_wt"],
            f", masked {row['spearman_masked']:.3f}" if masked else "",
        )
    wanted_set = set(wanted)
    return [r for r in rows if r["DMS_id"] in wanted_set]


# ---------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--masked", action="store_true", help="also masked marginals (slow)")
    parser.add_argument("--taxon", choices=TAXA, help="only the assays of one taxon")
    parser.add_argument("--data-dir", type=Path, default=Path("data/proteingym"))
    parser.add_argument("--max-tokens", type=int, default=None, help="lower on GPU OOM")
    parser.add_argument(
        "--output", type=Path, help="default: benchmarks/results/proteingym_all_<model>.tsv"
    )
    args = parser.parse_args()
    tag = model_tag(args.model) + ("_masked" if args.masked else "")
    args.output = args.output or Path(f"benchmarks/results/proteingym_all_{tag}.tsv")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("proteingym_all", "omnimissense"):
        logging.getLogger(name).setLevel(logging.INFO)

    reference_csv = fetch(f"{GITHUB_RAW}/{REFERENCE_FILE}", args.data_dir / "DMS_substitutions.csv")
    reference = {row["DMS_id"]: row for row in csv.DictReader(reference_csv.open())}
    published_csv = fetch(
        f"{GITHUB_RAW}/{PUBLISHED_FILE}", args.data_dir / "spearman_dms_level.csv"
    )
    published = {row["DMS ID"]: row for row in csv.DictReader(published_csv.open())}
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

    log.info("Loading all assays")
    assays = load_all_assays(parquet_files)
    wanted = [d for d in reference if not args.taxon or reference[d]["taxon"] == args.taxon]
    for dms_id in wanted:  # check every assay before spending GPU time
        expected = int(reference[dms_id]["DMS_total_number_mutants"])
        found = len(assays[dms_id][1]) if dms_id in assays else 0
        if found != expected:
            raise ValueError(f"{dms_id}: {found} mutants, reference file says {expected}")

    kwargs = {"max_tokens": args.max_tokens} if args.max_tokens else {}
    rows = run_benchmark(
        assays,
        reference,
        wanted,
        lambda: MaskedLMScorer.from_pretrained(args.model, **kwargs),
        args.output,
        args.masked,
    )
    summary = summarise(rows, published)
    summary_path = args.output.with_suffix(".summary.tsv")
    write_summary(summary_path, summary)
    print_summary(args.model, summary)
    if args.masked and args.model == ESM2_650M:
        implementation_check(rows, published)
    print(f"\nPer assay: {args.output}\nSummary:   {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
