"""Realistic dog benchmark, and comparison with the human-ortholog workaround.

A. Ranking ("needle in a haystack"). For 50 Dog10K breed dogs, the candidates are the rare
   missense variants each dog carries (ALT frequency < 1% among 1,872 dogs, canonical
   transcripts of autosomal coding genes). Each OMIA causal variant is added in turn to each
   dog's candidates and ranked by the score, as a geneticist would rank a sick dog's variants.
B. The same variants scored three ways: ESM-2 on the dog protein (omnimissense), ESM-2 on the
   human ortholog, and AlphaMissense (human). Human scores need the dog variant mapped onto
   the human protein: same gene symbol, global alignment, identical reference residue (the
   "mock variant" strategy). Variants that cannot be mapped get no human score.

Inputs (see benchmarks/README.md): the hand-downloaded OMIA CSV; the Dog10K SNV BCF and its
index in data/dog10k/; AlphaMissense_aa_substitutions.tsv.gz in data/alphamissense/
(CC BY-NC-SA 4.0); UniProt human reviewed proteins in data/human/. Only aggregate metrics are
written to the repository.

    uv run --group benchmarks python benchmarks/dog_realistic.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam
import pysam.bcftools
from Bio.Align import PairwiseAligner, substitution_matrices

sys.path.insert(0, str(Path(__file__).parent))
from blosum import BLOSUM62
from dog_omia import (
    DOG10K_SAMPLES,
    DOG_CATEGORIES,
    DogReference,
    ProteinVariant,
    causal_variants,
    fetch,
    load_reference,
    read_omia,
)
from stats import auroc, bootstrap_ci, bootstrap_diff_ci

from omnimissense.cache import ScoreCache, default_cache_dir
from omnimissense.defaults import DEFAULT_MODEL
from omnimissense.gff import open_text, parse_attributes
from omnimissense.pipeline import (
    LazyScorer,
    VariantStatus,
    annotate,
    cached_masked_scores,
    cached_wt_scores,
)
from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX, sequence_hash

log = logging.getLogger("dog_realistic")

DOG10K_BCF = Path("data/dog10k/AutoAndXPAR.SNPs.bcf")
ALPHAMISSENSE = Path("data/alphamissense/AlphaMissense_aa_substitutions.tsv.gz")
UNIPROT_HUMAN = Path("data/human/uniprot_human_reviewed.tsv")
#: Global alignment of very long proteins is slow and memory hungry; they are not mapped.
MAX_ALIGN_LENGTH = 6000

METHODS = ("dog_esm2_masked", "dog_esm2_wt", "human_esm2_masked", "alphamissense", "blosum62")


@dataclass(frozen=True)
class Candidate:
    """A scored dog missense variant (canonical transcript)."""

    gene: str
    transcript_id: str
    position: int
    ref: str
    alt: str


# ---------------------------------------------------------------------------- Dog10K


def write_canonical_gff3(source: Path, keep: set[str], output: Path) -> None:
    """Copy of an Ensembl GFF3 restricted to the given transcripts (genes are kept)."""
    with open_text(source) as handle, output.open("w") as out:
        for line in handle:
            if line.startswith("#"):
                if not line.startswith("##FASTA"):
                    out.write(line)
                continue
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 9:
                continue
            attributes = parse_attributes(columns[8])
            feature_id = attributes.get("ID", "")
            parents = attributes.get("Parent", "").split(",")
            if feature_id.startswith("gene:"):
                out.write(line)
            elif feature_id.startswith("transcript:"):
                if feature_id.removeprefix("transcript:") in keep:
                    out.write(line)
            elif any(p.removeprefix("transcript:") in keep for p in parents):
                out.write(line)


def rare_variants_of_dogs(
    cds_bed: Path, all_dogs: Path, chosen_dogs: list[str], max_af: float, workdir: Path
) -> dict[tuple[str, int, str, str], set[str]]:
    """Rare PASS SNVs (frequency among all dogs < max_af) carried by the chosen dogs.

    Returns ``{(chrom, pos, ref, alt): {dog, ...}}`` with Ensembl chromosome names.
    """
    rare_all = workdir / "rare_all_dogs.bcf"
    rare_chosen = workdir / "rare_chosen_dogs.bcf"
    chosen_file = workdir / "chosen_dogs.txt"
    chosen_file.write_text("\n".join(chosen_dogs) + "\n")
    if not rare_all.exists():
        log.info("Selecting rare coding SNVs among all dogs (local Dog10K BCF)")
        pysam.bcftools.view(
            "-S", str(all_dogs), "--force-samples", "-R", str(cds_bed), "-f", "PASS",
            "-c", "1:alt1", "-Q", f"{max_af}:alt1", "-Ob", "-o", str(rare_all), str(DOG10K_BCF),
            catch_stdout=False,
        )  # fmt: skip
    if not rare_chosen.exists():
        pysam.bcftools.view(
            "-S", str(chosen_file), "-c", "1:alt1", "-Ob", "-o", str(rare_chosen), str(rare_all),
            catch_stdout=False,
        )  # fmt: skip
    carriers: dict[tuple[str, int, str, str], set[str]] = defaultdict(set)
    previous = pysam.set_verbosity(0)
    try:
        reader = pysam.VariantFile(str(rare_chosen))
    finally:
        pysam.set_verbosity(previous)
    with reader:
        for record in reader:
            chrom = record.chrom.removeprefix("chr")
            for dog in chosen_dogs:
                for allele in set(record.samples[dog]["GT"]):
                    if allele:  # neither missing (None) nor reference (0)
                        carriers[(chrom, record.pos, record.ref, record.alts[allele - 1])].add(dog)
    return carriers


def write_sites_vcf(keys: Iterable[tuple[str, int, str, str]], fasta: Path, output: Path) -> None:
    with pysam.FastaFile(str(fasta)) as genome:
        contigs = list(zip(genome.references, genome.lengths, strict=True))
    lines = ["##fileformat=VCFv4.2"]
    lines += [f"##contig=<ID={c},length={n}>" for c, n in contigs]
    lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    order = {c: i for i, (c, _) in enumerate(contigs)}
    for chrom, pos, ref, alt in sorted(keys, key=lambda k: (order[k[0]], k[1], k[3])):
        lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.")
    output.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------- human mapping


def load_uniprot(path: Path) -> dict[str, tuple[str, str]]:
    """Gene symbol (upper case) -> (UniProt accession, sequence); primary names win."""
    primary, synonyms = {}, {}
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            entry = (row["Entry"], row["Sequence"])
            if row["Gene Names (primary)"]:
                primary.setdefault(row["Gene Names (primary)"].upper(), entry)
            for name in row["Gene Names (synonym)"].split():
                synonyms.setdefault(name.upper(), entry)
    return {**synonyms, **primary}


def alignment_map(dog: str, human: str, aligner: PairwiseAligner) -> np.ndarray:
    """For each dog residue, the aligned human residue index (0-based), or -1."""
    mapping = np.full(len(dog), -1, dtype=np.int64)
    alphabet = set(aligner.substitution_matrix.alphabet)

    def clean(seq: str) -> str:  # e.g. U (selenocysteine) -> X; same length, same positions
        return "".join(c if c in alphabet else "X" for c in seq)

    alignment = aligner.align(clean(dog), clean(human))[0]
    for (d0, d1), (h0, h1) in zip(*alignment.aligned, strict=True):
        mapping[d0:d1] = np.arange(h0, h1)
    return mapping


def make_aligner() -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


def map_to_human(
    variants: Iterable[ProteinVariant | Candidate],
    dog_sequences: dict[str, str],
    uniprot: dict[str, tuple[str, str]],
) -> dict[tuple[str, int, str], tuple[str, int, str, str]]:
    """``(transcript, position, alt)`` -> ``(accession, human position, ref, alt)`` when the
    gene has a human entry and the aligned human residue equals the dog reference residue."""
    aligner = make_aligner()
    maps: dict[tuple[str, str], np.ndarray | None] = {}
    mapped = {}
    for v in variants:
        human = uniprot.get(v.gene.upper())
        if human is None:
            continue
        accession, human_seq = human
        dog_seq = dog_sequences[v.transcript_id]
        key = (v.transcript_id, accession)
        if key not in maps:
            too_long = max(len(dog_seq), len(human_seq)) > MAX_ALIGN_LENGTH
            maps[key] = None if too_long else alignment_map(dog_seq, human_seq, aligner)
        mapping = maps[key]
        if mapping is None:
            continue
        index = int(mapping[v.position - 1])
        if index >= 0 and human_seq[index] == v.ref:
            mapped[(v.transcript_id, v.position, v.alt)] = (accession, index + 1, v.ref, v.alt)
    return mapped


def load_alphamissense(path: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Stream the AlphaMissense table, keeping ``(accession, "R106W")`` keys in ``wanted``."""
    accessions = {a for a, _ in wanted}
    found = {}
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#") or line.startswith("uniprot_id"):
                continue
            accession, _, rest = line.partition("\t")
            if accession not in accessions:
                continue
            variant, score, _ = rest.split("\t", 2)
            if (accession, variant) in wanted:
                found[(accession, variant)] = float(score)
    return found


# ---------------------------------------------------------------------------- metrics


def scored_fraction(score_dicts: list[dict[str, float]], method: str) -> float:
    if not score_dicts:
        return float("nan")
    return round(float(np.mean([np.isfinite(s[method]) for s in score_dicts])), 3)


def rank_in(candidate_scores: np.ndarray, score: float) -> float:
    """1-based rank of ``score`` among candidates (higher = more damaging; ties averaged)."""
    better = np.count_nonzero(candidate_scores > score)
    ties = np.count_nonzero(candidate_scores == score)
    return 1 + better + ties / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--omia-csv", type=Path, default=Path("data/omia/omia_dog_variants.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark_dog"))
    parser.add_argument("--dogs", type=int, default=50)
    parser.add_argument("--max-af", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-prefix", default="benchmarks/results/dog_realistic")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("dog_realistic", "dog_omia", "omnimissense"):
        logging.getLogger(name).setLevel(logging.INFO)
    for path in (args.omia_csv, DOG10K_BCF, ALPHAMISSENSE, UNIPROT_HUMAN):
        if not path.exists():
            sys.exit(f"Missing {path}: see benchmarks/README.md")
    work = args.data_dir / "realistic"
    work.mkdir(parents=True, exist_ok=True)

    ref: DogReference = load_reference()
    omia = read_omia(args.omia_csv)
    positives, mapping_counts = causal_variants(omia, ref)
    positives = [p for p in positives if p.group in ("pathogenic", "disease_unclassified")]
    canonical = {
        txs[0].transcript_id: gene for gene, txs in ref.genes.items() if txs[0].seqid.isdigit()
    }

    # ---- rare variants of 50 dogs
    samples = fetch(DOG10K_SAMPLES, args.data_dir / "dog10k_samples.txt")
    with samples.open() as handle:
        rows = [
            r
            for r in csv.DictReader(handle, delimiter="\t")
            if r["inVCF"] == "YES" and r["inAnalyzedSNV"] == "TRUE"
        ]
    all_dogs = work / "all_dogs.txt"
    all_dogs.write_text(
        "\n".join(r["sampleName"] for r in rows if r["Category"] in DOG_CATEGORIES) + "\n"
    )
    breed_dogs = sorted(r["sampleName"] for r in rows if r["Category"] == "Breed_Dogs")
    chosen = random.Random(args.seed).sample(breed_dogs, args.dogs)
    breeds = {r["sampleName"]: r["Breed/Type"] for r in rows}

    cds_bed = work / "canonical_cds.bed"
    cds_rows = sorted(
        (f"chr{ref.transcripts[t].seqid}", s.start - 1, s.end)
        for t in canonical
        for s in ref.transcripts[t].cds
    )
    cds_bed.write_text("".join(f"{c}\t{a}\t{b}\n" for c, a, b in cds_rows))
    carriers = rare_variants_of_dogs(cds_bed, all_dogs, chosen, args.max_af, work)
    log.info("%d rare SNVs carried by %d dogs", len(carriers), len(chosen))

    sites = work / "rare_sites.vcf"
    write_sites_vcf(carriers, ref.fasta, sites)
    canonical_gff3 = work / "canonical.gff3"
    if not canonical_gff3.exists():
        write_canonical_gff3(ref.gff3, set(canonical), canonical_gff3)

    cache = ScoreCache(default_cache_dir(), args.model)

    def scorer_factory():
        return MaskedLMScorer.from_pretrained(args.model)

    result = annotate(
        sites,
        ref.fasta,
        canonical_gff3,
        work / "rare.omnimissense.vcf.gz",
        work / "rare.omnimissense.tsv",
        cache=cache,
        scorer_factory=scorer_factory,
        model_name=args.model,
    )
    positive_keys = {(p.transcript_id, p.position, p.alt) for p in positives}
    candidates: dict[tuple[str, str, int, str], Candidate] = {}  # site key -> candidate
    for v in result.variants:
        c = v.consequence
        if v.status is not VariantStatus.OK or c.transcript_id not in canonical:
            continue
        if (c.transcript_id, c.protein_pos, c.alt_aa) in positive_keys:
            continue  # a causal variant carried by a healthy dog is not a candidate
        candidates[(v.chrom, v.pos, v.ref, v.alt)] = Candidate(
            canonical[c.transcript_id], c.transcript_id, c.protein_pos, c.ref_aa, c.alt_aa
        )
    per_dog: dict[str, list[Candidate]] = defaultdict(list)
    for site, dogs in carriers.items():
        if site in candidates:
            for dog in dogs:
                per_dog[dog].append(candidates[site])
    log.info(
        "Rare missense candidates: %d unique; median per dog %d",
        len(candidates),
        int(np.median([len(v) for v in per_dog.values()])),
    )

    # ---- common variants (from the dog_omia run) as a second negative set
    common = {}
    common_tsv = args.data_dir / "dog10k_common.omnimissense.tsv"
    if common_tsv.exists():
        with common_tsv.open() as handle:
            for r in csv.DictReader(handle, delimiter="\t"):
                if r["status"] != "ok" or r["transcript"] not in canonical:
                    continue
                key = (r["transcript"], int(r["protein_pos"]), r["alt_aa"])
                if key not in positive_keys:
                    common[key] = Candidate(
                        canonical[r["transcript"]], *key[:2], r["ref_aa"], key[2]
                    )
    else:
        log.warning("%s missing: run benchmarks/dog_omia.py first", common_tsv)

    # ---- scores on dog proteins
    unique_candidates = {(c.transcript_id, c.position, c.alt): c for c in candidates.values()}
    everything = (
        [Candidate(p.gene, p.transcript_id, p.position, p.ref, p.alt) for p in positives]
        + list(unique_candidates.values())
        + list(common.values())
    )
    dog_sequences = {v.transcript_id: ref.proteins[v.transcript_id].sequence for v in everything}
    lazy = LazyScorer(scorer_factory)
    wt = cached_wt_scores({sequence_hash(s): s for s in dog_sequences.values()}, cache, lazy)
    masked = cached_masked_scores(
        {(sequence_hash(dog_sequences[v.transcript_id]), v.position): dog_sequences[v.transcript_id]
         for v in everything},
        cache,
        lazy,
    )  # fmt: skip

    # ---- human ortholog mapping and human scores
    uniprot = load_uniprot(UNIPROT_HUMAN)
    human = map_to_human(everything, dog_sequences, uniprot)
    human_sequences = {acc: seq for acc, seq in uniprot.values()}
    human_queries = {
        (sequence_hash(human_sequences[a]), pos): human_sequences[a]
        for a, pos, _, _ in human.values()
    }
    human_masked = cached_masked_scores(human_queries, cache, lazy)
    am = load_alphamissense(
        ALPHAMISSENSE, {(a, f"{r}{pos}{alt}") for a, pos, r, alt in human.values()}
    )
    log.info(
        "Human mapping: %d / %d variants; AlphaMissense hits: %d",
        len(human),
        len(everything),
        len(am),
    )

    def scores(v: ProteinVariant | Candidate) -> dict[str, float]:
        """Predictors, higher = more likely damaging; -inf = not scored by that method."""
        seq = dog_sequences[v.transcript_id]
        h = sequence_hash(seq)
        out = {
            "dog_esm2_masked": -float(masked[(h, v.position)][AA_INDEX[v.alt]]),
            "dog_esm2_wt": -float(wt[h][0][v.position - 1, AA_INDEX[v.alt]]),
            "human_esm2_masked": -np.inf,
            "alphamissense": -np.inf,
            "blosum62": -float(BLOSUM62[v.ref, v.alt]),
        }
        mapped = human.get((v.transcript_id, v.position, v.alt))
        if mapped:
            accession, pos, r, alt = mapped
            hh = sequence_hash(human_sequences[accession])
            out["human_esm2_masked"] = -float(human_masked[(hh, pos)][AA_INDEX[alt]])
            out["alphamissense"] = am.get((accession, f"{r}{pos}{alt}"), -np.inf)
        return out

    causal_sets = {
        "pathogenic": [p for p in positives if p.group == "pathogenic"],
        "disease": positives,
    }
    causal_scores = {name: [scores(p) for p in ps] for name, ps in causal_sets.items()}
    rare_scores = [scores(c) for c in unique_candidates.values()]
    common_scores = [scores(c) for c in common.values()]

    # ---- AUROC: all variants (unscored = least damaging) and the subset scored by all methods
    auroc_rows = []
    for set_name, positive_scores in causal_scores.items():
        for neg_name, negative_scores in (("rare", rare_scores), ("common", common_scores)):
            if not negative_scores:
                continue
            labels = np.array([1] * len(positive_scores) + [0] * len(negative_scores))
            table = {
                m: np.array([s[m] for s in positive_scores + negative_scores]) for m in METHODS
            }
            all_scored = np.all([np.isfinite(table[m]) for m in METHODS], axis=0)
            for subset_name, mask in (
                ("all variants", np.ones_like(labels, bool)),
                ("scored by all methods", all_scored),
            ):
                y = labels[mask]
                row = {
                    "causal_set": set_name, "negatives": neg_name, "subset": subset_name,
                    "n_causal": int(y.sum()), "n_negative": int((1 - y).sum()),
                }  # fmt: skip
                for m in METHODS:
                    s = np.nan_to_num(table[m][mask], neginf=-1e9)
                    low, high = bootstrap_ci(y, s)
                    row[m] = round(auroc(y, s), 3)
                    row[f"{m}_ci95"] = f"{low:.3f}-{high:.3f}"
                d_low, d_high = bootstrap_diff_ci(
                    y,
                    np.nan_to_num(table["dog_esm2_masked"][mask], neginf=-1e9),
                    np.nan_to_num(table["alphamissense"][mask], neginf=-1e9),
                )
                row["diff_dog_esm2_vs_alphamissense_ci95"] = f"{d_low:+.3f} to {d_high:+.3f}"
                d_low, d_high = bootstrap_diff_ci(
                    y,
                    np.nan_to_num(table["dog_esm2_masked"][mask], neginf=-1e9),
                    np.nan_to_num(table["human_esm2_masked"][mask], neginf=-1e9),
                )
                row["diff_dog_vs_human_esm2_ci95"] = f"{d_low:+.3f} to {d_high:+.3f}"
                auroc_rows.append(row)

    # ---- ranking: each causal variant inserted among each dog's rare missense candidates
    candidate_scores = {
        dog: {m: np.array([scores(c)[m] for c in cands]) for m in METHODS}
        for dog, cands in per_dog.items()
    }
    ranking_rows = []
    for set_name, positive_scores in causal_scores.items():
        for m in METHODS:
            ranks = np.array(
                [
                    rank_in(
                        np.nan_to_num(candidate_scores[d][m], neginf=-1e9),
                        np.nan_to_num(s[m], neginf=-1e9),
                    )
                    for s in positive_scores
                    for d in candidate_scores
                ]
            )
            ranking_rows.append(
                {
                    "causal_set": set_name,
                    "method": m,
                    "n_causal": len(positive_scores),
                    "n_dogs": len(candidate_scores),
                    "median_rank": float(np.median(ranks)),
                    "top1": round(float((ranks <= 1).mean()), 3),
                    "top5": round(float((ranks <= 5).mean()), 3),
                    "top10": round(float((ranks <= 10).mean()), 3),
                    "causal_scored": round(
                        float(np.mean([np.isfinite(s[m]) for s in positive_scores])), 3
                    ),
                }
            )

    n_candidates = [len(v) for v in per_dog.values()]
    summary = {
        "omia_mapping": mapping_counts,
        "dogs": len(chosen),
        "breeds": len({breeds[d] for d in chosen}),
        "max_af_rare": args.max_af,
        "rare_missense_unique": len(unique_candidates),
        "rare_missense_per_dog_median": int(np.median(n_candidates)),
        "rare_missense_per_dog_range": f"{min(n_candidates)}-{max(n_candidates)}",
        "common_missense": len(common),
        "causal_pathogenic": len(causal_sets["pathogenic"]),
        "causal_disease": len(causal_sets["disease"]),
        "human_mappable_causal": scored_fraction(causal_scores["disease"], "human_esm2_masked"),
        "human_mappable_rare": scored_fraction(rare_scores, "human_esm2_masked"),
        "human_mappable_common": scored_fraction(common_scores, "human_esm2_masked"),
        "alphamissense_found_causal": scored_fraction(causal_scores["disease"], "alphamissense"),
        "alphamissense_found_rare": scored_fraction(rare_scores, "alphamissense"),
        "alphamissense_found_common": scored_fraction(common_scores, "alphamissense"),
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, table_rows in (("_auroc.tsv", auroc_rows), ("_ranking.tsv", ranking_rows)):
        with Path(f"{prefix}{suffix}").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(table_rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(table_rows)
    Path(f"{prefix}.summary.tsv").write_text("".join(f"{k}\t{v}\n" for k, v in summary.items()))

    print(
        "\n| causal | negatives | subset | n | dog ESM-2 masked | dog ESM-2 wt "
        "| human ESM-2 | AlphaMissense | BLOSUM62 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in auroc_rows:
        print(
            f"| {r['causal_set']} | {r['negatives']} | {r['subset']} "
            f"| {r['n_causal']}/{r['n_negative']} "
            + " | ".join(f"{r[m]:.3f}" for m in METHODS)
            + " |"
        )
    print("\n| causal | method | median rank | top-1 | top-5 | top-10 | causal scored |")
    print("|---|---|---|---|---|---|---|")
    for r in ranking_rows:
        print(
            f"| {r['causal_set']} | {r['method']} | {r['median_rank']:.1f} | {r['top1']:.0%} "
            f"| {r['top5']:.0%} | {r['top10']:.0%} | {r['causal_scored']:.0%} |"
        )
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
