"""Dog benchmark: known causal missense variants (OMIA) vs common dog missense variants (Dog10K).

Causal sets (OMIA dog variants whose effect is "missense", mapped to UU_Cfam_GSD_1.0 proteins
of Ensembl 116 by gene symbol and amino-acid change; the reference residue must match):
  pathogenic   AVCG classification P, LP or LP/P (curated; main analysis)
  disease      pathogenic + disease variants not yet classified (traits excluded, see TRAITS)
  trait        coat colour / hair / pattern variants (informational: altered protein, no disease)
Negatives: Dog10K single-nucleotide variants with ALT allele frequency >= 5% among 1,872
dogs (wolves and coyotes excluded), in the coding sequence of (a) the same genes as the
positives and (b) randomly chosen protein-coding genes; annotated with omnimissense itself.
Variants in both sets are removed from the negatives. Scores: masked marginal (main) and
wild-type marginal, ESM-2 650M; BLOSUM62 as a context-free baseline. Only aggregate metrics
are written to the repository.

OMIA data cannot be fetched by scripts (robots.txt disallows /download*). Download the CSV
by hand, see benchmarks/README.md, then run:

    uv run python benchmarks/dog_omia.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import re
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam
import pysam.bcftools
import pysam.utils

sys.path.insert(0, str(Path(__file__).parent))
from blosum import BLOSUM62
from models import model_tag
from stats import auroc, bootstrap_ci

from omnimissense.cache import ScoreCache, default_cache_dir
from omnimissense.defaults import DEFAULT_MODEL
from omnimissense.fasta import prepare_fasta
from omnimissense.gff import Transcript, read_transcripts
from omnimissense.pipeline import (
    LazyScorer,
    VariantStatus,
    annotate,
    cached_masked_scores,
    cached_wt_scores,
)
from omnimissense.proteome import Protein, ProteinStatus, build_proteome
from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX, sequence_hash

log = logging.getLogger("dog_omia")

ENSEMBL = "https://ftp.ensembl.org/pub/release-116"
GSD_FASTA = "Canis_lupus_familiarisgsd.UU_Cfam_GSD_1.0.dna.toplevel.fa.gz"
GSD_GFF3 = "Canis_lupus_familiarisgsd.UU_Cfam_GSD_1.0.116.gff3.gz"
GSD_URLS = {
    GSD_FASTA: f"{ENSEMBL}/fasta/canis_lupus_familiarisgsd/dna/{GSD_FASTA}",
    GSD_GFF3: f"{ENSEMBL}/gff3/canis_lupus_familiarisgsd/{GSD_GFF3}",
}
ROS_FASTA = "Canis_lupus_familiaris.ROS_Cfam_1.0.dna.toplevel.fa.gz"
ROS_GFF3 = "Canis_lupus_familiaris.ROS_Cfam_1.0.116.gff3.gz"
ROS_URLS = {
    ROS_FASTA: f"{ENSEMBL}/fasta/canis_lupus_familiaris/dna/{ROS_FASTA}",
    ROS_GFF3: f"{ENSEMBL}/gff3/canis_lupus_familiaris/{ROS_GFF3}",
}
DOG10K = "https://zenodo.org/records/8084059/files"
DOG10K_SNPS = f"{DOG10K}/AutoAndXPAR.SNPs.bcf"
DOG10K_SAMPLES = f"{DOG10K}/dog10K-alignment-sample-table.2022-02-23-v8.txt.UPDATESRA.txt"
DOG_CATEGORIES = {"Breed_Dogs", "Village_Dogs", "Mixed/Other"}

THREE_LETTER = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}  # fmt: skip
_P_THREE = re.compile(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
_P_ONE = re.compile(r"(?<![A-Za-z])([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])(?![a-z])")


def parse_protein_changes(text: str) -> list[tuple[str, int, str]]:
    """Single amino-acid substitutions in HGVS-like text: p.Arg106Trp, p.(Arg106Trp), p.R106W."""
    found = []
    for ref, pos, alt in _P_THREE.findall(text):
        if ref in THREE_LETTER and alt in THREE_LETTER and ref != alt:
            found.append((THREE_LETTER[ref], int(pos), THREE_LETTER[alt]))
    if not found:
        for ref, pos, alt in _P_ONE.findall(text.replace("p.", " ")):
            if ref != alt:
                found.append((ref, int(pos), alt))
    return found


@dataclass(frozen=True)
class ProteinVariant:
    gene: str
    transcript_id: str
    position: int  # 1-based
    ref: str
    alt: str
    label: int  # 1 = causal (OMIA), 0 = common (Dog10K)
    group: str  # OMIA category, or "same_gene" / "random_gene" for common variants


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s", url)
        tmp = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


#: OMIA phenotypes (dog missense, "Not currently evaluated") that are traits, not diseases.
#: Listed explicitly from the OMIA export used here; a modifier is excluded as well.
TRAITS = {
    "Black melanistic mask",
    "Brown",
    "Coat colour, reduced expression of eumelanin",
    "Coat colour, sable",
    "Coat colour, white or cream",
    "Curly coat",
    "Dilute",
    "Grizzle",
    "Harlequin",
    "Himalayan coat colour",
    "Liver",
    "Long hair",
    "Recessive black",
    "reduced hair shedding",
}
EXCLUDED = {"Modifier of copper toxicosis"}
PATHOGENIC_CLASSES = {"P", "LP", "LP/P"}


def read_omia(path: Path) -> list[dict[str, str]]:
    """Dog missense records of the OMIA variant CSV, with a ``category``.

    Categories: ``pathogenic`` (AVCG P/LP), ``disease_unclassified``, ``trait``.
    """
    records = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["Species Name"].strip() != "dog":
                continue
            if row["Variant Effect"].strip() != "missense":
                continue
            phenotype = row["Variant Phenotype"].strip()
            classification = row["Pathogenicity Classification"].strip()
            if classification in PATHOGENIC_CLASSES:
                category = "pathogenic"
            elif classification == "Not currently evaluated" and phenotype in TRAITS:
                category = "trait"
            elif classification == "Not currently evaluated" and phenotype not in EXCLUDED:
                category = "disease_unclassified"
            else:
                continue
            records.append(
                {
                    "id": row["OMIA Variant ID"],
                    "gene": row["Gene"].strip(),
                    "protein": row["p."],
                    "category": category,
                }
            )
    return records


def gene_index(transcripts: dict[str, Transcript]) -> dict[str, list[Transcript]]:
    """Coding transcripts by upper-case gene name, canonical first, then longest CDS."""
    index: dict[str, list[Transcript]] = defaultdict(list)
    for tx in transcripts.values():
        if tx.cds and tx.gene_name and tx.biotype == "protein_coding":
            index[tx.gene_name.upper()].append(tx)
    for txs in index.values():
        txs.sort(key=lambda t: (not t.canonical, -sum(s.end - s.start + 1 for s in t.cds)))
    return index


def map_omia(
    records: list[dict[str, str]],
    sources: list[tuple[dict[str, list[Transcript]], dict[str, Protein]]],
) -> tuple[list[ProteinVariant], dict[str, int]]:
    """Map OMIA protein changes onto proteins, trying each (genes, proteins) source in order."""
    counts = defaultdict(int)
    mapped = {}
    for record in records:
        changes = parse_protein_changes(record["protein"])
        if len(changes) != 1:
            counts["no_single_protein_change"] += 1
            continue
        ref, position, alt = changes[0]
        candidates = [
            (tx, proteins)
            for genes, proteins in sources
            for tx in genes.get(record["gene"].upper(), [])
        ]
        if not candidates:
            counts["gene_not_in_annotation"] += 1
            continue
        for tx, proteins in candidates:
            protein = proteins.get(tx.transcript_id)
            if protein and protein.status is ProteinStatus.OK:
                seq = protein.sequence
                if position <= len(seq) and seq[position - 1] == ref:
                    key = (tx.gene_name.upper(), position, alt)
                    mapped.setdefault(
                        key,
                        ProteinVariant(
                            tx.gene_name,
                            tx.transcript_id,
                            position,
                            ref,
                            alt,
                            1,
                            record["category"],
                        ),
                    )
                    counts["mapped"] += 1
                    break
        else:
            counts["reference_residue_mismatch"] += 1
    counts["mapped_unique"] = len(mapped)
    return list(mapped.values()), dict(counts)


def write_cds_bed(transcripts: list[Transcript], path: Path) -> None:
    """BED (Dog10K naming: chr1...) of CDS segments; autosomes and the X PAR only."""
    rows = sorted(
        (f"chr{tx.seqid}", s.start - 1, s.end)
        for tx in transcripts
        if tx.seqid.isdigit()
        for s in tx.cds
    )
    path.write_text("".join(f"{c}\t{a}\t{b}\n" for c, a, b in rows))


def _query_dog10k(bed: Path, dogs: Path, min_af: float, attempts: int = 3) -> str:
    """One remote bcftools query, retried: long range reads over HTTPS occasionally fail."""
    for attempt in range(1, attempts + 1):
        previous = pysam.set_verbosity(0)
        try:
            return pysam.bcftools.view(
                "-S", str(dogs), "--force-samples", "-R", str(bed), "-f", "PASS",
                "-q", f"{min_af}:alt1", "-G", "-Ov",
                f"{DOG10K_SNPS}##idx##{DOG10K_SNPS}.csi",
                catch_stdout=True,
            )  # fmt: skip
        except pysam.utils.SamtoolsError as exc:
            if attempt == attempts:
                raise
            log.warning("Dog10K query failed (%s), retrying (%d/%d)", exc, attempt, attempts)
            time.sleep(10 * attempt)
        finally:
            pysam.set_verbosity(previous)
    raise AssertionError("unreachable")


def extract_common_snvs(bed: Path, dogs: Path, fasta: Path, output: Path, min_af: float) -> None:
    """Remote-query Dog10K chromosome by chromosome (each part cached on disk) and keep PASS
    SNVs with ALT frequency >= min_af among dogs, renamed to Ensembl chromosome names."""
    regions: dict[str, list[str]] = defaultdict(list)
    for line in bed.read_text().splitlines():
        regions[line.split("\t")[0]].append(line)
    parts_dir = output.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    header: list[str] = []
    records: dict[tuple[str, ...], str] = {}
    for chrom in sorted(regions, key=lambda c: int(c.removeprefix("chr"))):
        part = parts_dir / f"{chrom}.vcf"
        if not part.exists():
            chrom_bed = parts_dir / f"{chrom}.bed"
            chrom_bed.write_text("\n".join(regions[chrom]) + "\n")
            log.info("Dog10K %s: %d regions", chrom, len(regions[chrom]))
            text = _query_dog10k(chrom_bed, dogs, min_af)
            tmp = part.with_suffix(".tmp")
            tmp.write_text(text)
            tmp.rename(part)
        for line in part.read_text().splitlines():
            if line.startswith("#"):
                if not header and line.startswith("#CHROM"):
                    header = [h for h in part.read_text().splitlines() if h.startswith("##")]
                    header = [h for h in header if not h.startswith("##contig")] + [line]
                continue
            fields = line.split("\t")
            fields[0] = fields[0].removeprefix("chr")  # Dog10K chr1.. -> Ensembl 1..
            records.setdefault(tuple(fields[:5]), "\t".join(fields))  # overlaps repeat records
    with pysam.FastaFile(str(fasta)) as genome:
        contigs = [
            f"##contig=<ID={c},length={n}>"
            for c, n in zip(genome.references, genome.lengths, strict=True)
        ]
    body = sorted(records.values(), key=lambda r: (int(r.split("\t")[0]), int(r.split("\t")[1])))
    output.write_text("\n".join(header[:-1] + contigs + header[-1:] + body) + "\n")
    log.info("%d common SNVs written to %s", len(records), output)


@dataclass
class DogReference:
    """UU_Cfam_GSD_1.0 (Dog10K's assembly) plus ROS_Cfam_1.0 as a gene-symbol fallback."""

    fasta: Path
    gff3: Path
    transcripts: dict[str, Transcript]
    genes: dict[str, list[Transcript]]
    ros_genes: dict[str, list[Transcript]]
    proteins: dict[str, Protein]  # GSD proteins of coding genes + ROS proteins of OMIA genes
    ros_fasta: Path
    ros_transcripts: dict[str, Transcript]


def load_reference() -> DogReference:
    gsd = Path("data/dog/gsd-ensembl-116")
    fasta = prepare_fasta(fetch(GSD_URLS[GSD_FASTA], gsd / GSD_FASTA))
    gff3 = fetch(GSD_URLS[GSD_GFF3], gsd / GSD_GFF3)
    transcripts = read_transcripts(gff3)
    genes = gene_index(transcripts)
    proteins = build_proteome(
        fasta, transcripts, [tx.transcript_id for txs in genes.values() for tx in txs]
    )
    ros = Path("data/dog/ensembl-116")
    ros_fasta = prepare_fasta(fetch(ROS_URLS[ROS_FASTA], ros / ROS_FASTA))
    ros_transcripts = read_transcripts(fetch(ROS_URLS[ROS_GFF3], ros / ROS_GFF3))
    return DogReference(
        fasta, gff3, transcripts, genes, gene_index(ros_transcripts), proteins, ros_fasta,
        ros_transcripts,
    )  # fmt: skip


def causal_variants(
    omia: list[dict[str, str]], ref: DogReference
) -> tuple[list[ProteinVariant], dict[str, int]]:
    """Map OMIA records to GSD proteins, falling back to ROS_Cfam_1.0 (adds ROS proteins to
    ``ref.proteins``)."""
    omia_genes = {r["gene"].upper() for r in omia}
    ros_proteins = build_proteome(
        ref.ros_fasta,
        ref.ros_transcripts,
        [tx.transcript_id for g in omia_genes for tx in ref.ros_genes.get(g, [])],
    )
    positives, counts = map_omia(omia, [(ref.genes, ref.proteins), (ref.ros_genes, ros_proteins)])
    counts["mapped_via_ros_cfam"] = sum(p.transcript_id in ros_proteins for p in positives)
    ref.proteins.update(ros_proteins)
    return positives, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--omia-csv", type=Path, default=Path("data/omia/omia_dog_variants.csv"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark_dog"))
    parser.add_argument("--min-af", type=float, default=0.05)
    parser.add_argument("--random-genes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output", type=Path, help="default: benchmarks/results/dog_omia_<model>.tsv"
    )
    args = parser.parse_args()
    args.output = args.output or Path(f"benchmarks/results/dog_omia_{model_tag(args.model)}.tsv")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("dog_omia", "omnimissense"):
        logging.getLogger(name).setLevel(logging.INFO)

    if not args.omia_csv.exists():
        sys.exit(f"Missing {args.omia_csv}: download it by hand, see benchmarks/README.md")
    data = args.data_dir
    samples = fetch(DOG10K_SAMPLES, data / "dog10k_samples.txt")
    ref = load_reference()
    fasta, gff3, genes = ref.fasta, ref.gff3, ref.genes
    omia = read_omia(args.omia_csv)
    positives, mapping_counts = causal_variants(omia, ref)
    proteins = ref.proteins
    log.info("OMIA deleterious dog missense records: %d; mapping: %s", len(omia), mapping_counts)

    # ---- negatives
    positive_genes = {p.gene.upper() for p in positives}
    # Random genes are drawn from all autosomal coding genes *before* removing the causal
    # genes, so the draw does not change when the causal set changes.
    autosomal = sorted(g for g, txs in genes.items() if txs[0].seqid.isdigit())
    drawn = random.Random(args.seed).sample(autosomal, min(args.random_genes, len(autosomal)))
    random_genes = set(drawn) - positive_genes
    # same-gene negatives only for genes present in the GSD annotation (Dog10K coordinates)
    chosen = [genes[g][0] for g in sorted((positive_genes & genes.keys()) | random_genes)]

    dogs = data / "dog10k_dogs.txt"
    with samples.open() as handle:
        names = [
            r["sampleName"]
            for r in csv.DictReader(handle, delimiter="\t")
            if r["inVCF"] == "YES"
            and r["inAnalyzedSNV"] == "TRUE"
            and r["Category"] in DOG_CATEGORIES
        ]
    dogs.write_text("\n".join(names) + "\n")
    bed = data / "cds_regions.bed"
    write_cds_bed(chosen, bed)
    # cached Dog10K extracts are keyed by the exact regions and frequency threshold
    regions_key = hashlib.sha256(f"{args.min_af}\n{bed.read_text()}".encode()).hexdigest()[:12]
    common_vcf = data / f"dog10k_common_{regions_key}.vcf"
    if not common_vcf.exists():
        extract_common_snvs(bed, dogs, fasta, common_vcf, args.min_af)

    cache = ScoreCache(default_cache_dir(), args.model)

    def scorer_factory():
        return MaskedLMScorer.from_pretrained(args.model)

    result = annotate(
        common_vcf,
        fasta,
        gff3,
        data / "dog10k_common.omnimissense.vcf.gz",
        data / "dog10k_common.omnimissense.tsv",
        cache=cache,
        scorer_factory=scorer_factory,
        model_name=args.model,
    )
    canonical = {
        genes[g][0].transcript_id: g for g in (positive_genes & genes.keys()) | random_genes
    }
    positive_keys = {(p.transcript_id, p.position, p.alt) for p in positives}
    negatives = {}
    for v in result.variants:
        tid = v.consequence.transcript_id
        if v.status is not VariantStatus.OK or tid not in canonical:
            continue
        key = (tid, v.consequence.protein_pos, v.consequence.alt_aa)
        if key in positive_keys:
            continue  # common in dogs and listed as causal: drop from the negatives
        group = "same_gene" if canonical[tid] in positive_genes else "random_gene"
        negatives[key] = ProteinVariant(
            canonical[tid], tid, key[1], v.consequence.ref_aa, key[2], 0, group
        )
    overlap = sum(
        1
        for v in result.variants
        if (v.consequence.transcript_id, v.consequence.protein_pos, v.consequence.alt_aa)
        in positive_keys
    )
    log.info("Negatives: %d (removed %d also listed as causal)", len(negatives), overlap)

    # ---- scores (masked marginal, wild-type marginal) for all variants, via the cache
    variants = positives + list(negatives.values())
    sequences = {p.transcript_id: proteins[p.transcript_id].sequence for p in variants}
    lazy = LazyScorer(scorer_factory)
    wt = cached_wt_scores({sequence_hash(s): s for s in sequences.values()}, cache, lazy)
    queries = {
        (sequence_hash(sequences[v.transcript_id]), v.position): sequences[v.transcript_id]
        for v in variants
    }
    masked = cached_masked_scores(queries, cache, lazy)

    labels = np.array([v.label for v in variants])
    categories = np.array([v.group for v in variants])
    gene_names = np.array([v.gene.upper() for v in variants])
    mm = np.empty(len(variants))
    wm = np.empty(len(variants))
    blosum = np.array([BLOSUM62[v.ref, v.alt] for v in variants], dtype=np.float64)
    in_protein = np.full(len(variants), np.nan)
    for i, v in enumerate(variants):
        seq = sequences[v.transcript_id]
        h = sequence_hash(seq)
        mm[i] = masked[(h, v.position)][AA_INDEX[v.alt]]
        matrix = wt[h][0]
        wm[i] = matrix[v.position - 1, AA_INDEX[v.alt]]
        if v.label == 1:
            # percentile among all 19 x L possible substitutions of the protein
            # (wild-type marginal; 0 = most damaging)
            not_wt = np.isfinite(matrix)
            for pos, aa in enumerate(seq):
                if aa in AA_INDEX:
                    not_wt[pos, AA_INDEX[aa]] = False
            in_protein[i] = float((matrix[not_wt] < wm[i]).mean())

    common = labels == 0
    causal_sets = {
        "pathogenic": categories == "pathogenic",
        "disease": np.isin(categories, ["pathogenic", "disease_unclassified"]),
        "trait": categories == "trait",
    }
    rows = []
    for set_name, causal in causal_sets.items():
        set_genes = list(set(gene_names[causal]))
        negative_groups = {
            "all common": common,
            "same genes": common & np.isin(gene_names, set_genes),
            "random genes": common & (categories == "random_gene"),
        }
        if set_name != "trait":
            negative_groups["OMIA traits"] = categories == "trait"
        ranks = in_protein[causal]
        for negatives_name, negative in negative_groups.items():
            if causal.sum() == 0 or negative.sum() == 0:
                continue
            mask = causal | negative
            y = causal[mask].astype(int)
            row = {
                "causal_set": set_name,
                "negatives": negatives_name,
                "n_causal": int(causal.sum()),
                "n_common": int(negative.sum()),
            }
            for score_name, score in (("masked", mm), ("wt", wm), ("blosum62", blosum)):
                predictor = -score[mask]  # lower LLR = more damaging
                low, high = bootstrap_ci(y, predictor)
                row[f"auroc_{score_name}"] = round(auroc(y, predictor), 3)
                row[f"auroc_{score_name}_ci95"] = f"{low:.3f}-{high:.3f}"
            row["causal_in_protein_percentile_median"] = round(float(np.median(ranks)), 3)
            row["causal_in_top10pct_of_protein"] = round(float((ranks < 0.10).mean()), 3)
            rows.append(row)

    summary = {
        "omia_dog_missense_records_used": len(omia),
        **{f"omia_{k}": v for k, v in mapping_counts.items()},
        **{f"causal_{k}": int(v.sum()) for k, v in causal_sets.items()},
        "min_af": args.min_af,
        "random_genes": len(random_genes),
        "causal_genes": len(positive_genes),
        "common_missense_same_genes": int((common & (categories == "same_gene")).sum()),
        "common_missense_random_genes": int((common & (categories == "random_gene")).sum()),
        "common_removed_as_causal": overlap,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    with args.output.with_suffix(".summary.tsv").open("w") as handle:
        handle.writelines(f"{k}\t{v}\n" for k, v in summary.items())

    print(
        "\n| causal set | negatives | causal | common | AUROC masked (95% CI) "
        "| AUROC wt (95% CI) | AUROC BLOSUM62 | causal in top 10% of protein |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['causal_set']} | {r['negatives']} | {r['n_causal']} | {r['n_common']} "
            f"| {r['auroc_masked']:.3f} ({r['auroc_masked_ci95']}) "
            f"| {r['auroc_wt']:.3f} ({r['auroc_wt_ci95']}) "
            f"| {r['auroc_blosum62']:.3f} "
            f"| {r['causal_in_top10pct_of_protein']:.0%} |"
        )
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
