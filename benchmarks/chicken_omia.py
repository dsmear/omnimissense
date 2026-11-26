"""Chicken benchmark: OMIA causal missense variants vs common chicken missense variants (EVA).

Chicken and human last shared an ancestor more than 300 million years ago (dog and human:
about 90). The benchmark asks whether the scores of the chicken protein itself hold up that
far from human, and compares them with the usual workaround for non-human species: mapping
the variant onto the human ortholog and scoring it there (the same model, or AlphaMissense).

Causal sets: OMIA chicken variants whose effect is "missense". OMIA numbers them on a RefSeq
protein (NP_...), which often differs from the Ensembl 116 GRCg7b protein (other start
codon, other isoform), so each change is located on the cited RefSeq protein (fetched from
NCBI) and carried to the Ensembl protein of the same gene by global alignment (canonical
transcript first); the aligned Ensembl residue must be the reference one. Where it is the
alternative one, the reference genome (a broiler) carries the variant allele, and the
variant is scored on the Ensembl protein with the reference residue restored.
  pathogenic  AVCG classification P, LP or LP/P
  disease     pathogenic + diseases not yet classified (traits excluded)
  trait       plumage colour and pattern, virus resistance, blood group (TRAITS)
  all         disease + trait: every variant OMIA lists as causal (main analysis: chicken
              has only four disease variants)
Negatives: PASS biallelic SNVs with ALT allele frequency >= 5% among the 96 Bangladeshi native
chickens of EVA study PRJEB78357 (three varieties, improved and unimproved lines; GRCg7b,
autosomes), missense in the canonical transcript of (a) the genes of the causal variants and
(b) 500 randomly chosen autosomal protein-coding genes; annotated with omnimissense itself.
Common variants at the site of a causal variant (in either direction) are removed.
Methods: the model on the chicken protein (omnimissense; masked and wild-type marginal); the
same model on the human ortholog and AlphaMissense, through the human mapping of
dog_realistic.py (gene symbol, global alignment, identical reference residue); BLOSUM62.
Variants without a human mapping count as least damaging for the human methods (as a
geneticist could not rank them); since causal variants map more often than common ones,
this favours the human methods, and the fair comparison is the AUROC on the subset that
every method scores (also reported).

OMIA data cannot be fetched by scripts (robots.txt disallows /download*): download the CSV by
hand, see benchmarks/README.md. The human methods need the AlphaMissense and UniProt files of
dog_realistic.py; without them they are skipped with a warning. Then:

    uv run --group benchmarks python benchmarks/chicken_omia.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysam
import pysam.bcftools
import pysam.utils

sys.path.insert(0, str(Path(__file__).parent))
from blosum import BLOSUM62
from dog_omia import ProteinVariant, fetch, gene_index, parse_protein_changes, read_omia
from models import model_tag
from stats import auroc, bootstrap_ci, bootstrap_diff_ci

from omnimissense.cache import ScoreCache, default_cache_dir
from omnimissense.defaults import DEFAULT_MODEL
from omnimissense.fasta import prepare_fasta
from omnimissense.gff import Transcript, prepare_annotation, read_transcripts
from omnimissense.pipeline import (
    LazyScorer,
    VariantStatus,
    annotate,
    cached_masked_scores,
    cached_wt_scores,
)
from omnimissense.proteome import Protein, ProteinStatus, build_proteome
from omnimissense.reference import load_reference, reference_key
from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX, sequence_hash

log = logging.getLogger("chicken_omia")

ENSEMBL = "https://ftp.ensembl.org/pub/release-116"
ASSEMBLY = "Gallus_gallus.bGalGal1.mat.broiler.GRCg7b"
FASTA = f"{ASSEMBLY}.dna.toplevel.fa.gz"
GFF3 = f"{ASSEMBLY}.116.gff3.gz"
URLS = {
    FASTA: f"{ENSEMBL}/fasta/gallus_gallus/dna/{FASTA}",
    GFF3: f"{ENSEMBL}/gff3/gallus_gallus/{GFF3}",
}
EVA = "https://ftp.ebi.ac.uk/pub/databases/eva/PRJEB78357"
POPULATION_VCF = f"{EVA}/BD_chicken_all_autosomes_passedSNPs_Gal7_96samples.vcf.gz"
POPULATION_INDEX = f"{EVA}/BD_chicken_all_autosomes_passedSNPs_Gal7_96samples.vcf.csi"

ALPHAMISSENSE = Path("data/alphamissense/AlphaMissense_aa_substitutions.tsv.gz")
UNIPROT_HUMAN = Path("data/human/uniprot_human_reviewed.tsv")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#: OMIA chicken phenotypes (missense, "Not currently evaluated") that are not diseases:
#: plumage colour and pattern, virus resistance (loss of a viral receptor) and a blood
#: group. Listed explicitly from the OMIA export used here (2026-11-26). The diseases are
#: analphalipoproteinaemia (P), muscular dystrophy and restricted ovulator (LP/P), and
#: trimethylaminuria (not evaluated).
TRAITS: set[str] = {
    "Barring, autosomal",
    "Blood group H",
    "Chocolate feathered",
    "Feather colour, buttercup",
    "Feather colour, extended black",
    "Feather colour, lavender",
    "Feather colour, mottled",
    "Feather colour, recessive white",
    "Feather colour, silver",
    "Feather colour, wheaten-like",
    "Five gray",
    "Resistance to avian sarcoma and leukosis viruses, subgroup A",
    "Resistance to avian sarcoma and leukosis viruses, subgroup B",
    "Resistance to myxovirus",
    "Sex-linked barring",
    "Sex-linked dilution",
}
EXCLUDED: set[str] = set()
_ACCESSION = re.compile(r"^\s*([NXY]P_\d+\.\d+)\s*:")

CHICKEN_METHODS = ("chicken_masked", "chicken_wt", "blosum62")
HUMAN_METHODS = ("human_masked", "human_wt", "alphamissense")
#: Paired AUROC differences reported with a bootstrap interval (when the human methods run).
DIFFERENCES = (
    ("chicken_masked", "human_masked"),
    ("chicken_wt", "human_wt"),
    ("chicken_masked", "alphamissense"),
    ("chicken_wt", "alphamissense"),
)


# ---------------------------------------------------------------------------- causal variants


def protein_accession(text: str) -> str | None:
    """RefSeq protein accession of an OMIA ``p.`` entry (``NP_989476.1:p.(E89K)``)."""
    match = _ACCESSION.match(text)
    return match.group(1) if match else None


def read_fasta(path: Path) -> dict[str, str]:
    """``{first word of the header: sequence}``."""
    sequences = {}
    for block in path.read_text().split(">")[1:]:
        header, *lines = block.splitlines()
        sequences[header.split()[0]] = "".join(lines).rstrip("*")
    return sequences


def fetch_refseq_proteins(accessions: set[str], path: Path) -> dict[str, str]:
    """RefSeq protein sequences from NCBI E-utilities, cached in ``path`` (FASTA)."""
    cached = read_fasta(path) if path.exists() else {}
    missing = sorted(accessions - cached.keys())
    if missing:
        log.info("Fetching %d RefSeq proteins from NCBI", len(missing))
        query = urllib.parse.urlencode(
            {"db": "protein", "id": ",".join(missing), "rettype": "fasta", "retmode": "text",
             "tool": "omnimissense-benchmark"}
        )  # fmt: skip
        with urllib.request.urlopen(f"{EUTILS}?{query}", timeout=120) as response:
            text = response.read().decode()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
        cached = read_fasta(path)
    return {a: cached[a] for a in accessions if a in cached}


def map_omia_via_refseq(
    records: list[dict[str, str]],
    genes: dict[str, list[Transcript]],
    proteins: dict[str, Protein],
    refseq: dict[str, str],
) -> tuple[list[ProteinVariant], dict[str, str], dict[str, int]]:
    """Map OMIA protein changes onto Ensembl proteins through the RefSeq protein they cite.

    The change is located on the cited RefSeq protein (whose residue must be the reference or,
    when the RefSeq sequence was built from the variant allele, the alternative one), then
    carried to each Ensembl protein of the gene (canonical first) by global alignment.
    Where the Ensembl residue is the alternative one, the reference genome carries the
    variant allele: the variant is scored on the Ensembl protein with the reference residue
    restored, registered under ``<transcript>#<ref><position>`` in the returned sequences.
    Entries without an accession are located by their position on the Ensembl protein.

    Returns the variants (unique per gene, position and alternative residue), the restored
    sequences and the counts of each outcome.
    """
    from dog_realistic import alignment_map, make_aligner  # Biopython (benchmarks group)

    aligner = make_aligner()
    counts: dict[str, int] = defaultdict(int)
    mapped: dict[tuple[str, int, str], ProteinVariant] = {}
    restored: dict[str, str] = {}
    for record in records:
        changes = parse_protein_changes(record["protein"])
        if len(changes) != 1:
            counts["no_single_protein_change"] += 1
            continue
        ref, position, alt = changes[0]
        candidates = [
            tx for tx in genes.get(record["gene"].upper(), [])
            if tx.transcript_id in proteins
            and proteins[tx.transcript_id].status is ProteinStatus.OK
        ]  # fmt: skip
        if not candidates:
            counts["gene_not_in_annotation"] += 1
            continue
        accession = protein_accession(record["protein"])
        source = refseq.get(accession) if accession else None
        if accession and source is None:
            counts["refseq_protein_not_found"] += 1
            continue
        if source is not None and (
            position > len(source) or source[position - 1] not in (ref, alt)
        ):
            counts["refseq_residue_mismatch"] += 1
            continue
        found = None
        for tx in candidates:
            sequence = proteins[tx.transcript_id].sequence
            if source is None:
                index = position - 1 if position <= len(sequence) else -1
            else:
                index = int(alignment_map(source, sequence, aligner)[position - 1])
            if index < 0:
                continue
            if sequence[index] == ref:
                found = (tx, tx.transcript_id, index + 1)
                break
            if sequence[index] == alt and source is not None:
                key = f"{tx.transcript_id}#{ref}{index + 1}"
                restored[key] = sequence[:index] + ref + sequence[index + 1 :]
                found = (tx, key, index + 1)
                counts["reference_genome_carries_alt"] += 1
                break
        if found is None:
            counts["ensembl_residue_mismatch"] += 1
            continue
        tx, transcript_key, ensembl_position = found
        counts["mapped"] += 1
        mapped.setdefault(
            (tx.gene_name.upper(), ensembl_position, alt),
            ProteinVariant(
                tx.gene_name, transcript_key, ensembl_position, ref, alt, 1, record["category"]
            ),
        )
    counts["mapped_unique"] = len(mapped)
    variants = list(mapped.values())
    used = {v.transcript_id for v in variants}
    return variants, {k: s for k, s in restored.items() if k in used}, dict(counts)


# ---------------------------------------------------------------------------- population


def index_url(vcf: str, index: str | None) -> str:
    """Index of a remote VCF: the one given, EVA's for the default VCF (named ``.vcf.csi``,
    which htslib does not find on its own), else ``<vcf>.csi``."""
    if index:
        return index
    return POPULATION_INDEX if vcf == POPULATION_VCF else f"{vcf}.csi"


def population_source(vcf: str, index: str | None, data_dir: Path) -> str:
    """htslib path of the population VCF: a local file as is, or a URL read remotely with a
    local copy of its index."""
    if not vcf.startswith(("http://", "https://", "ftp://")):
        return vcf
    url = index_url(vcf, index)
    local = fetch(url, data_dir / url.rsplit("/", 1)[-1])
    return f"{vcf}##idx##{local}"


def _query_common(source: str, bed: Path, min_af: float, attempts: int = 4) -> str:
    """Sites (no genotypes) of PASS biallelic SNVs with ALT frequency >= min_af in the BED
    regions; retried, as long remote reads occasionally fail."""
    for attempt in range(1, attempts + 1):
        previous = pysam.set_verbosity(0)
        try:
            return pysam.bcftools.view(
                "-R", str(bed), "-f", "PASS,.", "-m2", "-M2", "-v", "snps",
                "-q", f"{min_af}:alt1", "-G", "-Ov", source,
                catch_stdout=True,
            )  # fmt: skip
        except pysam.utils.SamtoolsError as exc:
            if attempt == attempts:
                raise
            log.warning("Population query failed (%s), retrying (%d/%d)", exc, attempt, attempts)
            time.sleep(10 * attempt)
        finally:
            pysam.set_verbosity(previous)
    raise AssertionError("unreachable")


def extract_common_snvs(source: str, bed: Path, fasta: Path, output: Path, min_af: float) -> int:
    """Query the population VCF sequence by sequence (each part cached on disk) and write the
    common SNVs, sorted, with the FASTA's contigs; returns the number of SNVs."""
    previous = pysam.set_verbosity(0)
    try:
        with pysam.VariantFile(source) as vcf:
            available = set(vcf.header.contigs)
    finally:
        pysam.set_verbosity(previous)
    with pysam.FastaFile(str(fasta)) as genome:
        contigs = list(zip(genome.references, genome.lengths, strict=True))
    order = {name: i for i, (name, _) in enumerate(contigs)}
    regions: dict[str, list[str]] = defaultdict(list)
    for line in bed.read_text().splitlines():
        chrom = line.split("\t")[0]
        if chrom in available:
            regions[chrom].append(line)
    parts = output.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    definitions: list[str] = []
    records: dict[tuple[str, ...], list[str]] = {}
    for chrom in sorted(regions, key=lambda c: order[c]):
        part = parts / f"{chrom}.vcf"
        if not part.exists():
            chrom_bed = parts / f"{chrom}.bed"
            chrom_bed.write_text("\n".join(regions[chrom]) + "\n")
            log.info("Population VCF, sequence %s: %d regions", chrom, len(regions[chrom]))
            tmp = part.with_suffix(".tmp")
            tmp.write_text(_query_common(source, chrom_bed, min_af))
            tmp.rename(part)
        for line in part.read_text().splitlines():
            if line.startswith("##INFO") or line.startswith("##FILTER"):
                if line not in definitions:
                    definitions.append(line)
            elif line and not line.startswith("#"):
                fields = line.split("\t")
                records.setdefault(tuple(fields[:5]), fields[:8])  # overlapping regions repeat
    body = sorted(records.values(), key=lambda f: (order[f[0]], int(f[1]), f[4]))
    header = ["##fileformat=VCFv4.2", *definitions]
    header += [f"##contig=<ID={name},length={length}>" for name, length in contigs]
    header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
    output.write_text("\n".join(header + ["\t".join(f) for f in body]) + "\n")
    log.info("%d common SNVs written to %s", len(body), output)
    return len(body)


def write_cds_bed(transcripts: list[Transcript], path: Path) -> None:
    """BED of the CDS segments, one line per segment (sorted)."""
    rows = sorted((tx.seqid, s.start - 1, s.end) for tx in transcripts for s in tx.cds)
    path.write_text("".join(f"{c}\t{a}\t{b}\n" for c, a, b in rows))


# ---------------------------------------------------------------------------- metrics


def in_protein_percentile(matrix: np.ndarray, sequence: str, score: float) -> float:
    """Share of the protein's 19 x L possible substitutions scoring lower (more damaging)."""
    not_wt = np.isfinite(matrix)
    for pos, aa in enumerate(sequence):
        if aa in AA_INDEX:
            not_wt[pos, AA_INDEX[aa]] = False
    return float((matrix[not_wt] < score).mean())


def auroc_row(
    labels: np.ndarray,
    table: dict[str, np.ndarray],
    methods: tuple[str, ...],
    differences: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    """AUROC with a 95% bootstrap interval per method (higher score = more damaging; -inf =
    not scored, ranked last) and paired differences."""
    y = np.asarray(labels, dtype=int)
    row: dict[str, object] = {"n_causal": int(y.sum()), "n_negative": int((1 - y).sum())}
    finite = {m: np.nan_to_num(table[m], neginf=-1e9) for m in methods}
    for m in methods:
        low, high = bootstrap_ci(y, finite[m])
        row[m] = round(auroc(y, finite[m]), 3)
        row[f"{m}_ci95"] = f"{low:.3f}-{high:.3f}"
    for a, b in differences:
        low, high = bootstrap_diff_ci(y, finite[a], finite[b])
        row[f"diff_{a}_vs_{b}"] = round(float(row[a]) - float(row[b]), 3)
        row[f"diff_{a}_vs_{b}_ci95"] = f"{low:+.3f} to {high:+.3f}"
    return row


# ---------------------------------------------------------------------------- main


def load_chicken(data_dir: Path) -> tuple[Path, Path, dict[str, Transcript]]:
    fasta = prepare_fasta(fetch(URLS[FASTA], data_dir / FASTA))
    gff3 = fetch(URLS[GFF3], data_dir / GFF3)
    return fasta, gff3, read_transcripts(gff3, fasta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--omia-csv", type=Path, default=Path("data/omia/omia_chicken_variants.csv")
    )
    parser.add_argument("--genome-dir", type=Path, default=Path("data/chicken/ensembl-116"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark_chicken"))
    parser.add_argument(
        "--population-vcf",
        default=POPULATION_VCF,
        help="population VCF (path or URL); default: EVA PRJEB78357, read remotely",
    )
    parser.add_argument("--population-index", help="index URL or path, for a remote VCF")
    parser.add_argument("--min-af", type=float, default=0.05)
    parser.add_argument("--random-genes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output", type=Path, help="default: benchmarks/results/chicken_omia_<model>.tsv"
    )
    args = parser.parse_args()
    args.output = args.output or Path(
        f"benchmarks/results/chicken_omia_{model_tag(args.model)}.tsv"
    )
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name in ("chicken_omia", "dog_omia", "omnimissense"):
        logging.getLogger(name).setLevel(logging.INFO)
    if not args.omia_csv.exists():
        sys.exit(f"Missing {args.omia_csv}: download it by hand, see benchmarks/README.md")
    human_files = ALPHAMISSENSE.exists() and UNIPROT_HUMAN.exists()
    if not human_files:
        log.warning(
            "%s or %s missing: human-ortholog methods skipped (see benchmarks/README.md)",
            ALPHAMISSENSE,
            UNIPROT_HUMAN,
        )
    methods = CHICKEN_METHODS + (HUMAN_METHODS if human_files else ())
    differences = DIFFERENCES if human_files else ()
    data = args.data_dir
    data.mkdir(parents=True, exist_ok=True)

    # ---- causal variants
    fasta, gff3, transcripts = load_chicken(args.genome_dir)
    genes = gene_index(transcripts)
    omia = read_omia(args.omia_csv, species="chicken", traits=TRAITS, excluded=EXCLUDED)
    omia_genes = {r["gene"].upper() for r in omia}
    canonical_ids = [txs[0].transcript_id for txs in genes.values()]
    proteins: dict[str, Protein] = build_proteome(
        fasta,
        transcripts,
        set(canonical_ids) | {tx.transcript_id for g in omia_genes for tx in genes.get(g, [])},
    )
    accessions = {a for r in omia if (a := protein_accession(r["protein"]))}
    refseq = fetch_refseq_proteins(accessions, data / "refseq_proteins.fa")
    positives, restored, mapping_counts = map_omia_via_refseq(omia, genes, proteins, refseq)
    log.info("OMIA chicken missense records: %d; mapping: %s", len(omia), mapping_counts)
    unmapped_genes = sorted(g for g in omia_genes if g not in genes)
    if unmapped_genes:
        log.info("OMIA genes without an Ensembl gene symbol: %s", ", ".join(unmapped_genes))

    # ---- negatives: common SNVs in the causal genes and in random autosomal genes
    positive_genes = {p.gene.upper() for p in positives}
    # drawn before removing the causal genes, so the draw does not depend on the causal set
    autosomal = sorted(
        g
        for g, txs in genes.items()
        if txs[0].seqid.isdigit() and proteins[txs[0].transcript_id].status is ProteinStatus.OK
    )
    drawn = random.Random(args.seed).sample(autosomal, min(args.random_genes, len(autosomal)))
    random_genes = set(drawn) - positive_genes
    negative_genes = positive_genes | random_genes
    canonical = {genes[g][0].transcript_id: g for g in negative_genes}
    bed = data / "cds_regions.bed"
    write_cds_bed([genes[g][0] for g in sorted(negative_genes)], bed)
    source = population_source(args.population_vcf, args.population_index, data)
    regions_key = hashlib.sha256(
        f"{args.population_vcf}\n{args.min_af}\n{bed.read_text()}".encode()
    ).hexdigest()[:12]
    common_vcf = data / f"population_common_{regions_key}.vcf"
    if not common_vcf.exists():
        extract_common_snvs(source, bed, fasta, common_vcf, args.min_af)

    cache = ScoreCache(default_cache_dir(), args.model)

    def scorer_factory():
        return MaskedLMScorer.from_pretrained(args.model)

    result = annotate(
        common_vcf,
        fasta,
        gff3,
        data / "population_common.omnimissense.vcf.gz",
        data / "population_common.omnimissense.tsv",
        cache=cache,
        scorer_factory=scorer_factory,
        model_name=args.model,
    )
    # a common variant at the site of a causal one (either direction) is not a negative
    positive_sites = {(p.transcript_id.split("#")[0], p.position) for p in positives}
    negatives: dict[tuple[str, int, str], ProteinVariant] = {}
    overlap = 0
    for v in result.variants:
        c = v.consequence
        if v.status is not VariantStatus.OK or c.transcript_id not in canonical:
            continue
        key = (c.transcript_id, c.protein_pos, c.alt_aa)
        if key[:2] in positive_sites:
            overlap += 1
            continue
        gene = canonical[c.transcript_id]
        group = "same_gene" if gene in positive_genes else "random_gene"
        negatives[key] = ProteinVariant(gene, key[0], key[1], c.ref_aa, key[2], 0, group)
    log.info("Negatives: %d (removed %d at causal sites)", len(negatives), overlap)

    # ---- scores on chicken proteins
    variants = positives + list(negatives.values())
    sequences = {
        v.transcript_id: restored.get(v.transcript_id) or proteins[v.transcript_id].sequence
        for v in variants
    }
    lazy = LazyScorer(scorer_factory)
    wt = cached_wt_scores({sequence_hash(s): s for s in sequences.values()}, cache, lazy)
    masked = cached_masked_scores(
        {(sequence_hash(sequences[v.transcript_id]), v.position): sequences[v.transcript_id]
         for v in variants},
        cache,
        lazy,
    )  # fmt: skip

    n = len(variants)
    table = {m: np.full(n, -np.inf) for m in methods}
    wt_llr = np.empty(n)
    in_protein = np.empty(n)
    for i, v in enumerate(variants):
        seq = sequences[v.transcript_id]
        h = sequence_hash(seq)
        matrix = wt[h][0]
        wt_llr[i] = matrix[v.position - 1, AA_INDEX[v.alt]]
        table["chicken_masked"][i] = -float(masked[(h, v.position)][AA_INDEX[v.alt]])
        table["chicken_wt"][i] = -wt_llr[i]
        table["blosum62"][i] = -float(BLOSUM62[v.ref, v.alt])
        in_protein[i] = in_protein_percentile(matrix, seq, wt_llr[i])

    # ---- human ortholog
    if human_files:
        from dog_realistic import load_alphamissense, load_uniprot, map_to_human

        uniprot = load_uniprot(UNIPROT_HUMAN)
        human = map_to_human(variants, sequences, uniprot)
        human_sequences = dict(uniprot.values())
        human_masked = cached_masked_scores(
            {(sequence_hash(human_sequences[a]), pos): human_sequences[a]
             for a, pos, _, _ in human.values()},
            cache,
            lazy,
        )  # fmt: skip
        human_wt = cached_wt_scores(
            {sequence_hash(human_sequences[a]): human_sequences[a] for a, *_ in human.values()},
            cache,
            lazy,
        )
        am = load_alphamissense(
            ALPHAMISSENSE, {(a, f"{r}{pos}{alt}") for a, pos, r, alt in human.values()}
        )
        for i, v in enumerate(variants):
            mapped = human.get((v.transcript_id, v.position, v.alt))
            if mapped is None:
                continue
            accession, pos, r, alt = mapped
            hh = sequence_hash(human_sequences[accession])
            table["human_masked"][i] = -float(human_masked[(hh, pos)][AA_INDEX[alt]])
            table["human_wt"][i] = -float(human_wt[hh][0][pos - 1, AA_INDEX[alt]])
            table["alphamissense"][i] = am.get((accession, f"{r}{pos}{alt}"), -np.inf)
        log.info("Human mapping: %d / %d variants; AlphaMissense: %d", len(human), n, len(am))

    # ---- proteome-wide percentile (omnimissense precompute on the same files), if built
    with tempfile.TemporaryDirectory(prefix="chicken-omia-") as tmp:
        codes = prepare_annotation(gff3, Path(tmp)).codes
    reference = load_reference(cache.directory, reference_key(gff3, fasta, codes))
    proteome_pct = reference.percentile(wt_llr) if reference is not None else None
    if reference is None:
        log.info("No percentile reference: run omnimissense precompute on the same files")

    # ---- AUROC
    labels = np.array([v.label for v in variants])
    groups = np.array([v.group for v in variants])
    gene_names = np.array([v.gene.upper() for v in variants])
    common = labels == 0
    causal_sets = {
        "pathogenic": groups == "pathogenic",
        "disease": np.isin(groups, ["pathogenic", "disease_unclassified"]),
        "trait": groups == "trait",
        "all": labels == 1,
    }
    all_scored = np.all([np.isfinite(table[m]) for m in methods], axis=0)
    subsets = {"all variants": np.ones(n, bool)}
    if human_files:
        subsets["scored by all methods"] = all_scored
    rows = []
    for set_name, causal in causal_sets.items():
        if not causal.any():
            continue
        negative_groups = {
            "all common": common,
            "same genes": common & np.isin(gene_names, list(set(gene_names[causal]))),
            "random genes": common & (groups == "random_gene"),
        }
        if set_name not in ("trait", "all"):
            negative_groups["OMIA traits"] = causal_sets["trait"]
        for negatives_name, negative in negative_groups.items():
            for subset_name, subset in subsets.items():
                mask = (causal | negative) & subset
                if not (causal & mask).any() or not (negative & mask).any():
                    continue
                row = {"causal_set": set_name, "negatives": negatives_name, "subset": subset_name}
                row.update(
                    auroc_row(causal[mask], {m: table[m][mask] for m in methods}, methods,
                              differences)
                )  # fmt: skip
                rows.append(row)

    summary: dict[str, object] = {
        "omia_chicken_missense_records_used": len(omia),
        **{f"omia_{k}": v for k, v in mapping_counts.items()},
        "omia_genes_without_symbol": ",".join(unmapped_genes),
        **{f"causal_{k}": int(v.sum()) for k, v in causal_sets.items()},
        "causal_genes": len(positive_genes),
        "min_af": args.min_af,
        "random_genes": len(random_genes),
        "common_missense_same_genes": int((common & (groups == "same_gene")).sum()),
        "common_missense_random_genes": int((common & (groups == "random_gene")).sum()),
        "common_removed_at_causal_sites": overlap,
    }
    for name, selection in (*causal_sets.items(), ("common", common)):
        if not selection.any():
            continue
        summary[f"in_protein_percentile_median_{name}"] = round(
            float(np.median(in_protein[selection])), 3
        )
        if proteome_pct is not None:
            values = proteome_pct[selection]
            summary[f"proteome_percentile_median_{name}"] = round(float(np.median(values)), 2)
            summary[f"proteome_percentile_below5_{name}"] = round(float((values < 5).mean()), 3)
        if human_files:
            summary[f"human_mappable_{name}"] = round(
                float(np.isfinite(table["human_masked"][selection]).mean()), 3
            )
            summary[f"alphamissense_found_{name}"] = round(
                float(np.isfinite(table["alphamissense"][selection]).mean()), 3
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t",
                                lineterminator="\n")  # fmt: skip
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".summary.tsv").write_text(
        "".join(f"{k}\t{v}\n" for k, v in summary.items())
    )

    print(f"\nModel: {args.model}")
    print("| causal | negatives | subset | causal/negative | " + " | ".join(methods) + " |")
    print("|---" * (4 + len(methods)) + "|")
    for r in rows:
        print(
            f"| {r['causal_set']} | {r['negatives']} | {r['subset']} "
            f"| {r['n_causal']}/{r['n_negative']} | "
            + " | ".join(f"{r[m]:.3f} ({r[f'{m}_ci95']})" for m in methods)
            + " |"
        )
    if differences:
        print("\nPaired AUROC differences (95% bootstrap interval):")
        for r in rows:
            diffs = "; ".join(f"{a}-{b} {r[f'diff_{a}_vs_{b}_ci95']}" for a, b in differences)
            print(f"  {r['causal_set']}/{r['negatives']}/{r['subset']}: {diffs}")
    print()
    for k, v in summary.items():
        print(f"{k}: {v}")

    # per causal variant: printed only (OMIA data are not written to the repository)
    common_wt = -table["chicken_wt"][common]
    print("\n| gene | change (Ensembl numbering) | set | wt LLR | share of common variants "
          "scoring higher | in-protein pct | proteome pct | human |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|")
    for i, v in enumerate(variants):
        if v.label != 1:
            continue
        restored_note = " (reference carries ALT)" if "#" in v.transcript_id else ""
        above = float((common_wt > wt_llr[i]).mean()) if common.any() else float("nan")
        human_note = (
            ("mapped" if np.isfinite(table["human_masked"][i]) else "not mapped")
            if human_files
            else "-"
        )
        pct = f"{proteome_pct[i]:.1f}" if proteome_pct is not None else "-"
        print(
            f"| {v.gene} | {v.ref}{v.position}{v.alt}{restored_note} | {v.group} "
            f"| {wt_llr[i]:.2f} | {above:.0%} | {in_protein[i]:.2f} | {pct} | {human_note} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
