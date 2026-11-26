# omnimissense

[![CI](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml/badge.svg)](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/dsmear/omnimissense/blob/main/LICENSE)

**Score missense variants in any species with protein language models, straight from a VCF.**

```bash
pip install omnimissense        # or: uv tool install omnimissense
omnimissense doctor             # checks bcftools (bundled), PyTorch, CUDA
omnimissense annotate --vcf my.vcf.gz --fasta genome.fa.gz --gff3 genes.gff3.gz -o my.oms.vcf.gz
```

Python ≥ 3.11; a CUDA GPU is strongly recommended (CPU works but is slow). bcftools is bundled
through pysam: nothing else to install. Full example: [Quick start](#quick-start-dog).

> **Status: alpha.** The pipeline works end to end on Ensembl and NCBI annotations (dog, yeast,
> *E. coli*: proteomes identical to the reference translations), reproduces published ESM-2
> results and separates known dog disease variants from common dog variants (see
> [Validation](#validation)).
> The scores measure how disruptive a substitution looks to the protein, **not** whether it
> causes disease: do not use them alone for breeding or clinical decisions.

## Why

For human variants, precomputed missense effect scores are everywhere (AlphaMissense,
ESM1b from Brandes et al. 2023, VESM, …). For other species there is almost nothing:
the usual workaround is to lift variants over to the human genome and borrow human scores,
which drops variants that have no alignable human counterpart and scores the wrong protein.

omnimissense scores each variant on the **species' own protein**, using only files that
exist for any sequenced genome:

| Input | What it is |
|---|---|
| VCF | the variants (differences from the reference genome) |
| FASTA | the reference genome sequence |
| GFF3 or GTF | the gene annotation (where genes, exons and coding sequences are): Ensembl, NCBI RefSeq, GENCODE, Prokka, Bakta, AUGUSTUS/BRAKER |

It outputs the same VCF (all records, all samples) with protein-language-model scores in the
`INFO` field, plus a TSV with one row per missense consequence. It can also write the scores
of every possible missense SNV of a genome, so that VCFs can later be annotated with bcftools
alone ([Annotating without a GPU](#annotating-without-a-gpu)).

## How it works

1. **REF check.** Every VCF record's REF allele is compared with the genome. Mismatching
   records are flagged (`OMS_REF_MISMATCH`) and skipped; if more than 10% mismatch, the run
   stops (VCF and FASTA are probably from different assemblies).
2. **Consequences** are called with [`bcftools csq`](https://samtools.github.io/bcftools/howtos/csq-calling.html)
   (bundled in pysam, nothing to install), one record at a time, for every ALT allele, with
   the genetic code of each sequence (one csq run per code, see
   [Genetic codes](#genetic-codes)).
3. **Proteins** are translated from the FASTA + GFF3, using the transcripts exactly as csq
   parsed them (`--dump-gff`), so both steps always agree on transcripts and coding segments.
   The reference amino acid reported by csq must match the translated protein; mismatches are
   reported (`ref_aa_mismatch`).
4. **Scores** come from a protein language model, by default **VESM 650M** (an ESM-2 650M
   re-trained for variant effects, MIT license; fp16 on GPU), as log-likelihood ratios
   `log P(alt) − log P(ref)` at the mutated position (more negative = more likely damaging):
   - `wt_marginal` (default): one forward pass per protein gives all 19 substitutions at
     every position; the whole proteome can be precomputed once (`omnimissense precompute`),
     after which annotating a VCF takes seconds.
   - `masked_marginal` (`--masked`): the position is masked and predicted from its context;
     one forward pass per position, computed only for the variants in the VCF. Usually the
     better mode for plain ESM-2, not for VESM (see Validation).
   - Proteins longer than 1022 residues are split into 1022-residue windows overlapping by
     511; each residue's scores are a weighted average over the windows that contain it, with
     weights fading near artificial window edges (scheme of Brandes et al. 2023).
5. **Cache.** Scores are stored on disk (Parquet) keyed by the SHA-256 of the protein
   sequence, so identical isoforms and repeated runs are never recomputed.

## Quick start: dog

The dog reference used by Ensembl (release 116, checked 2026-11-25) is **ROS_Cfam_1.0**
(GenBank GCA_014441545.1, genebuild 2020-10).

```bash
mkdir -p data/dog && cd data/dog
B=https://ftp.ensembl.org/pub/release-116
curl -O $B/fasta/canis_lupus_familiaris/dna/Canis_lupus_familiaris.ROS_Cfam_1.0.dna.toplevel.fa.gz
curl -O $B/gff3/canis_lupus_familiaris/Canis_lupus_familiaris.ROS_Cfam_1.0.116.gff3.gz
cd ../..

FASTA=data/dog/Canis_lupus_familiaris.ROS_Cfam_1.0.dna.toplevel.fa.gz
GFF3=data/dog/Canis_lupus_familiaris.ROS_Cfam_1.0.116.gff3.gz

# optional, once: score the whole proteome into the cache (default model: VESM 650M)
omnimissense precompute --fasta $FASTA --gff3 $GFF3

omnimissense annotate --vcf my_dogs.vcf.gz --fasta $FASTA --gff3 $GFF3 \
    -o my_dogs.omnimissense.vcf.gz          # also writes my_dogs.omnimissense.tsv

# optional: all scores of all coding transcripts in one Parquet file
omnimissense export --fasta $FASTA --gff3 $GFF3 -o dog_scores.parquet

# optional: every possible missense SNV at genome coordinates (no GPU needed afterwards)
omnimissense export-variants --fasta $FASTA --gff3 $GFF3 -o dog_snvs.vcf.gz
```

The plain-gzip FASTA distributed by Ensembl is recompressed once to BGZF (`.fa.bgz`) next to
the original, because random access needs it. From a development clone, prefix the commands
with `uv run`.

### Annotating without a GPU

`export-variants` writes every possible missense SNV of the canonical coding transcripts
(`--all-transcripts`: all of them) with its score, at genome coordinates, tabix-indexed:

- `.vcf.gz`: one record per SNV with the same `OMS` / `OMS_WT` / `OMS_WT_PCT` tags as
  `annotate`;
- `.tsv.gz`: one row per SNV and transcript (`#CHROM POS REF ALT gene transcript canonical
  protein_variant oms_wt oms_wt_percentile`), for R, pandas or `bcftools annotate -c`.

Any VCF can then be annotated with bcftools alone, in seconds. Split multi-allelic records
first (`bcftools annotate` matches one annotation record per VCF record, so without the split
only one ALT of a multi-allelic record gets its score), and index the result (`bcftools
annotate` with a VCF annotation file needs an indexed input):

```bash
bcftools norm -m -any -Oz -o my_dogs.split.vcf.gz my_dogs.vcf.gz
bcftools index -t my_dogs.split.vcf.gz
bcftools annotate -a dog_snvs.vcf.gz -c INFO/OMS,INFO/OMS_WT,INFO/OMS_WT_PCT -Oz -o my_dogs.oms.vcf.gz my_dogs.split.vcf.gz
```

The result is identical to `omnimissense annotate` for SNVs (checked on every SNV of 300
random genes in yeast and in *E. coli*; see Validation). `annotate` is still needed for
masked marginals and for what the file does not hold: MNVs, and transcripts that are not
exported. Changes of a start codon (start lost) or to a stop codon are not missense and are
not in the file. Sizes: dog (canonical transcripts) 73.8 M SNVs; *E. coli* 8.6 M SNVs,
96 MB (VCF) in 49 s; yeast (all transcripts) 19.3 M SNVs, 216 MB (VCF) or 153 MB (TSV) in
1–2 min, from cached scores.

### Other species

Any assembly with an Ensembl or NCBI RefSeq annotation works the same way, prokaryotes
included (Prokka and Bakta GFF3 too). From NCBI, take `<assembly>_genomic.fna.gz` and `<assembly>_genomic.gff.gz` from the
assembly's FTP folder (e.g. `genomes/all/GCF/000/146/045/GCF_000146045.2_R64/` for yeast);
transcript IDs are then NCBI's (`rna-NM_…`, `rna-XM_…`). The VCF must use the same sequence
names as the FASTA (NCBI: accessions such as `NC_001133.9`).

### Genetic codes

Proteins are translated, and consequences called, with the
[NCBI genetic code](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi) of each
sequence (the 26 tables bcftools csq has: 1–6, 9–16, 21–31, 33):

| Sequences | Code used |
|---|---|
| nuclear (and other non-mitochondrial) | `--genetic-code`, else the GFF3's `transl_table`, else 1 (standard) |
| mitochondrial (`MT`, `chrM`, … or NCBI `genome=mitochondrion`) | `--mito-genetic-code`, else the GFF3's `transl_table`, else **skipped** |

NCBI writes `transl_table` on the CDS features whenever the code is not the standard one, so
NCBI annotations need no option (e.g. yeast mitochondria: table 3; bacteria and plastids: 11;
*Candida*: 12). Ensembl GFF3 files carry no code: pass e.g. `--mito-genetic-code 2` for
vertebrate mitochondria (3 yeast, 5 invertebrates) or `--genetic-code 6` for ciliates.

## Precomputed scores

| Species | Assembly / annotation | Model | Dataset |
|---|---|---|---|
| Dog | ROS_Cfam_1.0, Ensembl 116 (43,524 coding transcripts) | VESM 650M, wild-type marginal | [DavideMacchia/omnimissense-dog-ros-cfam-1.0-vesm-650m](https://huggingface.co/datasets/DavideMacchia/omnimissense-dog-ros-cfam-1.0-vesm-650m) (Parquet, 1.55 GB, CC-BY-4.0) |

Produced with `omnimissense precompute` + `omnimissense export`; the dataset card documents the
schema, provenance and limitations.

## Output

**VCF `INFO` fields**

| Field | Number | Content |
|---|---|---|
| `OMS` | `.` | one entry per missense consequence: `Allele\|Gene\|Transcript\|Canonical\|ProteinPos\|RefAA\|AltAA\|WTMarginal\|MaskedMarginal\|Status` |
| `OMS_WT` | `A` | lowest (most damaging) wild-type-marginal score across transcripts, per ALT allele |
| `OMS_WT_PCT` | `A` | proteome-wide percentile of `OMS_WT` (see below); needs `omnimissense precompute` once |
| `OMS_MM` | `A` | lowest masked-marginal score across transcripts, per ALT allele (only with `--masked`) |
| `OMS_REF_MISMATCH` | flag | the REF allele does not match the genome; record not annotated |
| `BCSQ` | `.` | consequences from bcftools csq (all consequence types) |

**TSV columns:** `chrom pos ref alt gene transcript canonical biotype strand protein_pos
ref_aa alt_aa protein_length n_windows wt_marginal masked_marginal status wt_percentile`

**Percentiles.** The raw score (a log-likelihood ratio) has no fixed scale. `OMS_WT_PCT` and
`wt_percentile` give it one: the percent of all possible missense SNVs of the genome's
canonical transcripts (one value per SNV and transcript) that have a lower (more damaging)
score. `0.5` means "among the 0.5%
most damaging missense changes possible in this genome", for any gene, species or model. It
is a monotone transform of the score, so it changes no ranking. The reference distribution
is built by `omnimissense precompute` (or `export-variants`) and stored in the cache, keyed
by the GFF3, the genome and the genetic codes; without it the fields stay empty. A
percentile within each protein was considered and rejected: it discards what separates
genes (dog pathogenic variants sit at a median 34th percentile of their own protein, while
the raw score separates them from common variants with AUROC 0.977).

**Status values:** `ok`, `ref_aa_mismatch` (csq and the translated protein disagree),
`multi_aa_change` (the change alters the protein length), `nonstandard_aa`, `protein_unavailable`
(e.g. internal stop codon), `mitochondrial_skipped` (see limitations), `pseudogene` (CDS
flagged `pseudo=true` by NCBI: a disrupted gene). Only `ok` rows are scored.

## Performance (RTX 3070, 8 GB)

| Step | Measured |
|---|---|
| Wild-type marginals, 650M model fp16 (ESM-2 and VESM have the same size) | ~12,400 residues/s, 1.6 GiB peak VRAM |
| Whole dog proteome (`precompute`, ESM-2) | 43,092 unique proteins, 26.7 M residues in 36 min; cache 1.4 GB |
| Masked marginals (`--masked`) | 15–28 positions/s, depending on protein length |
| Annotate 127,270 dog variants (chr38, 0–5 Mb; 1,689 missense consequences), default settings | 13.5 s when the proteins are cached (25 s on a first run with 0.2.0; 1.2 GB RAM) |
| Same, with `--masked` (ESM-2) | 137 s on first run, 13 s when cached |
| Proteome translation (dog) | 22 s (after the one-off BGZF conversion of the genome, ~3 min) |
| Percentile reference (dog, scores cached) | 50 s for 73.9 M values (`precompute` with nothing left to score: 73 s) |
| `export-variants`, from cached scores (4 CPU cores, no GPU) | *E. coli* 8.6 M SNVs in 49 s; yeast 19.3 M SNVs in 1–2 min |
| ProteinGym, all 217 assays, wild-type marginals | ~15 s of GPU per model |

Dog annotation, translation and percentile-reference times were measured with the current
version; the model and `precompute` throughput rows with 0.2.0 (the scoring has not changed).

## Validation

Scripts and details: [benchmarks/README.md](https://github.com/dsmear/omnimissense/blob/main/benchmarks/README.md). Results, one file per
model: `benchmarks/results/`. Two models are benchmarked: **ESM-2 650M** and **VESM 650M**
(an ESM-2 re-trained for variant effects; scored with the wild-type marginal, as its authors
do).

**Implementation checks**

- **Coordinates:** a hand-designed synthetic genome (`tests/synthetic.py`) with genes on both
  strands and codons split by introns; every designed variant is recovered with the right
  protein position and amino acids.
- **Proteome:** the dog proteome translated from FASTA + GFF3 is identical to Ensembl's
  published peptides for 43,524 / 43,524 transcripts (the only exclusion is one
  selenoprotein, see limitations).
- **NCBI RefSeq annotation** (`benchmarks/ncbi_translation.py`): yeast *S. cerevisiae* R64
  (GCF_000146045.2), read without conversion; 6,021 / 6,021 proteins identical to NCBI's
  translation of the same CDS, including the 19 mitochondrial ones (yeast mitochondrial code,
  taken from the GFF3). The other 6 are ORFs that NCBI itself marks as pseudogenes (internal
  stop codons).
- **Prokaryote** (same script): *E. coli* K-12 MG1655 (GCF_000005845.2), whose CDS hang
  directly from genes; 4,300 / 4,300 proteins identical to NCBI's translation (table 11 from
  the GFF3), including the programmed frameshifts of *prfB*, *dnaX* and *copA* and the 3
  selenoproteins (selenocysteine from `transl_except`). Not scored: 18 NCBI pseudogenes.
- **Genome-coordinate export** (`export-variants`): on every possible SNV of all CDS bases of
  300 random transcripts, the exported rows equal `annotate`'s scored consequences (yeast
  955,675 / 955,675, *E. coli* 624,682 / 624,682; same transcripts, positions, amino acids and
  scores), and `bcftools annotate` with the exported VCF reproduces `annotate`'s `OMS` and
  `OMS_WT` tags exactly. Unit test: every SNV of the synthetic genome, both strands, codons
  split by introns, a 5'-incomplete CDS, a mitochondrial gene (table 2).
- **Genetic codes:** the 26 tables are bcftools csq's own (copied from its source), checked
  against Biopython's NCBI tables and, for every table, against the amino acids csq reports
  for every codon (`tests/test_genetic_code*.py`).
- **ProteinGym** (`benchmarks/proteingym.py`): Spearman correlation between scores and
  experimental fitness. Our ESM-2 masked marginals reproduce the values ProteinGym publishes
  for ESM-2 650M; ProteinGym publishes no VESM values.

| DMS assay | organism | length | mutants | ESM-2 (ours) | ESM-2 (published) | VESM (ours) |
|---|---|---|---|---|---|---|
| BLAT_ECOLX_Stiffler_2015 | *E. coli* | 286 | 4,996 | 0.732 | 0.731 | **0.774** |
| KCNJ2_MOUSE_Coyote-Maestas_2022_function | mouse | 428 | 6,963 | 0.382 | 0.382 | **0.390** |
| SC6A4_HUMAN_Young_2021 | human | 630 | 11,576 | 0.543 | 0.543 | **0.562** |
| UBE4B_MOUSE_Starita_2013 | mouse | 1,173 | 899 | **0.459** | 0.459 | 0.452 |
| NPC1_HUMAN_Erwood_2022_HEK293T | human | 1,278 | 637 | 0.695 | 0.694 | **0.725** |
| BRCA1_HUMAN_Findlay_2018 | human | 1,863 | 1,837 | 0.515 | 0.515 | **0.523** |

The last three are longer than 1022 residues and use our sliding windows.

**Does it hold far from human? ProteinGym, all 217 assays by taxon**
(`benchmarks/proteingym_all.py`): 2.47 M mutants in 186 proteins, Spearman aggregated as
ProteinGym does (per protein, then per taxon; overall: per function type). Published models
on the same assays. VESM and ESM-2 "(ours)" are scored by omnimissense with the wild-type
marginal (results: `benchmarks/results/proteingym_all_*.tsv`); the other columns are
ProteinGym's published scores (ESM-2: masked marginal).

| | assays (proteins) | **VESM 650M** | ESM-2 650M (ours) | ESM-2 650M (published) | ESM-C 600M | GEMME (alignments) | VenusREM (structure + alignments) |
|---|---|---|---|---|---|---|---|
| Overall (by function) | 217 (186) | **0.457** | 0.418 | 0.414 | 0.405 | 0.455 | 0.518 |
| Human | 96 (81) | **0.474** | 0.458 | 0.457 | 0.462 | 0.469 | 0.529 |
| Other eukaryotes | 40 (34) | **0.517** | 0.519 | 0.486 | 0.481 | 0.516 | 0.582 |
| Prokaryotes | 50 (43) | **0.481** | 0.464 | 0.458 | 0.459 | 0.467 | 0.549 |
| Viruses | 31 (28) | **0.426** | 0.260 | 0.261 | 0.241 | 0.472 | 0.492 |

- **No loss outside human for cellular life:** VESM scores eukaryotic and prokaryotic proteins
  at least as well as human ones, on par with alignment-based methods (GEMME), which need a
  deep multiple sequence alignment for every protein.
- **Viruses are the weak spot of single-sequence models** (ESM-2 0.260, ESM-C 0.241). VESM
  recovers most of it: VESM − ESM-2 with the same scoring +0.167 (95% interval +0.103 to
  +0.230 over proteins; better on 25 of 31 viral assays), so the gain comes from the model,
  not from the scoring mode. It stays below alignment-based methods (≈0.47): use viral
  scores with more care.
- VESM vs ESM-2, same scoring: +0.017 human (+0.003 to +0.030), −0.001 other eukaryotes,
  +0.017 prokaryotes (−0.004 to +0.040). Our ESM-2 wild-type numbers match ProteinGym's
  published ESM-2 (masked) within noise (overall +0.008, −0.002 to +0.018).
- Overall VESM would rank 17th of 98 on the ProteinGym leaderboard, level with the best
  single-sequence model (VespaG, 0.458, GPL); the models above it mostly also use protein
  structure and/or alignments.
- No label leakage, as far as VESM's released training code shows: it is trained on UniProtKB
  sequences with the ESM models' own scores as targets (no DMS, ClinVar or other experimental
  labels), and checkpoints are chosen by agreement with the ESM ensemble on held-out human
  proteins, not on ProteinGym.

**Dog benchmark** (`benchmarks/dog_omia.py`): OMIA dog missense variants mapped to dog
proteins, against Dog10K missense variants with allele frequency ≥ 5% among 1,872 dogs
(UU_Cfam_GSD_1.0, annotated by omnimissense). AUROC with 95% bootstrap interval; BLOSUM62 is a
context-free baseline (amino-acid similarity only).

| causal set | common variants | n causal / common | ESM-2 | VESM | BLOSUM62 |
|---|---|---|---|---|---|
| pathogenic (AVCG P/LP) | all | 49 / 521 | 0.962 (0.926–0.988) | **0.977** (0.962–0.991) | 0.678 |
| pathogenic (AVCG P/LP) | same genes | 49 / 33 | 0.947 (0.890–0.987) | **0.961** (0.922–0.989) | 0.613 |
| disease (P/LP + not yet classified) | all | 118 / 521 | 0.945 (0.919–0.969) | **0.956** (0.932–0.977) | 0.650 |
| disease | same genes | 118 / 104 | 0.938 (0.904–0.966) | **0.947** (0.915–0.972) | 0.647 |
| coat/hair **traits** | all | 15 / 521 | 0.968 (0.912–0.999) | 0.946 (0.842–0.999) | 0.728 |
| pathogenic **vs OMIA traits** | — | 49 / 15 | 0.437 (0.282–0.593) | 0.550 (0.403–0.703) | 0.437 |

How to read it:

- Both models separate known dog disease variants from common dog variants far better than
  amino-acid similarity alone, also within the same genes.
- **Trait variants** (coat colour, hair) look as disruptive as disease variants, and neither
  model can tell the two apart (AUROC 0.44–0.55, intervals include chance). The score measures
  damage to the protein, not disease. This also explains the low AUROC (0.55–0.61) reported
  for human-mapped predictors on OMIA when "non-deleterious" (often trait) variants are the
  negatives; our numbers are **not** directly comparable with that study.
- Common variants are an easy negative set (selection has removed damaging alleles from high
  frequency); the realistic test below uses rare variants instead.
- Small sets (49 curated pathogenic variants), OMIA ascertainment bias (variants found in
  candidate genes), and 133 of 176 OMIA records mappable to Ensembl proteins (gene symbol
  missing or different protein numbering for the rest).

**Realistic test and comparison with the human-ortholog workaround**
(`benchmarks/dog_realistic.py`). For 50 Dog10K dogs (49 breeds), each dog's candidates are
the rare missense variants it carries (allele frequency < 1%; median 126 per dog). Each
causal OMIA variant is inserted in turn among each dog's candidates and ranked. The same
variants are also scored the way people do today without a dog tool: map the variant to the
human ortholog (same gene, aligned residue must be identical) and use a human score: the same
model on the human protein, or AlphaMissense.

| 49 pathogenic variants × 50 dogs | median rank | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| **omnimissense + VESM** (dog protein) | **4** | 21% | 59% | **72%** |
| omnimissense + ESM-2 (dog protein) | 6 | 19% | 46% | 66% |
| VESM on the human ortholog | 4 | 22% | 61% | 73% |
| ESM-2 on the human ortholog | 6 | 19% | 49% | 67% |
| AlphaMissense (human) | 4 | 26% | 60% | 75% |
| BLOSUM62 | 40.5 | 0% | 1% | 13% |

| AUROC, pathogenic vs rare variants | all variants (49 / 7,581) | scored by every method (48 / 4,180) |
|---|---|---|
| **omnimissense + VESM** | **0.927** | **0.903** |
| omnimissense + ESM-2 | 0.913 | 0.882 |
| VESM on the human ortholog | 0.932 | 0.904 |
| ESM-2 on the human ortholog | 0.915 | 0.874 |
| AlphaMissense (human) | 0.936 | 0.912 |

What this says:

- **The model matters most.** VESM puts the causal variant in the top 10 of ~126 rare
  missense candidates in 72% of the trials (ESM-2: 66%), close to AlphaMissense (75%); the
  paired AUROC difference VESM − AlphaMissense is not significant (−0.035 to +0.019 on all
  variants). VESM is MIT-licensed and works for any species; AlphaMissense covers human
  proteins only and has a non-commercial license.
- **Dog protein vs human protein, same model:** a small gain for the dog protein with ESM-2
  (+0.007 to +0.019 AUROC; paired interval at or above zero in three of four comparisons), no
  difference with VESM (−0.008 to +0.006). Scoring the species' own protein is not, by
  itself, what makes the scores better.
- **Coverage:** the human workaround can score only 56% of the dogs' rare missense variants
  and 41% of the common ones (the human residue differs or there is no human entry), but 96%
  of the causal variants, which sit on conserved residues. Unscorable variants are mostly
  benign, so for ranking they cost the workaround little in the dog, a species close to
  human. omnimissense scores 100% of them, which should matter more in species far from human.

## Limitations

- The annotation must describe genes and CDS: Ensembl, NCBI RefSeq, Prokka and Bakta GFF3
  work; CDS that hang directly from a gene or have no parent (prokaryotes, NCBI vertebrate
  mitochondria) get a transcript feature (`tx-<CDS ID>`), and an embedded `##FASTA`
  section is ignored, in a normalized copy that bcftools csq reads. GTF (Ensembl, GENCODE,
  NCBI, AUGUSTUS/BRAKER) is converted to GFF3 the same way, with the stop codon (outside the
  CDS in GTF) merged back into the CDS; transcript IDs are the GTF's `transcript_id`.
- Canonical transcripts come from `Ensembl_canonical`, `MANE Select` or `RefSeq Select`
  tags; genes without any tagged transcript get their longest CDS.
- Mitochondrial transcripts whose genetic code is unknown (no `--mito-genetic-code`, no
  `transl_table` in the GFF3) are skipped and reported (`mitochondrial_skipped`).
- RNA editing (plant mitochondria and plastids) is not modelled: the protein is translated
  from the genome sequence.
- Selenocysteine and pyrrolysine are read from NCBI-style `transl_except` attributes (the
  residue becomes `U` / `O`; variants elsewhere in the protein are scored, the recoded codon
  itself is not). Without that attribute (e.g. Ensembl GFF3) a selenoprotein looks like a
  protein with an internal stop and is skipped (1 transcript in the dog).
- Consequences are computed one record at a time: two nearby variants on the same haplotype
  that change the same codon are scored separately.
- A variant changing several residues at once (an MNV over two codons, e.g. `4WL>4CM`) is
  scored as the sum of its single substitutions, as ProteinGym does for multiple mutants;
  its percentile places that sum on the single-SNV scale. In-frame insertions and deletions
  are not scored (csq reports them in `BCSQ`).
- Scores are raw log-likelihood ratios, not calibrated probabilities of pathogenicity; the
  percentile gives them a genome-wide scale, not a probability either.
- Viral proteins are scored less reliably than cellular ones (ProteinGym: VESM 0.43 on
  viruses vs 0.47–0.52 elsewhere; alignment-based methods ≈0.47 on viruses).

## Compatibility

Before 1.0 these are rules the project follows, not yet guarantees; 1.0 will make them
guarantees.

**Kept stable across 0.x versions** (a change needs a deprecation note in the changelog and a
new minor version):

- command names and their options: `annotate`, `precompute`, `export`, `export-variants`,
  `proteome`, `doctor`;
- VCF `INFO` tags (`OMS`, `OMS_WT`, `OMS_WT_PCT`, `OMS_MM`, `OMS_REF_MISMATCH`) and the order
  of the 10 fields of an `OMS` entry; new information comes as new tags;
- TSV columns of `annotate` and `export-variants`: append-only (new columns at the end,
  existing names and positions unchanged);
- status values keep their meaning (new ones can be added);
- the definition of the scores (log-likelihood ratios, wild-type and masked marginals, the
  long-protein windows).

**Not stable:** the Python API (modules and functions are internal), log messages, and the
on-disk cache layout. The cache is versioned (`<cache>/v1/…`): a change in how scores are
computed bumps the version, so an old cache is recomputed, never misread. The percentile
reference is keyed by the annotation, genome and genetic codes.

**Reproducibility.** Scores depend on the model (with its pinned revision), the genome, the
annotation, the genetic codes and the omnimissense version; output VCFs record the version and
the model (`##omnimissenseVersion`, `##omnimissenseModel`), `export` files all of them.
`pysam` is pinned to the 0.24 series (bcftools 1.24), because omnimissense relies on bcftools
csq's genetic-code tables and `--dump-gff` format; the pin moves only after the test suite
passes on a new version.

## Installation (development)

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dsmear/omnimissense.git
cd omnimissense
uv sync --all-groups           # all groups: the benchmark tests need Biopython
uv run omnimissense doctor     # bcftools (bundled), PyTorch, CUDA
```

## Development

```bash
uv run pre-commit install      # ruff lint/format + guards against large files and secrets
uv run pytest                  # fast tests, tiny random model, no download
uv run pytest -m gpu           # also runs the real ESM-2 650M (downloads ~2.6 GB once)
uv run pre-commit run --all-files
```

Never commit genomes, VCFs, model weights or caches: they are excluded by `.gitignore`, and a
pre-commit hook rejects any file larger than 500 KB. Decisions and their rationale are logged
in [NOTES.md](https://github.com/dsmear/omnimissense/blob/main/NOTES.md).

## Models and licenses

omnimissense is released under the [MIT License](https://github.com/dsmear/omnimissense/blob/main/LICENSE). Model weights are not
redistributed: they are downloaded from Hugging Face on first use. Select a model with
`--model`; the cache is kept separately per model.

| Model | Status | License |
|---|---|---|
| ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) | supported (reference model of the benchmarks) | MIT |
| other ESM-2 sizes (e.g. `facebook/esm2_t30_150M_UR50D`) | supported, same architecture | MIT |
| ESM1b (`facebook/esm1b_t33_650M_UR50S`) | should work, untested | MIT |
| VESM 650M (`ntranoslab/vesm/VESM_650M`) | **default**; re-trained ESM-2, applied on top of the ESM-2 checkpoint (revision pinned) | MIT |
| VESM 35M / 150M / 3B (`ntranoslab/vesm/VESM_…`) | supported, not benchmarked (3B needs > 8 GB of GPU memory) | MIT |
| VESM3, ESM3 | **not supported**: non-commercial license | — |

## Citation

If you use omnimissense, please also cite the model you scored with: for the default model,
VESM (Dinh, Jang, Zaitlen & Ntranos, 2025; <https://github.com/ntranoslab/vesm>), which is
built on ESM-2: Lin et al., *Evolutionary-scale prediction of atomic-level protein structure
with a language model*, Science 379, 1123–1130 (2023). The long-protein windowing follows Brandes et al.,
*Genome-wide prediction of disease variant effects with a deep protein language model*,
Nat Genet 55, 1512–1522 (2023).
