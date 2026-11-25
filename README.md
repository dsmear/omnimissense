# omnimissense

[![CI](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml/badge.svg)](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Score missense variants in any species with protein language models, straight from a VCF.**

> **Status: alpha.** The pipeline works end to end, reproduces published ESM-2 results and
> separates known dog disease variants from common dog variants (see [Validation](#validation)).
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
| GFF3 | the gene annotation (where genes, exons and coding sequences are), Ensembl-style |

It outputs the same VCF (all records, all samples) with protein-language-model scores in the
`INFO` field, plus a TSV with one row per missense consequence.

## How it works

1. **REF check.** Every VCF record's REF allele is compared with the genome. Mismatching
   records are flagged (`OMS_REF_MISMATCH`) and skipped; if more than 10% mismatch, the run
   stops (VCF and FASTA are probably from different assemblies).
2. **Consequences** are called with [`bcftools csq`](https://samtools.github.io/bcftools/howtos/csq-calling.html)
   (bundled in pysam, nothing to install), one record at a time, for every ALT allele.
3. **Proteins** are translated from the FASTA + GFF3. The reference amino acid reported by
   csq must match the translated protein; mismatches are reported (`ref_aa_mismatch`).
4. **Scores** come from ESM-2 650M (fp16 on GPU), as log-likelihood ratios
   `log P(alt) − log P(ref)` at the mutated position (more negative = more likely damaging):
   - `wt_marginal`: one forward pass per protein gives all 19 substitutions at every position;
     the whole proteome can be precomputed once (`omnimissense precompute`).
   - `masked_marginal`: the position is masked and predicted from its context; one forward
     pass per position, computed only for the variants in the VCF.
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

# optional, once: score the whole proteome into the cache
uv run omnimissense precompute --fasta $FASTA --gff3 $GFF3

uv run omnimissense annotate --vcf my_dogs.vcf.gz --fasta $FASTA --gff3 $GFF3 \
    -o my_dogs.omnimissense.vcf.gz          # also writes my_dogs.omnimissense.tsv
```

The plain-gzip FASTA distributed by Ensembl is recompressed once to BGZF (`.fa.bgz`) next to
the original, because random access needs it.

## Output

**VCF `INFO` fields**

| Field | Number | Content |
|---|---|---|
| `OMS` | `.` | one entry per missense consequence: `Allele\|Gene\|Transcript\|Canonical\|ProteinPos\|RefAA\|AltAA\|WTMarginal\|MaskedMarginal\|Status` |
| `OMS_WT` | `A` | lowest (most damaging) wild-type-marginal score across transcripts, per ALT allele |
| `OMS_MM` | `A` | lowest masked-marginal score across transcripts, per ALT allele |
| `OMS_REF_MISMATCH` | flag | the REF allele does not match the genome; record not annotated |
| `BCSQ` | `.` | consequences from bcftools csq (all consequence types) |

**TSV columns:** `chrom pos ref alt gene transcript canonical biotype strand protein_pos
ref_aa alt_aa protein_length n_windows wt_marginal masked_marginal status`

**Status values:** `ok`, `ref_aa_mismatch` (csq and the translated protein disagree),
`multi_aa_change` (several residues change), `nonstandard_aa`, `protein_unavailable`
(e.g. internal stop codon), `mitochondrial_skipped` (see limitations). Only `ok` rows are scored.

## Performance (RTX 3070, 8 GB)

| Step | Measured |
|---|---|
| Wild-type marginals, ESM-2 650M fp16 | ~12,400 residues/s, 1.6 GiB peak VRAM |
| Whole dog proteome (`precompute`) | 43,092 unique proteins, 26.7 M residues in 36 min; cache 1.4 GB |
| Masked marginals | 15–28 positions/s, depending on protein length |
| Annotate 127,270 dog variants (chr38, 0–5 Mb; 1,689 missense consequences) | 137 s on first run, 13 s when the scores are cached (1.4 GB RAM) |
| Proteome translation | ~10 s (after the one-off BGZF conversion of the genome, ~3 min) |

## Validation

Scripts and details: [benchmarks/README.md](benchmarks/README.md). Results: `benchmarks/results/`.

**Implementation checks**

- **Coordinates:** a hand-designed synthetic genome (`tests/synthetic.py`) with genes on both
  strands and codons split by introns; every designed variant is recovered with the right
  protein position and amino acids.
- **Proteome:** the dog proteome translated from FASTA + GFF3 is identical to Ensembl's
  published peptides for 43,524 / 43,524 transcripts (the only exclusion is one
  selenoprotein, see limitations).
- **ProteinGym** (`benchmarks/proteingym.py`): Spearman correlation between our masked-marginal
  scores and experimental fitness, against the value ProteinGym publishes for ESM-2 650M.

| DMS assay | organism | length | mutants | ours | published |
|---|---|---|---|---|---|
| BLAT_ECOLX_Stiffler_2015 | *E. coli* | 286 | 4,996 | 0.732 | 0.731 |
| KCNJ2_MOUSE_Coyote-Maestas_2022_function | mouse | 428 | 6,963 | 0.382 | 0.382 |
| SC6A4_HUMAN_Young_2021 | human | 630 | 11,576 | 0.543 | 0.543 |
| UBE4B_MOUSE_Starita_2013 | mouse | 1,173 | 899 | 0.459 | 0.459 |
| NPC1_HUMAN_Erwood_2022_HEK293T | human | 1,278 | 637 | 0.695 | 0.694 |
| BRCA1_HUMAN_Findlay_2018 | human | 1,863 | 1,837 | 0.515 | 0.515 |

The last three are longer than 1022 residues and use our sliding windows.

**Dog benchmark** (`benchmarks/dog_omia.py`): OMIA dog missense variants mapped to dog
proteins, against Dog10K missense variants with allele frequency ≥ 5% among 1,872 dogs
(UU_Cfam_GSD_1.0, annotated by omnimissense). AUROC with 95% bootstrap interval; BLOSUM62 is a
context-free baseline (amino-acid similarity only).

| causal set | common variants | n causal / common | ESM-2 masked | ESM-2 wild-type | BLOSUM62 |
|---|---|---|---|---|---|
| pathogenic (AVCG P/LP) | all | 49 / 521 | **0.962** (0.926–0.988) | 0.954 (0.918–0.983) | 0.678 |
| pathogenic (AVCG P/LP) | same genes | 49 / 33 | **0.947** (0.890–0.987) | 0.934 (0.869–0.981) | 0.613 |
| disease (P/LP + not yet classified) | all | 118 / 521 | 0.945 (0.919–0.969) | 0.943 (0.916–0.966) | 0.650 |
| disease | same genes | 118 / 104 | 0.938 (0.904–0.966) | 0.932 (0.897–0.961) | 0.647 |
| coat/hair **traits** | all | 15 / 521 | 0.968 (0.912–0.999) | 0.968 (0.915–0.998) | 0.728 |
| pathogenic **vs OMIA traits** | — | 49 / 15 | 0.437 (0.282–0.593) | 0.461 (0.306–0.620) | 0.437 |

How to read it:

- ESM-2 scores separate known dog disease variants from common dog variants far better than
  amino-acid similarity alone, also within the same genes.
- **Trait variants** (coat colour, hair) look as disruptive as disease variants, and the model
  cannot tell the two apart (AUROC 0.44, no better than chance). The score measures damage to
  the protein, not disease. This also explains the low AUROC (0.55–0.61) reported for
  human-mapped predictors on OMIA when "non-deleterious" (often trait) variants are the
  negatives; our numbers are **not** directly comparable with that study.
- Common variants are an easy negative set (selection has removed damaging alleles from high
  frequency). Finding the causal variant among the *rare* variants of one dog is harder and
  is not measured here. Within its own protein, the median pathogenic variant ranks at the
  32nd percentile of all possible substitutions (wild-type marginal); only 8% fall in the
  most damaging 10%.
- Small sets (49 curated pathogenic variants), OMIA ascertainment bias (variants found in
  candidate genes), and 133 of 176 OMIA records mappable to Ensembl proteins (gene symbol
  missing or different protein numbering for the rest).

**Realistic test and comparison with the human-ortholog workaround**
(`benchmarks/dog_realistic.py`). For 50 Dog10K dogs (49 breeds), each dog's candidates are
the rare missense variants it carries (allele frequency < 1%; median 126 per dog). Each
causal OMIA variant is inserted in turn among each dog's candidates and ranked. The same
variants are also scored the way people do today without a dog tool: map the variant to the
human ortholog (same gene, aligned residue must be identical) and use a human score: ESM-2 on
the human protein, or AlphaMissense.

| 49 pathogenic variants × 50 dogs | median rank | top-1 | top-5 | top-10 |
|---|---|---|---|---|
| **omnimissense** (ESM-2 on the dog protein) | 6 | 19% | 46% | 66% |
| ESM-2 on the human ortholog | 6 | 19% | 49% | 67% |
| AlphaMissense (human) | 4 | 26% | 60% | 75% |
| BLOSUM62 | 40.5 | 0% | 1% | 13% |

| AUROC, pathogenic vs rare variants | all variants (49 / 7,581) | scored by every method (48 / 4,180) |
|---|---|---|
| **omnimissense** | 0.913 | 0.882 |
| ESM-2 on the human ortholog | 0.915 | 0.874 |
| AlphaMissense (human) | 0.936 | 0.912 |

What this says:

- **Dog protein vs human protein, same model:** on the variants both can score, scoring the
  dog's own protein is slightly but consistently better (+0.007 to +0.019 AUROC; the paired
  95% interval is at or above zero in three of four comparisons, e.g. disease vs common
  +0.005 to +0.032, and −0.002 to +0.026 in the fourth).
- **Coverage:** the human workaround can score only 56% of the dogs' rare missense variants
  and 41% of the common ones (the human residue differs or there is no human entry), but 96%
  of the causal variants, which sit on conserved residues. Unscorable variants are mostly
  benign, so for ranking they cost the workaround little in the dog, a species close to
  human. omnimissense scores 100% of them.
- **AlphaMissense** (trained with structure and population data, human only, non-commercial
  license) ranks causal variants better than ESM-2 650M on either protein (top-10: 75% vs
  66%); the paired AUROC difference is significant in one of four comparisons.
- In a real dog genome, omnimissense puts the causal variant in the top 10 of ~126 rare
  missense candidates two times out of three.

## Limitations

- GFF3 must follow the Ensembl layout (`ID=gene:…`, `ID=transcript:…`), as required by
  bcftools csq. Other sources (e.g. NCBI RefSeq) must be converted first.
- Mitochondrial transcripts are skipped: mitochondria use a different genetic code and
  bcftools csq applies a single code to the whole genome.
- Selenoproteins (a TGA codon read as selenocysteine) look like proteins with an internal
  stop and are skipped (1 transcript in the dog).
- Consequences are computed one record at a time: two nearby variants on the same haplotype
  that change the same codon are scored separately.
- Variants changing several residues at once (MNVs) are reported but not scored.
- Scores are raw log-likelihood ratios, not calibrated probabilities of pathogenicity.

## Installation (development)

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11. A CUDA GPU is strongly
recommended (CPU works but is slow).

```bash
git clone https://github.com/dsmear/omnimissense.git
cd omnimissense
uv sync
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
in [NOTES.md](NOTES.md).

## Models and licenses

omnimissense is released under the [MIT License](LICENSE). Model weights are not
redistributed: they are downloaded from Hugging Face on first use. Select a model with
`--model`; the cache is kept separately per model.

| Model | Status | License |
|---|---|---|
| ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) | default | MIT |
| other ESM-2 sizes (e.g. `facebook/esm2_t30_150M_UR50D`) | supported, same architecture | MIT |
| ESM1b (`facebook/esm1b_t33_650M_UR50S`) | should work, untested | MIT |
| VESM 35M–3B (`ntranoslab/vesm`) | planned | MIT |
| VESM3, ESM3 | **not supported**: non-commercial license | — |

## Citation

If you use omnimissense, please also cite the model you scored with, e.g. ESM-2:
Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language
model*, Science 379, 1123–1130 (2023). The long-protein windowing follows Brandes et al.,
*Genome-wide prediction of disease variant effects with a deep protein language model*,
Nat Genet 55, 1512–1522 (2023).
