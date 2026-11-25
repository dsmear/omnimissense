# omnimissense

[![CI](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml/badge.svg)](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Score missense variants in any species with protein language models, straight from a VCF.**

> **Status: alpha (MVP).** The pipeline works end to end and its coordinate handling is
> tested, but the scores have **not been validated yet** on benchmarks (Phase 3). Do not use
> them for breeding or clinical decisions.

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

## Validation so far

- **Coordinates:** a hand-designed synthetic genome (`tests/synthetic.py`) with genes on both
  strands and codons split by introns; every designed variant is recovered with the right
  protein position and amino acids.
- **Proteome:** the dog proteome translated from FASTA + GFF3 is identical to Ensembl's
  published peptides for 43,524 / 43,524 transcripts (the only exclusion is one
  selenoprotein, see limitations).
- **Model:** with the real ESM-2 650M, the wild-type LLR is exactly 0 and a core
  substitution in ubiquitin (I44D) scores far below a conservative one (I44V).
- **Benchmarks (ProteinGym, dog disease variants): not done yet** (Phase 3).

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
