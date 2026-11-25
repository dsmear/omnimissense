# omnimissense

[![CI](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml/badge.svg)](https://github.com/dsmear/omnimissense/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Score missense variants in any species with protein language models, straight from a VCF.**

> **Status: pre-alpha.** The project scaffolding is in place; the scoring pipeline is under
> development. Nothing below the "Roadmap" section works yet.

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
| GFF3 | the gene annotation (where genes, exons and coding sequences are) |

It outputs the same VCF with protein-language-model scores in the `INFO` field, plus a TSV
summary. The first target species is the dog.

## Roadmap

- [x] Phase 0: prior-art review (see [NOTES.md](NOTES.md))
- [x] Phase 1: repository scaffolding
- [ ] Phase 2: MVP
  - consequence calling with an established tool (choice between `bcftools csq` and Ensembl VEP documented in NOTES.md)
  - ESM-2 650M scoring: wild-type marginals for the whole proteome (cached on disk by sequence hash), masked marginals for the variants in the VCF, sliding windows for proteins longer than 1022 residues
  - scores written to VCF `INFO` and TSV
- [ ] Phase 3: validation
  - reproduce published ESM-2 Spearman correlations on ProteinGym DMS assays
  - dog benchmark: known causal missense variants vs common, presumably benign variants (AUROC)

## Installation (development)

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
git clone https://github.com/dsmear/omnimissense.git
cd omnimissense
uv sync
uv run omnimissense --version
uv run omnimissense doctor     # checks bcftools, tabix, PyTorch and CUDA
```

## Development

```bash
uv run pre-commit install      # ruff lint/format + guards against large files and secrets
uv run pytest
uv run pre-commit run --all-files
```

Never commit genomes, VCFs, model weights or caches: they are excluded by `.gitignore`, and a
pre-commit hook rejects any file larger than 500 KB. Small hand-made test fixtures go in
`tests/data/`.

## Models and licenses

omnimissense is released under the [MIT License](LICENSE). Model weights are not
redistributed: they are downloaded from Hugging Face on first use.

| Model | Status | License |
|---|---|---|
| ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) | default (planned) | MIT |
| ESM-2 150M, ESM1b 650M | optional (planned) | MIT |
| VESM 35M–3B (`ntranoslab/vesm`) | optional (planned) | MIT |
| VESM3, ESM3 | **not supported**: non-commercial license | — |

## Citation

If you use omnimissense, please also cite the model you scored with, e.g. ESM-2:
Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language
model*, Science 379, 1123–1130 (2023).
