# NOTES — decisions and rationale

Log of project decisions. Each entry: date, decision, rationale, sources.
Unverified facts are marked as such.

---

## 2026-11-25 — Phase 0: prior-art review

**Verdict: exists PARTIALLY.** The building blocks exist (models, FASTA-based scorers, a
VCF→ESM script inside CADD), but no open-source tool with a permissive license:
takes VCF + FASTA + GFF3 of an arbitrary species → computes PLM scores on that species' own
protein → writes them to the VCF `INFO` field.

### What already exists

| Project | What it does | Arbitrary-species VCF | Writes to VCF | License | Last activity |
|---|---|---|---|---|---|
| kircherlab/CADD-scripts `esmScore` | ESM-1v (5-model ensemble, 350-aa window) on a VEP-annotated VCF + Ensembl peptide FASTA | Partly: the ESM script itself has nothing human-specific, but it needs a VEP-annotated VCF and an Ensembl-format peptide FASTA (transcript ID read from the 5th header field), so it only works for species/assemblies with Ensembl annotation, not with an arbitrary GFF3. The CADD pipeline around it is configured for GRCh37/GRCh38 (human) only | Yes (INFO) | Custom CADD license, **non-commercial only** | push 2026-11-11 |
| JSchlensok/VespaG | ESM-2 3B embeddings + small network trained on GEMME scores; FASTA → CSV/H5 | No VCF (FASTA only) | No | GPL-3.0 | push 2026-02-06 |
| ntranoslab/vesm | Re-trained ESM-2 models (35M–3B) + VESM3; precomputed scores for UniProtKB/hg19/hg38 | No VCF | No | MIT (VESM 35M–3B), non-commercial (VESM3) | push 2026-01-27 |
| ntranoslab/esm-variants | ESM1b, FASTA → CSV of all substitutions, windows for long proteins | No VCF | No | MIT | push 2023-09-20 (inactive) |
| MaizeGDB/PanEffect | JS viewer of precomputed ESM scores for maize | No (maize only, viewer) | No | MIT | push 2024-01-30 |
| Ensembl/VEP_plugins | 94 entries; AlphaMissense, EVE, PrimateAI, REVEL, dbNSFP (human ESM1b)… | Plugins = precomputed human scores | — | Apache-2.0 | — |
| nf-core/modules | bcftools/csq, ensemblvep, snpeff, fastvep, vepyr; the only ESM module is amulety/esm2 (antibodies, deprecated) | — | — | — | — |
| PyPI / bioconda | no PLM→VCF package found (plausible names probed) | — | — | — | — |

### State of the art for the dog

- Current practice: lift dog variants over to human and read precomputed human scores
  (AlphaMissense/ESM1b/CADD). In the genotype-first study (bioRxiv 2025, CanFam3.1) only
  1,107/1,644 missense variants (~67%) could be lifted to human.
- Open Vet J 2025 ("mock-variant strategy"): 284 OMIA "deleterious" + 90 "non-deleterious"
  variants mapped to human → AUROC 0.55–0.61, specificity 0.20–0.46. The authors note that
  many OMIA "non-deleterious" variants are trait variants (e.g. coat colour) →
  **be careful when building the benign set in Phase 3**.
- Ensembl VEP provides SIFT for dog (alignment-based; the archived documentation page cites
  SIFT 6.2.1 / UniRef90 2014_11; current coverage not verified).

### Differentiation (approved 2026-11-25)

1. Species-agnostic: only VCF + FASTA + GFF3; no Ensembl cache, no liftover to human.
2. Permissive license (vs CADD non-commercial, VespaG GPL).
3. Score on the species' own protein → also covers variants without an alignable human ortholog.
4. Proteome cache keyed by sequence hash → publishable by-product: precomputed scores for the
   dog proteome.
5. Swappable model backend: ESM-2 650M by default; VESM_650M (MIT, same architecture) and
   ESM1b as options to compare in Phase 3.
6. Interoperability instead of competition: export a tabix-indexed TSV usable with
   `bcftools annotate` and VEP `--custom`.

### Language and efficiency

- **Python.** Nearly all the cost is the model forward pass on the GPU (PyTorch → C++/CUDA).
  Everything else is delegated to C/C++ libraries: bcftools/htslib for consequence calling,
  pysam/cyvcf2 for VCF I/O, pyarrow for Parquet. Python is the glue. Models, ProteinGym and
  the bioinformatics ecosystem are Python-native.
- **Efficiency: decided in the architecture now, optimised later with measurements.**
  - Now, because they are hard to retrofit: cache by sequence hash, batching by length, fp16,
    swappable model (including a small one for a fast mode), distributable precomputed scores.
  - Later, only if profiling justifies it: torch.compile/SDPA, multi-GPU, possibly a Rust
    rewrite of the non-GPU parts (precedents: fastVEP, vepyr).
  - Phase 2 adds a timing benchmark to get real numbers.

### Limits of the search / unverified

- `gh` was not installed: GitHub was searched through the unauthenticated public API (name,
  description, README; not code).
- VespaG: the README says `pip install vespag`, but PyPI returns 404 for that name.
- VESM `UniProtKB_*_scores`: species coverage not verified (1.6 GB file not downloaded).
- The Frontiers 2024 (cat) paper and the dog preprint were only partly read (truncated text,
  supplementary material not read); for Open Vet J 2025 only the abstract was read.

---

## 2026-11-25 — Phase 1: repository setup

| Decision | Rationale |
|---|---|
| Name **omnimissense** (folder renamed from `zoomissense`) | Says what it does (missense, any species); not tied to a model or to animals only; free on PyPI, conda and GitHub (checked 2026-11-25). |
| License **MIT**, copyright "The omnimissense contributors" | Same license as ESM-2, ESM1b, VESM 35M–3B and ProteinGym; all runtime dependencies are permissive (PyTorch BSD-style, transformers/pyarrow Apache-2.0, pysam/cyvcf2/typer MIT). bcftools is MIT/Expat in its default build and is called as an external program anyway. VESM3/ESM3 are excluded (non-commercial). Model weights are never redistributed. A data license for published precomputed scores (CC-BY-4.0 or CC0) will be decided when we publish them. |
| Everything in **English** | International audience and contributors. |
| Python **≥ 3.11** | typer 0.29 and pre-commit 4.7 require 3.11. CI tests 3.11 and 3.13. |
| **uv** + `uv_build` backend, `src/` layout | Requested in the brief; `uv_build` is uv's native backend, no extra tool. |
| CLI with **typer** | Type-annotated options; first commands: `--version` and `doctor` (checks bcftools, tabix, PyTorch, CUDA). |
| Models loaded through **Hugging Face `transformers`** (`EsmForMaskedLM`), not `fair-esm` | `facebookresearch/esm` now redirects to `Biohub/esm-archive`, archived since 2024; the HF checkpoints (`facebook/esm2_t33_650M_UR50D`, MIT) are maintained and VESM loads the same way. Implemented in Phase 2. |
| Heavy dependencies (torch, transformers, pysam, pyarrow) added in Phase 2, not now | Keeps CI fast while there is no code that uses them. |
| **pre-commit**: ruff (check + format), `check-added-large-files --maxkb=500 --enforce-all`, `detect-private-key`, basic hygiene hooks | Enforces "never commit large data, weights or secrets" mechanically, not just by `.gitignore`. CI runs the same hooks, so local and CI checks are identical. |
| `.gitignore` excludes genomes, annotations, VCF/BAM, weights, caches, Parquet; whitelists `tests/data/` | Small hand-made fixtures for coordinate tests (Phase 2) must be committed. |
| `initial_prompt.md` kept out of git via `.git/info/exclude` | Personal brief, not project documentation. |
