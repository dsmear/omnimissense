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

---

## 2026-11-25 — Phase 2: MVP

| Decision | Rationale |
|---|---|
| **bcftools csq**, not Ensembl VEP, for consequence calling | Takes VCF + FASTA + GFF3 directly; fast (C); MIT; haplotype-aware if ever needed; and it ships **inside pysam** (pysam 0.24.1 bundles bcftools/htslib 1.24, `import pysam.bcftools`), so users and CI install nothing. VEP needs a Perl stack or Docker and a sorted, bgzipped, tabix-indexed GFF. Downside shared by both: csq only accepts Ensembl-style GFF3 ("Currently, only Ensembl GFF3 files are supported", bcftools manual) — a GFF3 normaliser is the main follow-up for "any species". |
| csq options `-l -s -` | `-l`: one record at a time, so unphased data works and each variant gets its own score. `-s -`: annotate every ALT allele regardless of genotypes. Checked on the synthetic genome that samples/genotypes pass through unchanged. |
| BCSQ format parsed: `Consequence\|gene\|transcript\|biotype\|strand\|aa_change\|dna_change` (e.g. `missense\|GENEA\|TXA\|protein_coding\|+\|2K>2R\|45A>G`) | Observed with bcftools 1.24; the manual's older examples lack the biotype field. Entries without a single-record amino-acid change (synonymous, splice_region, `@pos`, `a+b` haplotype changes) are ignored; `*missense` (downstream of a stop) is not missense. |
| Dog reference: **Ensembl release 116, ROS_Cfam_1.0** (GCA_014441545.1), `dna.toplevel.fa.gz` + `116.gff3.gz` | Verified via Ensembl REST (`/info/assembly`, `/info/data` → release 116) and FTP; BSD `sum` checksums matched Ensembl's CHECKSUMS for all files. |
| Own GFF3 reader + CDS translation | csq outputs only the changed residue; ESM needs the whole protein. The reader follows csq's ID conventions so both see the same transcripts. **Validated: 43,524/43,524 dog proteins identical to Ensembl `pep.all`** after one fix: non-ATG start codons (CTG/TTG, standard code) at a complete 5' end (first CDS phase 0) are translated as M, like Ensembl (177 proteins had L instead of M before). The one Ensembl peptide we lack is a selenoprotein (`U`). |
| **REF pre-check** against the FASTA; mismatching records flagged `OMS_REF_MISMATCH` and excluded from csq; error if >10% mismatch | bcftools csq aborts the whole run at the first REF mismatch ("the fasta reference does not match the VCF REF allele"). A high mismatch rate almost always means wrong assembly or chromosome naming (`chr1` vs `1`), so we fail with an explicit message. |
| **Reference amino-acid check**: csq's ref aa must equal our translated protein at that position, else status `ref_aa_mismatch`, no score, warning logged | Brief: "if it does not match, report it, don't ignore it". Catches disagreements between csq's and our translation (or annotation oddities). |
| Mitochondrial transcripts skipped (`mitochondrial_skipped`) | Verified in csq.c: a single genetic-code table is used genome-wide (default 0 = standard), so csq's consequences on MT would be wrong. (No MT coding transcripts in the dog GFF3 anyway.) Possible later: second csq run on MT with `-C 2`. |
| Scores = LLR `log P(alt) − log P(ref)`; log-softmax over the full vocabulary, then the 20 standard amino acids | The LLR is the same whether or not probabilities are renormalised over the 20 amino acids (same denominator). |
| Long proteins: windows of 1022 residues, overlap 511, sigmoid edge weights over the first/last 255 residues of artificial edges (scale = ramp/12.8 ≈ 20), weighted average of log-probabilities | Adapted from Brandes et al. 2023 (`esm-variants`, `get_intervals_and_weights`, MIT). Real protein ends are never down-weighted. Masked marginals use the same windows and weights. |
| Cache: Parquet, append-only part files sorted by `seq_hash`, row groups of 128; path `<cache>/v1/<model>/{wt_marginal,masked_marginal}/` | Keyed by SHA-256 of the protein, so isoforms with the same sequence and repeated runs are free. `v1` must be bumped if scoring changes (e.g. window parameters). Not safe for concurrent writers (documented). |
| Model loaded via `transformers.EsmForMaskedLM`, fp16 on CUDA, SDPA attention; own tokenisation from the tokenizer vocabulary | `facebookresearch/esm` is archived (Phase 1). Own tokenisation avoids per-sequence tokenizer overhead and lets tests use a tiny random model with the same alphabet. Requires transformers ≥ 4.56 for the `dtype=` argument. |
| Default `max_tokens` = 8192 | Measured on the RTX 3070 (400 random dog proteins): 4096 → 11,993 res/s, 8192 → 12,405, 16384 → 11,961, 24576 → 11,701; peak VRAM 1.4–2.4 GiB. The GPU is saturated already at small batches. |
| Output VCF = input VCF (all samples) + csq's `BCSQ` + `OMS*` fields | Users keep their data; BCSQ is useful context. The csq run uses a sites-only copy (faster with many samples) and is merged back in lockstep. |
| htslib "could not retrieve index" messages silenced when opening VCFs | Harmless for streaming reads of unindexed files, but alarming to users. |
| Tests: synthetic genome designed from protein sequences (expected effects known independently); tiny random ESM (no download) for scoring logic; real ESM-2 test behind `-m gpu` | Fast CI without GPU or downloads; the GPU test checks the real model end to end. |
| Cache lookups read only the Parquet row groups whose `seq_hash` min/max can contain a requested hash | First real run used 5.8 GB RAM because the dataset filter loaded the whole cache (1.4 GB on disk) before filtering. With row-group pruning a fully cached re-annotation of 127,270 dog variants takes 13 s and 1.4 GB RAM, with byte-identical output. |
| Logging: INFO only for `omnimissense.*`, third-party loggers at WARNING | httpx/Hugging Face printed every HTTP request at INFO. |
| Hugging Face now redirects `facebook/esm2_t33_650M_UR50D` → `biohub/esm2_t33_650M_UR50D` (observed 2026-11-25) | Same move as the GitHub repo. The `facebook/...` ID still resolves; kept as default because it is the ID cited in the literature. Cache paths use the requested ID. |
| Version 0.1.0 | First working MVP. |

### Phase 2 results (RTX 3070 8 GB, Ensembl 116 dog)

- Proteome: 43,525 coding transcripts → 43,524 OK (1 selenoprotein = internal stop) → 43,092 unique proteins, 26,655,640 residues.
- `precompute`: 2,148 s (35.8 min), 12,412 residues/s, peak RAM 4.9 GB, cache 1.4 GB.
- Demo: Ensembl 116 variation VCF, chromosome 38:0–5 Mb (127,270 records, sites only) → 0 REF mismatches, 598 records with missense, 1,689 missense consequences (128 transcripts, 52 genes), all status `ok`; 1,640 masked positions computed; 137 s first run, 13 s cached. Pearson(wt, masked) = 0.979.
- Most damaging in the demo: ATP2B4 G833R (masked −13.3), RAB7B L122R (−13.0), KCNT2 V823E (−12.8); least: conservative I→L / L→V (+7.6/+7.7). Plausible, but **not a validation** (Phase 3).
