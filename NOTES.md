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

---

## 2026-11-25 — Phase 3: validation

### ProteinGym sanity check (`benchmarks/proteingym.py`)

| Decision | Rationale |
|---|---|
| Data from the official Hugging Face dataset `OATML-Markslab/ProteinGym_v1` (revision `1ea2aa1…`), reference file and published scores from GitHub commit `144fe22…` | `marks.hms.harvard.edu` failed TLS verification ("unable to get local issuer certificate"); verification was not disabled. Same authors, pinned revisions. Mutant counts checked against the reference file. |
| Compare our **masked** marginal with ProteinGym's "ESM2 (650M)" column | ProteinGym's `scoring_ESM2_substitutions.sh` uses `scoring_strategy="masked-marginals"`. |
| 3 single-window assays (pass/fail, tolerance 0.03) + 3 assays > 1022 aa (informational) | ProteinGym scores long proteins with one window centred on the mutation (`get_optimal_window`); we average overlapping windows, so exact agreement is not expected there. |
| **Result:** differences ≤ 0.001 on all six assays (BLAT_ECOLX 0.732 vs 0.731, KCNJ2_MOUSE 0.382 vs 0.382, SC6A4_HUMAN 0.543 vs 0.543, UBE4B_MOUSE 0.459 vs 0.459, NPC1_HUMAN 0.695 vs 0.694, BRCA1_HUMAN 0.515 vs 0.515) | Implementation matches the reference; the windowing scheme gives the same correlations as ProteinGym's on these long proteins. |

### Dog benchmark (`benchmarks/dog_omia.py`)

| Decision | Rationale |
|---|---|
| OMIA CSV downloaded **by hand** by the user (`data/omia/omia_dog_variants.csv`, 2026-11-25) | OMIA `robots.txt` disallows `/download*`, `/curate*`, `/search*` for automated agents. OMIA terms: free, cite OMIA (DOI 10.25910/2AMR-PV70) and acknowledge the Sydney Informatics Hub / AHIDA; contact the curators about co-authorship for publications relying on the MySQL dump or extended data. No OMIA data are committed; only aggregate metrics. The OMIA VCF exports ("for maintainers of other databases") were not used. |
| Causal sets from the CSV columns actually present: `Variant Effect = missense`; **pathogenic** = AVCG `Pathogenicity Classification` P, LP or LP/P; **disease** = pathogenic + "Not currently evaluated" disease phenotypes; **trait** = coat/hair/pattern phenotypes (explicit list in the script); "Modifier of copper toxicosis" and "Out of scope" excluded | The planned "Deleterious?" column is not in the CSV export. 179 dog missense records: 64 P/LP, 114 not evaluated (of which 14 traits), 1 out of scope. |
| Mapping by gene symbol + protein change (`XP_…:p.(R1186Q)`), reference residue must match; canonical transcript first; GSD annotation first, ROS_Cfam_1.0 as fallback | OMIA coordinates are on mixed assemblies (CanFam3.1 mostly); protein-level mapping avoids liftover. ROS fallback recovered 21 variants whose gene symbol is missing from the GSD annotation. |
| Common set: **Dog10K** (CC-BY-4.0, Zenodo 8084059) PASS SNVs with ALT AF ≥ 5% among 1,872 dogs (breed, village, mixed; `inAnalyzedSNV`), read remotely with bcftools (`-S dogs -R cds.bed -f PASS -q 0.05:alt1 -G`, AC/AN recomputed on the subset); annotated with omnimissense on UU_Cfam_GSD_1.0 (Ensembl 116); canonical transcripts of the causal genes and of 500 random autosomal genes (seed 0) | The Dog10K BCF has genotypes but no INFO fields; it is on UU_Cfam_GSD_1.0, so the negatives run through the full omnimissense pipeline on a second assembly. Remote reads occasionally fail ("BCF read error"): one query per chromosome, cached, 3 retries. Variants both common and causal are dropped from the negatives. |
| Bug found and fixed during the run: random genes were drawn after removing the causal genes and the Dog10K extract was cached under a name that did not reflect the gene set | When the causal set changed, a stale extract was reused (random-gene negatives dropped from 361 to 34). Now: draw from all autosomal genes first, then remove causal genes; cache keyed by a hash of the regions and threshold. |
| BLOSUM62 added as a context-free baseline (verified identical to Biopython's matrix, 400/400 entries; Biopython not added as a dependency) | Puts the very high AUROC in context: amino-acid similarity alone reaches 0.68 (0.61 within the same genes). |
| Extra comparison: pathogenic vs OMIA trait variants | Reproduces the negative-set definition of the Open Vet J 2025 study (OMIA "non-deleterious" variants, often traits). |

### Dog results (ESM-2 650M masked marginal; Ensembl 116 GSD; Dog10K AF ≥ 5%; 494 random genes, seed 0)

- OMIA: 176 dog missense records used → 133 mapped (21 via ROS_Cfam_1.0), 25 reference-residue mismatches (different protein numbering), 16 gene symbols not found, 2 without a single protein change. Sets: 49 pathogenic, 118 disease, 15 trait. 3 causal variants are also common in Dog10K and were removed from the negatives.
- Pathogenic vs all common (49 / 521): AUROC **0.962** (0.926–0.988); same genes (49 / 33): 0.947 (0.890–0.987); BLOSUM62 0.678 / 0.613. Wild-type marginal slightly lower (0.954 / 0.934).
- Disease vs all common (118 / 521): 0.945 (0.919–0.969); same genes 0.938.
- Traits vs common: 0.968 — **pathogenic vs traits: 0.437 (0.282–0.593)**, same as BLOSUM62: the model measures protein disruption, not disease.
- In-protein rank of pathogenic variants (wild-type marginal): median 32nd percentile of all possible substitutions; 8% in the most damaging 10%.
- Interpretation and limits: common variants are an easy negative set (selection); the realistic task (causal among a dog's rare variants) is not measured; small curated set; OMIA ascertainment bias. The numbers are not comparable with the 0.55–0.61 of the human-mapping study, whose negatives were mostly trait variants; a same-set comparison with human-mapped scores is future work.

---

## 2026-11-25 — Phase 4 (follow-up 2): realistic ranking and human-ortholog comparison

| Decision | Rationale |
|---|---|
| Dog10K SNV BCF downloaded locally (5.5 GB, MD5 verified) instead of remote queries | Approved by the user. Genome-wide rare-variant extraction needs ~180k CDS regions: ~3 h remotely with intermittent failures, minutes locally. |
| 50 random breed dogs (seed 0); candidates = rare (ALT AF < 1% among 1,872 dogs, AC ≥ 1) PASS SNVs they carry, missense in canonical transcripts of autosomal coding genes | Simulates a diagnostic setting: rank the causal variant among the rare missense variants of a real dog genome ("spike-in" design used for human variant prioritisation). Annotation runs on a GFF3 restricted to canonical transcripts to halve the scoring work. |
| Causal variants that a healthy Dog10K dog carries are removed from that role as candidates | Otherwise the same variant would compete with itself. |
| Human mapping via reviewed UniProt entries (downloaded 2026-11-25) by gene symbol, Biopython global alignment (BLOSUM62, −10/−0.5), identical reference residue required; proteins > 6,000 aa skipped | Reproduces the "mock variant" approach of the 2025 study. Biopython is in a separate `benchmarks` dependency group, not a runtime dependency; CI installs all groups. |
| Human scores: ESM-2 650M masked marginal on the human protein (same model) and AlphaMissense (CC BY-NC-SA 4.0, approved by the user; used only for aggregate metrics, not redistributed) | ESM-2 on human isolates "dog protein vs human protein"; AlphaMissense is what a practitioner using the human workaround would actually use. AlphaMissense rows are looked up by exact `(accession, "R106W")` string, so a UniProt sequence change since 2021 cannot silently shift positions. |
| Unscored variants (no human mapping / not in AlphaMissense) are ranked last; AUROC reported on all variants and on the subset scored by every method, with a paired bootstrap interval for dog ESM-2 − AlphaMissense | Separates coverage (what the human workaround cannot score) from quality on the variants both can score. |
| `dog_omia.py` setup refactored into `load_reference()` / `causal_variants()` | Shared with `dog_realistic.py`; re-run confirmed byte-identical dog_omia results. |
| Bug fixed during the run: Biopython's BLOSUM62 alphabet has no `U` (selenocysteine in some human proteins) → letters outside the alphabet are replaced by `X` before alignment (same length, positions unchanged); test added | The first run crashed in the alignment step. |

### Phase 4 results (50 dogs, 49 breeds; ESM-2 650M; Ensembl 116 GSD; Dog10K)

- Rare candidates: 19,878 rare SNVs carried by the 50 dogs → 7,581 rare missense (canonical); median 126 per dog (range 47–609).
- Human mapping (identical residue): 95.8% of causal, 55.9% of rare, 40.8% of common variants; AlphaMissense found for 95.8% / 55.1% / 40.5%.
- Ranking, 49 pathogenic × 50 dogs: omnimissense median rank 6, top-10 66%; human ESM-2 6 / 67%; AlphaMissense 4 / 75%; BLOSUM62 40.5 / 13%. Disease set (118): omnimissense 8 / 57%, AlphaMissense 5 / 68%.
- AUROC pathogenic vs rare: omnimissense 0.913 (all) / 0.882 (scored by all); human ESM-2 0.915 / 0.874; AlphaMissense 0.936 / 0.912.
- Paired differences (scored-by-all subsets): dog − human ESM-2 = +0.008 (−0.002 to +0.026) pathogenic/rare, +0.019 (+0.002 to +0.041) pathogenic/common, +0.007 (+0.000 to +0.015) disease/rare, +0.017 (+0.005 to +0.032) disease/common. Dog ESM-2 − AlphaMissense: −0.030 (−0.077 to +0.009), −0.019 (−0.064 to +0.013), −0.033 (−0.065 to −0.003), −0.026 (−0.059 to +0.005).
- Conclusions: (1) scoring the species' own protein gives a small, consistent gain over the human ortholog with the same model; (2) the coverage advantage is real (100% vs ~56% of rare variants) but costs the human workaround little in dogs because unmappable variants are mostly benign — it should matter more in species far from human; (3) AlphaMissense is stronger than ESM-2 650M; a stronger permissive model (VESM, MIT) is the most promising lever.

---

## 2026-11-25 — Phase 4b: VESM (inserted before the release, at the user's request)

| Decision | Rationale |
|---|---|
| VESM support as model ID `ntranoslab/vesm/VESM_<size>`, Hugging Face revision pinned (`946749e…`) | VESM `.pth` files are partial state dicts: for VESM_650M, 87 tensors / 100 M parameters (embeddings, layers 28–32, LM head), all different from ESM-2 and all present in ESM-2 650M. They are loaded with `load_state_dict(strict=False)` on top of the base ESM-2; unexpected keys raise an error (so a wrong file cannot be silently ignored). The tied embedding/decoder tensors are identical in the VESM file, so loading both is safe (checked). |
| VESM3 refused with an explicit error; VESM_3B supported but not benchmarked | VESM3 is built on ESM3-open (non-commercial license). VESM_3B needs the 11 GB ESM-2 3B checkpoint and does not fit an 8 GB GPU with the desktop running. |
| Both wild-type and masked marginals reported for every model; human-ortholog wild-type marginal added to `dog_realistic.py` | The VESM authors score with one unmasked forward pass (wild-type marginal, `get_llrs` in their README); ESM-2 is usually best with masked marginals (ProteinGym's choice). Comparisons use the same scoring mode on dog and human. |
| Result files tagged by model (`benchmarks/models.py`) | Keeps ESM-2 and VESM results side by side; the ESM-2 files were regenerated from the cache under the new names (identical values). |

### Phase 4b results (VESM_650M, wild-type marginal unless stated)

- ProteinGym (6 assays): VESM wt 0.774 / 0.390 / 0.562 / 0.452 / 0.725 / 0.523 (mean 0.571) vs ESM-2 masked mean 0.554; VESM better on 5 of 6. VESM masked is lower than VESM wt on 4 of 6, as expected from how VESM was trained.
- Dog OMIA vs common: pathogenic 0.977 (0.962–0.991) vs ESM-2 0.962; same genes 0.961 vs 0.947; disease 0.956 vs 0.945. Pathogenic vs traits 0.550 (0.403–0.703): still no discrimination.
- Realistic ranking (49 pathogenic × 50 dogs): VESM dog median rank 4, top-10 72% (ESM-2 6 / 66%, AlphaMissense 4 / 75%). AUROC pathogenic vs rare 0.927 all / 0.903 scored-by-all (AlphaMissense 0.936 / 0.912); VESM − AlphaMissense −0.035 to +0.019 (n.s.).
- Dog vs human ortholog with VESM wt: −0.008 to +0.006 (no difference); with ESM-2 masked the small dog advantage remains.
- "Causal in the most damaging 10% of its protein" is 0% with VESM vs 8% with ESM-2, but the median in-protein percentile is similar (0.336 vs 0.319): VESM pushes the most conserved positions further into the tail; not a bug (LLR distributions checked on ubiquitin, no saturation, ~1,300 distinct values). This metric is not informative and is no longer quoted in the README.
- Recommendation: make VESM_650M the default and the wild-type marginal the primary score (it is also the cheap one: it comes from the precomputed proteome, while masked marginals dominate the annotation time).
- **Defaults changed (user decision, 2026-11-25): model VESM_650M, wild-type marginal only; masked marginals with `--masked`.** Best model in all dog benchmarks; the wild-type marginal is VESM's intended mode and comes from the precomputed proteome. Demo VCF (chr38, 127,270 records): 25 s first run / 12 s cached, vs 137 s / 13 s with masked ESM-2. `proteingym.py` keeps ESM-2 as default (published reference).

---

## 2026-11-25 — Item 3: release and published dog scores

| Decision | Rationale |
|---|---|
| Release workflow `.github/workflows/release.yml` on `v*` tags: tag/version check, tests, `uv build`, PyPI **trusted publishing** (environment `pypi`, OIDC, no token), GitHub release with the built files | User's choice. No long-lived secret in the repository; the publish step can be gated by a required reviewer on the `pypi` environment. Actions pinned to current releases (gh-action-pypi-publish v1.14.3, upload-artifact v7, download-artifact v8). |
| Version 0.2.0, "Development Status :: 3 - Alpha"; README links made absolute | Wheel/sdist checked: only code, README, LICENSE; `twine check` passes; wheel installed in an isolated env runs `omnimissense doctor`. Relative links would break on PyPI. |
| New `omnimissense export` command (Parquet, one row per coding transcript, flattened `L x 20` LLR matrix, provenance JSON in the file metadata) | Needed for the published dataset and useful for any species; rows processed in sequence-hash order so cache reads touch few row groups. |
| Published dog scores: **ROS_Cfam_1.0 only**, VESM 650M wild-type marginal, **Hugging Face**, **CC-BY-4.0** | User's choices. Dataset card kept in the repo (`datasets/dog_ros_cfam_1.0/README.md`) with exact provenance (Ensembl 116 checksums, VESM and ESM-2 revisions: VESM `946749e…`, ESM-2 650M served as `biohub/esm2_t33_650M_UR50D` `08e4846…`). |
| Dataset export checked: values identical to `omnimissense annotate` on the chr38 demo (5 spot checks); the dataset card's code example is executed against the file | The first card example (SOD1 position 40) was wrong and was caught this way. |
| Finding: Ensembl 116's canonical SOD1 model (ENSCAFT00845040105, 326 aa, gene spanning 128 kb) carries a 173-residue N-terminal extension in front of SOD1 (`…RVM` + `EMKAVCVLKG…`); the literature variant SOD1 p.E40K is residue 213 there (E213K, VESM wt LLR −8.43) | Explains one of the 25 OMIA "reference residue mismatch" losses: literature/RefSeq numbering vs Ensembl gene models. Possible improvement for the benchmark: match OMIA variants on sequence context (or align RefSeq `XP_` proteins) instead of raw positions. Documented in the dataset card. |
| Dataset published (user-confirmed, public): https://huggingface.co/datasets/DavideMacchia/omnimissense-dog-ros-cfam-1.0-vesm-650m, HF commit `050cdf1…`; Parquet SHA-256 `2a9c70e3…352613` identical locally and remotely; HF recognises the CC-BY-4.0 license | Integrity check after upload. |

---

## 2026-11-25 — GFF3 layouts beyond Ensembl

| Decision | Rationale |
|---|---|
| No GFF converter written; transcripts are read from **bcftools csq's own parse** of the GFF3 (`--dump-gff`) | Checked first what exists. bcftools ships `misc/gff2gff` (Perl, MIT, P. Danecek 2023: fills `ID`/`biotype`/`Name` from GENCODE-style attributes, adds no `gene:`/`transcript:` prefixes) and `misc/gff2gff.py` (GenBank-derived prokaryotic GFF, one transcript per gene, needs gffutils, no license notice); AGAT is a generic GFF/GTF normaliser (GPL-3); `gbk2bcfgff` converts GenBank for csq. But csq itself accepts more layouts since 1.17/1.18 (NEWS: "ids can be just 'XXX'", "More versatility in parsing various flavors of GFF"): the csq bundled in pysam 0.24.1 (bcftools 1.24) reads a synthetic NCBI RefSeq GFF3 (`gene-`/`rna-`/`cds-` IDs, `gene_biotype` on genes, no biotype on mRNAs) without conversion and calls all 9 designed missense variants correctly. The blocker was omnimissense's own reader, which required `ID=gene:`/`ID=transcript:`. Reading csq's dump makes csq and the translation agree on transcripts, IDs and CDS by construction, for every layout csq accepts. |
| Structure (transcripts, gene links, strand, CDS + phase) from the dump; biotype, gene name, canonical tags and mitochondrial sequences from the original GFF3 | The dump drops tags and rewrites biotypes (`nonsense_mediated_decay` → `NMD`, unknown → `protein_coding`) and fills a missing gene `Name` with the ID. Keeping the original values leaves the export (and the published dog dataset's columns) unchanged. Transcript IDs are csq's: Ensembl prefixes stripped (as before), NCBI IDs verbatim (`rna-XM_…`). |
| `annotate` reuses the dump of its main csq run; `precompute`, `export` and `proteome` run csq once on an empty VCF | No extra csq pass in `annotate`. The empty VCF must list the FASTA's contigs: bcftools 1.24 crashed (segmentation fault) on an empty VCF without `##contig` lines. |
| Canonical = `Ensembl_canonical`, `MANE Select` or `RefSeq Select` tag; genes with no tagged transcript (coding or not) get their longest CDS (ties: smallest ID) | NCBI annotations of most species have no canonical tag; RefSeq Select covers only some species/genes. The rule is per gene, so a gene whose Ensembl canonical is non-coding keeps its coding transcripts non-canonical (as before). The number of genes handled by the fallback is logged. |
| Mitochondrial sequences also recognised from NCBI `region` features with `genome=mitochondrion` | NCBI names sequences by accession (`NC_001224.1` for yeast MT), which the name list (`MT`, `chrM`, …) misses. Since csq also uses the standard code there, csq and the translation would agree and the reference-residue check would pass: MT proteins would be scored with wrong residues **silently**. |
| Layouts csq cannot read fail with an explicit hint (GTF → AGAT/gffread; prokaryotic CDS directly under genes → `misc/gff2gff.py`/`gbk2bcfgff`) | Checked: csq rejects a GFF3 whose CDS have a gene as parent ("No usable transcripts found"). |
| Dev setup: the Biopython benchmark test is skipped when Biopython is missing; README says `uv sync --all-groups` | Following the README (`uv sync`, then `uv run pytest`) stopped at test collection (`No module named 'Bio'`); CI uses `--all-groups`, so it did not show there. |
| Validation on a real NCBI RefSeq annotation: *S. cerevisiae* R64 (`GCF_000146045.2`, annotation SGD R64-5-1), files downloaded by hand by the user from the NCBI FTP (2026-11-25; ftp.ncbi.nlm.nih.gov is not reachable from the development container) | New script `benchmarks/ncbi_translation.py` compares every translated protein with NCBI's `*_translated_cds.faa.gz`, matched by CDS `protein_id`. |
| **Result:** 6,027 coding transcripts (= 6,027 NCBI translated CDS): **6,002 / 6,002 identical**; 6 internal stop = S288C "blocked reading frames" that NCBI itself marks `pseudo=true` and translates with the same internal stops (FLO8, SDL1, CCW22, CRS5, YIR043C, YOL153C); 19 mitochondrial (`NC_001224.1`, recognised through `genome=mitochondrion`) skipped. No RefSeq Select tags in this annotation: canonical = longest CDS for all 6,027 genes. | The csq-dump reader works on a real RefSeq file without conversion. |
| End-to-end `annotate` on yeast: 1,519 random coding SNVs (1,500 nuclear transcripts + all MT, seed 0), tiny random model | 0 REF mismatches; 1,162 missense consequences: 1,122 `ok`, **0 `ref_aa_mismatch`** (csq and the translation agree everywhere), 2 `protein_unavailable` (the pseudo ORFs YOL153C, CRS5), 38 `mitochondrial_skipped`. 11 s. |
| Regression check on the dog (Ensembl 116, ROS_Cfam_1.0), run by the user on their machine (2026-11-26): `omnimissense proteome` before (`main`) and after the change | Both: `ok 43524`, `internal_stop 1`. Old vs new proteomes **identical** (43,524 / 43,524, same IDs and sequences); new vs Ensembl `pep.all`: 43,524 / 43,524 identical, 1 Ensembl peptide missing (the known selenoprotein). No "no canonical-transcript tag" message was logged, so every dog coding gene has an `Ensembl_canonical` transcript and canonical flags are unchanged too. |

---

## 2026-11-26 — Genetic codes

| Decision | Rationale |
|---|---|
| Genetic code tables **copied from bcftools 1.24 `csq.c`** (`gencode_tables`, generated there from NCBI `gc.prt`) into `omnimissense/genetic_code.py`, extracted programmatically from the source | csq and our translation must use the same table, or the reference-residue check fails (or, worse, passes on wrongly translated proteins). csq has 26 NCBI tables: 1–6, 9–16, 21–31, 33 (no 7, 8: retired; no 32) plus its own table 0 (standard codons, ATG as the only start), its default. |
| Tables tested three ways: equal to Biopython's NCBI tables (codons, stops, starts; benchmark dependency group only); **behavioural test against csq** for all 26 tables (a gene with every sense codon, every third-base change, csq's reported amino acids = ours); the yeast data below | A future pysam may bundle a different bcftools; the behavioural test would catch a change. |
| `--genetic-code N` is validated before reaching csq | Found in the source: `gencode` is a C global and pysam runs bcftools in-process, so an unknown `-C` does not fail but **silently keeps the table of the previous call** (`-C 0`..`39` all "worked" in one process). |
| Per-sequence codes: `--genetic-code` (non-mitochondrial sequences) and `--mito-genetic-code` override; otherwise the `transl_table` attribute of each sequence's CDS in the GFF3; otherwise 1 for nuclear sequences and **skip** for mitochondrial ones (as before) | NCBI writes `transl_table` whenever the code is not 1 (yeast MT: 50 CDS with `transl_table=3`), so NCBI annotations work without options, including plastids (11) and alternative nuclear codes (*Candida* 12). Ensembl GFF3 has no code: options needed (e.g. `--mito-genetic-code 2` for vertebrates). An annotated table csq lacks (32) is an error asking for an override. |
| bcftools csq is **run once per genetic code** (records split by sequence when writing the sites-only file), outputs merged back in input order; `ScoredVariant.record_index` is now the input-VCF ordinal | csq applies one code per run. A genome with one code still gives a single run; the csq parse of the GFF3 (`--dump-gff`) comes from the first run. For the standard code csq keeps its default table 0 (no `-C`), so standard-code output is unchanged. |
| Tables 27, 28, 31: codons that are stops only at the end of a CDS (otherwise Q/Q/E) are translated as amino acids and listed as stops; a terminal codon that is a stop of the table is removed | Same as csq (amino acid from the code string, stop from the mask). Without this, table 28 proteins would end with a spurious Q. |
| Initiation codons per table (e.g. GTG in table 11, ATA in tables 2 and 3) read as M at a complete 5' end | Generalises the previous CTG/TTG rule (= table 1). |
| **Yeast R64 results:** proteome vs NCBI `translated_cds`: **6,021 / 6,021 identical**, now including the 19 mitochondrial proteins (table 3 from the GFF3); 6 NCBI pseudo ORFs as before. `annotate` on the same 1,519 random coding SNVs: 1,157 `ok` (1,122 nuclear as before + 35 mitochondrial), 2 `protein_unavailable` (pseudo ORFs), **0 `ref_aa_mismatch`**, 0 skipped. Nuclear TSV rows identical to the previous run (VCF scores differ only in the 6th significant digit: CPU batching of the tiny test model changed because MT proteins joined the batches). | |
| Dog regression check (user's machine, commit `4853d8f`): `omnimissense proteome` on Ensembl 116 ROS_Cfam_1.0 | `ok 43524`, `internal_stop 1`; **43,524 / 43,524 identical to Ensembl `pep.all`** (1 Ensembl peptide missing: the known selenoprotein). Standard-code genomes are unchanged by the genetic-code support. |
| CLI help: `[default: …]` notes escaped | Typer's rich markup swallowed them (`--tsv`, `--cache-dir`, `--device` showed no default) — pre-existing. |
| Not modelled: RNA editing (plant organelles), selenocysteine (`transl_except`), programmed frameshifts | Translation is from the genome sequence; such proteins are reported (internal stop) or differ from the mature protein. |

---

## 2026-11-26 — Prokaryotic GFF3

| Decision | Rationale |
|---|---|
| CDS without a transcript feature get one in a **normalized copy** of the GFF3 (`prepare_annotation`), instead of asking users to run a converter | csq rejects CDS whose parent is a gene ("No usable transcripts found"), the layout of NCBI prokaryotic annotations (PGAP), Prokka (`--addgenes`), Bakta, and NCBI vertebrate mitochondria (CDS `Parent=gene-ND1`, no mRNA). In mixed files csq does not fail: it silently ignores those CDS. Prokka without `--addgenes` writes CDS with no parent at all. Handling it per CDS covers all these cases. |
| Synthetic IDs: transcript `tx-<CDS ID without "cds-">` (e.g. `tx-WP_000001.1`), gene `gene-<same>` only for parentless CDS; made unique with `-2`, `-3`… | Readable and traceable to the protein accession or locus tag; never equal to an existing feature ID. Transcript biotype = the gene's `gene_biotype`/`biotype`, else `protein_coding`; gene name of a created gene = CDS `gene`, else `Name`, else `locus_tag`. |
| An embedded `##FASTA` section is dropped in the copy | Checked: csq fails on it ("Could not parse the line") and Prokka/Bakta always write one. |
| csq, its `--dump-gff` and the metadata pass all read the normalized copy; the copy is written only when needed | Same transcripts everywhere by construction; Ensembl and NCBI eukaryote files (no orphan CDS, no `##FASTA`) are used as they are. The scan for orphans replaces the genetic-code scan: still one extra pass over the GFF3, plus one write pass for prokaryotic files. Parents are assumed to precede children (true of Ensembl, NCBI, Prokka, Bakta). |
| Hint for unusable GFF3 updated: only GTF (AGAT/gffread) and GENCODE-style attributes (`misc/gff2gff`) still need a converter | |
| Tests: NCBI-prokaryote and Prokka-style synthetic files (proteins, IDs, gene names, table 11 from `transl_table`), a mixed file (only the orphan CDS changed), `##FASTA` alone, end-to-end `annotate` for both styles; yeast R64 unchanged (6,021 / 6,021) | |
| Real prokaryote: *E. coli* K-12 MG1655 (GCF_000005845.2), files downloaded by hand by the user from the NCBI FTP (2026-11-26). All 4,318 CDS hang from genes (`Parent=gene-b0001`) with `transl_table=11` | `benchmarks/ncbi_translation.py`: 4,318 transcripts added, all with table 11; **4,297 / 4,297 identical** to NCBI's `translated_cds`, including the programmed frameshifts (*prfB*, *dnaX*, *copA*: CDS segments that overlap by one base or are adjacent) and the 9 genes with several CDS (alternative starts, e.g. *infB*: one transcript per CDS); 3 selenoproteins (fdnG, fdoG, fdhF: `transl_except` Sec) with internal stop; 18 pseudogenes. |
| New status **`pseudogene`**: CDS with `pseudo=true` (NCBI) are not translated or scored | 14 of the 18 *E. coli* pseudo CDS (genes interrupted by IS elements, joined across the insertion) translate without internal stop and would have been scored as real proteins (e.g. in `precompute`/`export`). csq itself calls no missense in transcripts of biotype `pseudogene`; for pseudo CDS under protein-coding genes (yeast's 6 "blocked reading frames") variants are now reported as `pseudogene` instead of `protein_unavailable`. Ensembl files have no `pseudo=true`. The benchmark script matches pseudo CDS (no `protein_id`) by `locus_tag`. |
| End-to-end `annotate` on *E. coli*: 2,113 random coding SNVs (1,500 random CDS + 15 per segment of every multi-segment CDS, seed 0), tiny model | 0 REF mismatches; 1,224 missense consequences: 1,223 `ok`, **0 `ref_aa_mismatch`** (frameshift genes included: prfB 20, copA 15, dnaX 13 `ok`), 1 `protein_unavailable` (selenoprotein fdnG). 9 s. |
| Possible follow-up: selenocysteine from `transl_except` | Would recover selenoproteins (3 in *E. coli*, 1 in the dog, ~25 in human); variants at the Sec residue itself would remain unscorable (`U` is not one of the 20 amino acids). |

---

## 2026-11-26 — ProteinGym by taxon: do the scores hold far from human?

| Decision / finding | Rationale |
|---|---|
| New `benchmarks/proteingym_all.py`: all 217 substitution assays, multiple mutants scored additively, results aggregated per protein then per taxon / function type, published models summarised on the same assays, ours − ESM-2 650M with a bootstrap interval over proteins | The question for "any genome" is whether the model, not the pipeline, holds outside human proteins. ProteinGym labels every assay with a taxon (Human 96, other Eukaryote 40, Prokaryote 50, Virus 31). |
| Aggregation verified on the published data: with any published model in place of ours, the script reproduces ProteinGym's `Summary_performance` (overall and `Taxa_*`) to 0.001 for every model compared | ProteinGym averages assays per UniProt protein first; a plain mean over assays differs by up to 0.02. |
| Scoring: wild-type marginal by default (VESM's mode, one pass per protein); `--masked` optional | VESM is meant to be scored with wild-type marginals (Phase 4b); ESM-2's published ProteinGym numbers are masked marginals, hence the optional masked run, which doubles as a 217-assay implementation check. |
| Run on the user's GPU | The development container has no GPU and no access to huggingface.co (models and DMS data). |
| **Published finding** (ProteinGym summary, commit `144fe22…`, Spearman): single-sequence language models are on par with alignment-based models for human (ESM-2 650M 0.457, GEMME 0.469), other eukaryotes (0.486 / 0.516) and prokaryotes (0.458 / 0.467), but **fall on viruses**: ESM-2 650M 0.261, ESM-1v 0.294, ESM-C 600M 0.241, ESM-2 15B 0.333, vs GEMME 0.472, TranceptEVE L 0.461, VenusREM 0.492 | Viral proteins are sparse and fast-evolving in the training sets of single-sequence models. Expectation for omnimissense: animals, plants, fungi and bacteria fine; viruses weak unless VESM differs (to be measured). |
| **Result, VESM 650M wild-type marginal, all 217 assays** (run by the user on the RTX 3070, 2026-11-26) | Overall (by function) **0.457** (ESM-2 650M published 0.414; VESM − ESM-2 +0.043, 95% CI +0.024 to +0.063). Human 0.474 (+0.017, +0.001 to +0.034); other eukaryotes 0.517 (+0.031, −0.022 to +0.090); prokaryotes 0.481 (+0.023, −0.005 to +0.057); **viruses 0.426** (+0.165, +0.100 to +0.231). Same assays, published: GEMME 0.455 / 0.469 / 0.516 / 0.467 / 0.472; TranceptEVE L 0.456 / … / 0.461; VenusREM 0.518 / 0.529 / 0.582 / 0.549 / 0.492. Leaderboard position: overall 17th of 98 (human 14th, other eukaryotes 26th, prokaryotes 25th, viruses 21st); best published single-sequence model VespaG 0.458 (viruses: VESPA 0.443). |
| Interpretation | For cellular organisms the model does not degrade away from human: eukaryotic and prokaryotic proteins score at least as well as human ones, on par with alignment-based methods. Viruses remain the weakest taxon but VESM recovers most of ESM-2's deficit. |
| ESM-2 650M wild-type marginal, all 217 assays (user's GPU; per-assay files pushed by the user, `benchmarks/results/proteingym_all_{vesm,esm2}_650m*.tsv`) | Overall 0.418, human 0.458, other eukaryotes 0.519, prokaryotes 0.464, viruses 0.260; vs published ESM-2 masked +0.008 overall (−0.002 to +0.018): scoring mode makes little difference for ESM-2, and our implementation agrees with ProteinGym's. **VESM − ESM-2, same scoring, paired by protein:** all +0.036 (+0.021 to +0.052), human +0.017 (+0.003 to +0.030), other eukaryotes −0.001 (−0.029 to +0.025), prokaryotes +0.017 (−0.004 to +0.040), **viruses +0.167 (+0.103 to +0.230)**, VESM better on 25 of 31 viral assays. The virus gain is the model's. Wild-type marginals for all 217 assays: about 15 s of GPU time per model. |
| Caveats | (1) Resolved: see the ESM-2 wild-type row. (2) Checked later (see "VESM training data" below): no ProteinGym or other experimental labels in VESM's training or checkpoint selection. (3) DMS fitness is a proxy for "damaging", not for disease. |

---

## 2026-11-26 — Genome-coordinate export (`export-variants`)

| Decision | Rationale |
|---|---|
| New command `export-variants`: every possible missense SNV of the canonical (default) or all coding transcripts, with the wild-type-marginal score, at genome coordinates | The published dog dataset is protein-level (Parquet): using it on a VCF needs omnimissense and csq. A genome-coordinate file, like AlphaMissense's `hg38` TSV, lets anyone annotate with bcftools alone, without a GPU. |
| Two formats, by file name: `.vcf.gz` with `annotate`'s INFO tags (`OMS` per transcript, `OMS_WT` = lowest) and `.tsv.gz` (one row per SNV and transcript); BGZF + tabix (`.tbi`, or CSI when a sequence exceeds 2^29 bp) | The VCF makes `bcftools annotate -a … -c INFO/OMS,INFO/OMS_WT` (and VEP `--custom`) give exactly `annotate`'s tags; the TSV is the convenient table for R/pandas. |
| SNVs computed per transcript with numpy: CDS bases mapped to genome positions (strand, first phase), each of the 3 alternative bases of every codon translated with the transcript's genetic code; kept: missense only, the codon read as written (not the initiation codon turned into M), not the start codon at a complete 5' end (csq: start lost), not changes to a stop codon (csq: stop gained) | Matches csq's calls, verified rather than assumed: on every SNV of the synthetic genome (both strands, intron-split codons, 5'-incomplete CDS, table-2 mitochondrial gene) and of 300 random transcripts in yeast (955,675 rows) and *E. coli* (624,682 rows), the export equals `annotate`'s scored consequences exactly; `bcftools annotate` with the exported VCF reproduces `OMS` and `OMS_WT` exactly (yeast 945,891 SNVs, *E. coli* 618,109). |
| **Bug found and fixed** by the synthetic check: csq's `--dump-gff` trims a 5' partial codon and resets the first CDS phase to 0 | Since the dump-based reader (earlier today), a 5'-incomplete transcript looked complete: its first codon could be read as an initiation codon (CTG/TTG → M) and, in the export, excluded as a start. The original first-CDS phase is now read from the GFF3 (`Transcript.incomplete_5prime`). The dog check (43,524 / 43,524 identical to Ensembl) did not show it: no such transcript there starts with a non-ATG start codon. |
| Documented: `bcftools annotate -a <export>.vcf.gz` needs an indexed input VCF | Checked: on a bgzipped but unindexed input it fails ("could not load index"). The README uses `bcftools index -t` (any bcftools version) rather than `norm -W` (bcftools ≥ 1.17), with options before the file name (BSD getopt on macOS stops at the first non-option). |
| Documented: split multi-allelic records (`bcftools norm -m -any`) before `bcftools annotate` | Checked: on multi-allelic records bcftools annotate takes one matching annotation record, so only 367,680 of 945,891 alleles got a score; after the split, 945,891 / 945,891, identical to `annotate`. |
| Performance (4 CPU cores, cached scores) | *E. coli*: 8.6 M SNVs, 49 s, 96 MB VCF / 68 MB TSV. Yeast, all transcripts: 19.3 M SNVs, 118 s / 88 s, 216 MB / 153 MB. |
| Dog (user's machine, from the VESM 650M cache of `precompute`, no GPU work): ROS_Cfam_1.0, Ensembl 116, canonical transcripts | 20,566 transcripts, **73,800,615 missense SNVs** (73,901,519 SNV × transcript rows: a few SNVs fall in two overlapping canonical transcripts), `dog_ros_cfam_1.0.vesm_650m.snvs.vcf.gz` + index. Publishing it next to the Parquet dataset is the user's decision. |

---

## 2026-11-26 — Release 0.3.0 (prepared; not tagged)

| Decision | Rationale |
|---|---|
| 0.3.0, not 1.0.0 (user's decision after discussion) | 1.0 would freeze CLI options, `OMS*` tags, export columns and the cache layout. Not yet: the export formats and the NCBI/prokaryote transcript IDs are new and unused by others; omnimissense depends on bcftools internals; the score's presentation (raw LLR, no percentile or threshold) is still open. Criteria for 1.0: a "Compatibility" section stating what is stable, protection against pysam/bcftools changes, a decision on score presentation, and outside feedback on 0.3.0 without format changes. |
| `pysam>=0.24.1,<0.25` | pysam 0.24.1 (the latest, 2026-09-07) bundles bcftools 1.24: the genetic-code tables are copied from its `csq.c`, the `--dump-gff` format and the empty-VCF crash workaround were checked on it. A future pysam with another bcftools could change any of them silently for users (the CI tests would catch it, users would not). The bound moves after re-running the tests on the new version. |
| Release checks | `uv lock` changes only the version and the pysam specifier (no dependency upgrades). `uv build`: wheel and sdist contain all modules (including `genetic_code.py`, `variant_export.py`), README and LICENSE; `twine check` passes for both; the wheel installed in a separate environment reports 0.3.0, `doctor` sees bcftools 1.24 (pysam 0.24.1), and `proteome` on *E. coli* gives the benchmark's numbers (4,297 ok, 3 internal stop, 18 pseudogene). 183 tests pass. |
| Not tagged | Pushing tag `v0.3.0` publishes to PyPI (release workflow, trusted publishing): the user tags after the pull request is merged. |

---

## 2026-11-26 — Score presentation: proteome-wide percentile

| Decision | Rationale |
|---|---|
| Add a **proteome-wide percentile** next to the raw score (`OMS_WT_PCT`, `wt_percentile`, `oms_wt_percentile`): percent of all possible missense SNVs of the canonical transcripts of the genome annotation with a lower score | The raw LLR has no fixed scale (it differs between models and is hard to read). The percentile is a monotone transform: no ranking or AUROC changes, but "among the 0.5% most damaging missense changes possible in this genome" reads the same for every gene, species and model. Reference set = exactly the SNVs `export-variants` writes by default (missense only, codon-level rules of csq). |
| Rejected: a percentile **within each protein** | Dog benchmark (Phase 4b): pathogenic variants sit at a median 34th percentile of their own protein (VESM), 0% in the protein's most damaging 10%, while the raw score separates them from common variants with AUROC 0.977. A per-protein scale discards what separates damaging-in-essential-genes from damaging-anywhere. |
| Storage: a quantile table (10,901 levels: 1e-5 steps below 1%, 1e-4 above) in `<cache>/v1/<model>/reference/<key>.npz`; key = SHA-256 of the GFF3 bytes, the genome's sequence names and lengths, the genetic codes and a reference version | Small (≈ 175 kB) and exact enough (tests: within 0.002 percentage points of the exact rank below the 1st percentile, 0.02 elsewhere). A different annotation, genome or code never reuses the wrong reference. |
| Built by `precompute` (after scoring) and by `export-variants` if missing; `annotate` only reads it, and leaves the fields empty with a warning when it is absent | Building needs every canonical protein's scores; `annotate` normally scores only the proteins of the VCF. |
| Formats: new INFO tag and new **last** TSV column; existing fields, columns and the `OMS` entry layout unchanged | Additive change, safe for existing parsers (pre-1.0 rule: add, do not change). |
| Check on yeast R64 (tiny model, cached scores) | Reference from 19.4 M SNVs in 31 s, peak RSS 1.3 GB (with the model). On 955,675 random coding SNVs: median percentile 50.0 and 1.00% below the 1st percentile, as expected for random SNVs; `bcftools annotate` with the exported VCF reproduces `OMS_WT` and `OMS_WT_PCT` exactly (945,891 SNVs). Dog (73.8 M SNVs) expected ≈ 2 min. |
| Code: the SNV enumeration moved from `variant_export.py` to `snvs.py` | Shared by the export and the reference without an import cycle. |

---

## 2026-11-26 — Selenocysteine; csq state between calls

| Decision / finding | Rationale |
|---|---|
| Read `transl_except` (NCBI) on CDS features: `aa:Sec` → `U`, `aa:Pyl` → `O` at the codon's position; other values (e.g. `TERM`) ignored | Checked first how csq handles a selenoprotein (*E. coli* fdhF, Sec at residue 140): it ignores `transl_except`, calls a change of the Sec codon `stop_lost` (`140*>140L`), and calls the variants after it as ordinary missense with continued numbering (`456T>456I`). So reading the codon as `U` makes the whole protein scorable and consistent with csq; the `U` residue itself is never scored (not one of the 20 amino acids; csq does not call it missense). ESM has `U` and `O` tokens. |
| Results | *E. coli*: 4,300 / 4,300 proteins identical to NCBI's translated CDS (fdhF, fdnG, fdoG now included). Every SNV of the 3 selenoproteins: 18,107 missense consequences, all `ok`, identical between `annotate` and `export-variants`, none at the Sec residue. Yeast unchanged (6,021 / 6,021). The dog's selenoprotein stays skipped: the Ensembl GFF3 has no `transl_except` (support for Ensembl's own representation, if any, not checked). |
| **Bug found and fixed:** bcftools csq keeps state between calls in one Python process | Found by a test: after a csq call whose last record was at position 16, a call with a record at position 10 on the same sequence failed with "The file is not sorted, 1:16 comes before 1:10", even after an intermediate call on an empty VCF. Same family as the genetic-code table that survived between calls. Real-world risk: `annotate` runs csq once per genetic code, so a mitochondrion listed before other contigs (with variants on those contigs) could fail; repeated API calls (benchmarks) too. Each csq call now runs in a fresh `python -c` process (≈ 45 ms; the test suite goes from 2 s to 6 s). Regression test added. E. coli export-vs-annotate check rerun after the change: identical. |

---

## 2026-11-26 — GTF input

| Decision | Rationale |
|---|---|
| GTF converted to GFF3 inside `prepare_annotation` (`gtf_to_gff3`), detected from the attributes (`key "value";`) | AUGUSTUS/BRAKER, the usual annotators of new non-model genomes, write GTF, and Ensembl/GENCODE/NCBI also publish GTF; csq reads only GFF3. The converted file then goes through the same path (orphan CDS, codes, csq, dump, metadata). |
| Genes and transcripts are created from `gene_id`/`transcript_id` (spans from their features) when the GTF has no `gene`/`transcript` lines; IDs are the GTF's | AUGUSTUS GTF often has only exon/CDS lines. |
| `stop_codon` features merged into the CDS (adjacent or overlapping intervals joined), phases recomputed from the 5'-most CDS phase | GTF 2.2 CDS exclude the stop codon; GFF3 CDS include it. Merging makes GTF and GFF3 of the same genes identical (checked on the synthetic genome: same CDS segments, proteins and `annotate` results on both strands). A stop codon split over two exons stays a separate segment with its phase. |
| Kept: gene name (`gene_name`, else `gene`), biotypes (`gene_biotype`/`gene_type`, `transcript_biotype`/`transcript_type`; NCBI's `mRNA` → `protein_coding`), transcript `tag`s (canonical), CDS `transl_table`, `transl_except`, `pseudo`, `protein_id` | Everything downstream reads. |
| Real GTF: yeast R64 NCBI `GCF_000146045.2_R64_genomic.gtf.gz` (downloaded by the user): CDS without stop, separate `stop_codon`, `transl_table "3"` on the 50 mitochondrial CDS, `pseudo` on 6, `unassigned_transcript_…` IDs for the mitochondrial genes | **6,021 / 6,021 proteins identical** to NCBI's translated CDS (19 mitochondrial with table 3 from the GTF), 6 pseudogenes, 0 unmatched — the same as the GFF3. `annotate` on the 1,519 random coding SNVs: identical positions, amino acids, statuses and scores with GTF and GFF3 (1,157 `ok`, 2 `pseudogene`). `locus_tag` added to the CDS attributes kept from a GTF (the NCBI check matches pseudo CDS by it). |

---

## 2026-11-26 — Multi-residue changes (MNVs)

| Decision | Rationale |
|---|---|
| Score an equal-length multi-residue missense change as the **sum of its single substitutions** (wild-type and masked marginals), skipping unchanged residues; every reference residue must match the protein | Checked how csq reports MNVs: one within a codon is an ordinary single change (`80TG>AC` → `4W>4T`, already scored); one over two codons is `82GC>TA` → `4WL>4CM`, previously `multi_aa_change` and unscored. The additive score is ProteinGym's convention for multiple mutants with single-sequence models, the one `benchmarks/proteingym_all.py` uses (1.77 M multiple mutants in the 217 assays). |
| `multi_aa_change` now only for changes of protein length; in-frame indels stay unscored | csq calls in-frame indels `inframe_deletion`/`inframe_insertion`, not missense, so they never reach scoring; comparing sequences of different lengths is a different problem. |
| The percentile of an MNV is the summed score placed on the single-SNV reference | Kept for ranking convenience; documented as an approximation (the reference holds single SNVs). |
| Test: the synthetic MNV `4WL>4CM` gives exactly W4C + L5M (wild-type and masked) | |

---

## 2026-11-26 — VESM training data vs ProteinGym (leakage check)

| Finding | Source |
|---|---|
| VESM is trained on **UniProtKB sequences** (human `UPh` and non-human `UPnh` sets) with **targets produced by the ESM models themselves**: the minimum/aggregated log-likelihood ratios of a pool of 11 ESM models (round 1 `esmin`, then rounds on human and non-human data), and finally distillation from VESM_3B's LLRs. No DMS, ClinVar, AlphaMissense or other experimental labels appear in the training code or configs. | `ntranoslab/vesm` repository, commit `86d8bc7` (2025-10-15): `README.md`, `train.py`, `configs/*.json`, `data/data_generation.ipynb` (reads `UniProtKB_meta_data.pkl`, writes `ESM11_UniProt_min.pkl`) |
| **Checkpoint selection** uses an unsupervised proxy: Spearman correlation with the ESM-ensemble average (`esm11_avg`) on held-out human UniProt proteins (`ESM11_UPh_base_val.pkl`), minus 0.1 × the correlation with the base model. ProteinGym and ClinVar appear only in `inference.py` (evaluation). | `modelling/EsmCoDistiller.py` (`eval_epoch`), `train.py` |
| Not verifiable here: whether design choices (number of rounds, aggregation, thresholds, hyperparameters) were tuned by looking at benchmark results; the paper (Nature Methods 2026, doi:10.1038/s41592-026-03050-9) is not reachable from the container. ProteinGym proteins are surely among the UniProtKB training sequences, without labels — as for ESM-2 (UniRef) and every protein language model. | |
| Conclusion | Our ProteinGym-by-taxon numbers are not inflated by label leakage, as far as the released code shows. The dog benchmarks (OMIA, Dog10K) remain the fully independent evidence. |

---

## 2026-11-26 — Cache lookups and the dog percentile reference

| Finding / decision | Rationale |
|---|---|
| Dog `precompute` with every protein cached, building only the percentile reference, took **9 min 20 s** (user's machine); expected ≈ 2 min from yeast (19.4 M values in 31 s). The reference held 73,901,519 values = exactly the export's rows (SNV × transcript), as designed (an SNV in two overlapping canonical transcripts counts once per transcript); the docs now say so. | |
| Cause, measured: every `get_wt` call read about half of **every** cache file, whatever the number of proteins asked for: 224 ms for one protein on a 32-file cache (~7 ms per file), and the reference read the cache once per sequence (dog: hundreds of scaffolds × ~170 files). Row-group min/max pruning (Phase 2) cannot skip anything: hashes are random, so a row group of 128 sorted hashes spans almost the whole range. | The Phase 2 RAM fix (5.8 → 1.4 GB) came from reading row groups instead of whole files, not from skipping them. |
| Fix: an in-memory index `seq_hash → {(file, row group)}`, built from the `seq_hash` column only (0.03 s on yeast) and extended with new files; lookups read only the row groups that hold the requested hashes. Plus batching: the reference and `export-variants` read scores per batch of sequences (≤ 2 M residues) instead of per sequence. | Yeast cache (32 files): one protein 224 → 16 ms, 2,000 proteins 3.69 → 0.63 s, identical values; reference build 31 → 9 s. **Dog (user's machine): reference 9 min 20 s → 50 s** (same 73,901,519 values; `precompute` with nothing to score 73 s). |
| Dog, other timings with the 0.3.0+ code (user's machine): `proteome` 21.6 s (0.2.0: ~10 s; csq now parses the GFF3 too, plus the extra scan) | |
| Dog re-timed with the current code (user's machine, RTX 3070): `annotate` on the chr38 demo (127,270 records) 13.5 s with cached proteins (0.2.0: 12 s), 1,689 missense consequences all `ok` as before, percentiles present (e.g. LLR −0.73 → 99.8th percentile: common Ensembl variants are mostly benign); `proteome` 22 s. README performance table updated. | |

---

## 2026-11-26 — Release 0.4.0 (prepared; not tagged)

| Decision | Rationale |
|---|---|
| 0.4.0: minor version | New features (percentiles, selenocysteine, GTF, MNV scoring) and fixes; additive output changes only (a new `INFO` tag and new last TSV columns), as the Compatibility section requires; cache format unchanged (no `v1` bump: the index is built in memory). |
| Changelog tells upgraders to run `precompute` once | Percentiles need the reference; with scores already cached it takes about a minute for the dog. |
| Release checks | `uv lock` changes only the version. `uv build`: 19 modules including `reference.py` and `snvs.py`; `twine check` passes for wheel and sdist; the wheel installed in a separate environment reports 0.4.0, `doctor` sees bcftools 1.24 (pysam 0.24.1), `proteome` gives *E. coli* 4,300 ok + 18 pseudogene (selenoproteins included) and yeast from the NCBI GTF 6,021 ok + 6 pseudogene, the benchmark numbers; this also exercises the csq-in-a-subprocess path from an installed package. 204 tests pass. |
| Not tagged | Pushing `v0.4.0` publishes to PyPI: the user tags after the pull request is merged. |

---

## 2026-11-26 — Chicken benchmark: far from human (design)

| Decision | Rationale |
|---|---|
| Species: chicken (*Gallus gallus*) | The OMIA species farthest from human with a usable number of missense variants and an Ensembl assembly (GRCg7b, Ensembl 116: 14,069 gene symbols, 14,048 canonical proteins translate). Chicken–human: > 300 million years; dog–human: ≈ 90. Same design as the dog benchmark, so the two read side by side. |
| Common variants: EVA PRJEB78357, 96 Bangladeshi native chickens (3 varieties × improved/unimproved × 16), GRCg7b, autosomes, PASS SNVs; ALT frequency ≥ 5% from INFO/AC and AN (checked against the genotypes: 139 of 140 sampled sites agree within 0.01) | Same assembly as Ensembl 116 (no liftover). Village chickens of several varieties, closer to Dog10K's mix than a single breed. Alternatives checked: Ensembl 116 chicken VCF (no frequencies: EVA/dbSNP sites only), EVA PRJEB55239 (358 birds of one breed, Wenchang), PRJEB71347 (209 Chinese local chickens, 36 GB, unfiltered), GCRP (GRCg6a). EVA's accessioned VCFs are sites-only; the submitted 12 GB file has the INFO counts. |
| Remote reads of the 12 GB VCF, sequence by sequence, with a local copy of the index (EVA names it `.vcf.csi`) | Only the coding sequence of ~600 genes is needed. From this container the egress proxy drops long range reads to ftp.ebi.ac.uk, so the extraction was tested on a local copy of the first 400 MB (chromosome 1 up to 30.4 Mb); on the user's machine the remote read is expected to work as for Dog10K. `--population-vcf` takes a local file. |
| Human-ortholog comparison inside the same script (model on the human ortholog; AlphaMissense) | The question for a distant species is whether the human workaround still works: fewer genes map, and the aligned residue differs more often. The AUROC on all variants counts unmapped variants as least damaging, which favours the human methods (causal variants map more often than common ones; dog: 96% vs 41%); the subset scored by every method is the fair comparison. |
| Dry run (tiny random model, mock OMIA and human files, real genome and the local population extract): every step runs; human scores equal the chicken scores when the mock human protein is the chicken protein | Real numbers need VESM (GPU, user's machine) and the real OMIA export (by hand: OMIA's robots.txt). The trait list is filled from that export. |

## 2026-11-26 — Chicken benchmark: what OMIA holds, and mapping through RefSeq

| Finding / decision | Rationale |
|---|---|
| OMIA export of 2026-11-26 (1,834 likely causal variants, all species): **chicken has 23 missense rows, 21 distinct variants, of which 4 are diseases** (analphalipoproteinaemia ABCA1 P; muscular dystrophy WWP1 and restricted ovulator VLDLR LP/P; trimethylaminuria FMO3); the others are plumage colour and pattern (MC1R ×4, EDNRB2 ×3, SLC45A2 ×2, CDKN2A ×2, MLPH, TYRP1), virus resistance (CD320, TNFRSF10B, MX1) and a blood group (MCAM). All other non-mammals together: about 50 missense variants over some 40 species (Japanese quail 4, rock pigeon 4, ball python 3, parrots, fish, frogs), mostly pigmentation. | Chicken is the only distant species with enough variants, and even there a disease-only AUROC is meaningless (n = 4). Main analysis: all 21 causal variants vs common variants; in the dog benchmark, OMIA trait variants scored like disease variants against common ones (AUROC 0.95–0.97), so "causal" is a fair positive class for a score of protein disruption. Caveat: some MC1R alleles (extended black) are gain of function. |
| Map through the cited RefSeq protein, not by position | Matching OMIA's position on any Ensembl protein of the gene failed for 10 of 21 variants (and 3 matched only a non-canonical isoform): Ensembl's GRCg7b models often start upstream of RefSeq's (EDNRB2 +43, MLPH +67, FMO3 +121, CDKN2A +29 residues) or differ (CD320, TNFRSF10B). Aligning the RefSeq protein (fetched from NCBI by accession) to the Ensembl proteins maps all 21 (canonical transcript except where noted), each at a residue identical to OMIA's reference or alternative one. |
| Restore the reference residue when GRCg7b carries the variant allele (4 variants: MC1R M71T and E92K, SLC45A2 L347M, CDKN2A R10C; RefSeq itself carries the variant for three of them, as OMIA notes) | The broiler reference genome is a domestic bird carrying trait alleles. Scoring the variant needs the ancestral residue in the sequence; the restored protein is scored like any other. Population variants at the same site are the reverse change and are removed from the negatives. |
| Z chromosome | 5 of the 14 causal genes are Z-linked; the population VCF is autosomal, so they have no same-gene negatives (random-gene and all-common negatives are unaffected). |
| Dry run on the real OMIA export (tiny random model, chromosome 1 population extract): mapping 21/21 distinct, every step runs | Real numbers need VESM on the user's GPU. |
| Remote reads of the EVA VCF: one request per merged gene span (gaps < 20 kb), up to 4 sequences at a time, then a local filter to the CDS regions | The first version passed every CDS exon to `bcftools view -R`: about 5,900 separate range requests to EBI, which ran for over 10 minutes on chromosome 1 alone (810 exons) on the user's machine. Spans: 568 requests, ~9.3 Mb (~100 MB of the 12 GB file); identical SNVs on the local chromosome 1 extract (281 = 281). |

---

## 2026-11-26 — Public Python API

| Decision | Rationale |
|---|---|
| A small public module (`omnimissense.api`, re-exported by the package): `Model`, `ProteinScores`, `annotate`, `precompute`, `AnnotatedVariant(s)`, `VariantStatus`, `PrecomputeResult`, `ReferenceMismatchError`, `AMINO_ACIDS`; declared stable in *Compatibility*. Everything else stays internal. | Before, using omnimissense from code meant internal plumbing (a `ScoreCache`, a scorer factory, the model name three times, mandatory output files) and unlabelled `L × 20` matrices. The CLI is unchanged. |
| `AnnotatedVariant` fields = the TSV columns, same names and order, plus `protein_change` and `record_index`; internal details (`seq_hash`, `alt_index`, the nested csq `Consequence`) are not exposed | One vocabulary for the command line and the code; the TSV is already a stability promise. |
| `Model` loads the weights lazily (first score not in the cache) | A fully cached genome needs no GPU, as with the CLI; creating a `Model` is free. |
| Tables: `to_arrow()` always, `to_pandas()` with the optional extra `omnimissense[pandas]` (imported on call, clear error otherwise) | pyarrow is already a dependency (the cache); pandas is not installed today, even indirectly, and could clash with a user's pinned version. |
| `import omnimissense` imports the API on first attribute access (PEP 562) | The CLI keeps starting without torch (checked by a test). |
| `pipeline.annotate` takes `output_vcf` / `output_tsv` as optional | The API returns objects; files are written only when asked. The annotated VCF needs csq's temporary output, so it is written during the call (`output_vcf=`), not afterwards. |

## 2026-11-26 — Chicken benchmark: results (VESM 650M, user's machine)

| Finding | Numbers |
|---|---|
| Chicken proteome `precompute` (RTX 3070, VESM 650M fp16) | 44,876 coding transcripts, 42,089 unique proteins, 27.6 M residues in 49 min (9,384 residues/s); percentile reference 65,423,227 values (16,986 canonical transcripts) |
| Benchmark run | 21 causal variants mapped (4 with the ancestral residue restored); common: 54 in the 14 causal genes + 1,763 in 499 random genes; 5 common variants at causal sites removed |
| Main AUROC, all causal vs all common | VESM chicken wild-type 0.922 (0.868–0.966); VESM human ortholog 0.830; AlphaMissense 0.821 (0.715–0.919); VESM masked 0.793; BLOSUM62 0.564. Paired: chicken − AlphaMissense +0.101 (+0.005 to +0.204); chicken − human protein +0.092 (−0.006 to +0.198) |
| Scored by every method (16 / 416) | VESM chicken 0.888, human 0.870, AlphaMissense 0.815; differences not significant |
| By set | Traits (17): chicken 0.916 vs AlphaMissense 0.781 (+0.127 on the 12 both score, +0.045 to +0.227). Diseases (4): chicken 0.949 vs AlphaMissense 0.995 (−0.046, −0.086 to −0.006): too few to generalise |
| Human mappability | causal 76% (EDNRB2 ×3 has no mammalian ortholog; chicken CDKN2A = ARF only, UniProt's human CDKN2A is p16), common 23% (dog: 96% / 41%) |
| Percentiles | proteome-wide median: causal 35.6, common 82.0; below 5th: causal 9.5%, common 0%. In-protein median: causal 0.40, common 0.89 |
| Masked marginal | worse than wild-type (0.79 vs 0.92, pathogenic 3: 0.51 with a 0.02–1.00 interval); not investigated per variant (the masked mode is not the default); dog showed the same direction (0.948 vs 0.977) |

---

## 2026-11-26 — Release 0.5.0 (prepared; not tagged)

| Decision | Rationale |
|---|---|
| `--annotation` replaces `--gff3` (GFF3 or GTF), `--gff3` kept as an alias; `export-proteins` replaces `export`, kept as a hidden alias | Names reviewed before the 1.0 freeze: `--gff3` was misleading since GTF support (0.4.0) and `export` was easy to confuse with `export-variants`. Old names keep working, as the Compatibility rules require. The Python API (new, unreleased until now) uses `annotation=` from the start. Other options unchanged: the opposite defaults of `export-proteins` (all transcripts) and `export-variants` (canonical only) follow the output sizes. |
| Masked-marginal anomaly in the chicken benchmark not investigated before the release | `masked_marginals` re-read: window weights sum to 1, LLR against the wild-type residue; nothing wrong found. VESM is trained on unmasked LLR targets and its authors score with the wild-type marginal; the default is unchanged. To check per variant before 1.0. |
| Percentile threshold tables in the benchmarks (`*.percentiles.tsv`): share of causal and common variants below the 1st, 5th, 10th, 20th, 30th and 50th proteome-wide percentile | A guide to reading `OMS_WT_PCT`; the dog needs `precompute` on the GSD assembly (the Dog10K one), the chicken uses the reference already built. |
| Release checks | `uv lock` changes only the version; `uv build`: 20 modules including `api.py`; `twine check` passes for wheel and sdist; metadata has the `pandas` extra. In a fresh environment (`omnimissense[pandas]`, Python 3.13): `--version` 0.5.0, `doctor` sees bcftools 1.24; `proteome` with the old `--gff3` on *E. coli* gives 4,300 ok + 18 pseudogene, with `--annotation` on the yeast GTF 6,021 ok + 6 pseudogene; `export` is hidden from the help; the Python API (tiny model) scores a protein, runs `precompute` and `annotate` with percentiles and `to_pandas()`. 235 tests pass. |
| Not tagged | Pushing `v0.5.0` publishes to PyPI: the user tags after the pull request is merged. |
