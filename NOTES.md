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
| CLI help: `[default: …]` notes escaped | Typer's rich markup swallowed them (`--tsv`, `--cache-dir`, `--device` showed no default) — pre-existing. |
| Not modelled: RNA editing (plant organelles), selenocysteine (`transl_except`), programmed frameshifts | Translation is from the genome sequence; such proteins are reported (internal stop) or differ from the mature protein. |
