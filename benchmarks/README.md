# Benchmarks

Reproducible validation scripts. Downloaded data go to `data/` (never committed); only
aggregate results are written to `benchmarks/results/`, one file per model
(`*_esm2_650m.tsv`, `*_vesm_650m.tsv`). Every script takes `--model`; the default is the
omnimissense default (VESM 650M), except `proteingym.py`, which defaults to ESM-2 650M because
ProteinGym publishes reference values for it.

## 1. ProteinGym: implementation sanity check

```bash
uv run python benchmarks/proteingym.py
```

Scores every single mutant of six deep mutational scanning (DMS) assays and compares the
Spearman correlation with experimental fitness to the value ProteinGym publishes for the same
model (ESM-2 650M, masked marginals). Three assays are short proteins (one window: the
values must agree within 0.03, otherwise the script fails); three are longer than 1022
residues (ProteinGym uses one window centred on each mutation, omnimissense averages
overlapping windows: informational comparison).

Data: [ProteinGym](https://github.com/OATML-Markslab/ProteinGym) (MIT), DMS table from the
official Hugging Face dataset `OATML-Markslab/ProteinGym_v1` at revision `1ea2aa1…`, reference
file and published scores from the GitHub repository at commit `144fe22…` (pinned in the
script). Notin et al., *ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and
Design*, NeurIPS 2023.

## 2. Dog: OMIA causal variants vs common Dog10K variants

**Step 1 — download the OMIA variant table by hand.** OMIA's `robots.txt` disallows automated
access to `/download*`, so the script does not fetch it:

1. open <https://omia.org/results/?search_type=advanced&result_type=variant&singlelocus=yes&characterised=yes&species_id=9615>
2. click the link to download the table as CSV
3. save it as `data/omia/omia_dog_variants.csv`

**Step 2 — run:**

```bash
uv run python benchmarks/dog_omia.py
```

The script downloads the Ensembl 116 German Shepherd assembly (UU_Cfam_GSD_1.0, ~0.7 GB) and
reads Dog10K remotely, region by region (only the coding sequence of the selected genes).

- **Causal sets:** OMIA dog variants whose effect is "missense", mapped to UU_Cfam_GSD_1.0
  proteins by gene symbol and amino-acid change (the reference residue must match; canonical
  transcript tried first; genes missing from the GSD annotation are looked up in
  ROS_Cfam_1.0). `pathogenic` = AVCG classification P, LP or LP/P; `disease` = pathogenic +
  disease variants not yet classified; `trait` = coat colour, hair and pattern phenotypes
  (listed explicitly in the script).
- **Common set:** Dog10K PASS SNVs whose ALT allele frequency is ≥ 5% among 1,872 dogs
  (breed, village and mixed dogs; wolves and coyotes excluded), missense in the canonical
  transcript of (a) the genes of the causal set and (b) 500 autosomal protein-coding genes
  drawn at random (seed 0) minus the causal genes. Variants present in both sets are dropped
  from the common set.
- **Metrics:** AUROC (with 95% stratified bootstrap interval) of the masked- and
  wild-type-marginal scores and of BLOSUM62 (context-free baseline) against all common
  variants, common variants in the same genes, common variants in random genes and — as in
  the 2025 study that used OMIA "non-deleterious" variants as negatives — OMIA trait variants;
  percentile of each causal variant among all 19 × L possible substitutions of its protein
  (wild-type marginal; 0 = most damaging).

**Data sources and terms**

- OMIA — Online Mendelian Inheritance in Animals. Nicholas, F.W., Tammen, I., & Sydney
  Informatics Hub. Online Mendelian Inheritance in Animals (OMIA) [dataset].
  <https://omia.org/>. <https://doi.org/10.25910/2AMR-PV70>. OMIA data were made available by
  software support from the Sydney Informatics Hub, funded from the Ronald Bruce Anstee
  bequest to the Sydney School of Veterinary Science for the Anstee Hub for Inherited Diseases
  in Animals (AHIDA). OMIA data are not redistributed in this repository.
- Dog10K — Meadows et al., *Genome sequencing of 2000 canids by the Dog10K consortium
  advances the understanding of demography, genome function and architecture*, Genome Biology
  24, 187 (2023). Data: <https://zenodo.org/records/8084059> (CC-BY-4.0).
- Ensembl release 116, *Canis lupus familiaris* German Shepherd (UU_Cfam_GSD_1.0,
  GCA_011100685.1).

## 3. Dog, realistic: ranking among rare variants, and the human-ortholog workaround

```bash
uv run --group benchmarks python benchmarks/dog_realistic.py
```

Requires the OMIA CSV (section 2), a run of `dog_omia.py` (for the common-variant set) and
these downloads (about 6.7 GB; MD5 checksums from the Zenodo records):

| File | Destination | Source | License |
|---|---|---|---|
| `AutoAndXPAR.SNPs.bcf` + `.csi` (5.5 GB) | `data/dog10k/` | Dog10K, <https://zenodo.org/records/8084059> | CC-BY-4.0 |
| `AlphaMissense_aa_substitutions.tsv.gz` (1.2 GB) | `data/alphamissense/` | <https://zenodo.org/records/8208688> | CC BY-NC-SA 4.0 |
| `uniprot_human_reviewed.tsv` (12 MB) | `data/human/` | UniProt REST: `https://rest.uniprot.org/uniprotkb/stream?query=organism_id:9606+AND+reviewed:true&format=tsv&fields=accession,gene_primary,gene_synonym,sequence` | CC-BY-4.0 |

- **Ranking:** 50 Dog10K breed dogs (random, seed 0). Each dog's candidates are the rare
  missense variants it carries (PASS SNVs, ALT frequency < 1% among the 1,872 dogs, canonical
  transcripts of autosomal coding genes). Every OMIA causal variant (pathogenic and disease
  sets) is inserted in turn among each dog's candidates and ranked; median rank, top-1, top-5,
  top-10.
- **Methods:** ESM-2 650M on the dog protein (omnimissense, masked and wild-type marginal);
  ESM-2 650M on the human ortholog (masked marginal); AlphaMissense (human); BLOSUM62.
- **Human mapping ("mock variant"):** dog gene symbol → reviewed human UniProt entry (primary
  name, then synonyms); global alignment (BLOSUM62, gap open −10, extend −0.5); the aligned
  human residue must equal the dog reference residue. Proteins longer than 6,000 residues are
  not aligned. Unmapped variants have no human score and are ranked last.
- **AUROC:** causal vs rare and vs common variants, on all variants and on the subset scored by
  every method, with a paired bootstrap interval for the difference between dog ESM-2 and
  AlphaMissense.

AlphaMissense: Cheng et al., *Accurate proteome-wide missense variant effect prediction with
AlphaMissense*, Science 381, eadg7492 (2023). Predictions used only to compute aggregate
metrics; they are not redistributed.
