---
license: cc-by-4.0
pretty_name: Dog proteome missense variant effect scores (VESM 650M, ROS_Cfam_1.0)
tags:
  - biology
  - genomics
  - dog
  - canis-lupus-familiaris
  - missense
  - variant-effect-prediction
  - protein-language-model
  - VESM
  - ESM
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files: dog_ros_cfam_1.0_vesm_650m.parquet
---

# Dog proteome missense variant effect scores (VESM 650M, ROS_Cfam_1.0)

Precomputed scores for **every possible amino-acid substitution in every protein-coding
transcript of the dog** (*Canis lupus familiaris*, Ensembl reference assembly ROS_Cfam_1.0),
computed with [omnimissense](https://github.com/dsmear/omnimissense) and the VESM 650M protein
language model.

Each score is a log-likelihood ratio `log P(alt) − log P(ref)` at the substituted position
(wild-type marginal): **more negative = the substitution looks more disruptive to the
protein**. It is **not** a probability of disease (see *Limitations*).

## Contents

| | |
|---|---|
| File | `dog_ros_cfam_1.0_vesm_650m.parquet` (1.55 GB, Parquet, zstd) |
| Protein-coding transcripts | 43,524 (20,566 canonical) |
| Genes | 20,566 |
| Distinct protein sequences | 43,092 (26,655,640 residues; 506 M possible substitutions) |
| Proteins longer than 1022 residues (scored with sliding windows) | 6,435 transcripts; longest 27,097 residues |

## Provenance

| Item | Value |
|---|---|
| Genome | Ensembl release 116, `Canis_lupus_familiaris.ROS_Cfam_1.0.dna.toplevel.fa.gz` (BSD `sum` 14738 709328) |
| Annotation | Ensembl release 116, `Canis_lupus_familiaris.ROS_Cfam_1.0.116.gff3.gz` (BSD `sum` 62888 16627) |
| Assembly | ROS_Cfam_1.0, GenBank GCA_014441545.1 (genebuild 2020-10) |
| Model | VESM 650M, `ntranoslab/vesm` file `VESM_650M.pth`, revision `946749efe0c5bda4693884092df3a73c542752f3`, applied on top of ESM-2 650M (`facebook/esm2_t33_650M_UR50D`, served as `biohub/esm2_t33_650M_UR50D`, revision `08e4846e537177426273712802403f7ba8261b6c`) |
| Scoring | wild-type marginal (one forward pass per protein), fp16 on GPU; proteins longer than 1022 residues: 1022-residue windows overlapping by 511, sigmoid-weighted average (Brandes et al. 2023) |
| Software | omnimissense 0.2.0, command `omnimissense precompute` then `omnimissense export` |

Proteins are translated from the genome and the annotation by omnimissense; the translations
are identical to Ensembl's published peptides for 43,524 / 43,524 transcripts (one
selenoprotein, which has an internal TGA read as selenocysteine, is excluded). Mitochondrial
transcripts are not included.

## Schema

One row per protein-coding transcript.

| Column | Type | Description |
|---|---|---|
| `transcript_id` | string | Ensembl transcript ID (without version) |
| `gene_id`, `gene_name` | string | Ensembl gene ID and symbol (symbol may be empty) |
| `canonical` | bool | Ensembl canonical transcript of the gene |
| `biotype`, `seqid`, `strand` | string | transcript biotype, chromosome, strand |
| `protein_length` | int32 | number of residues (stop codon excluded) |
| `sequence` | string | protein sequence |
| `sequence_sha256` | string | SHA-256 of the sequence (identical isoforms share it) |
| `n_windows` | int16 | number of 1022-residue windows used (1 for most proteins) |
| `llr` | list<float32> | flattened `protein_length × 20` matrix, row-major; row *i* = residue *i* (0-based), columns in the order `ACDEFGHIKLMNPQRSTVWY`; 0 for the reference residue |

The file-level Parquet metadata key `omnimissense` holds the same provenance as JSON.

## Usage

```python
import numpy as np
import pyarrow.parquet as pq

AA = "ACDEFGHIKLMNPQRSTVWY"
row = pq.read_table(
    "dog_ros_cfam_1.0_vesm_650m.parquet",
    filters=[("transcript_id", "=", "ENSCAFT00845040105")],  # SOD1, canonical
).to_pylist()[0]
llr = np.array(row["llr"], dtype=np.float32).reshape(row["protein_length"], 20)

position, alt = 213, "K"  # 1-based position in THIS protein sequence
assert row["sequence"][position - 1] == "E"  # always check the reference residue
print(llr[position - 1, AA.index(alt)])  # about -8.43
```

**Protein numbering.** Positions refer to the Ensembl protein sequence stored in the file,
which can differ from the protein used in the literature. Example: the canonical SOD1
transcript of Ensembl 116 (ENSCAFT00845040105) carries a 173-residue N-terminal extension in
front of the SOD1 sequence, so the degenerative-myelopathy variant SOD1 p.E40K is residue 213
here (E213K, LLR −8.43). Always check the reference residue, or match on sequence context.

To annotate a VCF directly, use omnimissense itself (`omnimissense annotate`); it computes
the same scores and reuses its cache.

## Validation (summary)

From the omnimissense benchmarks (details and scripts in the repository):

- **Implementation:** ESM-2 650M scores computed by the same pipeline reproduce the ProteinGym
  published Spearman correlations within 0.001 on six deep mutational scanning assays.
- **Dog disease variants:** OMIA pathogenic/likely pathogenic dog missense variants vs common
  Dog10K dog variants (allele frequency ≥ 5%): AUROC 0.977 (95% CI 0.962–0.991); within the
  same genes 0.961.
- **Realistic ranking:** a known causal variant inserted among the rare missense variants of
  a real Dog10K dog (median 126) ranks in the top 10 in 72% of trials (median rank 4);
  AlphaMissense applied through human orthologs: 75%.

## Limitations

- The score measures how disruptive a substitution is to the protein, **not** whether it
  causes disease: coat-colour and other trait variants score as low as disease variants.
  Do not use the scores alone for breeding or clinical decisions.
- Scores are raw log-likelihood ratios, not calibrated probabilities; values are comparable
  across proteins only approximately.
- Annotation-dependent: transcripts, isoforms and protein numbering follow Ensembl 116;
  variant positions from other annotations (e.g. RefSeq `XP_` proteins) may be numbered
  differently.

## License and citation

Scores: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). Please cite:

- omnimissense — <https://github.com/dsmear/omnimissense>
- VESM — Dinh, Jang, Zaitlen & Ntranos, *VESM: Compressing the collective knowledge of ESM
  into a single protein language model* (model and code MIT)
- ESM-2 — Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with
  a language model*, Science 379, 1123–1130 (2023)
- Ensembl — Ensembl release 116 (genome and annotation)
