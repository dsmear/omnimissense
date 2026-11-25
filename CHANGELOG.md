# Changelog

## 0.2.0 — 2026-11-25

- **Default model: VESM 650M** (`ntranoslab/vesm/VESM_650M`, MIT), scored with the wild-type
  marginal; masked marginals are now opt-in (`--masked`). Annotation from a precomputed
  proteome takes seconds.
- VESM support (`ntranoslab/vesm/VESM_<size>`; VESM3 refused, non-commercial license).
- `omnimissense export`: wild-type-marginal scores of every coding transcript to one Parquet
  file with provenance metadata.
- Benchmarks: realistic ranking among the rare variants of 50 Dog10K dogs and comparison with
  the human-ortholog workaround (ESM-2/VESM on human proteins, AlphaMissense).
- Release workflow with PyPI trusted publishing.
- Published dataset: dog proteome scores (ROS_Cfam_1.0, VESM 650M) on Hugging Face,
  `DavideMacchia/omnimissense-dog-ros-cfam-1.0-vesm-650m` (CC-BY-4.0).

## 0.1.0 — 2026-11-25

- First working pipeline: REF check, bcftools csq consequences, Ensembl-style GFF3
  translation, ESM-2 wild-type and masked marginals with sliding windows, Parquet score cache,
  VCF `INFO` and TSV output; `annotate`, `precompute`, `proteome`, `doctor` commands.
- Validation: ProteinGym reproduction and dog OMIA vs Dog10K benchmark.
