# Changelog

## Unreleased

- **GFF3 from NCBI RefSeq** (and any other layout bcftools csq accepts), not only Ensembl:
  transcripts are now read from csq's own parse of the annotation (`--dump-gff`), so csq and
  the protein translation always see the same transcripts. Transcript IDs are csq's (Ensembl:
  unchanged; NCBI: e.g. `rna-XM_…`).
- Canonical transcripts: `Ensembl_canonical`, `MANE Select` or `RefSeq Select` tags; genes
  without any tagged transcript get their longest CDS.
- Mitochondrial sequences are also recognised from NCBI `region` features
  (`genome=mitochondrion`), since NCBI names sequences by accession (e.g. `NC_001224.1`).
- Clearer error for GFF3 layouts csq cannot read (GTF, prokaryotic CDS directly under genes),
  with the converters to use.
- `uv sync` without `--all-groups` no longer breaks test collection (Biopython test skipped).
- **Genetic codes:** `--genetic-code` and `--mito-genetic-code` (NCBI tables 1–6, 9–16,
  21–31, 33, as in bcftools csq); by default each sequence's `transl_table` from the GFF3
  (NCBI), else the standard code for nuclear sequences. Mitochondrial genes are now scored
  when their code is known (previously always skipped). bcftools csq is run once per code.
- CLI help now shows the `[default: …]` notes (they were swallowed as rich markup).
- **Prokaryotic GFF3** (NCBI, Prokka, Bakta) and NCBI vertebrate mitochondria: CDS without
  a transcript feature get one (`tx-<CDS ID>`, plus `gene-<CDS ID>` when the CDS has no
  parent) in a normalized copy of the GFF3, which bcftools csq and the translation read; an
  embedded `##FASTA` section (Prokka, Bakta), which made csq fail, is dropped.
- New status `pseudogene`: CDS flagged `pseudo=true` (NCBI) are neither translated nor scored
  (previously, interrupted pseudogenes that happened to translate were scored as proteins).

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
