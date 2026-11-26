# Changelog

## 0.3.0 — 2026-11-26

Any genome: NCBI and prokaryotic annotations, any genetic code, mitochondria, and a
genome-coordinate score file for annotating without a GPU.

**Annotations**

- **GFF3 from NCBI RefSeq**, not only Ensembl: transcripts are read from bcftools csq's own
  parse of the annotation (`--dump-gff`), so csq and the protein translation always see the
  same transcripts. Transcript IDs are csq's: Ensembl unchanged, NCBI e.g. `rna-XM_…`.
- **Prokaryotic GFF3** (NCBI, Prokka, Bakta) and NCBI vertebrate mitochondria: CDS without a
  transcript feature get one (`tx-<CDS ID>`, plus `gene-<CDS ID>` when the CDS has no parent)
  in a normalized copy of the GFF3; an embedded `##FASTA` section, which made csq fail, is
  dropped.
- Canonical transcripts: `Ensembl_canonical`, `MANE Select` or `RefSeq Select` tags; genes
  without any tagged transcript get their longest CDS.
- Clearer error for GFF3 layouts csq cannot read, with the converters to use.

**Genetic codes**

- `--genetic-code` and `--mito-genetic-code` (NCBI tables 1–6, 9–16, 21–31, 33, as in
  bcftools csq); by default each sequence's `transl_table` from the GFF3 (NCBI), else the
  standard code. bcftools csq is run once per code.
- Mitochondrial genes are scored when their code is known (previously always skipped);
  mitochondria are also recognised from NCBI `region` features (`genome=mitochondrion`).

**New command**

- `omnimissense export-variants`: every possible missense SNV of the coding transcripts at
  genome coordinates with its score, as a tabix-indexed VCF (`OMS`/`OMS_WT`, for
  `bcftools annotate` or VEP `--custom`) or TSV; equal to `annotate` on every SNV tested.

**Changes in behaviour**

- New status `pseudogene`: CDS flagged `pseudo=true` (NCBI) are neither translated nor scored
  (previously, interrupted pseudogenes that happened to translate were scored as proteins).
- `export` metadata records the genetic codes used.
- Dependency: `pysam>=0.24.1,<0.25` (bcftools 1.24), since omnimissense relies on csq's
  genetic-code tables and `--dump-gff` format.

**Fixes**

- A CDS starting inside a codon (non-zero first phase) lost that information in csq's dump;
  it is now taken from the GFF3, so such a first codon is never read as an initiation codon.
- CLI help shows the `[default: …]` notes (they were swallowed as rich markup).
- `uv sync` without `--all-groups` no longer breaks test collection.

**Validation and benchmarks**

- Proteomes identical to the reference translations: dog (Ensembl) 43,524 / 43,524, yeast
  (NCBI, mitochondria included) 6,021 / 6,021, *E. coli* (NCBI, prokaryote) 4,297 / 4,297.
- Genetic-code tables checked against Biopython and, for all 26 tables, against csq.
- `benchmarks/proteingym_all.py`: all 217 ProteinGym assays by taxon. VESM 650M: overall
  0.457 (ESM-2 650M 0.418), human 0.474, other eukaryotes 0.517, prokaryotes 0.481,
  viruses 0.426 (ESM-2 0.260).
- `benchmarks/ncbi_translation.py`: proteome check against NCBI's translated CDS.

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
