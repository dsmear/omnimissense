"""Command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from omnimissense import __version__
from omnimissense.defaults import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from omnimissense.environment import run_checks

app = typer.Typer(
    name="omnimissense",
    help="Score missense variants in any species with protein language models.",
    no_args_is_help=True,
    add_completion=False,
)

ModelOption = Annotated[
    str,
    typer.Option(
        help="ESM-style masked language model: a Hugging Face ID such as "
        "facebook/esm2_t33_650M_UR50D, or ntranoslab/vesm/VESM_<size>."
    ),
]
FastaOption = Annotated[
    Path,
    typer.Option(exists=True, dir_okay=False, help="Reference genome FASTA (plain, gzip or BGZF)."),
]
Gff3Option = Annotated[
    Path,
    typer.Option(
        exists=True, dir_okay=False, help="GFF3 annotation (Ensembl or NCBI RefSeq layout)."
    ),
]


def _check_genetic_code(value: int | None) -> int | None:
    from omnimissense.genetic_code import genetic_code

    if value is not None:
        try:
            genetic_code(value)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
    return value


GeneticCodeOption = Annotated[
    int | None,
    typer.Option(
        "--genetic-code",
        callback=_check_genetic_code,
        help="NCBI genetic code (translation table) of the nuclear genome, e.g. 6 ciliates, "
        "12 Candida, 11 bacteria \\[default: the GFF3's transl_table, else 1 = standard].",
    ),
]
MitoGeneticCodeOption = Annotated[
    int | None,
    typer.Option(
        "--mito-genetic-code",
        callback=_check_genetic_code,
        help="NCBI genetic code of the mitochondrial genome, e.g. 2 vertebrates, 3 yeast, "
        "5 invertebrates \\[default: the GFF3's transl_table; without one, mitochondrial "
        "transcripts are skipped].",
    ),
]
CacheOption = Annotated[
    Path | None,
    typer.Option(
        help="Score cache directory \\[default: $OMNIMISSENSE_CACHE or ~/.cache/omnimissense]."
    ),
]
DeviceOption = Annotated[
    str | None, typer.Option(help="Torch device, e.g. 'cuda' or 'cpu' \\[default: auto].")
]
MaxTokensOption = Annotated[
    int, typer.Option(help="Maximum padded tokens per GPU batch; lower it on out-of-memory errors.")
]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"omnimissense {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Score missense variants in any species with protein language models."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # INFO for our own progress messages only; third-party libraries stay at WARNING.
    logging.getLogger("omnimissense").setLevel(logging.DEBUG if verbose else logging.INFO)


def _scorer_factory(model: str, device: str | None, max_tokens: int):
    def factory():
        from omnimissense.scoring import MaskedLMScorer

        return MaskedLMScorer.from_pretrained(model, device=device, max_tokens=max_tokens)

    return factory


def _cache(cache_dir: Path | None, model: str):
    from omnimissense.cache import ScoreCache, default_cache_dir

    return ScoreCache(cache_dir or default_cache_dir(), model)


def _default_tsv(output: Path) -> Path:
    name = output.name
    for suffix in (".vcf.gz", ".vcf", ".bcf"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    return output.with_name(name + ".tsv")


@app.command()
def annotate(
    vcf: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="Input VCF/BCF.")],
    fasta: FastaOption,
    gff3: Gff3Option,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output VCF (.vcf, .vcf.gz or .bcf).")
    ],
    tsv: Annotated[
        Path | None, typer.Option(help="TSV summary \\[default: next to the output VCF].")
    ] = None,
    model: ModelOption = DEFAULT_MODEL,
    cache_dir: CacheOption = None,
    masked: Annotated[
        bool,
        typer.Option(
            help="Also compute masked-marginal scores (OMS_MM): one forward pass per variant "
            "position, much slower; the default score is the wild-type marginal (OMS_WT)."
        ),
    ] = False,
    device: DeviceOption = None,
    max_tokens: MaxTokensOption = DEFAULT_MAX_TOKENS,
    threads: Annotated[int, typer.Option(help="Extra bcftools threads.")] = 0,
    genetic_code: GeneticCodeOption = None,
    mito_genetic_code: MitoGeneticCodeOption = None,
) -> None:
    """Annotate missense variants of a VCF with protein language model scores."""
    from omnimissense.pipeline import annotate as run_annotate

    tsv = tsv or _default_tsv(output)
    result = run_annotate(
        vcf,
        fasta,
        gff3,
        output,
        tsv,
        cache=_cache(cache_dir, model),
        scorer_factory=_scorer_factory(model, device, max_tokens),
        model_name=model,
        masked=masked,
        threads=threads,
        genetic_code=genetic_code,
        mito_genetic_code=mito_genetic_code,
    )
    typer.echo(f"Records read:            {result.n_records}")
    typer.echo(f"REF allele mismatches:   {result.n_ref_mismatches}")
    typer.echo(f"Missense consequences:   {len(result.variants)}")
    for status, count in sorted(result.status_counts.items()):
        typer.echo(f"  {status:<22} {count}")
    typer.echo(f"VCF: {output}\nTSV: {tsv}")


@app.command()
def precompute(
    fasta: FastaOption,
    gff3: Gff3Option,
    model: ModelOption = DEFAULT_MODEL,
    cache_dir: CacheOption = None,
    device: DeviceOption = None,
    max_tokens: MaxTokensOption = DEFAULT_MAX_TOKENS,
    genetic_code: GeneticCodeOption = None,
    mito_genetic_code: MitoGeneticCodeOption = None,
) -> None:
    """Score every protein of an annotation (wild-type marginals) into the cache."""
    from omnimissense.pipeline import precompute as run_precompute

    result = run_precompute(
        fasta,
        gff3,
        cache=_cache(cache_dir, model),
        scorer_factory=_scorer_factory(model, device, max_tokens),
        genetic_code=genetic_code,
        mito_genetic_code=mito_genetic_code,
    )
    typer.echo(f"Coding transcripts:  {result.n_transcripts}")
    for status, count in sorted(result.status_counts.items()):
        typer.echo(f"  {status:<18} {count}")
    typer.echo(f"Unique proteins:     {result.n_unique_proteins}")
    typer.echo(f"Already cached:      {result.n_already_cached}")
    typer.echo(f"Scored now:          {result.n_scored} ({result.residues_scored} residues)")
    if result.n_scored:
        rate = result.residues_scored / result.seconds
        typer.echo(f"Time:                {result.seconds:.0f} s ({rate:.0f} residues/s)")


@app.command()
def export(
    fasta: FastaOption,
    gff3: Gff3Option,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output Parquet file.")],
    model: ModelOption = DEFAULT_MODEL,
    cache_dir: CacheOption = None,
    canonical_only: Annotated[
        bool,
        typer.Option(help="Only canonical transcripts (GFF3 tag, else the gene's longest CDS)."),
    ] = False,
    device: DeviceOption = None,
    max_tokens: MaxTokensOption = DEFAULT_MAX_TOKENS,
    genetic_code: GeneticCodeOption = None,
    mito_genetic_code: MitoGeneticCodeOption = None,
) -> None:
    """Write the wild-type-marginal scores of every coding transcript to a Parquet file."""
    from omnimissense.export import export_scores

    result = export_scores(
        fasta,
        gff3,
        output,
        cache=_cache(cache_dir, model),
        scorer_factory=_scorer_factory(model, device, max_tokens),
        model_name=model,
        canonical_only=canonical_only,
        genetic_code=genetic_code,
        mito_genetic_code=mito_genetic_code,
    )
    typer.echo(f"Transcripts:      {result.n_transcripts}")
    typer.echo(f"Unique proteins:  {result.n_unique_proteins} ({result.n_residues} residues)")
    typer.echo(f"Written to {output}")


@app.command()
def proteome(
    fasta: FastaOption,
    gff3: Gff3Option,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output protein FASTA.")],
    genetic_code: GeneticCodeOption = None,
    mito_genetic_code: MitoGeneticCodeOption = None,
) -> None:
    """Translate all coding transcripts and write the proteins to FASTA."""
    from collections import Counter

    from omnimissense.fasta import prepare_fasta
    from omnimissense.gff import read_transcripts, scan_genetic_codes
    from omnimissense.proteome import build_proteome, write_fasta

    fasta = prepare_fasta(fasta)
    codes = scan_genetic_codes(gff3, genetic_code, mito_genetic_code)
    proteins = build_proteome(fasta, read_transcripts(gff3, fasta, codes=codes))
    written = write_fasta(proteins.values(), output)
    for status, count in sorted(Counter(p.status.value for p in proteins.values()).items()):
        typer.echo(f"{status:<16} {count}")
    typer.echo(f"Wrote {written} proteins to {output}")


@app.command()
def doctor() -> None:
    """Report which dependencies are available (bcftools, PyTorch, CUDA)."""
    for check in run_checks():
        status = "ok" if check.ok else "--"
        typer.echo(f"[{status}] {check.name:<10} {check.detail}")
