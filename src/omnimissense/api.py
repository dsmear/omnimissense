"""The Python API: score proteins and annotate VCFs from code.

Everything here is importable from the package itself and is covered by the compatibility
promise of the README (section *Compatibility*)::

    import omnimissense as om

    model = om.Model()                        # VESM 650M; weights load on first use
    scores = model.score_protein("MKTAYIAKQRQISFVKSHFSRQ")
    scores["Y5D"]                             # log-likelihood ratio; lower = more damaging

    result = om.annotate("calls.vcf.gz", fasta="genome.fa", gff3="genes.gff3", model=model)
    for v in result:
        print(v.gene, v.protein_change, v.wt_marginal, v.wt_percentile, v.status)
    table = result.to_pandas()                # needs pandas: pip install 'omnimissense[pandas]'

Scores are log-likelihood ratios (natural log) of the alternative vs the reference amino
acid under the model: 0 = as likely as the reference residue, negative = less likely (more
damaging), positive = more likely. The *wild-type marginal* reads all 20 amino acids at
every position from one pass over the unmasked protein (omnimissense's default score); the
*masked marginal* masks the position first (one pass per position; slower, and in our
benchmarks no better). Every score is cached on disk, per model, by protein sequence.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import pyarrow as pa

from omnimissense.cache import ScoreCache, default_cache_dir
from omnimissense.defaults import DEFAULT_MAX_TOKENS, DEFAULT_MODEL
from omnimissense.pipeline import (
    LazyScorer,
    PrecomputeResult,
    ReferenceMismatchError,
    ScoredVariant,
    VariantStatus,
    cached_masked_scores,
    cached_wt_scores,
    write_tsv,
)
from omnimissense.pipeline import annotate as _annotate
from omnimissense.pipeline import precompute as _precompute
from omnimissense.scoring import MaskedLMScorer
from omnimissense.sequences import AA_INDEX, AMINO_ACIDS, sequence_hash
from omnimissense.windows import DEFAULT_OVERLAP, MAX_RESIDUES, plan_windows

if TYPE_CHECKING:
    import pandas

__all__ = [
    "AMINO_ACIDS",
    "AnnotatedVariant",
    "AnnotatedVariants",
    "Model",
    "PrecomputeResult",
    "ProteinScores",
    "ReferenceMismatchError",
    "VariantStatus",
    "annotate",
    "precompute",
]

PathLike = str | os.PathLike[str]

_THREE_LETTER = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
    "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
    "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}  # fmt: skip
_ONE = re.compile(r"^(?:p\.)?\(?([A-Z])(\d+)([A-Z])\)?$")
_THREE = re.compile(r"^(?:p\.)?\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$")


def _to_pandas(table: pa.Table) -> pandas.DataFrame:
    try:
        import pandas  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "to_pandas() needs pandas: pip install 'omnimissense[pandas]' "
            "(or use to_arrow(), which needs nothing more)"
        ) from exc
    return table.to_pandas()


def parse_substitution(change: str) -> tuple[str, int, str]:
    """``(ref, 1-based position, alt)`` of ``"Y5D"``, ``"p.Y5D"``, ``"p.(Tyr5Asp)"``…"""
    text = change.strip()
    if match := _ONE.match(text):
        ref, position, alt = match.groups()
    elif (match := _THREE.match(text)) and {match[1], match[3]} <= _THREE_LETTER.keys():
        ref, position, alt = _THREE_LETTER[match[1]], match[2], _THREE_LETTER[match[3]]
    else:
        raise ValueError(f"not a single amino-acid substitution: {change!r} (e.g. 'Y5D')")
    return ref, int(position), alt


# ---------------------------------------------------------------------------- proteins


@dataclass(frozen=True, eq=False)
class ProteinScores:
    """Scores of every single amino-acid substitution of one protein.

    ``llr[i, j]`` is the score of amino acid ``AMINO_ACIDS[j]`` at position ``i + 1``
    (``L x 20``; 0 for the reference residue; NaN where not scored: a non-standard reference
    residue, or a position not requested in masked mode). Read single values with
    ``scores["Y5D"]`` or ``scores.score(5, "D")``, which check the reference residue.
    """

    #: The protein, as scored (upper case, without a terminal ``*``).
    sequence: str
    #: Log-likelihood ratios, ``len(sequence) x 20``, columns in ``AMINO_ACIDS`` order.
    llr: np.ndarray
    #: ``"wt_marginal"`` or ``"masked_marginal"``.
    method: str
    #: Model that produced the scores.
    model: str
    #: Number of overlapping windows used (proteins longer than 1,022 residues need several).
    n_windows: int

    def __len__(self) -> int:
        return len(self.sequence)

    def score(self, position: int, alt: str) -> float:
        """Score of ``alt`` at the 1-based ``position``."""
        if not 1 <= position <= len(self.sequence):
            raise IndexError(f"position {position} outside the protein (1-{len(self.sequence)})")
        if alt not in AA_INDEX:
            raise ValueError(f"{alt!r} is not one of the 20 standard amino acids")
        value = float(self.llr[position - 1, AA_INDEX[alt]])
        if np.isnan(value):
            raise ValueError(
                f"position {position} has no score ("
                + (
                    f"reference residue {self.sequence[position - 1]!r} is not standard)"
                    if self.sequence[position - 1] not in AA_INDEX
                    else "not requested: pass it in positions=)"
                )
            )
        return value

    def __getitem__(self, change: str) -> float:
        """Score of a substitution written ``"Y5D"``, ``"p.Y5D"`` or ``"p.(Tyr5Asp)"``; the
        reference residue must match the protein."""
        ref, position, alt = parse_substitution(change)
        if not 1 <= position <= len(self.sequence):
            raise IndexError(f"position {position} outside the protein (1-{len(self.sequence)})")
        actual = self.sequence[position - 1]
        if actual != ref:
            raise ValueError(f"{change}: residue {position} of the protein is {actual}, not {ref}")
        return self.score(position, alt)

    def to_arrow(self) -> pa.Table:
        """One row per substitution (19 per scored position): ``position``, ``ref_aa``,
        ``alt_aa``, ``change`` (``"Y5D"``) and the score, in a column named after
        ``method``."""
        rows, cols = np.nonzero(np.isfinite(self.llr))
        ref = np.array([self.sequence[i] for i in rows], dtype=object)
        alt = np.array(list(AMINO_ACIDS), dtype=object)[cols]
        keep = ref != alt
        rows, ref, alt = rows[keep], ref[keep], alt[keep]
        values = self.llr[rows, cols[keep]]
        positions = rows + 1
        changes = [f"{r}{p}{a}" for r, p, a in zip(ref, positions, alt, strict=True)]
        return pa.table(
            {
                "position": pa.array(positions, pa.int32()),
                "ref_aa": pa.array(ref, pa.string()),
                "alt_aa": pa.array(alt, pa.string()),
                "change": pa.array(changes, pa.string()),
                self.method: pa.array(values, pa.float32()),
            }
        )

    def to_pandas(self) -> pandas.DataFrame:
        """``to_arrow()`` as a pandas DataFrame (needs pandas)."""
        return _to_pandas(self.to_arrow())


def _clean(sequence: str) -> str:
    sequence = sequence.strip().upper().removesuffix("*")
    if not sequence or not sequence.isalpha():
        raise ValueError("a protein sequence is a non-empty string of one-letter amino acids")
    return sequence


class Model:
    """A protein language model with its score cache.

    ``name`` is ``ntranoslab/vesm/VESM_<size>`` (default: VESM 650M) or a Hugging Face ESM
    checkpoint such as ``facebook/esm2_t33_650M_UR50D``. The weights are loaded the first
    time a score is not found in the cache (on ``device``: default the GPU if there is one),
    so creating a model is cheap and fully cached work never loads it. ``cache_dir`` defaults
    to ``$OMNIMISSENSE_CACHE`` or ``~/.cache/omnimissense``, shared with the command line.
    """

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        cache_dir: PathLike | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.name = name
        self.cache = ScoreCache(Path(cache_dir) if cache_dir else default_cache_dir(), name)
        self._scorer = LazyScorer(
            lambda: MaskedLMScorer.from_pretrained(name, device=device, max_tokens=max_tokens)
        )

    @classmethod
    def from_scorer(
        cls, scorer: MaskedLMScorer, *, name: str, cache_dir: PathLike | None = None
    ) -> Model:
        """Wrap an already loaded scorer (a custom or local model); ``name`` keys its cache."""
        model = cls(name, cache_dir=cache_dir)
        model._scorer = LazyScorer(lambda: scorer)
        return model

    def __repr__(self) -> str:
        return f"Model({self.name!r}, cache_dir={str(self.cache.directory.parents[1])!r})"

    def score_protein(
        self, sequence: str, *, masked: bool = False, positions: Iterable[int] | None = None
    ) -> ProteinScores:
        """Score every substitution of ``sequence``.

        Default: wild-type marginal (one forward pass per 1,022-residue window). With
        ``masked=True``: masked marginal at the 1-based ``positions`` (default all of them,
        one forward pass each).
        """
        if not masked:
            if positions is not None:
                raise ValueError("positions= is only used with masked=True")
            return self.score_proteins([sequence])[0]
        sequence = _clean(sequence)
        wanted = range(1, len(sequence) + 1) if positions is None else sorted(set(positions))
        for position in wanted:
            if not 1 <= position <= len(sequence):
                raise IndexError(f"position {position} outside the protein (1-{len(sequence)})")
        digest = sequence_hash(sequence)
        scorable = [p for p in wanted if sequence[p - 1] in AA_INDEX]
        vectors = cached_masked_scores(
            {(digest, p): sequence for p in scorable}, self.cache, self._scorer
        )
        llr = np.full((len(sequence), len(AMINO_ACIDS)), np.nan, dtype=np.float32)
        for position in scorable:
            llr[position - 1] = vectors[(digest, position)]
        n_windows = len(plan_windows(len(sequence), MAX_RESIDUES, DEFAULT_OVERLAP))
        return ProteinScores(sequence, llr, "masked_marginal", self.name, n_windows)

    def score_proteins(self, sequences: Iterable[str]) -> list[ProteinScores]:
        """Wild-type-marginal scores of many proteins at once (batched on the GPU), in order."""
        cleaned = [_clean(s) for s in sequences]
        unique = {sequence_hash(s): s for s in cleaned}
        found = cached_wt_scores(unique, self.cache, self._scorer)
        results = []
        for sequence in cleaned:
            llr, n_windows = found[sequence_hash(sequence)]
            results.append(ProteinScores(sequence, llr, "wt_marginal", self.name, n_windows))
        return results


def _model(model: Model | str | None) -> Model:
    if model is None:
        return Model()
    if isinstance(model, str):
        return Model(model)
    return model


# ---------------------------------------------------------------------------- variants


@dataclass(frozen=True)
class AnnotatedVariant:
    """One missense consequence: one ALT allele of a VCF record on one transcript.

    The fields are the columns of ``omnimissense annotate``'s TSV, in the same order, plus
    ``protein_change`` and ``record_index``. Scores are ``None`` unless ``status`` is
    ``"ok"``.
    """

    #: Sequence name, as in the VCF.
    chrom: str
    #: 1-based position of the VCF record.
    pos: int
    #: Reference allele of the VCF record.
    ref: str
    #: The ALT allele this consequence belongs to.
    alt: str
    #: Gene name (or ID when the annotation has no name).
    gene: str
    #: Transcript ID, as in the annotation.
    transcript: str
    #: Canonical transcript of its gene (Ensembl canonical, MANE or RefSeq Select tag;
    #: otherwise the longest CDS).
    canonical: bool
    #: Transcript biotype, as in the annotation.
    biotype: str
    #: ``"+"`` or ``"-"``.
    strand: str
    #: 1-based position of the (first) changed amino acid in the protein.
    protein_pos: int
    #: Reference amino acid(s); several for a change spanning codons.
    ref_aa: str
    #: Alternative amino acid(s).
    alt_aa: str
    #: Length of the protein, without the stop.
    protein_length: int | None
    #: Windows used to score the protein (1 up to 1,022 residues).
    n_windows: int | None
    #: Wild-type-marginal log-likelihood ratio (lower = more damaging); for a change of
    #: several residues, the sum of their scores.
    wt_marginal: float | None
    #: Masked-marginal log-likelihood ratio, when requested (``masked=True``).
    masked_marginal: float | None
    #: Why a consequence is or is not scored, e.g. ``"ok"``, ``"ref_aa_mismatch"``,
    #: ``"multi_aa_change"``, ``"pseudogene"`` (a ``VariantStatus``, equal to its string).
    status: VariantStatus
    #: Proteome-wide percentile of ``wt_marginal`` (0-100; 1 = among the 1% most damaging
    #: missense changes possible in this genome); ``None`` until ``precompute`` was run once
    #: for this genome, annotation and model.
    wt_percentile: float | None
    #: The substitution(s) in one-letter code, e.g. ``"R106W"``, ``"W4C+L5M"``; changes that
    #: insert or delete residues keep bcftools csq's notation (``"4WL>4C"``).
    protein_change: str
    #: 0-based ordinal of the record in the input VCF.
    record_index: int

    @property
    def is_scored(self) -> bool:
        return self.status is VariantStatus.OK


def _public(v: ScoredVariant) -> AnnotatedVariant:
    c = v.consequence
    substitutions = c.substitutions
    if substitutions:
        change = "+".join(f"{r}{p}{a}" for p, r, a in substitutions)
    else:
        change = f"{c.protein_pos}{c.ref_aa}>{c.protein_pos}{c.alt_aa}"
    return AnnotatedVariant(
        chrom=v.chrom,
        pos=v.pos,
        ref=v.ref,
        alt=v.alt,
        gene=c.gene,
        transcript=c.transcript_id,
        canonical=v.canonical,
        biotype=c.biotype,
        strand=c.strand,
        protein_pos=c.protein_pos,
        ref_aa=c.ref_aa,
        alt_aa=c.alt_aa,
        protein_length=v.protein_length,
        n_windows=v.n_windows,
        wt_marginal=v.wt_marginal,
        masked_marginal=v.masked_marginal,
        status=v.status,
        wt_percentile=v.wt_percentile,
        protein_change=change,
        record_index=v.record_index,
    )


_ARROW_TYPES = {
    "pos": pa.int64(),
    "canonical": pa.bool_(),
    "protein_pos": pa.int32(),
    "protein_length": pa.int32(),
    "n_windows": pa.int32(),
    "wt_marginal": pa.float64(),
    "masked_marginal": pa.float64(),
    "wt_percentile": pa.float64(),
    "record_index": pa.int64(),
}


class AnnotatedVariants(Sequence[AnnotatedVariant]):
    """The result of ``annotate``: a sequence of ``AnnotatedVariant`` in input order."""

    def __init__(
        self, scored: list[ScoredVariant], *, n_records: int, n_ref_mismatches: int, model: str
    ) -> None:
        self._scored = scored
        self._variants = [_public(v) for v in scored]
        #: Records in the input VCF.
        self.n_records = n_records
        #: Records whose REF disagrees with the genome (left unannotated).
        self.n_ref_mismatches = n_ref_mismatches
        #: Model that produced the scores.
        self.model = model

    def __len__(self) -> int:
        return len(self._variants)

    @overload
    def __getitem__(self, index: int) -> AnnotatedVariant: ...
    @overload
    def __getitem__(self, index: slice) -> list[AnnotatedVariant]: ...
    def __getitem__(self, index: Any) -> Any:
        return self._variants[index]

    def __iter__(self) -> Iterator[AnnotatedVariant]:
        return iter(self._variants)

    def __repr__(self) -> str:
        return (
            f"<AnnotatedVariants: {len(self)} missense consequences in {self.n_records} "
            f"records, {dict(self.status_counts())}>"
        )

    def status_counts(self) -> Counter[str]:
        """Number of consequences per status."""
        return Counter(v.status.value for v in self._variants)

    def to_arrow(self) -> pa.Table:
        """One row per consequence, one column per ``AnnotatedVariant`` field."""
        names = [f.name for f in fields(AnnotatedVariant)]
        columns = list(zip(*(astuple(v) for v in self._variants), strict=True)) or [
            () for _ in names
        ]
        arrays = {}
        for name, values in zip(names, columns, strict=True):
            if name == "status":
                values = [str(s) for s in values]
            arrays[name] = pa.array(list(values), _ARROW_TYPES.get(name, pa.string()))
        return pa.table(arrays)

    def to_pandas(self) -> pandas.DataFrame:
        """``to_arrow()`` as a pandas DataFrame (needs pandas)."""
        return _to_pandas(self.to_arrow())

    def write_tsv(self, path: PathLike) -> None:
        """Write the same TSV as ``omnimissense annotate --output-tsv``."""
        write_tsv(Path(path), self._scored)


def annotate(
    vcf: PathLike,
    *,
    fasta: PathLike,
    gff3: PathLike,
    model: Model | str | None = None,
    masked: bool = False,
    output_vcf: PathLike | None = None,
    output_tsv: PathLike | None = None,
    genetic_code: int | None = None,
    mito_genetic_code: int | None = None,
    threads: int = 0,
) -> AnnotatedVariants:
    """Score the missense variants of a VCF, as ``omnimissense annotate`` does.

    ``fasta`` is the reference genome (plain, gzip or BGZF) and ``gff3`` its annotation
    (GFF3 or GTF: Ensembl, NCBI RefSeq, Prokka, Bakta). ``model`` is a ``Model`` or a model
    name (default VESM 650M). With ``output_vcf`` (``.vcf.gz``, ``.bcf`` or ``.vcf``), the
    input VCF is also written with the ``BCSQ`` and ``OMS*`` INFO tags; with ``output_tsv``,
    the same table the command line writes. ``genetic_code`` / ``mito_genetic_code``: NCBI
    translation tables; default from the GFF3 (see the README, *Genetic codes*).

    Raises ``ReferenceMismatchError`` (a ``ValueError``) when most records disagree with
    the genome: a VCF called on another assembly, or with other sequence names.
    """
    model = _model(model)
    result = _annotate(
        Path(vcf),
        Path(fasta),
        Path(gff3),
        Path(output_vcf) if output_vcf is not None else None,
        Path(output_tsv) if output_tsv is not None else None,
        cache=model.cache,
        scorer_factory=model._scorer.get,
        model_name=model.name,
        masked=masked,
        threads=threads,
        genetic_code=genetic_code,
        mito_genetic_code=mito_genetic_code,
    )
    return AnnotatedVariants(
        result.variants,
        n_records=result.n_records,
        n_ref_mismatches=result.n_ref_mismatches,
        model=model.name,
    )


def precompute(
    fasta: PathLike,
    gff3: PathLike,
    *,
    model: Model | str | None = None,
    progress: bool = False,
    genetic_code: int | None = None,
    mito_genetic_code: int | None = None,
) -> PrecomputeResult:
    """Score every protein of an annotation into the cache and build the proteome-wide
    percentile reference (what ``omnimissense precompute`` does): afterwards ``annotate``
    on this genome needs no GPU and reports ``wt_percentile``."""
    model = _model(model)
    return _precompute(
        Path(fasta),
        Path(gff3),
        cache=model.cache,
        scorer_factory=model._scorer.get,
        progress=progress,
        genetic_code=genetic_code,
        mito_genetic_code=mito_genetic_code,
    )
