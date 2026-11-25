from typer.testing import CliRunner

from omnimissense import __version__, environment
from omnimissense.cli import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"omnimissense {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    for command in ("annotate", "precompute", "proteome", "doctor"):
        assert command in result.output


def test_doctor_lists_every_check() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    for name in ("python", "bcftools", "torch"):
        assert name in result.stdout


def test_bcftools_is_bundled_with_pysam() -> None:
    check = environment.check_bcftools()
    assert check.ok
    assert "pysam" in check.detail


def test_check_torch_when_not_installed(monkeypatch) -> None:
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _name: None)
    assert environment.check_torch() == [
        environment.Check("torch", ok=False, detail="not installed")
    ]


def test_proteome_command_writes_fasta(synthetic, tmp_path) -> None:
    output = tmp_path / "proteins.fa"
    result = runner.invoke(
        app,
        [
            "proteome",
            "--fasta",
            str(synthetic.fasta),
            "--gff3",
            str(synthetic.gff3),
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.read_text() == ">TXA\nMKRWLEHC\n>TXB\nMDPFGKWR\n"
