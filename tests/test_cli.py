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
    assert "doctor" in result.output


def test_doctor_lists_every_check() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    for name in ("python", "bcftools", "tabix", "torch"):
        assert name in result.stdout


def test_check_tool_reports_missing_tool(monkeypatch) -> None:
    monkeypatch.setattr(environment.shutil, "which", lambda _name: None)
    check = environment.check_tool("bcftools")
    assert not check.ok
    assert check.detail == "not found on PATH"


def test_check_tool_reports_version_of_found_tool(monkeypatch) -> None:
    monkeypatch.setattr(environment.shutil, "which", lambda name: f"/fake/bin/{name}")
    monkeypatch.setattr(environment, "_tool_version", lambda _path: "bcftools 1.22")
    check = environment.check_tool("bcftools")
    assert check.ok
    assert check.detail == "bcftools 1.22 (/fake/bin/bcftools)"


def test_check_torch_when_not_installed(monkeypatch) -> None:
    monkeypatch.setattr(environment.importlib.util, "find_spec", lambda _name: None)
    assert environment.check_torch() == [
        environment.Check("torch", ok=False, detail="not installed")
    ]
