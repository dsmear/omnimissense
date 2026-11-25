"""Command-line interface."""

from __future__ import annotations

from typing import Annotated

import typer

from omnimissense import __version__
from omnimissense.environment import run_checks

app = typer.Typer(
    name="omnimissense",
    help="Score missense variants in any species with protein language models.",
    no_args_is_help=True,
    add_completion=False,
)


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
) -> None:
    """Score missense variants in any species with protein language models."""


@app.command()
def doctor() -> None:
    """Report which external tools and optional dependencies are available."""
    for check in run_checks():
        status = "ok" if check.ok else "--"
        typer.echo(f"[{status}] {check.name:<10} {check.detail}")
