from __future__ import annotations

from pathlib import Path
import json
import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import load_config
from .data.download import fetch_dataset
from .data.verify import verify_dataset
from .validation.startup_gate import run_startup_gate
from .workflow import run_decomposition

app = typer.Typer(help="AorticTurb: reproducible analysis of aortic Tomo-PIV turbulence.")
dataset_app = typer.Typer(help="Acquire, verify, and inspect the curated input dataset.")
app.add_typer(dataset_app, name="dataset")
console = Console()


def _manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "dataset_manifest.json"


@app.command()
def version() -> None:
    console.print(f"AorticTurb {__version__}")


@dataset_app.command("fetch")
def dataset_fetch(
    output: Path = typer.Option(Path("data/hasler_obrist_tomopiv"), "--output", "-o"),
    record_id: int | None = typer.Option(None, "--record-id"),
    archive_name: str | None = typer.Option(None, "--archive-name"),
) -> None:
    path = fetch_dataset(
        output,
        _manifest_path(),
        record_id=record_id,
        archive_name=archive_name,
    )
    console.print(f"[green]Dataset ready:[/green] {path}")


@dataset_app.command("verify")
def dataset_verify(
    root: Path = typer.Option(..., "--root", exists=True, file_okay=False),
    checksums: bool = typer.Option(True, "--checksums/--no-checksums"),
) -> None:
    result = verify_dataset(root, verify_checksums=checksums)
    console.print_json(json.dumps(result.to_dict()))
    raise typer.Exit(code=0 if result.ok else 2)


@dataset_app.command("inspect")
def dataset_inspect(
    root: Path = typer.Option(..., "--root", exists=True, file_okay=False),
) -> None:
    result = verify_dataset(root, verify_checksums=False)
    table = Table(title="AorticTurb input dataset")
    table.add_column("Item")
    table.add_column("Observed")
    table.add_row("Velocity files", str(result.velocity_files))
    table.add_row("Geometry files", str(result.geometry_files))
    table.add_row("Structural verification", "PASS" if result.ok else "FAIL")
    table.add_row("Warnings", str(len(result.warnings)))
    console.print(table)
    if result.errors:
        for error in result.errors:
            console.print(f"[red]{error}[/red]")


@app.command()
def gate(
    root: Path = typer.Option(..., "--root", exists=True, file_okay=False),
) -> None:
    result = run_startup_gate(root)
    console.print_json(json.dumps(result.to_dict()))
    raise typer.Exit(code=0 if result.passed else 2)


@app.command()
def decompose(
    root: Path = typer.Option(..., "--root", exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs/decomposition"), "--output", "-o"),
) -> None:
    summary = run_decomposition(root, output)
    console.print_json(json.dumps(summary))


def _not_yet_driver(name: str) -> None:
    raise typer.BadParameter(
        f"{name} low-level scientific primitives are implemented, but the publication-grade "
        "driver is deliberately not enabled until decomposition outputs are reviewed. "
        "This enforces the AGENTS.md scientific sequence."
    )


@app.command()
def regime() -> None:
    _not_yet_driver("regime")


@app.command()
def kolmogorov() -> None:
    _not_yet_driver("kolmogorov")


@app.command()
def taylor() -> None:
    _not_yet_driver("taylor")


@app.command()
def anisotropy() -> None:
    _not_yet_driver("anisotropy")


@app.command()
def diagnostics() -> None:
    _not_yet_driver("diagnostics")


@app.command()
def classify() -> None:
    _not_yet_driver("classify")


@app.command("run-all")
def run_all(
    root: Path = typer.Option(..., "--root", exists=True, file_okay=False),
    output: Path = typer.Option(Path("outputs"), "--output", "-o"),
    config: Path | None = typer.Option(None, "--config"),
) -> None:
    cfg = load_config(config)
    console.print(
        "[yellow]run-all currently executes the startup gate and decomposition only.[/yellow]\n"
        "Downstream branches remain gated until decomposition is reviewed, by design."
    )
    run_decomposition(root, output / "01_laminar_vs_unsteady")
    console.print_json(json.dumps({"config": cfg, "status": "DECOMPOSITION_COMPLETE_REVIEW_REQUIRED"}))
