from __future__ import annotations

from importlib.resources import files
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
from .decomposition.driver import run_decomposition
from .regime.driver import run_regime
from .regime.sensitivity_driver import run_support_sensitivity
from .kolmogorov.driver import run_kolmogorov
from .taylor.driver import run_taylor
from .anisotropy.driver import run_anisotropy
from .diagnostics.driver import run_diagnostics
from .classification.driver import run_classification
from .workflow import run_all

app = typer.Typer(help="AorticTurb: reproducible analysis of aortic Tomo-PIV turbulence.", no_args_is_help=True)
dataset_app = typer.Typer(help="Acquire, verify, and inspect the curated input dataset.")
app.add_typer(dataset_app, name="dataset")
console = Console()


def _manifest_path() -> Path:
    return Path(str(files("aorticturb").joinpath("dataset_manifest.json")))


def _cfg(path: Path | None) -> dict:
    return load_config(path)


@app.command()
def version() -> None:
    console.print(f"AorticTurb {__version__}")


@dataset_app.command("fetch")
def dataset_fetch(output: Path = typer.Option(Path("data/hasler_obrist_tomopiv"), "--output", "-o"), record_id: int | None = typer.Option(None, "--record-id"), archive_name: str | None = typer.Option(None, "--archive-name")) -> None:
    path = fetch_dataset(output, _manifest_path(), record_id=record_id, archive_name=archive_name)
    console.print(f"[green]Dataset ready:[/green] {path}")


@dataset_app.command("verify")
def dataset_verify(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), checksums: bool = typer.Option(True, "--checksums/--no-checksums"), deep: bool = typer.Option(True, "--deep/--no-deep")) -> None:
    result = verify_dataset(root, verify_checksums=checksums, deep_content_check=deep)
    console.print_json(json.dumps(result.to_dict()))
    raise typer.Exit(code=0 if result.ok else 2)


@dataset_app.command("inspect")
def dataset_inspect(root: Path = typer.Option(..., "--root", exists=True, file_okay=False)) -> None:
    result = verify_dataset(root, verify_checksums=False, deep_content_check=True)
    table = Table(title="AorticTurb input dataset")
    table.add_column("Item")
    table.add_column("Observed")
    table.add_row("Velocity files", str(result.velocity_files))
    table.add_row("Geometry files", str(result.geometry_files))
    table.add_row("Structural/content verification", "PASS" if result.ok else "FAIL")
    table.add_row("Warnings", str(len(result.warnings)))
    console.print(table)
    for error in result.errors:
        console.print(f"[red]{error}[/red]")


@app.command()
def gate(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), checksums: bool = typer.Option(True, "--checksums/--no-checksums")) -> None:
    result = run_startup_gate(root, verify_checksums=checksums, deep_content_check=True)
    console.print_json(json.dumps(result.to_dict()))
    raise typer.Exit(code=0 if result.passed else 2)


@app.command()
def decompose(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    result = run_decomposition(root, output, _cfg(config))
    console.print_json(json.dumps(result))


@app.command()
def regime(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), decomposition: Path = typer.Option(..., "--decomposition", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    console.print_json(json.dumps(run_regime(root, decomposition, output, _cfg(config))))


@app.command("support-sensitivity")
def support_sensitivity(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), decomposition: Path = typer.Option(..., "--decomposition", exists=True, file_okay=False), regime_run: Path = typer.Option(..., "--regime", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    console.print_json(json.dumps(run_support_sensitivity(root, decomposition, regime_run, output, _cfg(config))))


@app.command()
def kolmogorov(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), decomposition: Path = typer.Option(..., "--decomposition", exists=True, file_okay=False), regime_run: Path = typer.Option(..., "--regime", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    console.print_json(json.dumps(run_kolmogorov(root, decomposition, regime_run, output, _cfg(config))))


@app.command()
def taylor(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), decomposition: Path = typer.Option(..., "--decomposition", exists=True, file_okay=False), regime_run: Path = typer.Option(..., "--regime", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    console.print_json(json.dumps(run_taylor(root, decomposition, regime_run, output, _cfg(config))))


@app.command()
def anisotropy(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), decomposition: Path = typer.Option(..., "--decomposition", exists=True, file_okay=False), regime_run: Path = typer.Option(..., "--regime", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    console.print_json(json.dumps(run_anisotropy(root, decomposition, regime_run, output, _cfg(config))))


@app.command()
def diagnostics(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), decomposition: Path = typer.Option(..., "--decomposition", exists=True, file_okay=False), regime_run: Path = typer.Option(..., "--regime", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    console.print_json(json.dumps(run_diagnostics(root, decomposition, regime_run, output, _cfg(config))))


@app.command()
def classify(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), regime_run: Path = typer.Option(..., "--regime", exists=True, file_okay=False), kolmogorov_run: Path = typer.Option(..., "--kolmogorov", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config"), taylor_run: Path | None = typer.Option(None, "--taylor"), anisotropy_run: Path | None = typer.Option(None, "--anisotropy"), diagnostics_run: Path | None = typer.Option(None, "--diagnostics")) -> None:
    console.print_json(json.dumps(run_classification(root, regime_run, kolmogorov_run, output, _cfg(config), taylor_dir=taylor_run, anisotropy_dir=anisotropy_run, diagnostics_dir=diagnostics_run)))


@app.command("run-all")
def run_all_command(root: Path = typer.Option(..., "--root", exists=True, file_okay=False), output: Path = typer.Option(Path("outputs"), "--output", "-o"), config: Path | None = typer.Option(None, "--config")) -> None:
    console.print_json(json.dumps(run_all(root, output, config_path=config)))
