from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
import csv
import hashlib
import json
import platform
import subprocess
import sys
from typing import Iterable

ANALYSIS_BRANCHES = {
    "01_laminar_vs_unsteady",
    "02_kolmogorov_obukhov",
    "03_taylor_hypothesis",
    "04_lumley_anisotropy",
    "04b_signed_interscale_transfer",
    "05_canonical_diagnostics",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_id() -> str:
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _package_versions() -> dict[str, str | None]:
    distributions = {
        "aorticturb": "aorticturb",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "PyYAML": "PyYAML",
        "requests": "requests",
        "typer": "typer",
        "rich": "rich",
    }
    versions: dict[str, str | None] = {}
    for label, dist in distributions.items():
        try:
            versions[label] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _installed_vcs_commit() -> str | None:
    """Return the exact VCS commit recorded by a PEP-610 VCS installation, if available."""
    try:
        distribution = importlib_metadata.distribution("aorticturb")
        raw = distribution.read_text("direct_url.json")
    except (importlib_metadata.PackageNotFoundError, OSError, ValueError):
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs_info = payload.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit = vcs_info.get("commit_id")
    return commit if isinstance(commit, str) and len(commit) >= 7 else None


def _git_commit() -> str | None:
    installed_commit = _installed_vcs_commit()
    if installed_commit is not None:
        return installed_commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        value = result.stdout.strip()
        return value if result.returncode == 0 and len(value) >= 7 else None
    except (OSError, subprocess.SubprocessError):
        return None


def create_run_directory(workspace_root: str | Path, branch: str, *, run_id: str | None = None) -> Path:
    """Create a run under the canonical AGENTS.md workspace tree."""
    run_id = run_id or new_run_id()
    root = Path(workspace_root)
    if branch in ANALYSIS_BRANCHES:
        base = root / "04_analysis" / branch
    elif branch == "05_results":
        base = root / "05_results"
    else:
        base = root / branch
    path = base / run_id
    path.mkdir(parents=True, exist_ok=False)
    (path / "figures").mkdir()
    (path / "processed").mkdir()
    (path / "logs").mkdir()
    return path


def write_json(path: Path, payload: dict | list) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_csv(path: Path, rows: list[dict]) -> Path:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def append_log(run_dir: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    with (run_dir / "logs" / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def write_input_hashes(run_dir: Path, dataset_root: Path, extra_inputs: Iterable[Path] = ()) -> Path:
    lines: list[str] = []
    ledger = dataset_root / "SHA256SUMS.txt"
    if ledger.exists():
        lines.append(f"# verified package ledger sha256 {sha256_file(ledger)}  SHA256SUMS.txt")
        lines.extend(line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())
    for path in extra_inputs:
        if path.exists() and path.is_file():
            lines.append(f"{sha256_file(path)}  {path}")
    out = run_dir / "input_hashes.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def initialize_run(
    run_dir: Path,
    *,
    branch: str,
    dataset_root: Path,
    config: dict,
    methods: str,
    extra_inputs: Iterable[Path] = (),
    warnings: list[str] | None = None,
) -> Path:
    write_json(run_dir / "config.json", config)
    (run_dir / "methods.md").write_text(methods.rstrip() + "\n", encoding="utf-8")
    write_input_hashes(run_dir, dataset_root, extra_inputs)
    manifest = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": _package_versions(),
        "git_commit": _git_commit(),
        "dataset_root": str(dataset_root),
        "config": config,
        "warnings": warnings or [],
        "status": "RUNNING",
    }
    append_log(run_dir, f"initialized branch={branch}")
    return write_json(run_dir / "run_manifest.json", manifest)


def finalize_run(run_dir: Path, *, status: str = "COMPLETE", warnings: list[str] | None = None) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    append_log(run_dir, f"finalized status={status}")
    outputs = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path == manifest_path:
            continue
        outputs.append({
            "path": str(path.relative_to(run_dir)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = status
    manifest["outputs"] = outputs
    if warnings:
        manifest.setdefault("warnings", []).extend(warnings)
    return write_json(manifest_path, manifest)
