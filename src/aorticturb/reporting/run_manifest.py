from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import sys


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_directory(output_root: str | Path, branch: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_root) / branch / f"run_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "figures").mkdir()
    return path


def write_run_manifest(
    run_dir: Path,
    *,
    branch: str,
    inputs: list[Path],
    config: dict,
    warnings: list[str] | None = None,
) -> Path:
    manifest = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "python": sys.version,
        "platform": platform.platform(),
        "inputs": [
            {"path": str(p), "sha256": sha256_file(p)}
            for p in inputs
            if p.exists() and p.is_file()
        ],
        "config": config,
        "warnings": warnings or [],
    }
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
