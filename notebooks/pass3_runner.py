from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

PHASES = ("t018", "t021", "t024")
REALIZATIONS = tuple(f"B{i:05d}.npy" for i in range(1, 17))
EXPECTED_POLICIES = {
    "legacy_global_fraction",
    "strict_mask_consistency",
    "fully_finite_local_windows",
    "strict_mask_and_local_windows",
}


def _fail(code: str, message: str) -> None:
    raise RuntimeError(f"{code}: {message}")


def _sha256_file(path: str | Path, chunk: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_ledger(path: str | Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            _fail("SOURCE_PROVENANCE_FAILED", f"malformed checksum line: {line}")
        records[parts[1].strip().lstrip("*")] = parts[0].lower()
    return records


def _copy_verified(src: str | Path, dst: str | Path, expected: str | None = None) -> str:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with src.open("rb") as fin, dst.open("wb") as fout:
        while True:
            block = fin.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            fout.write(block)
    streamed = digest.hexdigest()
    if expected and streamed.lower() != expected.lower():
        dst.unlink(missing_ok=True)
        _fail("COPY_HASH_MISMATCH", f"{src}: expected {expected}, got {streamed}")
    if _sha256_file(dst) != streamed:
        dst.unlink(missing_ok=True)
        _fail("COPY_HASH_MISMATCH", f"local write verification failed: {dst}")
    return streamed


def _repo_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _stage_dataset(drive_project_root: Path, local_root: Path) -> str:
    velocity_root = drive_project_root / "01_raw_data" / "M_medium_root" / "AAo"
    for phase in PHASES:
        observed = sorted(p.name for p in (velocity_root / phase).glob("*.npy"))
        if observed != list(REALIZATIONS):
            _fail(
                "DATASET_ROOT_INVALID",
                f"{phase} does not contain exactly B00001.npy–B00016.npy",
            )

    geometry_sources = {
        "M": drive_project_root / "02_metadata_geometry/M_medium_root/original_geometry/M_aortic_root.STL",
        "S": drive_project_root / "02_metadata_geometry/S_small_root/original_geometry/S_aortic_root.STL",
        "L": drive_project_root / "02_metadata_geometry/L_large_root/original_geometry/L_aortic_root.STL",
        "NS": drive_project_root / "02_metadata_geometry/NS_non_sinus_root/original_geometry/NS_aortic_root.STL",
    }
    for path in geometry_sources.values():
        if not path.is_file():
            _fail("DATASET_ROOT_INVALID", f"missing geometry {path}")

    historical_ledger = (
        drive_project_root / "00_provenance/checksums/final_acquired_SHA256SUMS.txt"
    )
    if not historical_ledger.is_file():
        _fail("SOURCE_PROVENANCE_FAILED", "missing final_acquired_SHA256SUMS.txt")
    historical = _parse_ledger(historical_ledger)

    by_basename: dict[str, set[str]] = defaultdict(set)
    for old_path, digest in historical.items():
        by_basename[Path(old_path).name].add(digest)

    if local_root.exists():
        if not str(local_root).startswith("/content/"):
            _fail("LOCAL_STAGING_INCOMPLETE", "refusing to delete non-/content path")
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=False)

    for name in ("METADATA.json", "AGENTS.md"):
        src = drive_project_root / name
        if not src.is_file():
            _fail("DATASET_ROOT_INVALID", f"missing {src}")
        _copy_verified(src, local_root / name)

    provenance_root = drive_project_root / "00_provenance"
    for src in sorted(p for p in provenance_root.rglob("*") if p.is_file()):
        _copy_verified(src, local_root / src.relative_to(drive_project_root))

    for phase in PHASES:
        for name in REALIZATIONS:
            rel = Path("01_raw_data") / "M_medium_root" / "AAo" / phase / name
            expected = historical.get(rel.as_posix())
            if expected is None:
                _fail("SOURCE_PROVENANCE_FAILED", f"ledger omits {rel.as_posix()}")
            _copy_verified(drive_project_root / rel, local_root / rel, expected)

    geometry_dest = {
        "M": Path("02_metadata_geometry/M_medium_root/original_geometry/M_aortic_root.STL"),
        "S": Path("02_metadata_geometry/S_small_root/original_geometry/S_aortic_root.STL"),
        "L": Path("02_metadata_geometry/L_large_root/original_geometry/L_aortic_root.STL"),
        "NS": Path("02_metadata_geometry/NS_non_sinus_root/original_geometry/NS_aortic_root.STL"),
    }
    for key, src in geometry_sources.items():
        candidates = by_basename[src.name]
        if len(candidates) != 1:
            _fail("SOURCE_PROVENANCE_FAILED", f"cannot uniquely resolve {src.name}")
        _copy_verified(src, local_root / geometry_dest[key], next(iter(candidates)))

    rows: list[str] = []
    for path in sorted(p for p in local_root.rglob("*") if p.is_file()):
        if path.name == "SHA256SUMS.txt" and path.parent == local_root:
            continue
        rows.append(f"{_sha256_file(path)}  {path.relative_to(local_root).as_posix()}")
    package_ledger = local_root / "SHA256SUMS.txt"
    package_ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return _sha256_file(package_ledger)


def run_pass3(
    *,
    expected_commit: str,
    drive_project_root: str | Path = "/content/drive/MyDrive/Hasler_Obrist_TomoPIV_Turbulence_Study",
    local_root: str | Path = "/content/aorticturb_input",
) -> dict:
    """Execute the agreed Pass-3 sequence without entering downstream branches."""
    from google.colab import drive
    import pandas as pd

    import aorticturb
    from aorticturb.config import load_config
    from aorticturb.data.verify import verify_dataset
    from aorticturb.decomposition.driver import run_decomposition
    from aorticturb.regime.driver import run_regime
    from aorticturb.regime.sensitivity_driver import run_support_sensitivity
    from aorticturb.validation.startup_gate import run_startup_gate

    repo_root = Path(__file__).resolve().parents[1]
    observed_commit = _repo_commit(repo_root)
    if observed_commit != expected_commit:
        _fail(
            "GIT_COMMIT_MISMATCH",
            f"expected {expected_commit}, got {observed_commit}",
        )
    os.chdir(repo_root)

    drive.mount("/content/drive")
    drive_project_root = Path(drive_project_root).resolve()
    drive_results_root = drive_project_root / "results"
    local_root = Path(local_root)

    if not drive_project_root.is_dir():
        _fail(
            "DATASET_ROOT_INVALID",
            "edit DRIVE_PROJECT_ROOT in the notebook if the mounted project path differs",
        )
    drive_results_root.mkdir(parents=True, exist_ok=True)
    probe = drive_results_root / ".write_test"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()

    free_gib = shutil.disk_usage("/content").free / (1024**3)
    if free_gib < 8:
        _fail("INSUFFICIENT_LOCAL_STORAGE", f"only {free_gib:.1f} GiB free")

    package_ledger_sha256 = _stage_dataset(drive_project_root, local_root)

    verification = verify_dataset(
        local_root,
        verify_checksums=True,
        deep_content_check=True,
    )
    if not verification.ok:
        _fail("AORTICTURB_VERIFY_FAILED", json.dumps(verification.to_dict()))

    startup = run_startup_gate(
        local_root,
        verify_checksums=True,
        deep_content_check=True,
    )
    if not startup.passed:
        _fail("STARTUP_GATE_FAILED", json.dumps(startup.to_dict()))

    config = load_config(None)
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    execution_id = (
        "colab_pass3_"
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_")
        + uuid.uuid4().hex[:8]
    )
    record_dir = drive_results_root / "colab_runs" / execution_id
    record_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "execution_id": execution_id,
        "pass": 3,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "aorticturb_commit": expected_commit,
        "aorticturb_version": aorticturb.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "drive_project_root": str(drive_project_root),
        "drive_results_root": str(drive_results_root),
        "local_package_ledger_sha256": package_ledger_sha256,
        "config_sha256": config_sha256,
        "scientific_sequence": [
            "phase_decomposition",
            "gate_a_legacy_primary",
            "support_policy_sensitivity",
        ],
        "software_versions": {
            name: importlib_metadata.version(name)
            for name in [
                "aorticturb",
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "PyYAML",
                "requests",
                "typer",
                "rich",
            ]
        },
    }
    (record_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (record_dir / "config_snapshot.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (record_dir / "dataset_verification.json").write_text(
        json.dumps(verification.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (record_dir / "startup_gate.json").write_text(
        json.dumps(startup.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        decomposition_summary = run_decomposition(
            local_root,
            drive_results_root,
            config,
        )
        decomposition_dir = Path(decomposition_summary["run_dir"])

        regime_summary = run_regime(
            local_root,
            decomposition_dir,
            drive_results_root,
            config,
        )
        regime_dir = Path(regime_summary["run_dir"])

        sensitivity_summary = run_support_sensitivity(
            local_root,
            decomposition_dir,
            regime_dir,
            drive_results_root,
            config,
        )
        sensitivity_dir = Path(sensitivity_summary["run_dir"])
    except Exception as exc:
        manifest["status"] = "PASS3_EXECUTION_FAILED"
        manifest["failed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        (record_dir / "execution_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    run_dirs = {
        "decomposition": decomposition_dir,
        "gate_a": regime_dir,
        "support_sensitivity": sensitivity_dir,
    }
    for label, run_dir in run_dirs.items():
        run_manifest_path = run_dir / "run_manifest.json"
        if not run_manifest_path.is_file():
            _fail("OUTPUT_VALIDATION_FAILED", f"missing {run_manifest_path}")
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest.get("status") != "COMPLETE":
            _fail("OUTPUT_VALIDATION_FAILED", f"{label} status is not COMPLETE")
        if not run_manifest.get("outputs"):
            _fail("OUTPUT_VALIDATION_FAILED", f"{label} has no recorded outputs")
        if run_manifest.get("git_commit") != expected_commit:
            _fail(
                "OUTPUT_VALIDATION_FAILED",
                f"{label} git_commit={run_manifest.get('git_commit')!r}; expected {expected_commit}",
            )

    decomposition_metrics = pd.read_csv(decomposition_dir / "metrics.csv")
    required_columns = {
        "phase",
        "support_fraction",
        "common_support_nodes",
        "union_support_nodes",
        "intersection_over_union_fraction",
        "masks_identical",
        "realizations",
    }
    missing = required_columns - set(decomposition_metrics.columns)
    if missing:
        _fail(
            "OUTPUT_VALIDATION_FAILED",
            f"decomposition metrics missing columns: {sorted(missing)}",
        )
    if set(decomposition_metrics["phase"]) != set(PHASES):
        _fail("OUTPUT_VALIDATION_FAILED", "decomposition phase set is incomplete")

    sensitivity_table = pd.read_csv(sensitivity_dir / "support_sensitivity.csv")
    if set(sensitivity_table["policy"]) != EXPECTED_POLICIES:
        _fail(
            "OUTPUT_VALIDATION_FAILED",
            f"unexpected support policies: {sorted(set(sensitivity_table['policy']))}",
        )
    if set(sensitivity_table["phase"]) != set(PHASES):
        _fail("OUTPUT_VALIDATION_FAILED", "sensitivity phase set is incomplete")

    support_columns = [
        "phase",
        "support_fraction",
        "common_support_nodes",
        "union_support_nodes",
        "intersection_over_union_fraction",
        "masks_identical",
        "realizations",
    ]
    sensitivity_columns = [
        "phase",
        "policy",
        "support_evaluable",
        "support_ready",
        "projected_decision",
        "primary_decision",
    ]

    manifest["status"] = "COMPLETE"
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["run_directories"] = {key: str(value) for key, value in run_dirs.items()}
    manifest["decomposition_support_metrics"] = decomposition_metrics[
        support_columns
    ].to_dict(orient="records")
    manifest["primary_gate_a"] = regime_summary["phases"]
    manifest["support_sensitivity_summary"] = sensitivity_summary["phases"]
    (record_dir / "execution_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completion = {
        "status": "COMPLETE",
        "execution_id": execution_id,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "aorticturb_commit": expected_commit,
        "config_sha256": config_sha256,
        "run_directories": manifest["run_directories"],
        "primary_gate_a_unchanged_by_sensitivity": True,
        "canonical_05_results_promoted": False,
    }
    (record_dir / "PASS3_RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if local_root.exists():
        if not str(local_root).startswith("/content/"):
            _fail("LOCAL_CLEANUP_BLOCKED", "refusing to remove path outside /content")
        shutil.rmtree(local_root)

    return {
        "execution_id": execution_id,
        "record_dir": str(record_dir),
        "run_directories": manifest["run_directories"],
        "support_table": decomposition_metrics[support_columns].to_dict(orient="records"),
        "sensitivity_table": sensitivity_table[sensitivity_columns].to_dict(orient="records"),
        "primary_gate_a": regime_summary["phases"],
        "support_sensitivity_summary": sensitivity_summary["phases"],
        "canonical_05_results_promoted": False,
    }
