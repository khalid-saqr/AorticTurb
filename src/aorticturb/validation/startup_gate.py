from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

from ..constants import PHASES
from ..data.verify import verify_dataset
from ..data.layout import provenance_manifest_paths, geometry_paths


@dataclass(frozen=True)
class StartupGate:
    passed: bool
    dataset_verification: dict
    metadata_integrity_flag: bool | None
    agents_present: bool
    contract_checks: dict[str, bool]

    def to_dict(self) -> dict:
        return asdict(self)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_startup_gate(root: str | Path, *, verify_checksums: bool = True, deep_content_check: bool = True) -> StartupGate:
    root = Path(root)
    verification = verify_dataset(root, verify_checksums=verify_checksums, deep_content_check=deep_content_check)
    metadata_integrity = None
    metadata_path = root / "METADATA.json"
    metadata = {}
    if metadata_path.exists():
        metadata = _load_json(metadata_path)
        metadata_integrity = metadata.get("integrity_verified")
    manifests = provenance_manifest_paths(root)
    acquisition = _load_json(manifests["acquisition"]) if manifests["acquisition"].exists() else {}
    pass5 = _load_json(manifests["pass5"]) if manifests["pass5"].exists() else {}
    selection = acquisition.get("selection", {})
    contract_checks = {
        "metadata_integrity_verified": metadata_integrity is True,
        "pass5_integrity_verified": pass5.get("integrity") is True or pass5.get("status") == "COMPLETE_AND_VERIFIED",
        "selection_geometry_M": selection.get("geometry") == "M",
        "selection_domain_AAo": selection.get("domain") == "AAo",
        "selection_phases_exact": tuple(selection.get("phases", [])) == PHASES,
        "selection_realizations_16": selection.get("realizations_per_phase") == 16,
        "canonical_M_geometry_present": geometry_paths(root)["M"].exists(),
        "raw_transformations_absent": metadata.get("velocity_acquisition", {}).get("raw_transformations_performed") is False,
        "source_nan_mask_preserved": metadata.get("velocity_acquisition", {}).get("source_nan_mask_preserved") is True,
    }
    agents_present = (root / "AGENTS.md").exists()
    passed = verification.ok and agents_present and all(contract_checks.values())
    return StartupGate(passed=passed, dataset_verification=verification.to_dict(), metadata_integrity_flag=metadata_integrity, agents_present=agents_present, contract_checks=contract_checks)
