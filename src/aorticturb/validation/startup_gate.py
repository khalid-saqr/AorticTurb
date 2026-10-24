from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

from ..data.verify import verify_dataset


@dataclass(frozen=True)
class StartupGate:
    passed: bool
    dataset_verification: dict
    metadata_integrity_flag: bool | None
    agents_present: bool

    def to_dict(self) -> dict:
        return asdict(self)


def run_startup_gate(root: str | Path, *, verify_checksums: bool = True) -> StartupGate:
    root = Path(root)
    verification = verify_dataset(root, verify_checksums=verify_checksums)

    metadata_integrity = None
    metadata_path = root / "METADATA.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_integrity = metadata.get("integrity_verified")

    agents_present = (root / "AGENTS.md").exists()
    passed = (
        verification.ok
        and agents_present
        and metadata_integrity is not False
    )

    return StartupGate(
        passed=passed,
        dataset_verification=verification.to_dict(),
        metadata_integrity_flag=metadata_integrity,
        agents_present=agents_present,
    )
