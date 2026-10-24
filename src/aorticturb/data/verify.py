from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

from ..constants import BYTES_PER_VELOCITY_FILE, PHASES, REALIZATIONS
from .layout import expected_geometry_paths, phase_dir, provenance_manifest_paths


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    velocity_files: int
    geometry_files: int

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relpath = line.split(maxsplit=1)
        records[relpath.strip().lstrip("*")] = digest.lower()
    return records


def verify_dataset(root: str | Path, *, verify_checksums: bool = True) -> VerificationResult:
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    velocity_count = 0

    for phase in PHASES:
        folder = phase_dir(root, phase)
        if not folder.is_dir():
            errors.append(f"Missing phase directory: {folder}")
            continue
        present = sorted(p.name for p in folder.glob("*.npy"))
        expected = list(REALIZATIONS)
        if present != expected:
            missing = sorted(set(expected) - set(present))
            extra = sorted(set(present) - set(expected))
            if missing:
                errors.append(f"{phase}: missing {missing}")
            if extra:
                errors.append(f"{phase}: unexpected {extra}")
        for name in expected:
            path = folder / name
            if not path.exists():
                continue
            velocity_count += 1
            if path.stat().st_size != BYTES_PER_VELOCITY_FILE:
                errors.append(
                    f"{path}: expected {BYTES_PER_VELOCITY_FILE} bytes, "
                    f"got {path.stat().st_size}"
                )

    geometry_paths = expected_geometry_paths(root)
    geometry_count = sum(p.exists() for p in geometry_paths)
    for path in geometry_paths:
        if not path.exists():
            errors.append(f"Missing geometry: {path}")

    for required in ("METADATA.json", "AGENTS.md"):
        if not (root / required).exists():
            errors.append(f"Missing required project file: {root / required}")

    for label, path in provenance_manifest_paths(root).items():
        if not path.exists():
            errors.append(f"Missing required provenance manifest ({label}): {path}")

    checksum_path = root / "SHA256SUMS.txt"
    checksums = parse_sha256sums(checksum_path)
    if verify_checksums:
        if not checksum_path.exists():
            errors.append(
                "Missing package-level SHA256SUMS.txt. The curated Zenodo archive must "
                "ship a checksum ledger for its exact extracted layout."
            )
        else:
            for relpath, expected in checksums.items():
                path = root / relpath
                if not path.exists():
                    errors.append(f"Checksum ledger references missing file: {relpath}")
                    continue
                observed = sha256_file(path)
                if observed.lower() != expected.lower():
                    errors.append(f"SHA-256 mismatch: {relpath}")
    elif not checksum_path.exists():
        warnings.append("SHA256SUMS.txt not found; checksum verification was not requested.")

    return VerificationResult(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        velocity_files=velocity_count,
        geometry_files=geometry_count,
    )
