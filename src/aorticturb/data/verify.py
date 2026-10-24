from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import numpy as np

from ..constants import BYTES_PER_VELOCITY_FILE, PHASES, REALIZATIONS
from .layout import expected_geometry_paths, expected_velocity_paths, phase_dir, provenance_manifest_paths, final_acquired_checksum_path
from .loader import load_velocity_field, split_field


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    velocity_files: int
    geometry_files: int
    deep_content_checked: bool

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
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed checksum line: {line}")
        digest, relpath = parts
        records[relpath.strip().lstrip("*")] = digest.lower()
    return records


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def verify_dataset(root: str | Path, *, verify_checksums: bool = True, deep_content_check: bool = True) -> VerificationResult:
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    velocity_count = 0
    expected_velocity = expected_velocity_paths(root)
    expected_velocity_rel = {_rel(root, p) for p in expected_velocity}
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
                errors.append(f"{path}: expected {BYTES_PER_VELOCITY_FILE} bytes, got {path.stat().st_size}")
    raw_root = root / "01_raw_data"
    if raw_root.exists():
        actual_velocity_rel = {_rel(root, p) for p in raw_root.rglob("*.npy")}
        unexpected = sorted(actual_velocity_rel - expected_velocity_rel)
        if unexpected:
            errors.append(f"Unexpected raw NPY files outside canonical subset: {unexpected}")
    geometry_paths = expected_geometry_paths(root)
    geometry_count = sum(p.exists() for p in geometry_paths)
    for path in geometry_paths:
        if not path.exists():
            errors.append(f"Missing geometry: {path}")
    for required in ("METADATA.json", "AGENTS.md"):
        if not (root / required).exists():
            errors.append(f"Missing required project file: {root / required}")
    manifests = provenance_manifest_paths(root)
    for label, path in manifests.items():
        if not path.exists():
            errors.append(f"Missing required provenance manifest ({label}): {path}")
    historical_ledger_path = final_acquired_checksum_path(root)
    if not historical_ledger_path.exists():
        errors.append(f"Missing historical final-acquired checksum ledger: {historical_ledger_path}")
    if (root / "METADATA.json").exists():
        metadata = _json(root / "METADATA.json")
        va = metadata.get("velocity_acquisition", {})
        if va.get("geometry") != "M" or va.get("domain") != "AAo":
            errors.append("METADATA.json does not identify canonical M/AAo velocity selection")
        if tuple(va.get("selected_phases", [])) != PHASES:
            errors.append("METADATA.json selected phases differ from t018,t021,t024")
        if va.get("realizations_per_phase") != 16 or va.get("total_files") != 48:
            errors.append("METADATA.json realization/file counts differ from canonical subset")
        if va.get("source_nan_mask_preserved") is not True:
            errors.append("METADATA.json does not affirm source NaN-mask preservation")
        expected_agents_sha = metadata.get("provenance", {}).get("agents_md_sha256")
        if expected_agents_sha and (root / "AGENTS.md").exists() and sha256_file(root / "AGENTS.md") != expected_agents_sha:
            errors.append("AGENTS.md SHA-256 differs from METADATA.json provenance")
    if manifests["pass5"].exists():
        pass5 = _json(manifests["pass5"])
        if pass5.get("integrity") is not True and pass5.get("status") != "COMPLETE_AND_VERIFIED":
            errors.append("Pass-5 provenance does not establish complete integrity")
    if manifests["acquisition"].exists():
        acq = _json(manifests["acquisition"])
        selection = acq.get("selection", {})
        if selection.get("geometry") != "M" or selection.get("domain") != "AAo":
            errors.append("Acquisition manifest selection is not M/AAo")
        if tuple(selection.get("phases", [])) != PHASES:
            errors.append("Acquisition manifest phases differ from canonical subset")
        if selection.get("realizations_per_phase") != 16:
            errors.append("Acquisition manifest does not specify 16 realizations per phase")
    checksum_path = root / "SHA256SUMS.txt"
    checksums: dict[str, str] = {}
    if checksum_path.exists():
        try:
            checksums = parse_sha256sums(checksum_path)
        except ValueError as exc:
            errors.append(str(exc))
    required_checksum_paths = expected_velocity_rel | {_rel(root, p) for p in geometry_paths}
    required_checksum_paths |= {"METADATA.json", "AGENTS.md"}
    required_checksum_paths |= {_rel(root, p) for p in manifests.values()}
    required_checksum_paths.add(_rel(root, historical_ledger_path))
    if verify_checksums:
        if not checksum_path.exists():
            errors.append("Missing package-level SHA256SUMS.txt. The curated Zenodo archive must ship an exact checksum ledger.")
        else:
            missing_from_ledger = sorted(required_checksum_paths - set(checksums))
            if missing_from_ledger:
                errors.append(f"Checksum ledger omits required canonical files: {missing_from_ledger}")
            for relpath, expected in checksums.items():
                path = root / relpath
                if not path.exists():
                    errors.append(f"Checksum ledger references missing file: {relpath}")
                    continue
                observed = sha256_file(path)
                if observed.lower() != expected.lower():
                    errors.append(f"SHA-256 mismatch: {relpath}")
            if historical_ledger_path.exists():
                historical = parse_sha256sums(historical_ledger_path)
                for relpath in sorted(expected_velocity_rel):
                    if relpath not in historical:
                        errors.append(f"Historical final-acquired ledger omits velocity file: {relpath}")
                    elif relpath in checksums and checksums[relpath] != historical[relpath]:
                        errors.append(f"Package/source provenance hash disagreement: {relpath}")
                by_basename: dict[str, list[str]] = {}
                for old_path, digest in historical.items():
                    by_basename.setdefault(Path(old_path).name, []).append(digest)
                for path in geometry_paths:
                    relpath = _rel(root, path)
                    source_digests = set(by_basename.get(path.name, []))
                    if len(source_digests) != 1:
                        errors.append(f"Historical final-acquired ledger cannot uniquely resolve geometry {path.name}")
                    elif relpath in checksums and checksums[relpath] not in source_digests:
                        errors.append(f"Package/source provenance hash disagreement: {relpath}")
    elif not checksum_path.exists():
        warnings.append("SHA256SUMS.txt not found; checksum verification was not requested.")
    if deep_content_check:
        reference_xyz = None
        for phase in PHASES:
            representative = phase_dir(root, phase) / REALIZATIONS[0]
            if not representative.exists() or representative.stat().st_size != BYTES_PER_VELOCITY_FILE:
                continue
            try:
                field = load_velocity_field(representative, mmap=True)
                xyz, uvw = split_field(field)
                if field.shape != (982600, 6):
                    errors.append(f"{representative}: expected representative shape (982600,6), got {field.shape}")
                if not np.isfinite(xyz).all():
                    errors.append(f"{representative}: coordinate columns contain non-finite entries")
                finite_components = np.isfinite(uvw)
                partial = finite_components.any(axis=1) & ~finite_components.all(axis=1)
                if partial.any():
                    errors.append(f"{representative}: partially finite velocity rows violate 3C support semantics")
                if reference_xyz is None:
                    reference_xyz = np.asarray(xyz)
                elif not np.array_equal(np.asarray(xyz), reference_xyz):
                    errors.append(f"{representative}: coordinate grid differs across selected phases")
            except Exception as exc:
                errors.append(f"Content validation failed for {representative}: {exc}")
    return VerificationResult(ok=not errors, errors=errors, warnings=warnings, velocity_files=velocity_count, geometry_files=geometry_count, deep_content_checked=deep_content_check)
