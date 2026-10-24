from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import requests

from .verify import verify_dataset


def _load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: zipfile.ZipFile, output: Path) -> None:
    root = output.resolve()
    for member in archive.infolist():
        target = (output / member.filename).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Unsafe archive member path: {member.filename}")
    archive.extractall(output)


def fetch_dataset(output: str | Path, manifest_path: str | Path, *, record_id: int | None = None, archive_name: str | None = None, timeout: int = 120) -> Path:
    manifest = _load_manifest(manifest_path)
    curated = manifest["curated_zenodo"]
    record_id = record_id or curated.get("record_id")
    archive_name = archive_name or curated.get("archive_name")
    if not record_id:
        raise RuntimeError("The curated Zenodo dataset has not been published in dataset_manifest.json. Provide --record-id explicitly or update the manifest after publication.")
    api_url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(api_url, timeout=timeout)
    response.raise_for_status()
    record = response.json()
    files = record.get("files", [])
    candidates = [f for f in files if f.get("key") == archive_name] if archive_name else [f for f in files if str(f.get("key", "")).lower().endswith(".zip")]
    if len(candidates) != 1:
        raise RuntimeError(f"Could not uniquely resolve curated archive. Available files: {[f.get('key') for f in files]}")
    selected = candidates[0]
    download_url = selected.get("links", {}).get("content") or selected.get("links", {}).get("self")
    if not download_url:
        raise RuntimeError("Zenodo record does not expose a downloadable content link.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_path = output.parent / selected["key"]
    with requests.get(download_url, stream=True, timeout=timeout) as stream:
        stream.raise_for_status()
        with archive_path.open("wb") as handle:
            shutil.copyfileobj(stream.raw, handle)
    expected_archive_sha = curated.get("archive_sha256")
    if expected_archive_sha:
        observed = _sha256(archive_path)
        if observed.lower() != expected_archive_sha.lower():
            archive_path.unlink(missing_ok=True)
            raise RuntimeError(f"Archive SHA-256 mismatch: expected {expected_archive_sha}, got {observed}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract(archive, output)
    dataset_root = output
    if not (dataset_root / "METADATA.json").exists():
        roots = [p for p in dataset_root.iterdir() if p.is_dir() and (p / "METADATA.json").exists()]
        if len(roots) == 1:
            dataset_root = roots[0]
    result = verify_dataset(dataset_root, verify_checksums=True, deep_content_check=True)
    if not result.ok:
        raise RuntimeError("Extracted dataset failed verification: " + "; ".join(result.errors))
    return dataset_root
