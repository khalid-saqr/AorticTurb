from pathlib import Path
import zipfile
import pytest

from aorticturb.data.download import _safe_extract


def test_zip_slip_is_rejected(tmp_path: Path):
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as z:
        z.writestr("../escape.txt", "no")
    with zipfile.ZipFile(archive_path) as z:
        with pytest.raises(RuntimeError):
            _safe_extract(z, tmp_path / "out")
