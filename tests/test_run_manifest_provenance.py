from __future__ import annotations

import json

from aorticturb.reporting import run_manifest


class _Distribution:
    def __init__(self, payload: dict | None):
        self.payload = payload

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return None if self.payload is None else json.dumps(self.payload)


def test_installed_vcs_commit_from_pep610(monkeypatch):
    commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setattr(
        run_manifest.importlib_metadata,
        "distribution",
        lambda name: _Distribution({"vcs_info": {"vcs": "git", "commit_id": commit}}),
    )
    assert run_manifest._installed_vcs_commit() == commit


def test_git_commit_prefers_installed_vcs_commit(monkeypatch):
    commit = "89abcdef0123456789abcdef0123456789abcdef"
    monkeypatch.setattr(run_manifest, "_installed_vcs_commit", lambda: commit)

    def _unexpected_subprocess(*args, **kwargs):
        raise AssertionError("git subprocess fallback must not run when PEP-610 commit is available")

    monkeypatch.setattr(run_manifest.subprocess, "run", _unexpected_subprocess)
    assert run_manifest._git_commit() == commit


def test_installed_vcs_commit_missing_metadata(monkeypatch):
    monkeypatch.setattr(run_manifest.importlib_metadata, "distribution", lambda name: _Distribution(None))
    assert run_manifest._installed_vcs_commit() is None
