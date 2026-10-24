from __future__ import annotations

from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any
import yaml


def _deep_merge(base: dict, update: dict) -> dict:
    out = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _default_config() -> dict[str, Any]:
    resource = files("aorticturb").joinpath("default_config.yaml")
    with resource.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise RuntimeError("Packaged default configuration is invalid")
    return data


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    base = _default_config()
    if path is None:
        return base
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        update = yaml.safe_load(handle)
    if not isinstance(update, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    config = _deep_merge(base, update)
    if tuple(config["analysis"]["phases"]) != ("t018", "t021", "t024"):
        raise ValueError("The canonical study phases are locked to t018,t021,t024")
    if int(config["analysis"]["realizations_per_phase"]) != 16:
        raise ValueError("The canonical study requires 16 realizations per phase")
    return config
