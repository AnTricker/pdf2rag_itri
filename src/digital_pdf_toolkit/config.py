from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: Path, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config root must be an object: {path}")
    return value


def load_config(root: Path, custom_path: Path | None = None) -> dict[str, Any]:
    """Resolve shipped defaults, optional local settings, and one override."""
    config_dir = root / "config"
    result = load_yaml(config_dir / "default.yml")
    result = deep_merge(result, load_yaml(config_dir / "surya2.yml"))
    result = deep_merge(result, load_yaml(config_dir / "marker.yml"))
    result = deep_merge(result, load_yaml(config_dir / "embedding.yml"))
    result = deep_merge(result, load_yaml(config_dir / "local.yml", required=False))
    if custom_path:
        result = deep_merge(result, load_yaml(custom_path))
    return result
