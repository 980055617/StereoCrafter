# =============================================
# File: /workspace/stereocraft/utils/config_utils.py
# ---------------------------------------------
# 目的: JSON設定読み込み
# =============================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_config_path(config: str, config_dir: str) -> Path:
    """Resolve a config identifier to an existing JSON file path."""
    config_path = Path(config).expanduser()
    if not config_path.suffix:
        config_path = config_path.with_suffix(".json")
    candidates: list[Path] = []
    if not config_path.is_absolute():
        base_dir = Path(config_dir).expanduser()
        candidates.append(base_dir / config_path)
    candidates.append(config_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Config file '{config}' not found. Searched: {searched}")


def load_json_config(config: str, config_dir: str) -> tuple[Path, dict[str, Any]]:
    """Load a JSON config file and return (path, dict payload)."""
    config_path = resolve_config_path(config, config_dir)
    with open(config_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{config_path}' must contain a JSON object at the top level.")
    return config_path, data
