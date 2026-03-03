from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List

import yaml


def _as_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def load_settings(path: str | Path = "configs/settings.yaml") -> Dict[str, Any]:
    path = _as_path(str(path))
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}