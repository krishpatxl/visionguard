from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def _json_safe(x: Any) -> Any:
    # Make numpy / pandas types JSON serializable without importing them
    t = type(x).__name__.lower()
    if "int" in t or "uint" in t:
        try:
            return int(x)
        except Exception:
            return x
    if "float" in t:
        try:
            return float(x)
        except Exception:
            return x
    if t in ("ndarray",):
        try:
            return x.tolist()
        except Exception:
            return str(x)
    return x


def _sanitize(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in payload.items():
        if isinstance(v, dict):
            out[k] = _sanitize(v)
        elif isinstance(v, list):
            out[k] = [_json_safe(i) for i in v]
        else:
            out[k] = _json_safe(v)
    return out


class EventLogger:
    def __init__(self, out_dir: str = "outputs", filename: str = "events.jsonl"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename

    def write(self, event_type: str, payload: Dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "type": event_type,
            **_sanitize(payload),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")