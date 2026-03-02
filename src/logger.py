import json
import time
from pathlib import Path
from typing import Any, Dict


def _json_safe(obj: Any) -> Any:
    """
    Converts numpy/scalar-ish types into plain Python types so json.dumps won't choke.
    """
    # numpy scalars typically support .item()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    return obj


class EventLogger:
    """
    Writes one JSON object per line. Easy to tail, grep, or load later.
    """
    def __init__(self, path: str = "outputs/events.jsonl"):
        Path("outputs").mkdir(exist_ok=True)
        self.path = path

    def write(self, event_type: str, payload: Dict[str, Any]) -> None:
        row = _json_safe({"ts": time.time(), "type": event_type, **payload})
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")