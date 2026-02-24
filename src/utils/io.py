"""File utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json(data: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
