from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def natural_key(path: Path) -> list[tuple[int, Any]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]
