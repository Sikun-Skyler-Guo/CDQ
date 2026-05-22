from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def load_env_file(path: Path) -> Dict[str, str]:
    """
    Minimal .env loader (avoids importing python-dotenv).
    Lines follow KEY=VALUE (value may be quoted). Comments start with '#'.
    """
    env_vars: Dict[str, str] = {}
    if not path.exists():
        return env_vars

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        env_vars[key] = value
    return env_vars
