"""Deterministic seeds and artifact I/O helpers."""
from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
                    encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)}")


def runtime_info() -> dict[str, str]:
    import matplotlib
    import seaborn
    import yaml

    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "seaborn": seaborn.__version__,
            "pyyaml": yaml.__version__}
