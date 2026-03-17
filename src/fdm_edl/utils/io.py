# SPDX-License-Identifier: GPL-3.0-or-later
"""I/O helpers for loading user model-parameter files."""

import json
from pathlib import Path
from typing import Any


def load_dict(file_path: str | Path) -> dict[str, Any]:
    """Load model parameters from a JSON or YAML file into a dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "YAML parsing requires PyYAML. Install it with `pip install pyyaml`."
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. Use .json, .yml, or .yaml."
        )

    if not isinstance(data, dict):
        raise ValueError("Model parameter file must decode to a dictionary.")
    return data


__all__ = ["load_dict"]
