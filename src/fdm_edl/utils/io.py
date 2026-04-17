# SPDX-License-Identifier: GPL-3.0-or-later
"""I/O helpers for loading user model-parameter files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


def load_dict(file_path: str | Path) -> dict[str, Any]:
    """
    Load model parameters from a JSON or YAML file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to a ``.json``, ``.yml``, or ``.yaml`` parameter file.

    Returns
    -------
    dict
        Dictionary of model parameters decoded from the file.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not point to an existing file.
    ValueError
        If the file extension is not supported, or if the decoded content
        is not a dictionary.
    ImportError
        If a YAML file is requested and *PyYAML* is not installed.
    """
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
