import json
import os
from typing import Any, cast

import numpy as np
from tenacity import retry, stop_after_attempt


def _deserialize_npz_value(value: np.ndarray) -> Any:
    """Deserialize numpy values loaded from NPZ files.

    - Preserve wrapped dicts stored as single-element object arrays.
    - Convert numeric arrays to lists for JSON-like compatibility.
    """
    if value.dtype == object:
        if value.size == 1:
            return value.item()
        return value.tolist()
    return value.tolist()


@retry(stop=stop_after_attempt(3))
def _save_data(data: dict[str, Any], filepath: str) -> None:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif ext == ".npz":
        processed = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                processed[key] = value
            elif isinstance(value, dict):
                # Wrap dicts to preserve structure when saving as object arrays.
                processed[key] = np.array([value], dtype=object)
            else:
                processed[key] = np.asarray(value)
        # Use cast to work around numpy's imprecise savez signature
        save_fn = cast("Any", np.savez)
        save_fn(filepath, **processed)
    else:
        raise ValueError(f"Unsupported format: {ext}")


@retry(stop=stop_after_attempt(3))
def _load_data(filepath: str) -> dict[str, Any]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data
    elif ext == ".npz":
        arrs = np.load(filepath, allow_pickle=True)
        # Convert NpzFile to dict - explicit type to satisfy mypy
        result: dict[str, Any] = {
            k: _deserialize_npz_value(v) if isinstance(v, np.ndarray) else v
            for k, v in arrs.items()
        }
        return result
    else:
        raise ValueError(f"Unsupported format: {ext}")


class DataSerializer:
    @staticmethod
    def save(data: dict[str, Any], filepath: str) -> None:
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string.")
        _save_data(data, filepath)

    @staticmethod
    def load(filepath: str) -> dict[str, Any]:
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string.")
        return _load_data(filepath)
