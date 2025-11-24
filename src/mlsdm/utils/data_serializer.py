import json
import os
from typing import Any

import numpy as np
from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(3))
def _save_data(data: dict[str, Any], filepath: str) -> None:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif ext == ".npz":
        processed = {k: np.asarray(v) for k, v in data.items()}
        # Type ignore: numpy.savez has imprecise type signature for **kwargs
        np.savez(filepath, **processed)  # type: ignore[arg-type]
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
            k: v.tolist() if isinstance(v, np.ndarray) else v
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
