import json
import os
from typing import Any, Dict

import numpy as np
from tenacity import retry, stop_after_attempt


@retry(stop=stop_after_attempt(3))  # type: ignore[misc]
def _save_data(data: Dict[str, Any], filepath: str) -> None:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif ext == ".npz":
        processed = {k: np.asarray(v) for k, v in data.items()}
        np.savez(filepath, **processed)
    else:
        raise ValueError(f"Unsupported format: {ext}")


@retry(stop=stop_after_attempt(3))  # type: ignore[misc]
def _load_data(filepath: str) -> Dict[str, Any]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            result: Dict[str, Any] = json.load(f)
            return result
    elif ext == ".npz":
        arrs = np.load(filepath, allow_pickle=True)
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in arrs.items()}
    else:
        raise ValueError(f"Unsupported format: {ext}")


class DataSerializer:
    @staticmethod
    def save(data: Dict[str, Any], filepath: str) -> None:
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string.")
        _save_data(data, filepath)

    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        if not isinstance(filepath, str):
            raise TypeError("Filepath must be a string.")
        result: Dict[str, Any] = _load_data(filepath)
        return result
