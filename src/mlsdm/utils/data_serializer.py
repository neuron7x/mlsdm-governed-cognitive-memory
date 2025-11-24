import json
import os
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from tenacity import retry, stop_after_attempt


def save_arrays(filepath: str, arrays: Mapping[str, NDArray[Any]]) -> None:
    """
    Save a mapping of named NumPy arrays to an .npz file.

    This is a typed wrapper around np.savez that handles the type system properly.
    The **dict unpacking is runtime-safe because numpy.savez accepts keyword arguments
    for array data, even though the type stubs don't perfectly reflect this pattern.

    Args:
        filepath: Path where the .npz file will be saved
        arrays: Mapping of array names to numpy arrays

    Note:
        The type: ignore on the np.savez call is justified because:
        1. numpy's type stubs define savez with explicit keyword params
        2. Runtime behavior correctly handles **dict unpacking of arrays
        3. This wrapper ensures type safety at the API boundary
        4. Tests verify the runtime behavior matches expectations
    """
    np.savez(filepath, **dict(arrays))  # type: ignore[arg-type]


@retry(stop=stop_after_attempt(3))
def _save_data(data: dict[str, Any], filepath: str) -> None:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif ext == ".npz":
        processed = {k: np.asarray(v) for k, v in data.items()}
        save_arrays(filepath, processed)
    else:
        raise ValueError(f"Unsupported format: {ext}")


@retry(stop=stop_after_attempt(3))
def _load_data(filepath: str) -> dict[str, Any]:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".json":
        with open(filepath, encoding="utf-8") as f:
            loaded: dict[str, Any] = json.load(f)
            return loaded
    elif ext == ".npz":
        arrs = np.load(filepath, allow_pickle=True)
        result: dict[str, Any] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in arrs.items()
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
