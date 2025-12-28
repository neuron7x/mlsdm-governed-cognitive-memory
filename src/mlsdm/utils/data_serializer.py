import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, cast

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


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories for the target path if they do not exist."""
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _atomic_write(path: Path, suffix: str, writer: Callable[[str], None]) -> None:
    """Write data to a temporary file and atomically replace the target."""
    _ensure_parent_dir(path)
    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix, dir=path.parent, delete=False
    )
    temp_name = temp_file.name
    temp_file.close()
    try:
        writer(temp_name)
        os.replace(temp_name, str(path))
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@retry(stop=stop_after_attempt(3))
def _save_data(data: dict[str, Any], filepath: str) -> None:
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".json":
        def _write_json(temp_name: str) -> None:
            with open(temp_name, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        _atomic_write(path, ".json", _write_json)
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

        def _write_npz(temp_name: str) -> None:
            save_fn(temp_name, **processed)

        _atomic_write(path, ".npz", _write_npz)
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
