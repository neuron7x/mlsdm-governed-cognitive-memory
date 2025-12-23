from pathlib import Path

import pytest

from scripts.check_dependency_drift import check_dependency_drift


def _write_sample_files(tmp_path: Path, pyproject: str, requirements: str, lock: str) -> tuple[Path, Path, Path]:
    py_path = tmp_path / "pyproject.toml"
    req_path = tmp_path / "requirements.txt"
    lock_path = tmp_path / "uv.lock"
    py_path.write_text(pyproject)
    req_path.write_text(requirements)
    lock_path.write_text(lock)
    return py_path, req_path, lock_path


def test_dependency_drift_passes_when_in_sync(tmp_path: Path) -> None:
    pyproject = """
[project]
dependencies = ["fastapi>=0.115.0"]
[project.optional-dependencies]
dev = ["pytest>=8.3.0"]
"""
    requirements = "fastapi>=0.115.0\npytest>=8.3.0\n"
    uv_lock = """
version = 1
revision = 3

[[package]]
name = "fastapi"
version = "0.115.0"

[[package]]
name = "pytest"
version = "8.3.0"
"""
    py_path, req_path, lock_path = _write_sample_files(tmp_path, pyproject, requirements, uv_lock)

    assert check_dependency_drift(py_path, req_path, lock_path) == []


def test_dependency_drift_reports_mismatch(tmp_path: Path) -> None:
    pyproject = """
[project]
dependencies = ["fastapi>=0.115.0"]
"""
    requirements = "fastapi>=0.115.0\n"
    uv_lock = """
version = 1
revision = 3

[[package]]
name = "fastapi"
version = "0.114.0"
"""
    py_path, req_path, lock_path = _write_sample_files(tmp_path, pyproject, requirements, uv_lock)

    errors = check_dependency_drift(py_path, req_path, lock_path)

    assert not any("Specifier drift" in message for message in errors)
    assert any("not in" in message for message in errors)
