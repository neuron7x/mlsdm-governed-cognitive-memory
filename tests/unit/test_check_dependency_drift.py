#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import check_dependency_drift  # noqa: E402


def _write_pyproject(path: Path, *, deps: list[str], optional: dict[str, list[str]], allowlist: list[str]) -> None:
    optional_blocks = "\n".join(
        f'{group} = [\n' + ",\n".join(f'    "{item}"' for item in items) + "\n]" for group, items in optional.items()
    )
    content = f"""
[project]
dependencies = [
{",".join(f'"{d}"' for d in deps)}
]

[project.optional-dependencies]
{optional_blocks}

[tool.mlsdm.dependency_drift]
allowlist = [
{",".join(f'"{a}"' for a in allowlist)}
]
checked_optional_groups = [
{",".join(f'"{g}"' for g in optional)}
]
"""
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _write_requirements(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_drift_check_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pyproject = tmp_path / "pyproject.toml"
    requirements = tmp_path / "requirements.txt"
    deps = ["foo>=1.0"]
    optional = {"extras": ["bar>=2.0"]}
    allowlist = ["baz>=3.0"]
    _write_pyproject(pyproject, deps=deps, optional=optional, allowlist=allowlist)
    _write_requirements(requirements, ["foo>=1.0", "bar>=2.0", "baz>=3.0"])

    check_dependency_drift.PYPROJECT = pyproject
    check_dependency_drift.REQUIREMENTS = requirements

    rc = check_dependency_drift.main()
    captured = capsys.readouterr()

    assert rc == 0
    assert "drift check passed" in captured.out.lower()


def test_drift_check_fail_missing_dep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pyproject = tmp_path / "pyproject.toml"
    requirements = tmp_path / "requirements.txt"
    deps = ["foo>=1.0"]
    optional = {"extras": ["bar>=2.0"]}
    allowlist = ["baz>=3.0"]
    _write_pyproject(pyproject, deps=deps, optional=optional, allowlist=allowlist)
    _write_requirements(requirements, ["foo>=1.0", "baz>=3.0"])  # missing bar

    check_dependency_drift.PYPROJECT = pyproject
    check_dependency_drift.REQUIREMENTS = requirements

    rc = check_dependency_drift.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "missing from requirements.txt" in captured.out.lower()
    assert "bar>=2.0" in captured.out
