#!/usr/bin/env python3
"""Export requirements.txt from pyproject.toml dependencies.

This script ensures requirements.txt stays in sync with pyproject.toml.
Run this to regenerate requirements.txt when dependencies change.

Usage:
    python scripts/ci/export_requirements.py
    python scripts/ci/export_requirements.py --check  # CI mode: fail if drift detected
"""
from __future__ import annotations

import argparse
import re
import sys
from itertools import zip_longest

# Python 3.11+ has tomllib in stdlib, earlier versions need tomli backport
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError as e:
        raise ImportError(
            "tomli package is required for Python <3.11. "
            "Install it with: pip install tomli"
        ) from e

from pathlib import Path
from typing import Any, Iterable

# Project root is two levels up from this script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
PREFERRED_OPTIONAL_GROUP_ORDER = [
    "test",
    "dev",
    "docs",
    "observability",
    "embeddings",
    "neurolang",
    "visualization",
]


def _normalize_package_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"[_.]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized


def _normalize_excluded_packages(excluded_packages: dict[str, str]) -> dict[str, str]:
    return {
        _normalize_package_name(name): reason for name, reason in excluded_packages.items()
    }


EXCLUDED_PACKAGES: dict[str, str] = _normalize_excluded_packages(
    {
        "jupyter": "excluded from requirements.txt to avoid pip-audit failures via nbconvert",
        "jupyter_core": "excluded from requirements.txt to avoid pip-audit failures via nbconvert",
    }
)


def load_pyproject(path: Path) -> dict[str, Any]:
    """Load pyproject.toml data using tomllib."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def parse_pyproject_deps(pyproject_data: dict[str, Any]) -> dict[str, Any]:
    """Parse dependencies from pyproject.toml data."""
    project = pyproject_data.get("project", {})
    core_deps = list(project.get("dependencies", []) or [])
    optional_deps = {
        group: list(deps or [])
        for group, deps in (project.get("optional-dependencies", {}) or {}).items()
    }
    return {"core": core_deps, "optional": optional_deps}


def _format_group_list(groups: Iterable[str]) -> str:
    group_list = ", ".join(groups)
    return group_list if group_list else "none"


def _title_case_group(group: str) -> str:
    return group.replace("-", " ").title()


def _normalize_dependency_name(dep: str) -> str:
    name = re.split(r"[<>=!~;\[]", dep, maxsplit=1)[0].strip()
    return _normalize_package_name(name)


def filter_excluded_dependencies(deps: Iterable[str]) -> list[str]:
    return [dep for dep in deps if _normalize_dependency_name(dep) not in EXCLUDED_PACKAGES]


def _format_excluded_packages(excluded_packages: dict[str, str]) -> list[str]:
    if not excluded_packages:
        return ["# Excluded packages (not exported): none"]
    excluded_lines = ["# Excluded packages (not exported):"]
    for name in sorted(excluded_packages):
        excluded_lines.append(f"# - {name}: {excluded_packages[name]}")
    return excluded_lines


def _order_optional_groups(optional_groups: Iterable[str]) -> list[str]:
    remaining = set(optional_groups)
    ordered = [group for group in PREFERRED_OPTIONAL_GROUP_ORDER if group in remaining]
    remaining.difference_update(ordered)
    ordered.extend(sorted(remaining))
    return ordered


def _normalize_requirement(dep: str) -> str:
    dep = dep.strip()
    name_part = re.split(r"[<>=!~;\[]", dep, maxsplit=1)[0]
    normalized_name = _normalize_package_name(name_part)
    remainder = dep[len(name_part) :].strip().lower()
    return f"{normalized_name}{remainder}"


def _ensure_trailing_newline(content: str) -> str:
    normalized = content.replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def generate_requirements(deps: dict[str, Any]) -> str:
    """Generate requirements.txt content from parsed dependencies."""
    optional_groups = _order_optional_groups(deps["optional"].keys())
    group_list = _format_group_list(optional_groups)
    header = """\
# GENERATED FILE - DO NOT EDIT MANUALLY
# This file is auto-generated from pyproject.toml dependencies.
# Regenerate with: python scripts/ci/export_requirements.py
#
# MLSDM Full Installation Requirements
#
# This file includes core dependencies and all optional dependency groups
# discovered in pyproject.toml (deduplicated across groups).
# Optional dependency groups discovered in pyproject.toml (ordered): {group_list}
# Optional dependency groups included in this file: all ({group_list})
#
# For minimal installation: pip install -e .
# For embeddings support: pip install -e ".[embeddings]"
# For full dev install: pip install -r requirements.txt
# Security floor pins for indirect dependencies are included at the end.
#
""".format(group_list=group_list)
    lines = [header]
    lines.extend(_format_excluded_packages(EXCLUDED_PACKAGES))
    lines.append("")

    lines.append("# Core Dependencies (from pyproject.toml [project.dependencies])")
    seen: set[str] = set()
    for dep in sorted(deps["core"], key=str.lower):
        normalized_dep = _normalize_requirement(dep)
        if normalized_dep in seen:
            continue
        seen.add(normalized_dep)
        lines.append(dep)
    lines.append("")

    for group in optional_groups:
        title = _title_case_group(group)
        lines.append(
            f"# Optional {title} (from pyproject.toml [project.optional-dependencies].{group})"
        )
        lines.append(f"# Install with: pip install \".[{group}]\"")
        for dep in sorted(filter_excluded_dependencies(deps["optional"][group]), key=str.lower):
            normalized_dep = _normalize_requirement(dep)
            if normalized_dep in seen:
                continue
            seen.add(normalized_dep)
            lines.append(dep)
        lines.append("")

    lines.append("# Security: Pin minimum versions for indirect dependencies with known vulnerabilities")
    for dep in [
        "certifi>=2025.11.12",
        "cryptography>=46.0.3",
        "jinja2>=3.1.6",
        "urllib3>=2.6.2",
        "setuptools>=80.9.0",
        "idna>=3.11",
    ]:
        normalized_dep = _normalize_requirement(dep)
        if normalized_dep in seen:
            continue
        seen.add(normalized_dep)
        lines.append(dep)
    lines.append("")

    return _ensure_trailing_newline("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Export requirements.txt from pyproject.toml")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check mode: fail if requirements.txt differs from generated",
    )
    args = parser.parse_args()

    if not PYPROJECT_PATH.exists():
        print(f"ERROR: pyproject.toml not found at {PYPROJECT_PATH}", file=sys.stderr)
        return 1

    pyproject_data = load_pyproject(PYPROJECT_PATH)
    deps = parse_pyproject_deps(pyproject_data)
    generated = generate_requirements(deps)

    if args.check:
        if not REQUIREMENTS_PATH.exists():
            print("ERROR: requirements.txt does not exist", file=sys.stderr)
            return 1

        current = REQUIREMENTS_PATH.read_text(encoding="utf-8")
        current_normalized = _ensure_trailing_newline(current)
        generated_normalized = _ensure_trailing_newline(generated)

        if current_normalized != generated_normalized:
            print("ERROR: Dependency drift detected!", file=sys.stderr)
            print("", file=sys.stderr)
            print("requirements.txt is out of sync with pyproject.toml", file=sys.stderr)
            print("Run: python scripts/ci/export_requirements.py", file=sys.stderr)
            print("", file=sys.stderr)

            current_lines = current_normalized.splitlines()
            generated_lines = generated_normalized.splitlines()
            for idx, (cur, gen) in enumerate(zip_longest(current_lines, generated_lines, fillvalue="")):
                if cur != gen:
                    print(f"First difference at line {idx + 1}:", file=sys.stderr)
                    print(f"  expected: {gen or '<missing>'}", file=sys.stderr)
                    print(f"  found:    {cur or '<missing>'}", file=sys.stderr)
                    break
            return 1

        print("✓ requirements.txt is in sync with pyproject.toml")
        return 0

    # Write mode
    REQUIREMENTS_PATH.write_text(generated, encoding="utf-8")
    print(f"✓ Generated {REQUIREMENTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
