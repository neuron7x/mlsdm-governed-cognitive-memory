#!/usr/bin/env python3
"""Fail the build if requirements.txt diverges from declared dependencies.

The canonical source of truth is pyproject.toml. We allow an explicit
allowlist for requirements-only pins under [tool.mlsdm.dependency_drift.allowlist].
"""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"


def _normalize(lines: list[str]) -> set[str]:
    normalized: set[str] = set()
    for raw in lines:
        candidate = raw.split("#", 1)[0].strip()
        if candidate:
            normalized.add(candidate)
    return normalized


def main() -> int:
    with PYPROJECT.open("rb") as fp:
        pyproject = tomllib.load(fp)
    project = pyproject["project"]

    declared = _normalize(list(project.get("dependencies", [])))

    drift_cfg = pyproject.get("tool", {}).get("mlsdm", {}).get("dependency_drift", {})
    optional_groups = project.get("optional-dependencies", {})
    groups_to_check = drift_cfg.get("checked_optional_groups") or optional_groups.keys()

    for group_name in groups_to_check:
        extra = optional_groups.get(group_name, [])
        declared.update(_normalize(list(extra)))

    allowlist = drift_cfg.get("allowlist", [])
    declared.update(_normalize(list(allowlist)))

    recorded = _normalize(REQUIREMENTS.read_text().splitlines())

    missing_in_requirements = declared - recorded
    extras_in_requirements = recorded - declared

    if missing_in_requirements or extras_in_requirements:
        print("Dependency drift detected between pyproject.toml and requirements.txt")
        if missing_in_requirements:
            print("  Missing from requirements.txt:")
            for dep in sorted(missing_in_requirements):
                print(f"    - {dep}")
        if extras_in_requirements:
            print("  Unexpected in requirements.txt (not declared in pyproject):")
            for dep in sorted(extras_in_requirements):
                print(f"    - {dep}")
        return 1

    print("Dependency drift check passed: requirements.txt matches pyproject.toml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
