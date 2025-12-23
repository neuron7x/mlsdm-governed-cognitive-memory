"""Detect dependency drift between pyproject.toml, requirements.txt, and uv.lock."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def _load_requirements(requirements_path: Path) -> list[Requirement]:
    lines = requirements_path.read_text().splitlines()
    requirements: list[Requirement] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirements.append(Requirement(stripped))
    return requirements


def _load_pyproject(pyproject_path: Path) -> tuple[list[Requirement], dict[str, list[Requirement]]]:
    data = tomllib.loads(pyproject_path.read_text())
    project = data.get("project", {})
    base_deps = [Requirement(dep) for dep in project.get("dependencies", [])]
    optional_raw = project.get("optional-dependencies", {}) or {}
    optional_deps = {group: [Requirement(dep) for dep in deps] for group, deps in optional_raw.items()}
    return base_deps, optional_deps


def _load_uv_lock(lock_path: Path) -> dict[str, str]:
    data = tomllib.loads(lock_path.read_text())
    packages: dict[str, str] = {}
    for package in data.get("package", []):
        name = package.get("name")
        version = package.get("version")
        if name and version:
            packages[name.lower()] = version
    return packages


def _ensure_present(
    expected: list[Requirement],
    observed: list[Requirement],
    context: str,
) -> list[str]:
    errors: list[str] = []
    observed_map = {req.name.lower(): req for req in observed}
    for requirement in expected:
        match = observed_map.get(requirement.name.lower())
        if match is None:
            errors.append(f"[{context}] Missing dependency: {requirement}")
            continue
        if str(requirement.specifier) != str(match.specifier):
            errors.append(
                f"[{context}] Specifier drift for {requirement.name}: "
                f"{match.specifier} (observed) vs {requirement.specifier} (expected)"
            )
    return errors


def _check_lock_satisfies(requirements: list[Requirement], locked: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for req in requirements:
        locked_version = locked.get(req.name.lower())
        if locked_version is None:
            errors.append(f"[uv.lock] Missing locked package for requirement: {req}")
            continue
        specifier: SpecifierSet = req.specifier
        if specifier and not specifier.contains(locked_version, prereleases=True):
            errors.append(
                f"[uv.lock] Locked version mismatch for {req.name}: "
                f"{locked_version} not in {specifier}"
            )
    return errors


def check_dependency_drift(pyproject_path: Path, requirements_path: Path, lock_path: Path) -> list[str]:
    base_deps, optional_deps = _load_pyproject(pyproject_path)
    requirements = _load_requirements(requirements_path)
    locked = _load_uv_lock(lock_path)

    errors: list[str] = []
    errors.extend(_ensure_present(base_deps, requirements, "pyproject:base"))
    for group, deps in optional_deps.items():
        errors.extend(_ensure_present(deps, requirements, f"pyproject:optional:{group}"))
    errors.extend(_check_lock_satisfies(requirements, locked))
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    requirements_path = repo_root / "requirements.txt"
    lock_path = repo_root / "uv.lock"

    missing_files = [path for path in (pyproject_path, requirements_path, lock_path) if not path.exists()]
    if missing_files:
        for path in missing_files:
            print(f"✗ Missing file: {path}")
        return 1

    errors = check_dependency_drift(pyproject_path, requirements_path, lock_path)
    if errors:
        print("✗ Dependency drift detected:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("✓ Dependencies are in sync between pyproject.toml, requirements.txt, and uv.lock.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
