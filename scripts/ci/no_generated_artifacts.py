#!/usr/bin/env python3
"""Fail fast when generated artifacts are committed."""
from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.git_changed_files import list_changed_files

FORBIDDEN_PATTERNS: tuple[str, ...] = (
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    ".coverage",
    ".coverage.*",
    "coverage.xml",
    "coverage.json",
    "htmlcov/**",
    "dist/**",
    "build/**",
    "*.egg-info/**",
    "junit*.xml",
    "artifacts/tmp/**",
)

DB_PATTERNS: tuple[str, ...] = (
    "*.db",
    "*.sqlite",
    "*.sqlite3",
)

ALLOWED_PREFIXES: tuple[str, ...] = (
    "artifacts/evidence/",
    "artifacts/baseline/",
)

DB_ALLOWED_PREFIXES: tuple[str, ...] = (
    "assets/",
    "docs/",
    "examples/",
    "data/",
)

ALLOWED_EXACT: tuple[str, ...] = (
    "artifacts/README.md",
    "artifacts/evidence/README.md",
)


def _is_allowlisted(path: str) -> bool:
    return path in ALLOWED_EXACT or any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _is_allowed_db_file(path: str, db_match: bool | None = None) -> bool:
    if db_match is None:
        db_match = _matches_forbidden(path, DB_PATTERNS)
    return db_match and any(path.startswith(prefix) for prefix in DB_ALLOWED_PREFIXES)


def _matches_forbidden(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def find_forbidden(paths: Iterable[str]) -> set[str]:
    forbidden_files = set()
    for path in paths:
        if _is_allowlisted(path):
            continue
        forbidden_match = _matches_forbidden(path, FORBIDDEN_PATTERNS)
        is_db_file = _matches_forbidden(path, DB_PATTERNS)
        if _is_allowed_db_file(path, is_db_file):
            continue
        if forbidden_match or is_db_file:
            forbidden_files.add(path)
    return forbidden_files


def main() -> int:
    changed_files, refs_checked = list_changed_files()
    if not changed_files:
        refs = ", ".join(refs_checked) if refs_checked else "none"
        print(f"No changed files detected; checked refs: {refs}. Skipping generated artifact check.")
        return 0

    forbidden_files = find_forbidden(changed_files)

    if forbidden_files:
        print("Generated artifacts or local caches detected in changed files:")
        for path in sorted(forbidden_files):
            print(f"- {path}")
        allowed_dirs = ", ".join(sorted(set([*ALLOWED_PREFIXES, *DB_ALLOWED_PREFIXES])))
        print(
            "\nRemove these files from commits or ensure they live in allowed directories "
            f"({allowed_dirs})."
        )
        return 1

    print("No forbidden generated artifacts found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
