#!/usr/bin/env python3
"""Verify deterministic installs and lockfile alignment."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class DeterminismError(RuntimeError):
    """Raised when deterministic install checks fail."""


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise DeterminismError(f"Command failed: {' '.join(command)}")


def _ensure_uv() -> None:
    if shutil.which("uv") is None:
        raise DeterminismError("uv is required for determinism checks. Install with: pip install uv")


def check_determinism() -> None:
    _ensure_uv()
    _run(["uv", "lock", "--check"])
    _run(["python", "scripts/ci/export_requirements.py", "--check"])
    _run(["uv", "sync", "--frozen", "--all-extras"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify deterministic installs and lockfile alignment")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        check_determinism()
    except DeterminismError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("✓ Determinism checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
