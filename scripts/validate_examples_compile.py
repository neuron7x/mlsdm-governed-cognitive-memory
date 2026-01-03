#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import py_compile
import sys


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples"
    if not examples_dir.exists():
        print("examples directory not found", file=sys.stderr)
        return 1

    failures: list[str] = []
    for path in sorted(examples_dir.glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{path}: {exc.msg}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"Compiled {len(list(examples_dir.glob('*.py')))} example scripts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
