#!/usr/bin/env python3
"""Reject f-string logging to enforce structured logging patterns."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_ROOTS = [PROJECT_ROOT / "src"]
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


class LoggingHygieneError(RuntimeError):
    """Raised when logging hygiene validation fails."""


def _is_logger_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in LOG_METHODS
        and isinstance(func.value, ast.Name)
        and func.value.id == "logger"
    )


def _check_file(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_logger_call(node):
            if node.args and isinstance(node.args[0], ast.JoinedStr):
                failures.append(f"{path}:{node.lineno} uses f-string logging")
    return failures


def check_logging() -> None:
    failures: list[str] = []
    for root in SOURCE_ROOTS:
        for path in root.rglob("*.py"):
            failures.extend(_check_file(path))
    if failures:
        raise LoggingHygieneError("\n".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate logging hygiene")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        check_logging()
    except (LoggingHygieneError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("✓ Logging hygiene checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
