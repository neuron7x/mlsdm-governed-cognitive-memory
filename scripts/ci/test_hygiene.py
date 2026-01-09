#!/usr/bin/env python3
"""Ensure pytest skips/xfails include reasons with issue links."""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_ROOT = PROJECT_ROOT / "tests"
ISSUE_LINK_PATTERN = re.compile(r"https?://")

SKIP_MARKS = {"skip", "skipif", "xfail"}
SKIP_FUNCTIONS = {"skip", "xfail"}


class HygieneError(RuntimeError):
    """Raised when test hygiene validation fails."""


def _is_pytest_mark_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in SKIP_MARKS
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "mark"
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "pytest"
    )


def _is_pytest_skip_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in SKIP_FUNCTIONS
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )


def _extract_reason(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "reason" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    if node.args:
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            return first_arg.value
    return None


def _check_file(path: Path) -> list[str]:
    failures: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (_is_pytest_mark_call(node) or _is_pytest_skip_call(node)):
            reason = _extract_reason(node)
            if not reason:
                failures.append(f"{path}: Missing reason for pytest skip/xfail")
                continue
            if not ISSUE_LINK_PATTERN.search(reason):
                failures.append(f"{path}: Reason missing issue link: {reason!r}")
    return failures


def check_hygiene() -> None:
    failures: list[str] = []
    for path in TEST_ROOT.rglob("*.py"):
        failures.extend(_check_file(path))
    if failures:
        raise HygieneError("\n".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pytest skip/xfail hygiene")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        check_hygiene()
    except (HygieneError, SyntaxError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("✓ Test hygiene checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
