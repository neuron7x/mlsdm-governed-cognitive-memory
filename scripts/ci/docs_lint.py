#!/usr/bin/env python3
"""Lightweight docs lint to detect duplicated headings within a file."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOC_PATHS = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "docs"]
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class DocsLintError(RuntimeError):
    """Raised when docs lint checks fail."""


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug


def _iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in DOC_PATHS:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for md_file in sorted(path.rglob("*.md")):
                if "docs/archive" in str(md_file):
                    continue
                files.append(md_file)
    return files


def _check_file(path: Path) -> list[str]:
    failures: list[str] = []
    seen: dict[tuple[str, ...], int] = {}
    in_code_block = False
    heading_stack: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = HEADING_RE.match(line)
        if not match:
            continue
        level = len(match.group(1))
        header_text = match.group(2)
        slug = _slugify(header_text)
        if not slug:
            continue
        while len(heading_stack) >= level:
            heading_stack.pop()
        heading_stack.append(slug)
        key = tuple(heading_stack)
        if key in seen:
            failures.append(
                f"{path}:{lineno} duplicate heading path '{' > '.join(heading_stack)}' "
                f"(first at line {seen[key]})"
            )
        else:
            seen[key] = lineno
    return failures


def lint_docs() -> None:
    failures: list[str] = []
    for path in _iter_markdown_files():
        failures.extend(_check_file(path))
    if failures:
        raise DocsLintError("\n".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lint docs for duplicate headings")
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        lint_docs()
    except DocsLintError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("✓ Docs lint checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
