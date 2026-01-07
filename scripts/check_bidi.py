"""
Fail-fast scanner for hidden/bidirectional Unicode control characters.

Intended for CI guard to prevent accidental introduction of bidi controls.
Scans tracked files (git ls-files) and exits non-zero on finding disallowed
code points.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BIDI_CODEPOINTS = {
    "\u202a": "LEFT-TO-RIGHT EMBEDDING",
    "\u202b": "RIGHT-TO-LEFT EMBEDDING",
    "\u202c": "POP DIRECTIONAL FORMATTING",
    "\u202d": "LEFT-TO-RIGHT OVERRIDE",
    "\u202e": "RIGHT-TO-LEFT OVERRIDE",
    "\u2066": "LEFT-TO-RIGHT ISOLATE",
    "\u2067": "RIGHT-TO-LEFT ISOLATE",
    "\u2068": "FIRST STRONG ISOLATE",
    "\u2069": "POP DIRECTIONAL ISOLATE",
    "\u200e": "LEFT-TO-RIGHT MARK",
    "\u200f": "RIGHT-TO-LEFT MARK",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ref_exists(ref: str, repo_root: Path) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _changed_files() -> list[str]:
    repo_root = _repo_root()
    base_ref = os.environ.get("GITHUB_BASE_REF")
    candidates = []
    if base_ref:
        candidates.append(f"origin/{base_ref}")
    candidates.extend(["origin/main", "HEAD~1"])

    for ref in candidates:
        if not _ref_exists(ref, repo_root):
            continue
        try:
            diff = subprocess.check_output(
                [
                    "git",
                    "diff",
                    "--name-only",
                    "--diff-filter=ACMRTUXB",
                    f"{ref}...HEAD",
                ],
                cwd=repo_root,
                text=True,
            )
        except subprocess.CalledProcessError:
            diff = ""
        files = [line for line in diff.splitlines() if line.strip()]
        if files:
            return files

    try:
        fallback = subprocess.check_output(
            [
                "git",
                "show",
                "--pretty=format:",
                "--name-only",
                "--diff-filter=ACMRTUXB",
                "HEAD",
            ],
            cwd=repo_root,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line for line in fallback.splitlines() if line.strip()]


def find_bidi_issues(text: str) -> list[str]:
    failures: list[str] = []
    for ch, name in BIDI_CODEPOINTS.items():
        if ch in text:
            failures.append(f"contains {name} (U+{ord(ch):04X})")
    return failures


def main() -> int:
    repo_root = _repo_root()
    changed_files = _changed_files()
    if not changed_files:
        print("No changed files detected; skipping bidirectional control scan.")
        return 0

    failures: list[str] = []
    for rel in changed_files:
        path = repo_root / rel
        if path.is_dir() or not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary or non-utf8

        for issue in find_bidi_issues(text):
            failures.append(f"{rel}: {issue}")

    if failures:
        print("ERROR: Disallowed bidirectional control characters found in diff:")
        for line in failures:
            print(f" - {line}")
        return 1

    print("✓ No bidirectional control characters detected in changed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
