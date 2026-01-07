"""
Fail-fast scanner for hidden/bidirectional Unicode control characters.

Intended for CI guard to prevent accidental introduction of bidi controls.
Scans tracked files (git ls-files) and exits non-zero on finding disallowed
code points.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.git_changed_files import list_changed_files, repo_root as repo_root_path

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


def find_bidi_issues(text: str) -> list[str]:
    failures: list[str] = []
    for ch, name in BIDI_CODEPOINTS.items():
        if ch in text:
            failures.append(f"contains {name} (U+{ord(ch):04X})")
    return failures


def main() -> int:
    repo_root = repo_root_path()
    changed_files, refs_checked = list_changed_files()
    if not changed_files:
        refs = ", ".join(refs_checked) if refs_checked else "none"
        print(f"No changed files detected; checked refs: {refs}. Skipping bidirectional control scan.")
        return 0

    failures: list[str] = []
    missing_on_disk: list[str] = []
    for rel in changed_files:
        path = repo_root / rel
        if path.is_dir():
            continue
        if not path.exists():
            missing_on_disk.append(rel)
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary or non-utf8

        for issue in find_bidi_issues(text):
            failures.append(f"{rel}: {issue}")

    if missing_on_disk:
        print(
            "Warning: git reported changed files that are missing on disk: "
            + ", ".join(sorted(missing_on_disk))
        )

    if failures:
        print("ERROR: Disallowed bidirectional control characters found in changed files:")
        for line in failures:
            print(f" - {line}")
        return 1

    print("✓ No bidirectional control characters detected in changed files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
