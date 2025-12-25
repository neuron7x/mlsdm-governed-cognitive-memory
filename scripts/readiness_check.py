#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

ROOT = Path(__file__).resolve().parent.parent
READINESS_PATH = ROOT / "docs" / "status" / "READINESS.md"
MAX_AGE_DAYS = 14
LAST_UPDATED_PATTERN = r"Last updated:\s*(\d{4}-\d{2}-\d{2})"
DEFAULT_SCOPED_PREFIXES = ("src/", "tests/", "config/", "deploy/", ".github/workflows/")
SCOPED_PREFIXES = (
    tuple(
        prefix.strip()
        for prefix in os.environ.get("READINESS_SCOPED_PREFIXES", "").split(",")
        if prefix.strip()
    )
    or DEFAULT_SCOPED_PREFIXES
)
MAX_LISTED_FILES = 10


class GitDiffResult(NamedTuple):
    files: list[str]
    success: bool
    error: str | None = None


class DiffOutcome(NamedTuple):
    files: list[str]
    base_ref: str
    success: bool
    error: str | None = None


def log_error(message: str) -> None:
    print(f"::error::{message}")


def ensure_readiness_file() -> bool:
    if READINESS_PATH.exists():
        return True
    log_error(f"Missing readiness file: {READINESS_PATH}")
    return False


def parse_last_updated() -> datetime | None:
    content = READINESS_PATH.read_text(encoding="utf-8")
    match = re.search(LAST_UPDATED_PATTERN, content)
    if not match:
        log_error("Last updated date not found in docs/status/READINESS.md")
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        log_error("Last updated date is not in YYYY-MM-DD format")
        return None


def last_updated_is_fresh(last_updated: datetime) -> bool:
    age_days = (datetime.now(timezone.utc) - last_updated).days
    if age_days > MAX_AGE_DAYS:
        log_error(
            f"docs/status/READINESS.md is {age_days} days old (limit: {MAX_AGE_DAYS}). "
            "Update the Last updated field with current evidence."
        )
        return False
    return True


def run_git_diff(ref: str) -> GitDiffResult:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{ref}..HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return GitDiffResult([], False, result.stderr.strip())
    return GitDiffResult(
        [line.strip() for line in result.stdout.splitlines() if line.strip()], True, None
    )


def working_tree_diff() -> GitDiffResult:
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return GitDiffResult([], False, result.stderr.strip())
    return GitDiffResult(
        [line.strip() for line in result.stdout.splitlines() if line.strip()], True, None
    )


def ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def collect_changed_files() -> DiffOutcome:
    had_git_errors = False
    last_error: str | None = None
    refs_to_try: list[str] = []
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    base_ref_env = os.environ.get("GITHUB_BASE_REF")
    ref_name = os.environ.get("GITHUB_REF_NAME")

    if event_name == "pull_request" and base_ref_env:
        refs_to_try.append(f"origin/{base_ref_env}")
    if ref_name:
        refs_to_try.append(f"origin/{ref_name}")
    refs_to_try.extend(["origin/main", "main"])

    for candidate in refs_to_try:
        if ref_exists(candidate):
            diff_result = run_git_diff(candidate)
            had_git_errors = had_git_errors or not diff_result.success
            last_error = diff_result.error or last_error
            if diff_result.success:
                return DiffOutcome(diff_result.files, candidate, True, None)

    if ref_exists("HEAD^"):
        diff_result = run_git_diff("HEAD^")
        had_git_errors = had_git_errors or not diff_result.success
        last_error = diff_result.error or last_error
        if diff_result.success:
            return DiffOutcome(diff_result.files, "HEAD^", True, None)

    diff_result = working_tree_diff()
    had_git_errors = had_git_errors or not diff_result.success
    last_error = diff_result.error or last_error
    if diff_result.success:
        return DiffOutcome(diff_result.files, "working-tree", True, None)

    guidance = (
        "Unable to determine changed files. Git history may be shallow; "
        "ensure actions/checkout uses fetch-depth: 0 or fetch the base ref."
    )
    if last_error:
        guidance += f" git error: {last_error}"
    if had_git_errors:
        return DiffOutcome([], "", False, guidance)
    return DiffOutcome([], "", False, "Unable to determine changed files.")


def is_scoped(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in SCOPED_PREFIXES) or Path(
        normalized
    ).name.startswith("Dockerfile")


def readiness_updated(changed_files: Iterable[str]) -> bool:
    return any(Path(f).as_posix() == "docs/status/READINESS.md" for f in changed_files)


def main() -> int:
    if not ensure_readiness_file():
        return 1

    last_updated = parse_last_updated()
    if last_updated is None or not last_updated_is_fresh(last_updated):
        return 1

    diff_outcome = collect_changed_files()
    if not diff_outcome.success:
        log_error(diff_outcome.error or "Failed to compute git diff.")
        return 1

    print(
        f"Readiness diff base: {diff_outcome.base_ref or 'unknown'}; "
        f"scope prefixes: {', '.join(SCOPED_PREFIXES)}"
    )

    scoped_changes = [f for f in diff_outcome.files if is_scoped(f)]

    if scoped_changes and not readiness_updated(diff_outcome.files):
        scoped_list = ", ".join(scoped_changes[:MAX_LISTED_FILES])
        if len(scoped_changes) > MAX_LISTED_FILES:
            scoped_list += f", ... (+{len(scoped_changes) - MAX_LISTED_FILES} more)"
        log_error(
            "Code/test/config/workflow changes detected without updating docs/status/READINESS.md. "
            f"Touched files: {scoped_list}"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
