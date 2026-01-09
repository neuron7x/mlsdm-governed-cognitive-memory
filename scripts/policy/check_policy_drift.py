#!/usr/bin/env python3
"""Detect policy drift and require explicit approval token."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_FILE = PROJECT_ROOT / "policy" / "debt_prevention_thresholds.json"
APPROVAL_FILE = PROJECT_ROOT / "policy" / "policy_drift_approval.json"


class PolicyDriftError(RuntimeError):
    """Raised when policy drift checks fail."""


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)


def _git_ref_exists(ref: str) -> bool:
    result = _run(["git", "rev-parse", "--verify", ref])
    return result.returncode == 0


def _fetch_base_ref(ref: str) -> None:
    _run(["git", "fetch", "origin", ref, "--depth", "1"])


def _changed_files(base_ref: str) -> set[str]:
    result = _run(["git", "diff", "--name-only", f"{base_ref}...HEAD"])
    if result.returncode != 0:
        raise PolicyDriftError(f"Failed to diff against {base_ref}: {result.stderr.strip()}")
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _resolve_base_ref(base_ref: str) -> str:
    if _git_ref_exists(base_ref):
        return base_ref
    _fetch_base_ref(base_ref.replace("origin/", ""))
    if _git_ref_exists(base_ref):
        return base_ref
    fetch_head = "FETCH_HEAD"
    if _git_ref_exists(fetch_head):
        return fetch_head
    raise PolicyDriftError(f"Unable to resolve base ref {base_ref}")


def _load_json(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PolicyDriftError(f"Invalid JSON in {path}: {exc}") from exc


def check_policy_drift(base_ref: str, output_path: Path | None) -> None:
    if not POLICY_FILE.exists():
        raise PolicyDriftError(f"Missing policy file: {POLICY_FILE}")
    if not APPROVAL_FILE.exists():
        raise PolicyDriftError(f"Missing approval file: {APPROVAL_FILE}")

    resolved_ref = _resolve_base_ref(base_ref)
    changed = _changed_files(resolved_ref)
    policy_changed = str(POLICY_FILE.relative_to(PROJECT_ROOT)) in changed
    approval_changed = str(APPROVAL_FILE.relative_to(PROJECT_ROOT)) in changed

    approval = _load_json(APPROVAL_FILE)
    approval_token = approval.get("approval_token", "UNSET")
    approved_by = approval.get("approved_by", "UNSET")
    approved_at = approval.get("approved_at", "UNSET")

    if policy_changed:
        if not approval_changed:
            raise PolicyDriftError("Policy thresholds changed without updating policy_drift_approval.json")
        if approval_token == "UNSET" or approved_by == "UNSET" or approved_at == "UNSET":
            raise PolicyDriftError("Approval token, approved_by, and approved_at must be set for policy drift")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "base_ref": resolved_ref,
            "policy_changed": policy_changed,
            "approval_changed": approval_changed,
            "approval_token": None if approval_token == "UNSET" else approval_token,
            "approved_by": None if approved_by == "UNSET" else approved_by,
            "approved_at": None if approved_at == "UNSET" else approved_at,
        }
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check policy drift and enforce approval token")
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Base ref to diff against (default: origin/<GITHUB_BASE_REF or main>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_ref = args.base_ref
    if not base_ref:
        base_branch = os.getenv("GITHUB_BASE_REF")
        base_ref = f"origin/{base_branch}" if base_branch else "origin/main"
    try:
        check_policy_drift(base_ref, args.output)
    except PolicyDriftError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("✓ Policy drift checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
