#!/usr/bin/env python3
"""Run pip-audit, store raw output, and fail on high/critical findings."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
POLICY_PATH = PROJECT_ROOT / "policy" / "debt_prevention_thresholds.json"
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"


class SecurityAuditError(RuntimeError):
    """Raised when security audit checks fail."""


def _load_policy() -> dict[str, object]:
    if not POLICY_PATH.exists():
        raise SecurityAuditError(f"Missing policy file: {POLICY_PATH}")
    try:
        return json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SecurityAuditError(f"Invalid JSON in {POLICY_PATH}: {exc}") from exc


def _run_pip_audit() -> tuple[int, str, str]:
    if shutil.which("pip-audit") is None:
        raise SecurityAuditError("pip-audit is required. Install with: pip install pip-audit")
    cmd = [
        "pip-audit",
        "--requirement",
        str(REQUIREMENTS_PATH),
        "--format",
        "json",
        "--progress-spinner=off",
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout, result.stderr


def _severity_blocklist(policy: dict[str, object]) -> set[str]:
    audit_policy = policy.get("security_audit", {}) if isinstance(policy, dict) else {}
    severities = audit_policy.get("fail_severities", ["HIGH", "CRITICAL"])
    if isinstance(severities, list):
        return {str(item).upper() for item in severities}
    return {"HIGH", "CRITICAL"}


def _extract_failures(payload: dict[str, object], blocklist: set[str]) -> list[str]:
    failures: list[str] = []
    dependencies = payload.get("dependencies", []) if isinstance(payload, dict) else []
    if not isinstance(dependencies, list):
        return failures
    for dep in dependencies:
        if not isinstance(dep, dict):
            continue
        name = dep.get("name", "unknown")
        for vuln in dep.get("vulns", []) or []:
            if not isinstance(vuln, dict):
                continue
            severity = str(vuln.get("severity", "unknown")).upper()
            if severity in blocklist:
                vuln_id = vuln.get("id", "unknown")
                failures.append(f"{name}: {vuln_id} ({severity})")
    return failures


def run_audit(output_path: Path) -> None:
    policy = _load_policy()
    exit_code, stdout, stderr = _run_pip_audit()
    if stderr:
        sys.stderr.write(stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stdout, encoding="utf-8")

    try:
        payload = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError as exc:
        raise SecurityAuditError(f"pip-audit output is not valid JSON: {exc}") from exc

    blocklist = _severity_blocklist(policy)
    failures = _extract_failures(payload, blocklist)

    if failures:
        details = "\n".join(f"- {entry}" for entry in failures)
        raise SecurityAuditError(f"High/Critical vulnerabilities found:\n{details}")

    if exit_code not in (0, 1):
        raise SecurityAuditError(f"pip-audit failed with exit code {exit_code}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pip-audit and enforce severity thresholds")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "security" / "pip-audit.json",
        help="Path to write raw pip-audit JSON output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_audit(args.output)
    except SecurityAuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("✓ Security audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
