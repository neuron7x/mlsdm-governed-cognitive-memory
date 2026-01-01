#!/usr/bin/env python3
"""Capture reproducible evidence snapshot (coverage + JUnit logs only).

Writes artifacts to: artifacts/evidence/YYYY-MM-DD/<short_sha>/
Required outputs:
  - coverage/coverage.xml
  - pytest/junit.xml
  - logs/*.log
  - manifest.json (schema_version: evidence-v1)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

SCHEMA_VERSION = "evidence-v1"


class CaptureError(Exception):
    """Raised when evidence capture fails."""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _prefer_uv(command: List[str]) -> List[str]:
    """Prefix command with `uv run` if available to mirror CI."""
    if shutil.which("uv"):
        return ["uv", "run", *command]
    return command


def run_command(command: List[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(
        "COMMAND: "
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + result.stdout
        + "\nSTDERR:\n"
        + result.stderr
        + f"\nEXIT CODE: {result.returncode}\n",
        encoding="utf-8",
    )
    return result


def _uv_lock_sha256() -> str:
    lock = repo_root() / "uv.lock"
    if not lock.exists():
        return "missing"
    try:
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    except Exception:
        return "unreadable"


def capture_coverage(
    evidence_dir: Path,
    commands: list[str],
    produced: list[Path],
    coverage_path: Path,
    coverage_log: Path | None,
) -> None:
    dest_dir = evidence_dir / "coverage"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not coverage_path.exists():
        raise CaptureError("coverage.xml not found at expected path")
    shutil.copy(coverage_path, dest_dir / "coverage.xml")
    produced.append(dest_dir / "coverage.xml")
    if coverage_log and coverage_log.exists():
        logs_dir = evidence_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        dest_log = logs_dir / "coverage_gate.log"
        shutil.copy(coverage_log, dest_log)
        produced.append(dest_log)


def capture_pytest_junit(
    evidence_dir: Path,
    commands: list[str],
    produced: list[Path],
    junit_path: Path,
    unit_log: Path | None,
) -> None:
    dest_dir = evidence_dir / "pytest"
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not junit_path.exists():
        raise CaptureError("junit.xml not found at expected path")
    shutil.copy(junit_path, dest_dir / "junit.xml")
    produced.append(dest_dir / "junit.xml")
    if unit_log and unit_log.exists():
        logs_dir = evidence_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        dest_log = logs_dir / "unit_tests.log"
        shutil.copy(unit_log, dest_log)
        produced.append(dest_log)


def capture_env(evidence_dir: Path, produced: list[Path]) -> None:
    env_dir = evidence_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    py_path = env_dir / "python_version.txt"
    uv_lock_path = env_dir / "uv_lock_sha256.txt"
    py_path.write_text(sys.version.split()[0] + "\n", encoding="utf-8")
    uv_lock_path.write_text(_uv_lock_sha256() + "\n", encoding="utf-8")
    produced.extend([py_path, uv_lock_path])


def write_manifest(
    evidence_dir: Path,
    sha: str,
    commands: Iterable[str],
    produced: Iterable[Path],
    ok: bool,
    errors: list[str],
) -> None:
    produced_paths = {path for path in produced if path.exists()}
    produced_paths.add(evidence_dir / "manifest.json")
    all_files = sorted({str(path.relative_to(evidence_dir)) for path in evidence_dir.rglob("*") if path.is_file()})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": sha,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "python_version": sys.version.split()[0],
        "commands": list(commands),
        "produced_files": sorted(
            {str(path.relative_to(evidence_dir)) for path in produced_paths}
        ),
        "files": all_files,
        "ok": ok,
        "partial": not ok,
        "errors": errors,
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture evidence snapshot")
    parser.add_argument(
        "--from-ci",
        action="store_true",
        help="Reuse artifacts already produced by CI steps instead of re-running coverage/tests",
    )
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=repo_root() / "coverage.xml",
        help="Path to coverage.xml generated earlier in the job",
    )
    parser.add_argument(
        "--coverage-log",
        type=Path,
        default=repo_root() / "coverage-gate.log",
        help="Path to coverage gate log to include if present",
    )
    parser.add_argument(
        "--junit-path",
        type=Path,
        default=repo_root() / "reports" / "junit.xml",
        help="Path to JUnit XML generated earlier in the job",
    )
    parser.add_argument(
        "--unit-log",
        type=Path,
        default=None,
        help="Optional path to unit test log to include",
    )
    return parser.parse_args()


def _run_commands_when_needed(args: argparse.Namespace, commands: list[str], produced: list[Path], evidence_dir: Path) -> None:
    if args.from_ci:
        return

    coverage_log = evidence_dir / "logs" / "coverage_gate.log"
    command = _prefer_uv(["bash", "./coverage_gate.sh"])
    commands.append(" ".join(command))
    result_cov = run_command(command, coverage_log)

    reports_dir = repo_root() / "reports"
    reports_dir.mkdir(exist_ok=True)
    junit_path = reports_dir / "junit-unit.xml"
    if junit_path.exists():
        junit_path.unlink()
    unit_log = evidence_dir / "logs" / "unit_tests.log"
    command_unit = _prefer_uv(
        [
            "python",
            "-m",
            "pytest",
            "tests/unit",
            "-q",
            "--junitxml",
            str(junit_path),
            "--maxfail=1",
        ]
    )
    commands.append(" ".join(command_unit))
    result_unit = run_command(command_unit, unit_log)

    # Update paths for downstream copy
    args.coverage_path = repo_root() / "coverage.xml"
    args.coverage_log = coverage_log
    args.junit_path = junit_path
    args.unit_log = unit_log

    produced.extend([coverage_log, unit_log])

    if result_cov.returncode != 0:
        raise CaptureError(f"coverage gate failed; see {coverage_log.relative_to(evidence_dir)}")
    if result_unit.returncode != 0:
        raise CaptureError(f"unit tests failed; see {unit_log.relative_to(evidence_dir)}")


def main() -> int:
    args = parse_args()
    root = repo_root()
    os.chdir(root)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    sha_full = git_sha()
    short_sha = sha_full[:12] if sha_full != "unknown" else "unknown"
    base_dir = root / "artifacts" / "evidence"
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="evidence-", dir=base_dir))
    evidence_dir = temp_dir

    commands: list[str] = []
    produced: list[Path] = []
    errors: list[str] = ["capture not completed"]
    # Write initial manifest to ensure partial snapshots exist
    write_manifest(evidence_dir, sha_full, commands, produced, ok=False, errors=errors)

    try:
        _run_commands_when_needed(args, commands, produced, evidence_dir)
        capture_coverage(evidence_dir, commands, produced, args.coverage_path, args.coverage_log)
        capture_pytest_junit(evidence_dir, commands, produced, args.junit_path, args.unit_log)
        capture_env(evidence_dir, produced)
        errors = []
        ok = True
    except CaptureError as exc:
        ok = False
        errors = [str(exc)]
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Logs preserved at {temp_dir}", file=sys.stderr)
    finally:
        write_manifest(evidence_dir, sha_full, commands, produced, ok=ok, errors=errors)

    final_parent = root / "artifacts" / "evidence" / date_str
    final_parent.mkdir(parents=True, exist_ok=True)
    final_dir = final_parent / short_sha
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(str(temp_dir), final_dir)

    print(f"Evidence captured at {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
