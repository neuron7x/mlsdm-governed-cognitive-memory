#!/usr/bin/env python3
"""Validate committed evidence snapshots for integrity and doc alignment."""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


REQUIRED_FILES = [
    "coverage/coverage.xml",
    "pytest/junit.xml",
    "benchmarks/benchmark-metrics.json",
    "memory/memory_footprint.json",
    "manifest.json",
]

FORBIDDEN_PATTERNS = [
    ".env",
    ".pem",
    ".key",
    "id_rsa",
    "token",
    "secret",
]


@dataclass
class CoverageStats:
    percent: float


@dataclass
class TestStats:
    total: int
    failures: int
    errors: int
    skipped: int


def get_repo_root() -> Path:
    """Return repository root by walking parents for .git."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return current.parent.parent.parent


def find_latest_snapshot(repo_root: Path) -> Path:
    """Locate the latest evidence snapshot (by date then sha)."""
    evidence_dir = repo_root / "artifacts" / "evidence"
    if not evidence_dir.exists():
        raise FileNotFoundError(f"No evidence directory found at {evidence_dir}")

    date_dirs = sorted(
        [p for p in evidence_dir.iterdir() if p.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", p.name)]
    )
    if not date_dirs:
        raise FileNotFoundError(f"No dated evidence snapshots found under {evidence_dir}")

    latest_date = date_dirs[-1]
    sha_dirs = sorted([p for p in latest_date.iterdir() if p.is_dir()])
    if not sha_dirs:
        raise FileNotFoundError(f"No SHA snapshots found under {latest_date}")

    return sha_dirs[-1]


def validate_manifest(manifest_path: Path) -> list[str]:
    """Ensure manifest exists and contains required fields."""
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"Missing manifest: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"Manifest is not valid JSON: {exc}"]
    except OSError as exc:  # pragma: no cover - I/O failure
        return [f"Manifest could not be read: {exc}"]

    for field in ("timestamp_utc", "git_sha", "python_version", "platform"):
        if field not in manifest:
            errors.append(f"Manifest missing field: {field}")

    return errors


def validate_benchmark_schema(benchmark_path: Path) -> list[str]:
    """Validate benchmark metrics JSON schema."""
    errors: list[str] = []
    try:
        payload = json.loads(benchmark_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"benchmark-metrics.json invalid JSON: {exc}"]
    except OSError as exc:  # pragma: no cover - I/O failure
        return [f"benchmark-metrics.json could not be read: {exc}"]

    required_top = ("timestamp", "commit", "metrics")
    for field in required_top:
        if field not in payload:
            errors.append(f"benchmark-metrics.json missing field: {field}")

    metrics = payload.get("metrics", {})
    if "max_p95_ms" not in metrics:
        errors.append("benchmark-metrics.json missing metrics.max_p95_ms")

    return errors


def parse_coverage(coverage_path: Path) -> CoverageStats:
    """Parse coverage.xml and return percent line rate."""
    try:
        tree = ET.parse(coverage_path)
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Unable to parse coverage XML at {coverage_path}: {exc}") from exc

    root = tree.getroot()
    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        raise ValueError("coverage.xml missing line-rate attribute")
    return CoverageStats(percent=round(float(line_rate) * 100, 2))


def parse_junit(junit_path: Path) -> TestStats:
    """Parse junit.xml and return aggregate stats."""
    try:
        tree = ET.parse(junit_path)
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Unable to parse junit XML at {junit_path}: {exc}") from exc
    tests = failures = errors = skipped = 0
    for suite in tree.iterfind(".//testsuite"):
        tests += int(suite.attrib.get("tests", 0))
        failures += int(suite.attrib.get("failures", 0))
        errors += int(suite.attrib.get("errors", 0))
        skipped += int(suite.attrib.get("skipped", 0))
    return TestStats(total=tests, failures=failures, errors=errors, skipped=skipped)


def check_required_files(snapshot: Path) -> list[str]:
    """Ensure required files exist within the snapshot."""
    missing = []
    for rel in REQUIRED_FILES:
        if not (snapshot / rel).exists():
            missing.append(rel)
    return missing


def check_forbidden_files(snapshot: Path, max_size_mb: int = 20) -> list[str]:
    """Detect forbidden file types or oversized files in evidence."""
    violations: list[str] = []
    max_bytes = max_size_mb * 1024 * 1024
    for path in snapshot.rglob("*"):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if any(pat in lower_name for pat in FORBIDDEN_PATTERNS):
            violations.append(f"Forbidden filename {path.relative_to(snapshot)}")
        if path.stat().st_size > max_bytes:
            violations.append(
                f"Oversized file {path.relative_to(snapshot)} ({path.stat().st_size} bytes > {max_bytes})"
            )
    return violations


def extract_doc_claims(content: str) -> dict[str, object]:
    """Extract numeric claims from docs/METRICS_SOURCE.md."""
    claims: dict[str, object] = {}
    cov_match = re.search(r"Actual Coverage[^\n]*?([0-9]+(?:\.[0-9]+)?)%", content, re.IGNORECASE)
    if cov_match:
        claims["coverage_pct"] = float(cov_match.group(1))

    tests_match = re.search(
        r"Total Tests[^\n]*?total:\s*(\d+)\s*,\s*failures:\s*(\d+)\s*,\s*errors:\s*(\d+)\s*,\s*skipped:\s*(\d+)",
        content,
        re.IGNORECASE,
    )
    if tests_match:
        claims["tests"] = {
            "total": int(tests_match.group(1)),
            "failures": int(tests_match.group(2)),
            "errors": int(tests_match.group(3)),
            "skipped": int(tests_match.group(4)),
        }

    path_match = re.search(
        r"artifacts/evidence/\d{4}-\d{2}-\d{2}/[a-fA-F0-9]{7,40}",
        content,
    )
    if path_match:
        claims["snapshot_path"] = path_match.group(0)

    return claims


def compare_with_docs(
    repo_root: Path,
    snapshot: Path,
    coverage: CoverageStats,
    tests: TestStats,
    fail_on_mismatch: bool = True,
) -> list[str]:
    """Compare computed metrics with documented claims."""
    metrics_source = repo_root / "docs" / "METRICS_SOURCE.md"
    if not metrics_source.exists():
        return [f"Missing metrics source doc: {metrics_source}"]

    content = metrics_source.read_text()
    claims = extract_doc_claims(content)
    errors: list[str] = []

    cov_claim = claims.get("coverage_pct")
    if isinstance(cov_claim, (int, float)) and abs(float(cov_claim) - coverage.percent) > 0.01:
        msg = (
            f"Coverage mismatch: docs={float(cov_claim)}%, "
            f"evidence={coverage.percent}% (snapshot {snapshot.name})"
        )
        errors.append(msg)

    tests_claim = claims.get("tests")
    if isinstance(tests_claim, dict):
        total = int(tests_claim.get("total", tests.total))
        failures = int(tests_claim.get("failures", tests.failures))
        errors_count = int(tests_claim.get("errors", tests.errors))
        skipped = int(tests_claim.get("skipped", tests.skipped))
        if (
            total != tests.total
            or failures != tests.failures
            or errors_count != tests.errors
            or skipped != tests.skipped
        ):
            errors.append(
                "Test metrics mismatch: "
                f"docs={{'total': {total}, 'failures': {failures}, 'errors': {errors_count}, 'skipped': {skipped}}}, "
                f"evidence={{'total': {tests.total}, 'failures': {tests.failures}, "
                f"'errors': {tests.errors}, 'skipped': {tests.skipped}}}"
            )

    doc_snapshot_obj = claims.get("snapshot_path")
    if isinstance(doc_snapshot_obj, str) and doc_snapshot_obj not in str(snapshot):
        errors.append(
            f"Documented snapshot path '{doc_snapshot_obj}' does not match validated snapshot '{snapshot}'."
        )

    if errors and not fail_on_mismatch:
        for err in errors:
            print(f"[warn] {err}")
        return []
    return errors


def validate_snapshot(
    snapshot: Path, fail_on_doc_mismatch: bool = True, max_size_mb: int = 20
) -> tuple[bool, list[str], CoverageStats | None, TestStats | None]:
    """Validate a snapshot and return (ok, errors, coverage, tests)."""
    errors: list[str] = []

    missing = check_required_files(snapshot)
    if missing:
        errors.append(f"Missing required files: {missing}")
        return False, errors, None, None

    manifest_errors = validate_manifest(snapshot / "manifest.json")
    errors.extend(manifest_errors)

    benchmark_errors = validate_benchmark_schema(snapshot / "benchmarks" / "benchmark-metrics.json")
    errors.extend(benchmark_errors)

    try:
        coverage = parse_coverage(snapshot / "coverage" / "coverage.xml")
    except ValueError as exc:
        errors.append(str(exc))
        return False, errors, None, None

    try:
        tests = parse_junit(snapshot / "pytest" / "junit.xml")
    except ValueError as exc:
        errors.append(str(exc))
        return False, errors, None, None

    errors.extend(compare_with_docs(get_repo_root(), snapshot, coverage, tests, fail_on_doc_mismatch))

    forbidden = check_forbidden_files(snapshot, max_size_mb=max_size_mb)
    if forbidden:
        errors.append(f"Forbidden or oversized evidence files detected: {forbidden}")

    return len(errors) == 0, errors, coverage, tests


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an evidence snapshot for integrity.")
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Path to the snapshot root (e.g., artifacts/evidence/<date>/<sha>). Defaults to latest.",
    )
    parser.add_argument(
        "--allow-doc-mismatch",
        action="store_true",
        help="Do not fail if docs claims differ from evidence; emit warnings instead.",
    )
    parser.add_argument(
        "--max-size-mb",
        type=int,
        default=20,
        help="Maximum allowed file size inside evidence (MB).",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print computed metrics summary.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = get_repo_root()
    snapshot = args.snapshot or find_latest_snapshot(repo_root)

    ok, errors, coverage, tests = validate_snapshot(
        snapshot,
        fail_on_doc_mismatch=not args.allow_doc_mismatch,
        max_size_mb=args.max_size_mb,
    )

    if args.print_summary and coverage and tests:
        print(
            f"Snapshot: {snapshot}\n"
            f"Coverage: {coverage.percent}%\n"
            f"Tests: total={tests.total}, failures={tests.failures}, "
            f"errors={tests.errors}, skipped={tests.skipped}"
        )

    if not ok:
        for err in errors:
            print(f"[error] {err}")
        return 1

    print(f"Evidence snapshot OK: {snapshot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
