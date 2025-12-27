#!/usr/bin/env python3
"""Validate evidence snapshot completeness and integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import defusedxml.ElementTree as ET


REQUIRED_FILES = [
    "manifest.json",
    "coverage/coverage.xml",
    "pytest/junit.xml",
    "benchmarks/benchmark-metrics.json",
    "memory/memory_footprint.json",
]
BENCHMARK_METRIC_KEYS = ("max_p95_ms", "preflight_p95_ms", "e2e_small_p95_ms")


class EvidenceError(Exception):
    """Raised when evidence validation fails."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"Invalid JSON in {path}: {exc}") from exc


def _validate_manifest(path: Path) -> None:
    data = _load_json(path)
    for key in ("timestamp_utc", "git_sha", "files"):
        if key not in data:
            raise EvidenceError(f"manifest.json missing required key '{key}'")
    if not isinstance(data["files"], list) or not all(
        isinstance(item, str) for item in data["files"]
    ):
        raise EvidenceError("manifest.json 'files' must be a list of strings")
    if "commands" in data and not (
        isinstance(data["commands"], list)
        and all(isinstance(cmd, str) for cmd in data["commands"])
    ):
        raise EvidenceError("manifest.json 'commands' must be a list of strings when present")


def _secure_parser() -> ET.XMLParser:
    parser = ET.XMLParser()
    try:
        parser.parser.UseForeignDTD(False)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        parser.entity.clear()  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return parser


def _parse_coverage_percent(path: Path) -> float:
    try:
        parser = _secure_parser()
        root = ET.parse(path, parser=parser).getroot()
    except ET.ParseError as exc:
        raise EvidenceError(f"Invalid coverage XML at {path}: {exc}") from exc

    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        raise EvidenceError("coverage.xml missing 'line-rate' attribute")

    try:
        rate = float(line_rate)
    except ValueError as exc:
        raise EvidenceError(f"coverage.xml line-rate not numeric: {line_rate}") from exc

    if rate < 0 or rate > 1:
        raise EvidenceError(f"coverage.xml line-rate out of bounds: {rate}")

    return rate * 100.0


def _aggregate_testsuites(element: ET.Element) -> tuple[int, int, int, int]:
    tests = int(element.attrib.get("tests", 0))
    failures = int(element.attrib.get("failures", 0))
    errors = int(element.attrib.get("errors", 0))
    # Some JUnit emitters use 'skipped', others use legacy 'skip'
    skipped = int(element.attrib.get("skipped", element.attrib.get("skip", 0)))

    for child in element.findall("testsuite"):
        child_totals = _aggregate_testsuites(child)
        tests += child_totals[0]
        failures += child_totals[1]
        errors += child_totals[2]
        skipped += child_totals[3]

    return tests, failures, errors, skipped


def _parse_junit_totals(path: Path) -> tuple[int, int, int, int]:
    try:
        parser = _secure_parser()
        root = ET.parse(path, parser=parser).getroot()
    except ET.ParseError as exc:
        raise EvidenceError(f"Invalid JUnit XML at {path}: {exc}") from exc

    if root.tag == "testsuites":
        totals = _aggregate_testsuites(root)
    elif root.tag == "testsuite":
        totals = _aggregate_testsuites(root)
    else:
        raise EvidenceError(f"Unexpected root tag in junit.xml: {root.tag}")

    tests, failures, errors, skipped = totals
    if tests <= 0:
        raise EvidenceError("junit.xml reports zero tests")
    if any(value < 0 for value in totals):
        raise EvidenceError("junit.xml contains negative counters")
    if failures + errors > tests:
        raise EvidenceError("junit.xml failures/errors exceed total tests")

    return totals


def _validate_benchmark_metrics(path: Path) -> None:
    data = _load_json(path)
    for key in ("timestamp", "commit", "metrics"):
        if key not in data:
            raise EvidenceError(f"benchmark-metrics.json missing '{key}'")
    metrics = data["metrics"]
    if not isinstance(metrics, dict):
        raise EvidenceError("benchmark-metrics.json 'metrics' must be an object")
    for metric_key in BENCHMARK_METRIC_KEYS:
        if metric_key not in metrics:
            raise EvidenceError(f"benchmark-metrics.json missing metric '{metric_key}'")
        try:
            value = float(metrics[metric_key])
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"benchmark-metrics.json metric '{metric_key}' is not numeric") from exc
        if value < 0:
            raise EvidenceError(f"benchmark-metrics.json metric '{metric_key}' is negative")


def _validate_memory_footprint(path: Path) -> None:
    data = _load_json(path)
    for key in ("pelm_mb", "controller_mb"):
        if key not in data:
            raise EvidenceError(f"memory_footprint.json missing '{key}'")
        try:
            value = float(data[key])
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"memory_footprint.json '{key}' is not numeric") from exc
        if value <= 0:
            raise EvidenceError(f"memory_footprint.json '{key}' must be positive")


def verify_snapshot(evidence_dir: Path) -> None:
    if not evidence_dir.is_dir():
        raise EvidenceError(f"Evidence directory not found: {evidence_dir}")

    missing = [rel for rel in REQUIRED_FILES if not (evidence_dir / rel).exists()]
    if missing:
        raise EvidenceError(f"Missing required evidence files: {', '.join(missing)}")

    _validate_manifest(evidence_dir / "manifest.json")
    coverage_percent = _parse_coverage_percent(evidence_dir / "coverage" / "coverage.xml")
    junit_totals = _parse_junit_totals(evidence_dir / "pytest" / "junit.xml")
    _validate_benchmark_metrics(evidence_dir / "benchmarks" / "benchmark-metrics.json")
    _validate_memory_footprint(evidence_dir / "memory" / "memory_footprint.json")

    print(f"✓ Evidence snapshot valid: {evidence_dir}")
    print(f"  Coverage: {coverage_percent:.2f}%")
    tests, failures, errors, skipped = junit_totals
    print(
        f"  Tests: {tests} (failures={failures}, errors={errors}, skipped={skipped})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify evidence snapshot integrity")
    parser.add_argument(
        "--evidence-dir",
        required=True,
        type=Path,
        help="Path to evidence snapshot directory (artifacts/evidence/<date>/<sha>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        verify_snapshot(args.evidence_dir)
    except EvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
