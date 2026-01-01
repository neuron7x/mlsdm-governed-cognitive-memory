#!/usr/bin/env python3
"""Validate evidence snapshot completeness and integrity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Tuple

try:
    import defusedxml.ElementTree as ET
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    import xml.etree.ElementTree as ET  # type: ignore

SCHEMA_VERSION = "evidence-v1"
REQUIRED_FILES = (
    "manifest.json",
    "coverage/coverage.xml",
    "pytest/junit.xml",
    "env/python_version.txt",
    "env/uv_lock_sha256.txt",
)
MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB per file
MAX_TOTAL_BYTES = 5 * 1024 * 1024  # 5 MB total snapshot
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key
    re.compile(r"(?i)aws_secret_access_key"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),  # GitHub token
    re.compile(r"-----BEGIN PRIVATE KEY-----"),
    re.compile(r"api[_-]?key\s*[:=]\s*[A-Za-z0-9_\-]{10,}"),
]


class EvidenceError(Exception):
    """Raised when evidence validation fails."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"manifest is not valid JSON: {path}") from exc


def _validate_manifest(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    required_keys = (
        "schema_version",
        "git_sha",
        "date_utc",
        "python_version",
        "commands",
        "produced_files",
        "files",
        "ok",
        "partial",
        "errors",
    )
    for key in required_keys:
        if key not in data:
            raise EvidenceError(f"manifest.json missing required key '{key}'")

    if data["schema_version"] != SCHEMA_VERSION:
        raise EvidenceError(
            f"manifest.json schema_version is '{data['schema_version']}', expected '{SCHEMA_VERSION}'"
        )

    if not isinstance(data["commands"], list) or not all(isinstance(cmd, str) for cmd in data["commands"]):
        raise EvidenceError("manifest.json 'commands' must be a list of strings")

    if not isinstance(data["produced_files"], list) or not all(
        isinstance(path, str) for path in data["produced_files"]
    ):
        raise EvidenceError("manifest.json 'produced_files' must be a list of relative paths")

    if not isinstance(data.get("files"), list) or not all(isinstance(path, str) for path in data["files"]):
        raise EvidenceError("manifest.json 'files' must be a list of relative paths")

    for status_key in ("ok", "partial"):
        if not isinstance(data.get(status_key), bool):
            raise EvidenceError(f"manifest.json '{status_key}' must be a boolean")
    if not isinstance(data.get("errors"), list) or not all(isinstance(err, str) for err in data["errors"]):
        raise EvidenceError("manifest.json 'errors' must be a list of strings")

    return data


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
        root = ET.parse(path, parser=_secure_parser()).getroot()
    except ET.ParseError as exc:
        raise EvidenceError(f"coverage.xml is not valid XML: {exc}") from exc

    line_rate = root.attrib.get("line-rate")
    if line_rate is None:
        raise EvidenceError("coverage.xml missing 'line-rate' attribute")
    try:
        rate = float(line_rate)
    except ValueError as exc:
        raise EvidenceError(f"coverage.xml line-rate must be numeric (got {line_rate!r})") from exc
    if rate < 0 or rate > 1:
        raise EvidenceError(f"coverage.xml line-rate out of bounds: {rate}")
    return rate * 100.0


def _aggregate_tests(element: ET.Element) -> Tuple[int, int, int, int]:
    tests = int(element.attrib.get("tests", 0))
    failures = int(element.attrib.get("failures", 0))
    errors = int(element.attrib.get("errors", 0))
    skipped = int(element.attrib.get("skipped", element.attrib.get("skip", 0)))
    for child in element.findall("testsuite"):
        c_tests, c_failures, c_errors, c_skipped = _aggregate_tests(child)
        tests += c_tests
        failures += c_failures
        errors += c_errors
        skipped += c_skipped
    return tests, failures, errors, skipped


def _parse_junit(path: Path) -> Tuple[int, int, int, int]:
    try:
        root = ET.parse(path, parser=_secure_parser()).getroot()
    except ET.ParseError as exc:
        raise EvidenceError(f"junit.xml is not valid XML: {exc}") from exc

    if root.tag not in {"testsuite", "testsuites"}:
        raise EvidenceError(f"junit.xml has unexpected root tag '{root.tag}'")

    totals = _aggregate_tests(root)
    if totals[0] <= 0:
        raise EvidenceError("junit.xml reports zero tests")
    return totals


def _check_required_files(evidence_dir: Path) -> None:
    missing = [rel for rel in REQUIRED_FILES if not (evidence_dir / rel).exists()]
    if missing:
        raise EvidenceError(f"Missing required evidence files: {', '.join(missing)}")


def _ensure_paths_within_evidence(evidence_dir: Path, paths: Iterable[str]) -> None:
    for rel in paths:
        candidate = (evidence_dir / rel).resolve()
        if not str(candidate).startswith(str(evidence_dir.resolve())):
            raise EvidenceError(f"Manifest path escapes evidence directory: {rel}")
        if not candidate.exists():
            raise EvidenceError(f"Manifest lists missing file: {rel}")


def _check_sizes(evidence_dir: Path) -> None:
    total = 0
    for file_path in evidence_dir.rglob("*"):
        if not file_path.is_file():
            continue
        size = file_path.stat().st_size
        total += size
        if size > MAX_FILE_BYTES:
            raise EvidenceError(f"Evidence file too large (> {MAX_FILE_BYTES} bytes): {file_path}")
    if total > MAX_TOTAL_BYTES:
        raise EvidenceError(f"Evidence snapshot too large (> {MAX_TOTAL_BYTES} bytes total)")


def _scan_secrets(evidence_dir: Path) -> None:
    for file_path in evidence_dir.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise EvidenceError(f"Secret-like pattern found in {file_path}")


def verify_snapshot(evidence_dir: Path) -> None:
    if not evidence_dir.is_dir():
        raise EvidenceError(f"Evidence directory not found: {evidence_dir}")

    _check_required_files(evidence_dir)
    manifest = _validate_manifest(evidence_dir / "manifest.json")
    if not manifest.get("ok"):
        raise EvidenceError("manifest marks snapshot as not ok/complete")
    _ensure_paths_within_evidence(evidence_dir, manifest["produced_files"])
    _ensure_paths_within_evidence(evidence_dir, manifest["files"])
    _check_sizes(evidence_dir)
    _scan_secrets(evidence_dir)
    coverage_percent = _parse_coverage_percent(evidence_dir / "coverage" / "coverage.xml")
    tests, failures, errors, skipped = _parse_junit(evidence_dir / "pytest" / "junit.xml")

    print(f"✓ Evidence snapshot valid: {evidence_dir}")
    print(f"  Schema: {manifest['schema_version']}")
    print(f"  Coverage: {coverage_percent:.2f}%")
    print(f"  Tests: {tests} (failures={failures}, errors={errors}, skipped={skipped})")


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
