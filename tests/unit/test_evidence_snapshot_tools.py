from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.evidence.verify_evidence_snapshot import SCHEMA_VERSION, EvidenceError, verify_snapshot


def _build_snapshot(tmp_path: Path) -> Path:
    evidence_dir = tmp_path / "artifacts" / "evidence" / "2026-01-01" / "abcdef123456"
    (evidence_dir / "coverage").mkdir(parents=True, exist_ok=True)
    (evidence_dir / "pytest").mkdir(parents=True, exist_ok=True)
    (evidence_dir / "logs").mkdir(parents=True, exist_ok=True)
    (evidence_dir / "env").mkdir(parents=True, exist_ok=True)

    (evidence_dir / "coverage" / "coverage.xml").write_text(
        '<coverage line-rate="0.80" branch-rate="0.0"></coverage>', encoding="utf-8"
    )
    (evidence_dir / "pytest" / "junit.xml").write_text(
        '<testsuite name="unit" tests="2" failures="0" errors="0" skipped="0"></testsuite>',
        encoding="utf-8",
    )
    (evidence_dir / "env" / "python_version.txt").write_text("3.11.0\n", encoding="utf-8")
    (evidence_dir / "env" / "uv_lock_sha256.txt").write_text("hash\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": "abcdef1234567890",
        "date_utc": "2026-01-01",
        "python_version": "3.11.0",
        "commands": ["uv run bash ./coverage_gate.sh", "uv run python -m pytest tests/unit -q --junitxml=..."],
        "produced_files": [
            "coverage/coverage.xml",
            "pytest/junit.xml",
            "logs/coverage_gate.log",
            "env/python_version.txt",
            "env/uv_lock_sha256.txt",
        ],
        "files": [
            "coverage/coverage.xml",
            "pytest/junit.xml",
            "logs/coverage_gate.log",
            "env/python_version.txt",
            "env/uv_lock_sha256.txt",
            "manifest.json",
        ],
        "ok": True,
        "partial": False,
        "errors": [],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (evidence_dir / "logs" / "coverage_gate.log").write_text("ok", encoding="utf-8")
    return evidence_dir


def test_verify_snapshot_passes_for_minimal_valid_snapshot(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path)
    verify_snapshot(snapshot)


def test_verify_snapshot_fails_when_coverage_missing(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path)
    (snapshot / "coverage" / "coverage.xml").unlink()
    with pytest.raises(EvidenceError):
        verify_snapshot(snapshot)


def test_verify_snapshot_rejects_secret_pattern(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path)
    (snapshot / "logs" / "unit_tests.log").write_text("AWS_SECRET_ACCESS_KEY=abc", encoding="utf-8")
    with pytest.raises(EvidenceError):
        verify_snapshot(snapshot)


def test_verify_snapshot_rejects_oversized_file(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path)
    big_file = snapshot / "logs" / "big.bin"
    big_file.write_bytes(b"0" * (3 * 1024 * 1024))
    with pytest.raises(EvidenceError):
        verify_snapshot(snapshot)


def test_verify_snapshot_rejects_path_traversal(tmp_path: Path) -> None:
    snapshot = _build_snapshot(tmp_path)
    manifest_path = snapshot / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["produced_files"].append("../outside.txt")
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(EvidenceError):
        verify_snapshot(snapshot)


def test_capture_from_ci_creates_snapshot(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    coverage_xml = repo_root / "coverage.xml"
    junit_dir = repo_root / "reports"
    junit_dir.mkdir(exist_ok=True)
    junit_xml = junit_dir / "junit.xml"
    coverage_log = repo_root / "coverage-gate.log"

    coverage_xml.write_text('<coverage line-rate="0.90"></coverage>', encoding="utf-8")
    junit_xml.write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"></testsuite>', encoding="utf-8")
    coverage_log.write_text("ok", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evidence/capture_evidence.py",
            "--from-ci",
            "--coverage-path",
            str(coverage_xml),
            "--coverage-log",
            str(coverage_log),
            "--junit-path",
            str(junit_xml),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr

    evidence_root = repo_root / "artifacts" / "evidence"
    date_dirs = sorted([d for d in evidence_root.iterdir() if d.is_dir()])
    assert date_dirs
    latest_date = date_dirs[-1]
    sha_dirs = sorted([d for d in latest_date.iterdir() if d.is_dir()])
    assert sha_dirs
    snapshot_dir = sha_dirs[-1]

    try:
        verify_snapshot(snapshot_dir)
    finally:
        shutil.rmtree(snapshot_dir.parent, ignore_errors=True)
        coverage_xml.unlink(missing_ok=True)
        junit_xml.unlink(missing_ok=True)
        coverage_log.unlink(missing_ok=True)
