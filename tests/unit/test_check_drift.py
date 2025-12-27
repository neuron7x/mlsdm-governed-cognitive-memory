from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_coverage_xml(path: Path, line_rate: float) -> None:
    path.write_text(
        f"""<?xml version='1.0' encoding='UTF-8'?>
<coverage line-rate="{line_rate}" branch-rate="0.0" version="1.0" timestamp="0">
</coverage>
"""
    )


def _write_junit(path: Path, tests: int, failures: int = 0, errors: int = 0) -> None:
    path.write_text(
        f"""<?xml version='1.0' encoding='UTF-8'?>
<testsuite name="root" tests="{tests}" failures="{failures}" errors="{errors}" skipped="0"></testsuite>
"""
    )


def _write_benchmarks(path: Path, max_p95: float) -> None:
    payload = {
        "metrics": {
            "max_p95_ms": max_p95,
            "preflight_p95_ms": 0.01,
            "e2e_small_p95_ms": 0.1,
        }
    }
    path.write_text(json.dumps(payload))


def _write_memory(path: Path, pelm: float, controller: float) -> None:
    payload = {"pelm_mb": pelm, "controller_mb": controller}
    path.write_text(json.dumps(payload))


def _baseline(path: Path, **overrides: float) -> None:
    baseline = {
        "metrics": {
            "coverage_percent": 80.0,
            "unit_tests_total": 100,
            "test_failures": 0,
            "max_p95_ms": 1.0,
            "memory_mb": {"pelm": 30.0, "controller": 20.0},
        }
    }
    baseline["metrics"].update(overrides)
    path.write_text(json.dumps(baseline))


def _run_check(baseline: Path, evidence_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/evidence/check_drift.py",
            "--baseline",
            str(baseline),
            "--evidence-dir",
            str(evidence_dir),
        ],
        cwd=Path(__file__).resolve().parent.parent.parent,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_check_drift_passes_within_thresholds(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _baseline(baseline_path)

    evidence_dir = tmp_path / "evidence"
    (evidence_dir / "coverage").mkdir(parents=True)
    (evidence_dir / "pytest").mkdir()
    (evidence_dir / "benchmarks").mkdir()
    (evidence_dir / "memory").mkdir()

    _write_coverage_xml(evidence_dir / "coverage" / "coverage.xml", 0.80 - 0.001)
    _write_junit(evidence_dir / "pytest" / "junit.xml", tests=100, failures=0)
    _write_benchmarks(evidence_dir / "benchmarks" / "benchmark-metrics.json", max_p95=1.05)
    _write_memory(evidence_dir / "memory" / "memory_footprint.json", pelm=31.0, controller=21.0)

    result = _run_check(baseline_path, evidence_dir)
    assert result.returncode == 0, result.stderr


def test_check_drift_fails_on_coverage_drop(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    _baseline(baseline_path, coverage_percent=90.0)

    evidence_dir = tmp_path / "evidence"
    (evidence_dir / "coverage").mkdir(parents=True)
    (evidence_dir / "pytest").mkdir()
    (evidence_dir / "benchmarks").mkdir()
    (evidence_dir / "memory").mkdir()

    _write_coverage_xml(evidence_dir / "coverage" / "coverage.xml", 0.80)  # 80%
    _write_junit(evidence_dir / "pytest" / "junit.xml", tests=100, failures=0)
    _write_benchmarks(evidence_dir / "benchmarks" / "benchmark-metrics.json", max_p95=1.0)
    _write_memory(evidence_dir / "memory" / "memory_footprint.json", pelm=30.0, controller=20.0)

    result = _run_check(baseline_path, evidence_dir)
    assert result.returncode == 1
    assert "Coverage drop" in result.stdout
