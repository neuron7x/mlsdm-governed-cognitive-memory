#!/usr/bin/env python3
"""Enforce evidence drift policy against a canonical baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from scripts.evidence.verify_evidence_snapshot import (
    _load_json,
    _parse_coverage_percent,
    _parse_junit_totals,
)


DEFAULT_COVERAGE_DROP = 0.5  # percent
DEFAULT_MAX_P95_REGRESSION = 10.0  # percent
DEFAULT_MEMORY_INCREASE = 2.0  # MB


class DriftError(Exception):
    """Raised when drift exceeds allowed limits."""


def _baseline_metrics(baseline_path: Path) -> dict[str, float]:
    data = _load_json(baseline_path)
    metrics_block = data.get("metrics") or data.get("evidence", {}).get("metrics", {})
    baseline_block = data.get("baseline_metrics", {})

    def _get(key: str) -> Any:
        return metrics_block.get(key) or data.get(key) or baseline_block.get(key)

    memory_block = metrics_block.get("memory_mb") or data.get("memory_mb") or {}

    def _coerce_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    metrics = {
        "coverage_percent": _coerce_float(_get("coverage_percent")),
        "unit_tests_total": _coerce_int(_get("unit_tests_total") or _get("tests_total")),
        "test_failures": _coerce_int(_get("test_failures") or data.get("failures", 0)),
        "max_p95_ms": _coerce_float(
            _get("max_p95_ms") or baseline_block.get("e2e_heavy_p95_ms")
        ),
        "pelm_mb": _coerce_float(memory_block.get("pelm") or _get("pelm_mb")),
        "controller_mb": _coerce_float(memory_block.get("controller") or _get("controller_mb")),
    }

    missing = [k for k, v in metrics.items() if v is None]
    if missing:
        raise DriftError(f"Baseline missing required metrics: {', '.join(missing)}")
    return metrics


def _evidence_metrics(evidence_dir: Path) -> dict[str, float]:
    coverage = _parse_coverage_percent(evidence_dir / "coverage" / "coverage.xml")
    tests, failures, errors, _skipped = _parse_junit_totals(
        evidence_dir / "pytest" / "junit.xml"
    )

    benchmarks = _load_json(evidence_dir / "benchmarks" / "benchmark-metrics.json")
    max_p95 = float(benchmarks["metrics"]["max_p95_ms"])

    memory = _load_json(evidence_dir / "memory" / "memory_footprint.json")
    pelm = float(memory["pelm_mb"])
    controller = float(memory["controller_mb"])

    return {
        "coverage_percent": coverage,
        "unit_tests_total": tests,
        "test_failures": failures + errors,
        "max_p95_ms": max_p95,
        "pelm_mb": pelm,
        "controller_mb": controller,
    }


def _compare_metrics(
    baseline: dict[str, float],
    current: dict[str, float],
    *,
    max_coverage_drop: float,
    max_p95_regression: float,
    max_memory_increase: float,
    allow_test_drop: bool,
) -> list[str]:
    messages: list[str] = []

    coverage_delta = baseline["coverage_percent"] - current["coverage_percent"]
    if coverage_delta > max_coverage_drop:
        messages.append(
            f"✗ Coverage drop {coverage_delta:.2f}% exceeds allowed {max_coverage_drop:.2f}% "
            f"(baseline {baseline['coverage_percent']:.2f}%, current {current['coverage_percent']:.2f}%)"
        )
    else:
        messages.append(
            f"✓ Coverage OK (baseline {baseline['coverage_percent']:.2f}%, "
            f"current {current['coverage_percent']:.2f}%, drop {coverage_delta:.2f}%)"
        )

    tests_delta = baseline["unit_tests_total"] - current["unit_tests_total"]
    if tests_delta > 0:
        prefix = "⚠️  Tests decreased" if allow_test_drop else "✗ Tests decreased"
        messages.append(
            f"{prefix} by {tests_delta} (baseline {baseline['unit_tests_total']}, "
            f"current {current['unit_tests_total']})"
        )
    else:
        messages.append(
            f"✓ Tests count OK (baseline {baseline['unit_tests_total']}, current {current['unit_tests_total']})"
        )

    if current["test_failures"] > baseline["test_failures"]:
        messages.append(
            f"✗ Test failures increased (baseline {baseline['test_failures']}, current {current['test_failures']})"
        )
    else:
        messages.append(
            f"✓ Test failures OK (baseline {baseline['test_failures']}, current {current['test_failures']})"
        )

    allowed_p95 = baseline["max_p95_ms"] * (1 + max_p95_regression / 100)
    if current["max_p95_ms"] > allowed_p95:
        messages.append(
            f"✗ max_p95_ms regression: {current['max_p95_ms']:.3f}ms > "
            f"allowed {allowed_p95:.3f}ms (baseline {baseline['max_p95_ms']:.3f}ms, "
            f"regression limit {max_p95_regression:.1f}%)"
        )
    else:
        messages.append(
            f"✓ max_p95_ms OK (baseline {baseline['max_p95_ms']:.3f}ms, "
            f"current {current['max_p95_ms']:.3f}ms, limit {max_p95_regression:.1f}%)"
        )

    for comp in ("pelm_mb", "controller_mb"):
        allowed = baseline[comp] + max_memory_increase
        if current[comp] > allowed:
            messages.append(
                f"✗ {comp} regression: {current[comp]:.2f}MB > "
                f"allowed {allowed:.2f}MB (baseline {baseline[comp]:.2f}MB, "
                f"limit +{max_memory_increase:.2f}MB)"
            )
        else:
            messages.append(
                f"✓ {comp} OK (baseline {baseline[comp]:.2f}MB, current {current[comp]:.2f}MB, "
                f"limit +{max_memory_increase:.2f}MB)"
            )

    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check evidence drift against baseline")
    parser.add_argument("--baseline", required=True, type=Path, help="Path to baseline JSON file")
    parser.add_argument(
        "--evidence-dir",
        required=True,
        type=Path,
        help="Path to produced evidence directory (output of make evidence)",
    )
    parser.add_argument(
        "--max-coverage-drop",
        type=float,
        default=DEFAULT_COVERAGE_DROP,
        help="Allowed coverage drop in percentage points (default: 0.5)",
    )
    parser.add_argument(
        "--max-p95-regression",
        type=float,
        default=DEFAULT_MAX_P95_REGRESSION,
        help="Allowed regression of max_p95_ms in percent (default: 10.0)",
    )
    parser.add_argument(
        "--max-memory-increase",
        type=float,
        default=DEFAULT_MEMORY_INCREASE,
        help="Allowed memory increase in MB per component (default: 2.0MB)",
    )
    parser.add_argument(
        "--allow-test-drop-note",
        type=str,
        help="Allow test total decrease (e.g., documented PR allowlist). Provide note for audit trail.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        baseline = _baseline_metrics(args.baseline)
        current = _evidence_metrics(args.evidence_dir)
        messages = _compare_metrics(
            baseline,
            current,
            max_coverage_drop=args.max_coverage_drop,
            max_p95_regression=args.max_p95_regression,
            max_memory_increase=args.max_memory_increase,
            allow_test_drop=bool(args.allow_test_drop_note),
        )
    except DriftError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        import traceback

        traceback.print_exc()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("=== Evidence Drift Report ===")
    for msg in messages:
        print(msg)

    failed = any(msg.startswith("✗") for msg in messages)
    if failed and args.allow_test_drop_note:
        # Allow only the test-count decrease case to be tolerated when allowlisted
        failed = any(
            msg.startswith("✗")
            for msg in messages
            if not msg.startswith("✗ Tests decreased")
        )

    if failed:
        print("\nDrift check FAILED")
        return 1

    print("\nDrift check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
