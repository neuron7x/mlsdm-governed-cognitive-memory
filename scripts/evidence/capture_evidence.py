#!/usr/bin/env python3
"""Capture reproducible evidence snapshot.

This script generates a complete evidence snapshot containing:
- coverage.xml (from coverage_gate.sh)
- junit.xml (unit + state tests)
- benchmark-metrics.json (in check_benchmark_drift.py schema)
- memory_footprint.json (PELM + controller)
- Environment metadata (python, platform, uv.lock hash)

Usage:
    uv run python scripts/evidence/capture_evidence.py
    make evidence
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_uv_lock_sha256(repo_root: Path) -> str:
    """Compute SHA256 of uv.lock for reproducibility tracking."""
    uv_lock = repo_root / "uv.lock"
    if not uv_lock.exists():
        return "not_found"
    try:
        content = uv_lock.read_bytes()
        return hashlib.sha256(content).hexdigest()
    except Exception:
        return "error"


def run_coverage_gate(repo_root: Path, evidence_dir: Path) -> bool:
    """Run coverage gate and capture output.

    Returns:
        True if coverage gate passed, False otherwise.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Running coverage gate...")
    print("=" * 70)

    coverage_dir = evidence_dir / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)

    log_path = coverage_dir / "coverage.log"
    result = subprocess.run(
        ["bash", "./coverage_gate.sh"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    # Write log
    with open(log_path, "w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)
        f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")

    # Copy coverage.xml if it exists
    src_coverage = repo_root / "coverage.xml"
    if src_coverage.exists():
        shutil.copy(src_coverage, coverage_dir / "coverage.xml")
        print(f"✓ Captured coverage.xml")
    else:
        print("⚠ coverage.xml not found")

    if result.returncode == 0:
        print("✓ Coverage gate passed")
    else:
        print(f"⚠ Coverage gate returned exit code {result.returncode}")

    return result.returncode == 0


def run_pytest_junit(repo_root: Path, evidence_dir: Path) -> bool:
    """Run unit + state tests and generate JUnit XML.

    Returns:
        True if tests passed, False otherwise.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Running unit + state tests with JUnit output...")
    print("=" * 70)

    pytest_dir = evidence_dir / "pytest"
    pytest_dir.mkdir(parents=True, exist_ok=True)
    junit_path = pytest_dir / "junit.xml"

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/unit", "tests/state",
            "-q", "--tb=short",
            f"--junitxml={junit_path}",
            "--maxfail=1",
            "-m", "not slow",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    if junit_path.exists():
        print(f"✓ Generated junit.xml ({junit_path.stat().st_size} bytes)")
    else:
        print("⚠ junit.xml not generated")

    if result.returncode == 0:
        print("✓ All tests passed")
    else:
        print(f"⚠ Tests returned exit code {result.returncode}")
        # Print summary for visibility
        lines = result.stdout.strip().split("\n")
        for line in lines[-10:]:
            print(f"  {line}")

    return result.returncode == 0


def generate_benchmark_metrics(repo_root: Path, evidence_dir: Path) -> bool:
    """Generate benchmark-metrics.json in check_benchmark_drift.py schema.

    Returns:
        True if generation succeeded, False otherwise.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Generating benchmark metrics...")
    print("=" * 70)

    benchmarks_dir = evidence_dir / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    # Import benchmark functions to compute metrics
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "src"))

    try:
        from benchmarks.test_neuro_engine_performance import (
            benchmark_end_to_end_latency_heavy_load,
            benchmark_end_to_end_latency_small_load,
            benchmark_pre_flight_latency,
        )

        git_sha = get_git_sha()
        timestamp = datetime.now(timezone.utc).isoformat()

        # Run benchmarks (each returns dict with p50, p95, p99, min, max, mean)
        print("  Running pre-flight latency benchmark...")
        preflight_stats = benchmark_pre_flight_latency()

        print("  Running small load benchmark...")
        small_load_stats = benchmark_end_to_end_latency_small_load()

        print("  Running heavy load benchmark...")
        heavy_load_results = benchmark_end_to_end_latency_heavy_load()

        # Find max P95 across all heavy load scenarios
        max_p95 = 0.0
        for _key, stats in heavy_load_results.items():
            if stats["p95"] > max_p95:
                max_p95 = stats["p95"]

        # SLO threshold (from baseline.json)
        slo_threshold_ms = 500.0
        slo_compliant = max_p95 < slo_threshold_ms

        # Build metrics JSON in expected schema
        metrics_json = {
            "timestamp": timestamp,
            "commit": git_sha,
            "slo_compliant": slo_compliant,
            "metrics": {
                "max_p95_ms": round(max_p95, 3),
                "preflight_p95_ms": round(preflight_stats["p95"], 3),
                "e2e_small_p95_ms": round(small_load_stats["p95"], 3),
            },
            "details": {
                "preflight": preflight_stats,
                "small_load": small_load_stats,
                "heavy_load": heavy_load_results,
            },
        }

        metrics_path = benchmarks_dir / "benchmark-metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"✓ Generated benchmark-metrics.json")

        # Save raw latency data too
        raw_path = benchmarks_dir / "raw_neuro_engine_latency.json"
        with open(raw_path, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "commit": git_sha,
                "preflight": preflight_stats,
                "small_load": small_load_stats,
                "heavy_load": heavy_load_results,
            }, f, indent=2)
        print(f"✓ Generated raw_neuro_engine_latency.json")

        if slo_compliant:
            print(f"✓ SLO compliant (max P95: {max_p95:.3f}ms < {slo_threshold_ms}ms)")
        else:
            print(f"⚠ SLO not met (max P95: {max_p95:.3f}ms >= {slo_threshold_ms}ms)")

        return True

    except Exception as e:
        print(f"✗ Benchmark generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def measure_memory(repo_root: Path, evidence_dir: Path) -> bool:
    """Measure memory footprint and generate JSON.

    Returns:
        True if measurement succeeded, False otherwise.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Measuring memory footprint...")
    print("=" * 70)

    memory_dir = evidence_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    json_path = memory_dir / "memory_footprint.json"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/measure_memory_footprint.py",
            "--json-out", str(json_path),
            "--seed", "42",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    if json_path.exists():
        print(f"✓ Generated memory_footprint.json")
    else:
        print("⚠ memory_footprint.json not generated")
        print(f"  stdout: {result.stdout[-500:]}")
        print(f"  stderr: {result.stderr[-500:]}")

    return result.returncode == 0


def capture_env_metadata(repo_root: Path, evidence_dir: Path) -> None:
    """Capture environment metadata."""
    print("\n" + "=" * 70)
    print("STEP 5: Capturing environment metadata...")
    print("=" * 70)

    env_dir = evidence_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)

    # Python version
    (env_dir / "python_version.txt").write_text(platform.python_version() + "\n")
    print(f"  Python: {platform.python_version()}")

    # Platform info
    (env_dir / "uname.txt").write_text(platform.platform() + "\n")
    print(f"  Platform: {platform.platform()}")

    # uv.lock hash
    lock_hash = get_uv_lock_sha256(repo_root)
    (env_dir / "uv_lock_sha256.txt").write_text(lock_hash + "\n")
    print(f"  uv.lock SHA256: {lock_hash[:16]}...")

    print("✓ Environment metadata captured")


def create_manifest(evidence_dir: Path) -> None:
    """Create manifest.json with snapshot metadata."""
    print("\n" + "=" * 70)
    print("STEP 6: Creating manifest...")
    print("=" * 70)

    git_sha = get_git_sha()
    timestamp = datetime.now(timezone.utc).isoformat()

    manifest = {
        "timestamp_utc": timestamp,
        "git_sha": git_sha,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "commands": [
            "bash ./coverage_gate.sh",
            "pytest tests/unit tests/state -q --junitxml=... --maxfail=1",
            "benchmarks/test_neuro_engine_performance.py (imported functions)",
            "benchmarks/measure_memory_footprint.py --json-out ... --seed 42",
        ],
        "files": sorted([str(p.relative_to(evidence_dir)) for p in evidence_dir.rglob("*") if p.is_file()]),
    }

    manifest_path = evidence_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Created manifest.json")
    print(f"  Timestamp: {timestamp}")
    print(f"  Git SHA: {git_sha}")
    print(f"  Files captured: {len(manifest['files'])}")


def main() -> int:
    """Main entry point."""
    print("=" * 70)
    print("MLSDM Evidence Snapshot Capture")
    print("=" * 70)

    # Determine paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(repo_root)

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    git_sha = get_git_sha()
    short_sha = git_sha[:12] if git_sha != "unknown" else "unknown"

    evidence_dir = repo_root / "artifacts" / "evidence" / date_str / short_sha
    evidence_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEvidence directory: {evidence_dir.relative_to(repo_root)}")
    print(f"Date: {date_str}")
    print(f"Git SHA: {git_sha}")

    # Capture evidence
    success = True

    if not run_coverage_gate(repo_root, evidence_dir):
        success = False

    if not run_pytest_junit(repo_root, evidence_dir):
        success = False

    if not generate_benchmark_metrics(repo_root, evidence_dir):
        success = False

    if not measure_memory(repo_root, evidence_dir):
        success = False

    capture_env_metadata(repo_root, evidence_dir)
    create_manifest(evidence_dir)

    print("\n" + "=" * 70)
    if success:
        print("✓ EVIDENCE SNAPSHOT COMPLETE")
    else:
        print("⚠ EVIDENCE SNAPSHOT COMPLETE (with warnings)")
    print("=" * 70)
    print(f"\nEvidence saved to: {evidence_dir.relative_to(repo_root)}")
    print("\nTo check benchmark drift:")
    print(f"  python scripts/check_benchmark_drift.py {evidence_dir.relative_to(repo_root)}/benchmarks/benchmark-metrics.json")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
