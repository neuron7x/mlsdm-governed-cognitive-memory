"""Tests for additional scripts in the scripts/ and benchmarks/ directories.

These tests verify that scripts can be imported and their main functions
can be invoked with --help without errors.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_export_openapi_help() -> None:
    """Test that export_openapi.py --help works."""
    result = subprocess.run(
        [sys.executable, "scripts/export_openapi.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "Export OpenAPI specification" in result.stdout


def test_run_effectiveness_suite_help() -> None:
    """Test that run_effectiveness_suite.py --help works."""
    result = subprocess.run(
        [sys.executable, "scripts/run_effectiveness_suite.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "effectiveness" in result.stdout.lower()


def test_security_audit_help() -> None:
    """Test that security_audit.py --help works."""
    result = subprocess.run(
        [sys.executable, "scripts/security_audit.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "security" in result.stdout.lower() or "Security" in result.stdout


def test_run_calibration_benchmarks_help() -> None:
    """Test that run_calibration_benchmarks.py --help works."""
    result = subprocess.run(
        [sys.executable, "scripts/run_calibration_benchmarks.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "Calibration" in result.stdout or "calibration" in result.stdout.lower()


def test_generate_effectiveness_charts_help() -> None:
    """Test that generate_effectiveness_charts.py --help works."""
    result = subprocess.run(
        [sys.executable, "scripts/generate_effectiveness_charts.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "effectiveness" in result.stdout.lower() or "charts" in result.stdout.lower()


def test_benchmark_fractal_pelm_gpu_help() -> None:
    """Test that benchmark_fractal_pelm_gpu.py --help works."""
    result = subprocess.run(
        [sys.executable, "benchmarks/benchmark_fractal_pelm_gpu.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "FractalPELMGPU" in result.stdout or "benchmark" in result.stdout.lower()


def test_measure_memory_footprint_help() -> None:
    """Test that measure_memory_footprint.py --help works."""
    result = subprocess.run(
        [sys.executable, "benchmarks/measure_memory_footprint.py", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent.parent,
    )
    assert result.returncode == 0
    assert "Memory" in result.stdout or "memory" in result.stdout.lower()
