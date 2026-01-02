"""Unit tests for scripts/ci/generate_evidence_inputs.py."""

import json
import subprocess
from pathlib import Path


def test_generate_evidence_inputs_with_all_files(tmp_path: Path) -> None:
    """Test evidence inputs generation when all files exist."""
    # Get the repo root (where the script actually is)
    repo_root = Path(__file__).parent.parent.parent

    # Create test files in tmp_path
    (tmp_path / "coverage.xml").write_text("<coverage/>")
    (tmp_path / "coverage-gate.log").write_text("log")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "junit.xml").write_text("<testsuite/>")

    # Execute script from repo root but with tmp_path as working directory
    result = subprocess.run(
        ["python3", str(repo_root / "scripts/ci/generate_evidence_inputs.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Verify
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    output = Path("/tmp/evidence-inputs.json")
    assert output.exists(), "Output file not created"

    inputs = json.loads(output.read_text())
    assert inputs["coverage_xml"] == "coverage.xml"
    assert inputs["coverage_log"] == "coverage-gate.log"
    assert inputs["junit_xml"] == "reports/junit.xml"

    # Check stderr output
    assert "✓ Generated" in result.stderr
    assert "✓ coverage_xml: coverage.xml" in result.stderr


def test_generate_evidence_inputs_missing_coverage(tmp_path: Path) -> None:
    """Test evidence inputs generation with missing coverage files."""
    # Get the repo root
    repo_root = Path(__file__).parent.parent.parent

    # Setup - only create junit.xml in tmp_path
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "junit.xml").write_text("<testsuite/>")

    # Execute
    result = subprocess.run(
        ["python3", str(repo_root / "scripts/ci/generate_evidence_inputs.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    # Verify
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    output = Path("/tmp/evidence-inputs.json")
    assert output.exists()

    inputs = json.loads(output.read_text())
    assert inputs["coverage_xml"] is None
    assert inputs["coverage_log"] is None
    assert inputs["junit_xml"] == "reports/junit.xml"

    # Check stderr output indicates missing files
    assert "✗ coverage_xml: missing" in result.stderr
    assert "✗ coverage_log: missing" in result.stderr
    assert "✓ junit_xml: reports/junit.xml" in result.stderr


def test_generate_evidence_inputs_from_repo_root() -> None:
    """Test script execution from actual repository root."""
    # This test runs from the actual repo, so we can't control file presence
    # but we can verify the script executes without errors
    repo_root = Path(__file__).parent.parent.parent

    result = subprocess.run(
        ["python3", str(repo_root / "scripts/ci/generate_evidence_inputs.py")],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    # Should succeed regardless of file presence
    assert result.returncode == 0, f"Script failed: {result.stderr}"
    assert "Generated /tmp/evidence-inputs.json" in result.stderr

    # Verify output is valid JSON
    output = Path("/tmp/evidence-inputs.json")
    assert output.exists()
    inputs = json.loads(output.read_text())

    # Verify expected keys exist
    assert "coverage_xml" in inputs
    assert "coverage_log" in inputs
    assert "junit_xml" in inputs

    # Values should be either string or None
    for key, value in inputs.items():
        assert value is None or isinstance(value, str), f"Invalid type for {key}: {type(value)}"
