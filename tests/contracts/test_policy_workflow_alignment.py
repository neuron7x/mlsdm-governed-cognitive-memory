"""
Test suite to validate policy-workflow-script alignment.

This test ensures that policy files, CI workflows, and scripts remain synchronized
to prevent truth-alignment drift.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not installed")
class TestPolicyWorkflowAlignment:
    """Test that policy files match workflow and script reality."""

    @pytest.fixture
    def repo_root(self) -> Path:
        """Get repository root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def security_policy(self, repo_root: Path) -> dict:
        """Load security baseline policy."""
        policy_path = repo_root / "policy" / "security-baseline.yaml"
        with open(policy_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_policy_required_checks_exist(
        self, repo_root: Path, security_policy: dict
    ) -> None:
        """Verify all required checks reference existing files/commands."""
        required_checks = security_policy.get("required_checks", [])
        assert len(required_checks) > 0, "Policy must define required checks"

        for check in required_checks:
            check_name = check.get("name")

            # Check workflow file exists if specified
            if "workflow_file" in check:
                workflow_file = repo_root / check["workflow_file"]
                assert workflow_file.exists(), (
                    f"{check_name}: Workflow file not found: {check['workflow_file']}"
                )

            # Check script exists if specified
            if "script" in check:
                script_path = repo_root / check["script"].lstrip("./")
                assert script_path.exists(), (
                    f"{check_name}: Script not found: {check['script']}"
                )

    def test_coverage_threshold_consistency(
        self, repo_root: Path, security_policy: dict
    ) -> None:
        """Verify coverage threshold is consistent between policy and coverage_gate.sh."""
        # Get policy coverage threshold
        required_checks = security_policy.get("required_checks", [])
        coverage_check = next(
            (c for c in required_checks if c.get("name") == "coverage_gate"),
            None
        )
        assert coverage_check is not None, "coverage_gate check not found in policy"

        policy_threshold = coverage_check.get("minimum_coverage")
        assert policy_threshold is not None, "minimum_coverage not specified in policy"

        # Parse coverage_gate.sh for default threshold
        coverage_script = repo_root / "coverage_gate.sh"
        assert coverage_script.exists(), "coverage_gate.sh not found"

        with open(coverage_script, encoding="utf-8") as f:
            content = f.read()

        # Look for COVERAGE_MIN default value
        # Format: COVERAGE_MIN="${COVERAGE_MIN:-65}"
        import re
        match = re.search(r'COVERAGE_MIN="\$\{COVERAGE_MIN:-(\d+)\}"', content)
        assert match is not None, "COVERAGE_MIN default not found in coverage_gate.sh"

        script_threshold = int(match.group(1))

        # Both should be in percentage (not ratio)
        # Policy should match script default
        assert policy_threshold == script_threshold, (
            f"Coverage threshold mismatch: "
            f"policy={policy_threshold}%, script default={script_threshold}%"
        )

    def test_security_modules_exist(
        self, repo_root: Path, security_policy: dict
    ) -> None:
        """Verify referenced security modules exist in codebase."""
        security_reqs = security_policy.get("security_requirements", {})

        # Check input validation modules
        input_val = security_reqs.get("input_validation", {})
        modules_to_check = [
            input_val.get("llm_safety_module"),
            input_val.get("payload_scrubber_module"),
        ]

        for module_path in modules_to_check:
            if not module_path:
                continue

            # Convert module path to file path
            # e.g., mlsdm.security.llm_safety -> src/mlsdm/security/llm_safety.py
            parts = module_path.split(".")
            if parts[0] == "mlsdm":
                file_path = repo_root / "src" / "/".join(parts)
                py_path = file_path.parent / f"{file_path.name}.py"
                init_path = file_path / "__init__.py"

                assert py_path.exists() or init_path.exists(), (
                    f"Module not found: {module_path} "
                    f"(checked {py_path} and {init_path})"
                )

    def test_scrubber_implementation_exists(self, repo_root: Path, security_policy: dict) -> None:
        """Verify payload scrubber implementation exists."""
        security_reqs = security_policy.get("security_requirements", {})
        data_protection = security_reqs.get("data_protection", {})

        scrubber_module = data_protection.get("scrubber_implementation")
        assert scrubber_module is not None, "scrubber_implementation not specified"

        # Convert to file path
        parts = scrubber_module.split(".")
        if parts[0] == "mlsdm":
            file_path = repo_root / "src" / "/".join(parts)
            py_path = file_path.parent / f"{file_path.name}.py"
            init_path = file_path / "__init__.py"

            assert py_path.exists() or init_path.exists(), (
                f"Scrubber implementation not found: {scrubber_module}"
            )

    def test_policy_validator_script_passes(self, repo_root: Path) -> None:
        """Verify the policy validator script passes."""
        validator_script = repo_root / "scripts" / "validate_policy_config.py"
        assert validator_script.exists(), "validate_policy_config.py not found"

        # Run the validator
        result = subprocess.run(
            ["python", str(validator_script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        # Should pass (exit 0) and report success
        assert result.returncode == 0, (
            f"Policy validation failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "All critical validations passed" in result.stdout, (
            "Policy validation did not report success"
        )

    def test_sast_workflows_exist(self, repo_root: Path, security_policy: dict) -> None:
        """Verify SAST scan workflows exist and are configured correctly."""
        required_checks = security_policy.get("required_checks", [])

        # Find SAST-related checks
        sast_checks = [
            check for check in required_checks
            if check.get("name") in ["bandit", "semgrep", "codeql"]
        ]

        assert len(sast_checks) > 0, "No SAST checks found in policy"

        for check in sast_checks:
            workflow_file = check.get("workflow_file")
            if workflow_file:
                workflow_path = repo_root / workflow_file
                assert workflow_path.exists(), (
                    f"{check['name']}: Workflow file not found: {workflow_file}"
                )

                # Read workflow and verify job exists
                with open(workflow_path, encoding="utf-8") as f:
                    workflow_content = yaml.safe_load(f)

                jobs = workflow_content.get("jobs", {})
                check_name = check.get("name")

                # Verify job name exists in workflow (case-insensitive match)
                job_names = [name.lower() for name in jobs]
                assert check_name.lower() in job_names, (
                    f"{check_name}: Job not found in {workflow_file}"
                )
