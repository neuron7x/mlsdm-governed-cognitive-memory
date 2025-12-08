"""Dependency health checks to prevent accidental removal of critical packages.

This module provides tests that guard against accidental removal or misconfiguration
of essential development and production dependencies.
"""

import os
from pathlib import Path


class TestDependencyHealth:
    """Test that critical dependencies are properly configured."""

    @staticmethod
    def _read_requirements_file(filename: str) -> list[str]:
        """Read a requirements file and return list of package lines.

        Args:
            filename: Name of the requirements file (relative to repo root)

        Returns:
            List of non-empty, non-comment lines
        """
        repo_root = Path(__file__).parent.parent.parent
        req_file = repo_root / filename
        if not req_file.exists():
            return []

        with open(req_file) as f:
            lines = f.readlines()

        # Filter out comments and empty lines
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("-r")
        ]

    def test_pytest_in_dev_requirements(self) -> None:
        """Test that pytest is in dev requirements."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("pytest" in req.lower() for req in dev_reqs), (
            "pytest not found in requirements-dev.txt - testing framework is required"
        )

    def test_mypy_in_dev_requirements(self) -> None:
        """Test that mypy is in dev requirements."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("mypy" in req.lower() for req in dev_reqs), (
            "mypy not found in requirements-dev.txt - type checking is required"
        )

    def test_ruff_in_dev_requirements(self) -> None:
        """Test that ruff is in dev requirements."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("ruff" in req.lower() for req in dev_reqs), (
            "ruff not found in requirements-dev.txt - linting is required"
        )

    def test_hypothesis_in_dev_requirements(self) -> None:
        """Test that hypothesis is in dev requirements."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("hypothesis" in req.lower() for req in dev_reqs), (
            "hypothesis not found in requirements-dev.txt - property-based testing is required"
        )

    def test_opentelemetry_in_dev_requirements(self) -> None:
        """Test that OpenTelemetry is in dev requirements for full testing."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        has_otel_api = any("opentelemetry-api" in req.lower() for req in dev_reqs)
        has_otel_sdk = any("opentelemetry-sdk" in req.lower() for req in dev_reqs)
        assert has_otel_api and has_otel_sdk, (
            "OpenTelemetry (api and sdk) not found in requirements-dev.txt - "
            "full observability testing requires OTEL packages"
        )

    def test_opentelemetry_not_in_core_requirements(self) -> None:
        """Test that OpenTelemetry is NOT in core requirements (it's optional)."""
        core_reqs = self._read_requirements_file("requirements.txt")
        has_otel = any("opentelemetry" in req.lower() for req in core_reqs)
        assert not has_otel, (
            "OpenTelemetry found in requirements.txt - "
            "it should be optional, not a core dependency. "
            "Users should install with `pip install mlsdm[tracing]` for OTEL support."
        )

    def test_core_dependencies_present(self) -> None:
        """Test that critical core dependencies are present."""
        core_reqs = self._read_requirements_file("requirements.txt")

        # Critical packages that must be in core
        critical_packages = ["numpy", "fastapi", "pydantic", "pyyaml"]

        for pkg in critical_packages:
            assert any(pkg in req.lower() for req in core_reqs), (
                f"{pkg} not found in requirements.txt - this is a critical core dependency"
            )

    def test_dev_requirements_includes_core(self) -> None:
        """Test that requirements-dev.txt includes requirements.txt."""
        repo_root = Path(__file__).parent.parent.parent
        dev_req_file = repo_root / "requirements-dev.txt"

        if not dev_req_file.exists():
            raise AssertionError("requirements-dev.txt not found")

        with open(dev_req_file) as f:
            content = f.read()

        assert "-r requirements.txt" in content, (
            "requirements-dev.txt should include requirements.txt via '-r requirements.txt'"
        )

    def test_pytest_cov_in_dev_requirements(self) -> None:
        """Test that pytest-cov is in dev requirements for coverage."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("pytest-cov" in req.lower() for req in dev_reqs), (
            "pytest-cov not found in requirements-dev.txt - coverage reporting is required"
        )

    def test_pytest_asyncio_in_dev_requirements(self) -> None:
        """Test that pytest-asyncio is in dev requirements for async tests."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("pytest-asyncio" in req.lower() for req in dev_reqs), (
            "pytest-asyncio not found in requirements-dev.txt - async testing is required"
        )

    def test_httpx_in_dev_requirements(self) -> None:
        """Test that httpx is in dev requirements for API testing."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("httpx" in req.lower() for req in dev_reqs), (
            "httpx not found in requirements-dev.txt - HTTP client for testing is required"
        )

    def test_locust_in_dev_requirements(self) -> None:
        """Test that locust is in dev requirements for load testing."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        assert any("locust" in req.lower() for req in dev_reqs), (
            "locust not found in requirements-dev.txt - load testing tool is required"
        )

    def test_types_packages_in_dev_requirements(self) -> None:
        """Test that type stubs are in dev requirements for mypy."""
        dev_reqs = self._read_requirements_file("requirements-dev.txt")
        has_types = any("types-" in req.lower() for req in dev_reqs)
        assert has_types, (
            "No types-* packages found in requirements-dev.txt - "
            "type stubs are needed for mypy type checking"
        )
