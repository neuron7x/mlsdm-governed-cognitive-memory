"""
Tests for CI Health Monitor integration.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from mlsdm.integrations import CIHealthMonitor, CIStatus


class TestCIHealthMonitor:
    """Test CI health monitoring integration."""

    def test_initialization(self) -> None:
        """Test monitor initialization."""
        monitor = CIHealthMonitor(
            github_token="ghp_test",
            repository="owner/repo",
            failure_threshold=3,
            recovery_cooldown_seconds=60,
        )

        assert monitor.github_token == "ghp_test"
        assert monitor.repository == "owner/repo"
        assert monitor.failure_threshold == 3
        assert monitor._consecutive_failures == 0

    def test_get_latest_workflow_status_success(self) -> None:
        """Test fetching successful workflow status."""
        monitor = CIHealthMonitor(github_token="ghp_test", repository="owner/repo")

        mock_response = {
            "workflow_runs": [
                {
                    "id": 123,
                    "conclusion": "success",
                    "status": "completed",
                }
            ]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            status = monitor.get_latest_workflow_status()
            assert status == CIStatus.SUCCESS

    def test_check_health_tracks_consecutive_failures(self) -> None:
        """Test that check_health tracks consecutive failures."""
        monitor = CIHealthMonitor(
            github_token="ghp_test", repository="owner/repo", failure_threshold=3
        )

        mock_response = {
            "workflow_runs": [{"id": 123, "conclusion": "failure", "status": "completed"}]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            # First failure
            health = monitor.check_health()
            assert health["consecutive_failures"] == 1

            # Third failure - should trigger recovery
            health = monitor.check_health()
            health = monitor.check_health()
            assert health["consecutive_failures"] == 3
            assert health["should_recover"]

    def test_recovery_callback(self) -> None:
        """Test recovery callback execution."""
        monitor = CIHealthMonitor(failure_threshold=2)

        callback_called = []

        def recovery_callback() -> None:
            callback_called.append(True)

        monitor.register_recovery_callback(recovery_callback)

        # Trigger recovery
        monitor._consecutive_failures = 3
        monitor._last_recovery_time = None

        result = monitor.trigger_recovery()
        assert result is True
        assert len(callback_called) == 1

    def test_missing_configuration(self) -> None:
        """Test behavior with missing GitHub configuration."""
        monitor = CIHealthMonitor()  # No token or repo

        status = monitor.get_latest_workflow_status()
        assert status == CIStatus.UNKNOWN
