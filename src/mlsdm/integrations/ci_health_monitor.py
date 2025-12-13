"""
CI Health Monitoring Integration

Monitors CI pipeline health and integrates with auto-recovery mechanisms.
Provides hooks for GitHub Actions, GitLab CI, and other CI platforms.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests


class CIStatus(Enum):
    """CI workflow status."""

    SUCCESS = "success"
    FAILURE = "failure"
    PENDING = "pending"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class CIWorkflowRun:
    """CI workflow run metadata."""

    workflow_id: str
    run_id: str
    status: CIStatus
    conclusion: Optional[str]
    created_at: datetime
    updated_at: datetime
    html_url: str
    repository: str


class CIHealthMonitor:
    """
    CI Health Monitor with auto-recovery integration.

    Monitors CI pipeline health and provides hooks for auto-recovery
    when deployment health degrades.

    Example:
        >>> monitor = CIHealthMonitor(
        ...     github_token="ghp_xxx",
        ...     repository="neuron7x/mlsdm",
        ... )
        >>> status = monitor.get_latest_workflow_status()
        >>> if status == CIStatus.FAILURE:
        ...     monitor.trigger_recovery()
    """

    def __init__(
        self,
        github_token: Optional[str] = None,
        repository: Optional[str] = None,
        failure_threshold: int = 3,
        recovery_cooldown_seconds: int = 300,
        enable_auto_recovery: bool = False,
    ) -> None:
        """
        Initialize CI health monitor.

        Args:
            github_token: GitHub API token for authentication
            repository: Repository in format "owner/repo"
            failure_threshold: Number of consecutive failures before triggering recovery
            recovery_cooldown_seconds: Cooldown period between recovery attempts
            enable_auto_recovery: Enable automatic recovery workflow dispatch
        """
        self.github_token = github_token
        self.repository = repository
        self.failure_threshold = failure_threshold
        self.recovery_cooldown_seconds = recovery_cooldown_seconds
        self.enable_auto_recovery = enable_auto_recovery

        self.logger = logging.getLogger(__name__)
        self._consecutive_failures = 0
        self._last_recovery_time: Optional[float] = None
        self._recovery_callbacks: List[Callable[[], None]] = []

    def register_recovery_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called when recovery is triggered.

        Args:
            callback: Function to call on recovery trigger
        """
        self._recovery_callbacks.append(callback)

    def get_latest_workflow_status(
        self, workflow_name: str = "ci-neuro-cognitive-engine.yml"
    ) -> CIStatus:
        """
        Get the latest workflow run status from GitHub Actions.

        Args:
            workflow_name: Name of the workflow file

        Returns:
            CI status enum
        """
        if not self.github_token or not self.repository:
            self.logger.warning("GitHub token or repository not configured")
            return CIStatus.UNKNOWN

        try:
            url = f"https://api.github.com/repos/{self.repository}/actions/workflows/{workflow_name}/runs"
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            if not data.get("workflow_runs"):
                return CIStatus.UNKNOWN

            latest_run = data["workflow_runs"][0]
            conclusion = latest_run.get("conclusion")

            if conclusion == "success":
                return CIStatus.SUCCESS
            elif conclusion == "failure":
                return CIStatus.FAILURE
            elif conclusion == "cancelled":
                return CIStatus.CANCELLED
            elif latest_run.get("status") == "in_progress":
                return CIStatus.PENDING
            else:
                return CIStatus.UNKNOWN

        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch workflow status: {e}")
            return CIStatus.UNKNOWN

    def check_health(self) -> Dict[str, Any]:
        """
        Check overall CI health status.

        Returns:
            Health status dictionary with metrics
        """
        status = self.get_latest_workflow_status()

        if status == CIStatus.FAILURE:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0

        health_status = {
            "latest_status": status.value,
            "consecutive_failures": self._consecutive_failures,
            "should_recover": self._should_trigger_recovery(),
        }

        # Auto-recovery if enabled
        if self.enable_auto_recovery and self._should_trigger_recovery():
            self.trigger_recovery()

        return health_status

    def _should_trigger_recovery(self) -> bool:
        """Check if recovery should be triggered."""
        if self._consecutive_failures < self.failure_threshold:
            return False

        # Check cooldown
        if self._last_recovery_time is not None:
            elapsed = time.time() - self._last_recovery_time
            if elapsed < self.recovery_cooldown_seconds:
                return False

        return True

    def trigger_recovery(self) -> bool:
        """
        Trigger recovery workflow or callbacks.

        Returns:
            True if recovery was triggered successfully
        """
        if not self._should_trigger_recovery():
            self.logger.info("Recovery not needed or in cooldown")
            return False

        self.logger.warning(
            f"Triggering recovery after {self._consecutive_failures} consecutive failures"
        )

        # Call registered callbacks
        for callback in self._recovery_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"Recovery callback failed: {e}")

        self._last_recovery_time = time.time()
        self._consecutive_failures = 0

        return True
