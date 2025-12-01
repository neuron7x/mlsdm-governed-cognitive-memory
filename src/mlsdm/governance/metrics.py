"""
MLSDM Governance Metrics Module.

Provides metrics collection and export for governance decisions,
including counters for total decisions, blocked content, modified content,
and escalated content, broken down by operational mode.

Integration:
    Metrics are automatically recorded by the enforcer module.
    Export to Prometheus via the existing metrics infrastructure.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Counters
# =============================================================================


@dataclass
class GovernanceCounters:
    """Thread-safe counters for governance metrics.

    Attributes:
        total_decisions: Total governance decisions made
        allowed_total: Total allowed decisions
        blocked_total: Total blocked decisions
        modified_total: Total modified decisions
        escalated_total: Total escalated decisions
        per_mode: Counters broken down by mode
        per_rule: Counters broken down by rule ID
    """

    total_decisions: int = 0
    allowed_total: int = 0
    blocked_total: int = 0
    modified_total: int = 0
    escalated_total: int = 0
    per_mode: dict[str, dict[str, int]] = field(default_factory=dict)
    per_rule: dict[str, int] = field(default_factory=dict)


class GovernanceMetrics:
    """Thread-safe governance metrics collector.

    Usage:
        >>> metrics = GovernanceMetrics()
        >>> metrics.record_decision("block", "R001", "normal")
        >>> print(metrics.get_summary())
    """

    _instance: GovernanceMetrics | None = None
    _lock: threading.Lock

    def __new__(cls) -> GovernanceMetrics:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._counters = GovernanceCounters()
            cls._instance._lock = threading.Lock()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance for testing."""
        cls._instance = None

    def record_decision(
        self,
        action: str,
        rule_id: str | None,
        mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a governance decision.

        Args:
            action: Decision action ("allow", "block", "modify", "escalate")
            rule_id: ID of the rule that triggered the decision
            mode: Operational mode
            metadata: Additional metadata (optional)
        """
        with self._lock:
            self._counters.total_decisions += 1

            # Increment action-specific counter
            if action == "allow":
                self._counters.allowed_total += 1
            elif action == "block":
                self._counters.blocked_total += 1
            elif action == "modify":
                self._counters.modified_total += 1
            elif action == "escalate":
                self._counters.escalated_total += 1

            # Validate action before per-mode tracking
            valid_actions = {"allow", "block", "modify", "escalate"}

            # Increment per-mode counter
            if mode not in self._counters.per_mode:
                self._counters.per_mode[mode] = {
                    "total": 0,
                    "allowed": 0,
                    "blocked": 0,
                    "modified": 0,
                    "escalated": 0,
                }
            self._counters.per_mode[mode]["total"] += 1
            # Only increment known action keys
            if action in valid_actions:
                # Map action to counter key (allow -> allowed, etc.)
                action_key = f"{action}ed" if action != "allow" else "allowed"
                if action == "modify":
                    action_key = "modified"
                elif action == "escalate":
                    action_key = "escalated"
                elif action == "block":
                    action_key = "blocked"
                self._counters.per_mode[mode][action_key] = (
                    self._counters.per_mode[mode].get(action_key, 0) + 1
                )

            # Increment per-rule counter
            if rule_id:
                self._counters.per_rule[rule_id] = (
                    self._counters.per_rule.get(rule_id, 0) + 1
                )

        logger.debug(
            "Recorded governance decision: action=%s rule_id=%s mode=%s",
            action,
            rule_id,
            mode,
        )

    def get_counters(self) -> GovernanceCounters:
        """Get current counter values (thread-safe copy).

        Returns:
            Copy of current counters
        """
        with self._lock:
            return GovernanceCounters(
                total_decisions=self._counters.total_decisions,
                allowed_total=self._counters.allowed_total,
                blocked_total=self._counters.blocked_total,
                modified_total=self._counters.modified_total,
                escalated_total=self._counters.escalated_total,
                per_mode=dict(self._counters.per_mode),
                per_rule=dict(self._counters.per_rule),
            )

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary as a dictionary.

        Returns:
            Dictionary with all metrics
        """
        counters = self.get_counters()
        return {
            "total_decisions": counters.total_decisions,
            "allowed_total": counters.allowed_total,
            "blocked_total": counters.blocked_total,
            "modified_total": counters.modified_total,
            "escalated_total": counters.escalated_total,
            "per_mode": counters.per_mode,
            "per_rule": counters.per_rule,
            "block_rate": (
                counters.blocked_total / counters.total_decisions
                if counters.total_decisions > 0
                else 0.0
            ),
        }

    def reset_counters(self) -> None:
        """Reset all counters to zero."""
        with self._lock:
            self._counters = GovernanceCounters()
        logger.info("Governance metrics counters reset")


# =============================================================================
# Prometheus Integration
# =============================================================================


def register_prometheus_metrics() -> dict[str, Any]:
    """Register governance metrics with Prometheus.

    Returns:
        Dictionary of registered Prometheus metrics objects

    Note:
        This function is optional and only works if prometheus_client is installed.
    """
    try:
        from prometheus_client import Counter, Gauge
    except ImportError:
        logger.warning("prometheus_client not installed, skipping Prometheus registration")
        return {}

    metrics = {
        "governance_decisions_total": Counter(
            "mlsdm_governance_decisions_total",
            "Total governance decisions",
            ["action", "mode", "rule_id"],
        ),
        "governance_blocked_total": Counter(
            "mlsdm_governance_blocked_total",
            "Total blocked governance decisions",
            ["mode", "rule_id"],
        ),
        "governance_modified_total": Counter(
            "mlsdm_governance_modified_total",
            "Total modified governance decisions",
            ["mode"],
        ),
        "governance_escalated_total": Counter(
            "mlsdm_governance_escalated_total",
            "Total escalated governance decisions",
            ["mode", "priority"],
        ),
        "governance_mode": Gauge(
            "mlsdm_governance_mode",
            "Current operational mode (1=normal, 2=cautious, 3=emergency)",
        ),
    }

    logger.info("Registered governance metrics with Prometheus")
    return metrics


# =============================================================================
# Logging Helpers
# =============================================================================


def log_governance_event(
    action: str,
    rule_id: str | None,
    mode: str,
    reason: str,
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Log a governance event with structured fields.

    This function logs governance events in a format suitable for
    structured log aggregation (e.g., ELK, Loki).

    Args:
        action: Decision action
        rule_id: Rule ID (if any)
        mode: Operational mode
        reason: Decision reason
        correlation_id: Request correlation ID
        metadata: Additional metadata (sanitized, no PII)
    """
    # Build log entry (no sensitive content)
    log_entry = {
        "event_type": "governance_decision",
        "action": action,
        "rule_id": rule_id,
        "mode": mode,
        "reason": reason,
    }

    if correlation_id:
        log_entry["correlation_id"] = correlation_id

    if metadata:
        # Only include safe metadata fields
        safe_fields = ["log_level", "escalation_channel", "escalation_priority"]
        log_entry["metadata"] = {
            k: v for k, v in metadata.items() if k in safe_fields
        }

    # Select log level based on action
    if action in ("block", "escalate"):
        logger.warning("Governance event: %s", log_entry)
    else:
        logger.info("Governance event: %s", log_entry)


# =============================================================================
# Public API
# =============================================================================


def get_metrics() -> GovernanceMetrics:
    """Get the singleton GovernanceMetrics instance.

    Returns:
        GovernanceMetrics instance
    """
    return GovernanceMetrics()


def record_decision(
    action: str,
    rule_id: str | None,
    mode: str,
    reason: str | None = None,
    correlation_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Convenience function to record a governance decision.

    This records the decision in metrics and logs it.

    Args:
        action: Decision action
        rule_id: Rule ID (if any)
        mode: Operational mode
        reason: Decision reason (optional)
        correlation_id: Request correlation ID (optional)
        metadata: Additional metadata (optional)
    """
    metrics = get_metrics()
    metrics.record_decision(action, rule_id, mode, metadata)

    if reason:
        log_governance_event(action, rule_id, mode, reason, correlation_id, metadata)
