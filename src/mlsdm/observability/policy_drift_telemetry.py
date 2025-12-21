"""Policy Drift Telemetry for Moral Filter Monitoring.

This module provides Prometheus metrics for detecting and tracking
policy drift in adaptive moral filters. Supports automated alerting
for dangerous threshold changes.

Metrics:
    - moral_threshold_current: Current threshold value per filter
    - moral_threshold_drift_rate: Rate of threshold change per operation
    - moral_threshold_violations: Boundary violations (MIN/MAX)
    - moral_ema_deviation: Deviation from target 0.5
    - moral_drift_events: Total drift events by severity

Usage:
    from mlsdm.observability.policy_drift_telemetry import record_threshold_change

    record_threshold_change(
        filter_id="production",
        old_threshold=0.5,
        new_threshold=0.52,
        ema_value=0.55
    )
"""

from prometheus_client import Counter, Gauge

# Metrics for policy drift detection
moral_threshold_gauge = Gauge(
    "mlsdm_moral_threshold_current",
    "Current moral filter threshold value",
    ["filter_id"],
)

moral_threshold_drift_rate = Gauge(
    "mlsdm_moral_threshold_drift_rate",
    "Rate of threshold change per operation",
    ["filter_id"],
)

moral_threshold_violations = Counter(
    "mlsdm_moral_threshold_violations_total",
    "Total boundary violations (MIN/MAX)",
    ["filter_id", "boundary"],
)

moral_ema_deviation = Gauge(
    "mlsdm_moral_ema_deviation",
    "Deviation of EMA from target 0.5",
    ["filter_id"],
)

moral_drift_events = Counter(
    "mlsdm_moral_drift_events_total",
    "Total policy drift events detected",
    ["filter_id", "severity"],
)


def record_threshold_change(
    filter_id: str, old_threshold: float, new_threshold: float, ema_value: float
) -> None:
    """Record threshold change and calculate drift.

    Args:
        filter_id: Unique identifier for the moral filter
        old_threshold: Previous threshold value
        new_threshold: New threshold value
        ema_value: Current EMA acceptance rate

    Side Effects:
        Updates Prometheus metrics for monitoring and alerting.
        Tracks current threshold, boundary violations, EMA deviation,
        and drift event severity.
    """
    # Update current value
    moral_threshold_gauge.labels(filter_id=filter_id).set(new_threshold)

    # Calculate drift magnitude
    drift = abs(new_threshold - old_threshold)
    moral_threshold_drift_rate.labels(filter_id=filter_id).set(drift)

    # Check for boundary violations
    if new_threshold <= 0.30:
        moral_threshold_violations.labels(filter_id=filter_id, boundary="MIN").inc()
    elif new_threshold >= 0.90:
        moral_threshold_violations.labels(filter_id=filter_id, boundary="MAX").inc()

    # EMA deviation from healthy 0.5
    deviation = abs(ema_value - 0.5)
    moral_ema_deviation.labels(filter_id=filter_id).set(deviation)

    # Detect drift severity (using absolute threshold differences)
    if drift > 0.1:  # CRITICAL: >0.1 absolute change
        moral_drift_events.labels(filter_id=filter_id, severity="critical").inc()
    elif drift > 0.05:  # WARNING: >0.05 absolute change
        moral_drift_events.labels(filter_id=filter_id, severity="warning").inc()
