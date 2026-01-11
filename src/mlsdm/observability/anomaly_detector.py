"""Compute observability anomaly scores from runtime telemetry."""

from __future__ import annotations

import logging
from typing import Any

from . import policy_drift_telemetry
from .memory_telemetry import MemoryMetricsExporter, get_memory_metrics_exporter
from .metrics import MetricsExporter, get_metrics_exporter

logger = logging.getLogger(__name__)


def compute_observability_anomaly_score(
    *,
    metrics_exporter: MetricsExporter | None = None,
    memory_exporter: MemoryMetricsExporter | None = None,
) -> float:
    """Compute a normalized anomaly score using observability telemetry.

    Returns:
        Score in [0.0, 1.0], where higher means more anomalous.
    """
    system_score = 0.0
    memory_score = 0.0
    policy_score = 0.0

    metrics_values: dict[str, Any] = {}
    if metrics_exporter is None:
        try:
            metrics_exporter = get_metrics_exporter()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load metrics exporter")
            metrics_exporter = None

    if metrics_exporter is not None:
        try:
            metrics_values = metrics_exporter.get_current_values()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to read metrics exporter values")

    if metrics_values:
        emergency_active = 1.0 if metrics_values.get("emergency_shutdown_active", 0.0) >= 1.0 else 0.0
        cognitive_emergency = _scale(metrics_values.get("cognitive_emergency_total", 0.0), 1.0)
        inflight_pressure = _scale(metrics_values.get("requests_inflight", 0.0), 50.0)
        bulkhead_pressure = max(
            _scale(metrics_values.get("bulkhead_queue_depth", 0.0), 50.0),
            _scale(metrics_values.get("bulkhead_rejected_total", 0.0), 10.0),
        )
        http_pressure = _scale(metrics_values.get("http_requests_in_flight", 0.0), 100.0)
        system_score = max(
            emergency_active,
            cognitive_emergency,
            inflight_pressure,
            bulkhead_pressure,
            http_pressure,
        )

    if memory_exporter is None:
        try:
            memory_exporter = get_memory_metrics_exporter()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to load memory metrics exporter")
            memory_exporter = None

    if memory_exporter is not None:
        try:
            pelm_utilization_ratio = _get_metric_value(memory_exporter.pelm_utilization_ratio)
            pelm_capacity_used = _get_metric_value(memory_exporter.pelm_capacity_used)
            pelm_capacity_total = _get_metric_value(memory_exporter.pelm_capacity_total)
            pelm_corruption_total = _get_metric_sum(memory_exporter.pelm_corruption_total)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to read memory metrics exporter values")
        else:
            utilization_ratio = pelm_utilization_ratio
            if pelm_capacity_total > 0:
                utilization_ratio = max(utilization_ratio, pelm_capacity_used / pelm_capacity_total)
            memory_pressure = _ramp(utilization_ratio, start=0.8, span=0.2)
            corruption_score = 1.0 if pelm_corruption_total > 0 else 0.0
            memory_score = max(memory_pressure, corruption_score)

    policy_drift_score = _scale(_get_metric_max(policy_drift_telemetry.moral_threshold_drift_rate), 0.1)
    ema_deviation_score = _scale(_get_metric_max(policy_drift_telemetry.moral_ema_deviation), 0.2)
    threshold_violation_score = (
        1.0 if _get_metric_sum(policy_drift_telemetry.moral_threshold_violations) > 0 else 0.0
    )
    critical_drift_events = _get_metric_sum_by_label(
        policy_drift_telemetry.moral_drift_events, "severity", "critical"
    )
    warning_drift_events = _get_metric_sum_by_label(
        policy_drift_telemetry.moral_drift_events, "severity", "warning"
    )
    drift_event_score = 1.0 if critical_drift_events > 0 else (0.6 if warning_drift_events > 0 else 0.0)

    policy_score = max(
        policy_drift_score,
        ema_deviation_score,
        threshold_violation_score,
        drift_event_score,
    )

    composite_score = 0.5 * system_score + 0.25 * memory_score + 0.25 * policy_score
    return _clamp(composite_score)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _scale(value: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return _clamp(float(value) / threshold)


def _ramp(value: float, *, start: float, span: float) -> float:
    if span <= 0:
        return 0.0
    return _clamp((float(value) - start) / span)


def _get_metric_value(metric: Any) -> float:
    values = [sample.value for family in metric.collect() for sample in family.samples]
    if not values:
        return 0.0
    return float(values[-1])


def _get_metric_sum(metric: Any) -> float:
    return float(sum(sample.value for family in metric.collect() for sample in family.samples))


def _get_metric_max(metric: Any) -> float:
    values = [sample.value for family in metric.collect() for sample in family.samples]
    return float(max(values, default=0.0))


def _get_metric_sum_by_label(metric: Any, label: str, label_value: str) -> float:
    total = 0.0
    for family in metric.collect():
        for sample in family.samples:
            if sample.labels.get(label) == label_value:
                total += float(sample.value)
    return total
