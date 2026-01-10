import logging
from typing import Any

from mlsdm.cognition.moral_filter_v2 import MoralFilterV2


def test_moral_filter_logs_boundary_cases(caplog) -> None:
    """Debug logging should capture boundary evaluations near thresholds."""
    caplog.set_level(logging.DEBUG, logger="mlsdm.cognition.moral_filter_v2")
    moral_filter = MoralFilterV2(initial_threshold=0.31)

    result = moral_filter.evaluate(moral_filter.MIN_THRESHOLD + 0.005)

    assert result is False
    assert any(
        "MoralFilterV2 boundary case" in record.message for record in caplog.records
    )


def test_compute_moral_value_updates_metadata() -> None:
    """Metadata and context should capture pattern counts during scoring."""
    moral_filter = MoralFilterV2()
    metadata: dict[str, Any] = {}
    context: dict[str, Any] = {}

    score = moral_filter.compute_moral_value(
        "help others and avoid harm", metadata=metadata, context=context
    )

    assert 0.0 <= score <= 1.0
    assert metadata["harmful_count"] == 1
    assert metadata["positive_count"] == 1
    assert context["metadata"]["harmful_count"] == 1
    assert context["metadata"]["positive_count"] == 1


def test_adapt_increases_threshold() -> None:
    """Positive signals above dead-band should raise threshold.

    The adaptive controller uses EMA with alpha=0.1 and dead-band=0.1.
    Starting from equilibrium (ema=0.5), three consecutive positive
    signals are required to exceed the dead-band and trigger adaptation.
    """
    moral_filter = MoralFilterV2(initial_threshold=0.5)

    for _ in range(3):
        moral_filter.adapt(accepted=True)

    assert moral_filter.threshold > 0.5
    assert moral_filter.ema_accept_rate > 0.63


def test_record_drift_logs_warning(caplog) -> None:
    """Significant drift should emit a warning and record history."""
    caplog.set_level(logging.WARNING, logger="mlsdm.cognition.moral_filter_v2")
    moral_filter = MoralFilterV2(initial_threshold=0.5)

    old = moral_filter.threshold
    new = old + 0.06  # triggers warning tier

    moral_filter._record_drift(old, new)

    assert any("Significant drift" in record.message for record in caplog.records)
    assert moral_filter._drift_history[-1] == new
