import logging

from mlsdm.cognition.moral_filter_v2 import MoralFilterV2


def test_moral_filter_logs_boundary_cases(caplog) -> None:
    """Debug logging should capture boundary evaluations near thresholds."""
    caplog.set_level(logging.DEBUG, logger="mlsdm.cognition.moral_filter_v2")
    moral_filter = MoralFilterV2(initial_threshold=0.31)

    result = moral_filter.evaluate(moral_filter.MIN_THRESHOLD + 0.005)

    assert result is False
    assert any("boundary case" in record.message for record in caplog.records)


def test_compute_moral_value_updates_metadata() -> None:
    """Metadata and context should capture pattern counts during scoring."""
    moral_filter = MoralFilterV2()
    metadata: dict[str, int] = {}
    context: dict[str, dict[str, int]] = {}

    score = moral_filter.compute_moral_value(
        "help others and avoid harm", metadata=metadata, context=context
    )

    assert 0.0 <= score <= 1.0
    assert metadata["harmful_count"] == 1
    assert metadata["positive_count"] == 1
    assert context["metadata"]["harmful_count"] == 1
    assert context["metadata"]["positive_count"] == 1
