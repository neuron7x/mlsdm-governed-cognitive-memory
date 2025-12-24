from tests.perf.test_golden_path_perf import percentile


def test_percentile_uses_nearest_rank_definition() -> None:
    data = [1.0, 2.0, 3.0, 4.0]

    # ceil(p * n) - 1 -> ceil(0.5 * 4) - 1 = 1 (0-based index for second element)
    assert percentile(data, 0.50) == 2.0
    # ceil(p * n) - 1 clamped to last element for high percentiles
    assert percentile(data, 0.95) == 4.0
    # Empty input should return zero without raising
    assert percentile([], 0.50) == 0.0
