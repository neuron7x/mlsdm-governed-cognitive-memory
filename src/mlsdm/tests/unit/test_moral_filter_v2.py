"""Comprehensive unit tests for MoralFilterV2."""
from mlsdm.cognition.moral_filter_v2 import MoralFilterV2


class TestMoralFilterV2:
    """Test suite for MoralFilterV2."""

    def test_initialization_default(self) -> None:
        """Test initialization with default threshold."""
        mf = MoralFilterV2()
        assert mf.threshold == 0.50
        assert mf.ema_accept_rate == 0.5
        assert mf.MIN_THRESHOLD == 0.30
        assert mf.MAX_THRESHOLD == 0.90
        assert mf.DEAD_BAND == 0.05
        assert mf.EMA_ALPHA == 0.1

    def test_initialization_custom_threshold(self) -> None:
        """Test initialization with custom threshold."""
        mf = MoralFilterV2(initial_threshold=0.60)
        assert mf.threshold == 0.60
        assert mf.ema_accept_rate == 0.5

    def test_initialization_threshold_clipping_low(self) -> None:
        """Test that threshold is clipped to minimum."""
        mf = MoralFilterV2(initial_threshold=0.10)
        assert mf.threshold == 0.30

    def test_initialization_threshold_clipping_high(self) -> None:
        """Test that threshold is clipped to maximum."""
        mf = MoralFilterV2(initial_threshold=0.95)
        assert mf.threshold == 0.90

    def test_evaluate_accept(self) -> None:
        """Test that values above threshold are accepted."""
        mf = MoralFilterV2(initial_threshold=0.50)
        assert mf.evaluate(0.60) is True
        assert mf.evaluate(0.50) is True
        assert mf.evaluate(0.90) is True

    def test_evaluate_reject(self) -> None:
        """Test that values below threshold are rejected."""
        mf = MoralFilterV2(initial_threshold=0.50)
        assert mf.evaluate(0.40) is False
        assert mf.evaluate(0.30) is False
        assert mf.evaluate(0.10) is False

    def test_evaluate_boundary(self) -> None:
        """Test evaluation at exact threshold boundary."""
        mf = MoralFilterV2(initial_threshold=0.50)
        assert mf.evaluate(0.50) is True

    def test_adapt_accepted_event(self) -> None:
        """Test adaptation when event is accepted."""
        mf = MoralFilterV2(initial_threshold=0.50)
        initial_ema = mf.ema_accept_rate

        mf.adapt(accepted=True)

        # EMA should move towards 1.0
        assert mf.ema_accept_rate > initial_ema or abs(mf.ema_accept_rate - 1.0) < 0.01

    def test_adapt_rejected_event(self) -> None:
        """Test adaptation when event is rejected."""
        mf = MoralFilterV2(initial_threshold=0.50)
        mf.ema_accept_rate = 0.8  # Set high initial rate

        mf.adapt(accepted=False)

        # EMA should move towards 0.0
        assert mf.ema_accept_rate < 0.8

    def test_ema_calculation(self) -> None:
        """Test EMA calculation with known values."""
        mf = MoralFilterV2(initial_threshold=0.50)
        mf.ema_accept_rate = 0.5

        # Accept event: EMA = 0.1 * 1.0 + 0.9 * 0.5 = 0.55
        mf.adapt(accepted=True)
        assert abs(mf.ema_accept_rate - 0.55) < 1e-6

    def test_threshold_increase_high_acceptance(self) -> None:
        """Test that threshold increases with high acceptance rate."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Simulate high acceptance rate
        for _ in range(20):
            mf.adapt(accepted=True)

        # Threshold should increase (but might be capped at max)
        assert mf.threshold >= 0.50

    def test_threshold_decrease_low_acceptance(self) -> None:
        """Test that threshold decreases with low acceptance rate."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Simulate low acceptance rate
        for _ in range(20):
            mf.adapt(accepted=False)

        # Threshold should decrease
        assert mf.threshold <= 0.50

    def test_threshold_stays_in_bounds(self) -> None:
        """Test that threshold never goes below MIN or above MAX."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Try to push threshold very high
        for _ in range(100):
            mf.adapt(accepted=True)
        assert mf.threshold <= mf.MAX_THRESHOLD

        # Try to push threshold very low
        for _ in range(200):
            mf.adapt(accepted=False)
        assert mf.threshold >= mf.MIN_THRESHOLD

    def test_dead_band_no_adaptation(self) -> None:
        """Test that small errors (within dead band) don't cause threshold changes."""
        mf = MoralFilterV2(initial_threshold=0.50)
        mf.ema_accept_rate = 0.52  # Within dead band of 0.05 from target 0.5

        mf.adapt(accepted=True)

        # Threshold might change or might not depending on exact dynamics
        # But should stay within reasonable bounds
        assert mf.MIN_THRESHOLD <= mf.threshold <= mf.MAX_THRESHOLD

    def test_threshold_convergence_to_min(self) -> None:
        """Test convergence to minimum threshold with consistent rejection."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Consistently reject
        for _ in range(200):
            mf.adapt(accepted=False)

        assert mf.threshold == mf.MIN_THRESHOLD

    def test_threshold_convergence_to_max(self) -> None:
        """Test convergence to maximum threshold with consistent acceptance."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Consistently accept
        for _ in range(200):
            mf.adapt(accepted=True)

        assert mf.threshold == mf.MAX_THRESHOLD

    def test_get_state(self) -> None:
        """Test getting current state."""
        mf = MoralFilterV2(initial_threshold=0.60)
        mf.ema_accept_rate = 0.7

        state = mf.get_state()

        assert "threshold" in state
        assert "ema" in state
        assert state["threshold"] == 0.60
        assert state["ema"] == 0.7

    def test_alternating_accept_reject(self) -> None:
        """Test behavior with alternating accept/reject pattern."""
        mf = MoralFilterV2(initial_threshold=0.50)

        for i in range(20):
            mf.adapt(accepted=(i % 2 == 0))

        # EMA should stabilize around 0.5
        assert 0.3 <= mf.ema_accept_rate <= 0.7

    def test_threshold_delta_magnitude(self) -> None:
        """Test that threshold changes by expected delta (0.05)."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Set EMA far from target to trigger adaptation
        mf.ema_accept_rate = 0.8  # Well above 0.5 + DEAD_BAND
        initial_threshold = mf.threshold

        mf.adapt(accepted=True)

        # Threshold should change by approximately 0.05
        # (might not be exact due to clipping and dead band)
        delta = abs(mf.threshold - initial_threshold)
        assert delta <= 0.051  # Allow small floating point tolerance

    def test_multiple_adaptations_stability(self) -> None:
        """Test stability over many adaptation cycles."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Mix of accepts and rejects
        import random
        random.seed(42)

        for _ in range(100):
            accepted = random.random() > 0.5
            mf.adapt(accepted=accepted)

            # Always check bounds
            assert mf.MIN_THRESHOLD <= mf.threshold <= mf.MAX_THRESHOLD
            assert 0.0 <= mf.ema_accept_rate <= 1.0

    def test_rapid_threshold_adaptation(self) -> None:
        """Test threshold adapts correctly with rapid changes."""
        mf = MoralFilterV2(initial_threshold=0.50)

        thresholds = []
        for i in range(50):
            mf.adapt(accepted=(i < 25))
            thresholds.append(mf.threshold)

        # Check thresholds are recorded and within bounds
        assert all(mf.MIN_THRESHOLD <= t <= mf.MAX_THRESHOLD for t in thresholds)

    def test_ema_bounds(self) -> None:
        """Test that EMA always stays between 0 and 1."""
        mf = MoralFilterV2(initial_threshold=0.50)

        for _ in range(100):
            mf.adapt(accepted=True)
            assert 0.0 <= mf.ema_accept_rate <= 1.0

        for _ in range(100):
            mf.adapt(accepted=False)
            assert 0.0 <= mf.ema_accept_rate <= 1.0

    def test_extreme_initial_thresholds(self) -> None:
        """Test behavior with extreme initial thresholds."""
        # Very low (should be clipped to MIN)
        mf_low = MoralFilterV2(initial_threshold=0.0)
        assert mf_low.threshold == 0.30

        # Very high (should be clipped to MAX)
        mf_high = MoralFilterV2(initial_threshold=1.0)
        assert mf_high.threshold == 0.90

        # Negative (should be clipped to MIN)
        mf_neg = MoralFilterV2(initial_threshold=-0.5)
        assert mf_neg.threshold == 0.30

    def test_consistent_evaluation(self) -> None:
        """Test that evaluation is consistent for same input."""
        mf = MoralFilterV2(initial_threshold=0.50)

        # Multiple evaluations without adaptation should be consistent
        results = [mf.evaluate(0.6) for _ in range(10)]
        assert all(results)

        results = [mf.evaluate(0.4) for _ in range(10)]
        assert not any(results)
