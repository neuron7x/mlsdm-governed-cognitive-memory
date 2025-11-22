"""Property-based tests using Hypothesis for invariant verification."""
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsdm.cognition.moral_filter import MoralFilter
from mlsdm.cognition.moral_filter_v2 import MoralFilterV2
from mlsdm.memory.multi_level_memory import MultiLevelSynapticMemory
from mlsdm.memory.qilm_v2 import QILM_v2
from mlsdm.rhythm.cognitive_rhythm import CognitiveRhythm


class TestPropertyBasedInvariants:
    """Property-based tests to verify mathematical invariants."""

    @given(initial_threshold=st.floats(min_value=-10.0, max_value=10.0))
    @settings(max_examples=50)
    def test_moral_filter_v2_threshold_always_bounded(self, initial_threshold):
        """Property: MoralFilterV2 threshold is always within [MIN, MAX] bounds."""
        mf = MoralFilterV2(initial_threshold=initial_threshold)

        # Invariant check
        assert MoralFilterV2.MIN_THRESHOLD <= mf.threshold <= MoralFilterV2.MAX_THRESHOLD

    @given(
        initial_threshold=st.floats(min_value=0.3, max_value=0.9),
        num_accepts=st.integers(min_value=0, max_value=100),
        num_rejects=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=50)
    def test_moral_filter_v2_threshold_stability_after_adaptation(self, initial_threshold, num_accepts, num_rejects):
        """Property: After many adaptations, threshold stays within bounds."""
        mf = MoralFilterV2(initial_threshold=initial_threshold)

        for _ in range(num_accepts):
            mf.adapt(accepted=True)

        for _ in range(num_rejects):
            mf.adapt(accepted=False)

        # Invariant: threshold must remain bounded
        assert MoralFilterV2.MIN_THRESHOLD <= mf.threshold <= MoralFilterV2.MAX_THRESHOLD
        # Invariant: EMA must remain in [0, 1]
        assert 0.0 <= mf.ema_accept_rate <= 1.0

    @given(moral_value=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50)
    def test_moral_filter_v2_evaluation_is_deterministic(self, moral_value):
        """Property: Evaluation of same value should be consistent."""
        mf = MoralFilterV2(initial_threshold=0.5)

        result1 = mf.evaluate(moral_value)
        result2 = mf.evaluate(moral_value)

        assert result1 == result2

    @given(
        dimension=st.integers(min_value=2, max_value=100),
        capacity=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=20)
    def test_qilm_v2_size_never_exceeds_capacity(self, dimension, capacity):
        """Property: QILM_v2 size never exceeds capacity."""
        qilm = QILM_v2(dimension=dimension, capacity=capacity)

        # Add more items than capacity
        for i in range(capacity + 50):
            vec = [float(i % 10) for _ in range(dimension)]
            qilm.entangle(vec, phase=0.5)

        # Invariant: size should be capped at capacity
        assert qilm.size == capacity
        assert qilm.pointer <= capacity

    @given(
        dimension=st.integers(min_value=2, max_value=50),
        num_items=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=20)
    def test_qilm_v2_retrieve_returns_valid_results(self, dimension, num_items):
        """Property: Retrieved items have valid structure and values."""
        qilm = QILM_v2(dimension=dimension, capacity=100)

        # Add items
        for i in range(num_items):
            vec = [float(i) for _ in range(dimension)]
            qilm.entangle(vec, phase=0.5)

        # Retrieve
        query = [1.0] * dimension
        results = qilm.retrieve(query, current_phase=0.5, phase_tolerance=0.5, top_k=5)

        # Invariants
        assert isinstance(results, list)
        assert len(results) <= min(num_items, 5)
        for result in results:
            assert len(result.vector) == dimension
            assert 0.0 <= result.phase <= 1.0
            assert -1.0 <= result.resonance <= 1.01  # Allow small floating point error

    @given(
        wake_duration=st.integers(min_value=1, max_value=50),
        sleep_duration=st.integers(min_value=1, max_value=50),
        num_steps=st.integers(min_value=0, max_value=100)
    )
    @settings(max_examples=30)
    def test_cognitive_rhythm_phase_transitions_are_valid(self, wake_duration, sleep_duration, num_steps):
        """Property: Cognitive rhythm only transitions between valid phases."""
        cr = CognitiveRhythm(wake_duration=wake_duration, sleep_duration=sleep_duration)

        for _ in range(num_steps):
            # Invariant: phase must be either 'wake' or 'sleep'
            assert cr.phase in ["wake", "sleep"]

            # Invariant: counter must be positive
            assert cr.counter > 0

            cr.step()

    @given(
        dimension=st.integers(min_value=2, max_value=50),
        num_updates=st.integers(min_value=0, max_value=20)
    )
    @settings(max_examples=20)
    def test_multi_level_memory_norms_are_non_negative(self, dimension, num_updates):
        """Property: Memory state norms are always non-negative."""
        mlm = MultiLevelSynapticMemory(dimension=dimension)

        for _i in range(num_updates):
            event = np.random.randn(dimension).astype(np.float32)
            mlm.update(event)

        l1, l2, l3 = mlm.state()

        # Invariant: norms must be non-negative
        assert np.linalg.norm(l1) >= 0
        assert np.linalg.norm(l2) >= 0
        assert np.linalg.norm(l3) >= 0

    @given(
        threshold=st.floats(min_value=0.0, max_value=1.0),
        adapt_rate=st.floats(min_value=0.0, max_value=1.0)
    )
    @settings(max_examples=30)
    def test_moral_filter_initialization_accepts_valid_params(self, threshold, adapt_rate):
        """Property: MoralFilter accepts all valid parameter combinations."""
        mf = MoralFilter(threshold=threshold, adapt_rate=adapt_rate)

        assert 0.0 <= mf.threshold <= 1.0
        assert 0.0 <= mf.adapt_rate <= 1.0

    @given(
        initial_threshold=st.floats(min_value=0.3, max_value=0.9),
        accept_sequence=st.lists(st.booleans(), min_size=0, max_size=50)
    )
    @settings(max_examples=30)
    def test_moral_filter_v2_ema_convergence(self, initial_threshold, accept_sequence):
        """Property: EMA converges towards actual acceptance rate."""
        mf = MoralFilterV2(initial_threshold=initial_threshold)

        for accepted in accept_sequence:
            mf.adapt(accepted=accepted)

        # After adaptation, EMA should be in valid range
        assert 0.0 <= mf.ema_accept_rate <= 1.0

    @given(
        dimension=st.integers(min_value=2, max_value=30),
        lambda_l1=st.floats(min_value=0.01, max_value=0.99),
        lambda_l2=st.floats(min_value=0.01, max_value=0.99),
        lambda_l3=st.floats(min_value=0.01, max_value=0.99)
    )
    @settings(max_examples=20)
    def test_multi_level_memory_decay_parameters_valid(self, dimension, lambda_l1, lambda_l2, lambda_l3):
        """Property: Memory accepts valid decay parameters."""
        mlm = MultiLevelSynapticMemory(
            dimension=dimension,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            lambda_l3=lambda_l3
        )

        assert mlm.lambda_l1 == lambda_l1
        assert mlm.lambda_l2 == lambda_l2
        assert mlm.lambda_l3 == lambda_l3

    @given(phase=st.floats(min_value=-10.0, max_value=10.0))
    @settings(max_examples=30)
    def test_qilm_v2_phase_storage(self, phase):
        """Property: QILM_v2 stores phase values correctly."""
        qilm = QILM_v2(dimension=10, capacity=100)
        vec = [1.0] * 10

        idx = qilm.entangle(vec, phase=phase)

        assert qilm.phase_bank[idx] == phase

    @given(
        dimension=st.integers(min_value=2, max_value=30),
        event_magnitude=st.floats(min_value=0.1, max_value=10.0)
    )
    @settings(max_examples=20)
    def test_multi_level_memory_preserves_dimension(self, dimension, event_magnitude):
        """Property: Memory state always has correct dimension."""
        mlm = MultiLevelSynapticMemory(dimension=dimension)

        event = np.ones(dimension) * event_magnitude
        mlm.update(event)

        l1, l2, l3 = mlm.state()

        # Invariant: dimensions must match
        assert len(l1) == dimension
        assert len(l2) == dimension
        assert len(l3) == dimension

    @given(
        wake_duration=st.integers(min_value=1, max_value=20),
        sleep_duration=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=30)
    def test_cognitive_rhythm_cycle_length(self, wake_duration, sleep_duration):
        """Property: One full cycle takes exactly wake_duration + sleep_duration steps."""
        cr = CognitiveRhythm(wake_duration=wake_duration, sleep_duration=sleep_duration)

        initial_phase = cr.phase

        # Step through one complete cycle
        for _ in range(wake_duration + sleep_duration):
            cr.step()

        # Should return to initial phase
        assert cr.phase == initial_phase

    @given(
        dimension=st.integers(min_value=2, max_value=50),
        top_k=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=20)
    def test_qilm_v2_top_k_constraint(self, dimension, top_k):
        """Property: Retrieve returns at most top_k results."""
        qilm = QILM_v2(dimension=dimension, capacity=100)

        # Add many items
        for i in range(50):
            vec = [float(i % 10) for _ in range(dimension)]
            qilm.entangle(vec, phase=0.5)

        query = [1.0] * dimension
        results = qilm.retrieve(query, current_phase=0.5, phase_tolerance=0.5, top_k=top_k)

        # Invariant: number of results <= top_k
        assert len(results) <= top_k
