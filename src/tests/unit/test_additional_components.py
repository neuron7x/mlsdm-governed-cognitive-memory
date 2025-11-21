"""Additional tests to improve coverage for remaining components."""
import numpy as np
import pytest

from src.cognition.moral_filter import MoralFilter
from src.cognition.ontology_matcher import OntologyMatcher
from src.memory.multi_level_memory import MultiLevelSynapticMemory
from src.memory.qilm_module import QILM
from src.rhythm.cognitive_rhythm import CognitiveRhythm
from src.utils.metrics import MetricsCollector


class TestMoralFilterAdditional:
    """Additional tests for MoralFilter to increase coverage."""

    def test_evaluate_invalid_moral_value_low(self):
        """Test that invalid moral values raise ValueError."""
        mf = MoralFilter()
        with pytest.raises(ValueError, match="Moral value must be between"):
            mf.evaluate(-0.1)

    def test_evaluate_invalid_moral_value_high(self):
        """Test that invalid moral values raise ValueError."""
        mf = MoralFilter()
        with pytest.raises(ValueError, match="Moral value must be between"):
            mf.evaluate(1.5)

    def test_adapt_invalid_accept_rate_low(self):
        """Test that invalid accept rate raises ValueError."""
        mf = MoralFilter()
        with pytest.raises(ValueError, match="Accept rate must be between"):
            mf.adapt(-0.1)

    def test_adapt_invalid_accept_rate_high(self):
        """Test that invalid accept rate raises ValueError."""
        mf = MoralFilter()
        with pytest.raises(ValueError, match="Accept rate must be between"):
            mf.adapt(1.5)

    def test_initialization_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="Threshold and adapt_rate must be between"):
            MoralFilter(threshold=1.5)

    def test_initialization_invalid_adapt_rate(self):
        """Test that invalid adapt_rate raises ValueError."""
        with pytest.raises(ValueError, match="Threshold and adapt_rate must be between"):
            MoralFilter(adapt_rate=1.5)

    def test_to_dict(self):
        """Test to_dict method."""
        mf = MoralFilter(threshold=0.6, adapt_rate=0.08, min_threshold=0.25, max_threshold=0.85)
        d = mf.to_dict()
        
        assert d["threshold"] == 0.6
        assert d["adapt_rate"] == 0.08
        assert d["min_threshold"] == 0.25
        assert d["max_threshold"] == 0.85


class TestOntologyMatcherAdditional:
    """Additional tests for OntologyMatcher to increase coverage."""

    def test_invalid_ontology_vectors_not_2d(self):
        """Test that 1D array raises ValueError."""
        with pytest.raises(ValueError, match="ontology_vectors must be a 2D NumPy array"):
            OntologyMatcher(np.array([1, 2, 3]))

    def test_invalid_ontology_vectors_not_ndarray(self):
        """Test that non-ndarray raises ValueError."""
        with pytest.raises(ValueError, match="ontology_vectors must be a 2D NumPy array"):
            OntologyMatcher([[1, 2, 3]])

    def test_labels_length_mismatch(self):
        """Test that mismatched labels raise ValueError."""
        vectors = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValueError, match="Length of labels must match"):
            OntologyMatcher(vectors, labels=["A"])

    def test_match_invalid_dimension(self):
        """Test that dimension mismatch raises ValueError."""
        vectors = np.array([[1, 0, 0], [0, 1, 0]])
        matcher = OntologyMatcher(vectors)
        
        with pytest.raises(ValueError, match="Event vector must be a NumPy array of dimension"):
            matcher.match(np.array([1, 0]))

    def test_match_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        vectors = np.array([[1, 0, 0], [0, 1, 0]])
        matcher = OntologyMatcher(vectors)
        
        with pytest.raises(ValueError, match="Metric must be"):
            matcher.match(np.array([1, 0, 0]), metric="invalid")

    def test_match_empty_ontology(self):
        """Test matching with empty ontology."""
        vectors = np.zeros((0, 3))
        matcher = OntologyMatcher(vectors)
        
        label, score = matcher.match(np.array([1, 0, 0]))
        assert label is None
        assert score == 0.0

    def test_match_euclidean(self):
        """Test euclidean metric matching."""
        vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        matcher = OntologyMatcher(vectors, labels=["X", "Y", "Z"])
        
        label, score = matcher.match(np.array([1.1, 0, 0]), metric="euclidean")
        assert label == "X"
        assert score < 0  # Euclidean returns negative distance

    def test_match_zero_norm_event(self):
        """Test matching with zero-norm event vector."""
        vectors = np.array([[1, 0, 0], [0, 1, 0]])
        matcher = OntologyMatcher(vectors)
        
        label, score = matcher.match(np.zeros(3))
        assert label is None
        assert score == 0.0

    def test_match_zero_norm_ontology(self):
        """Test matching with zero-norm ontology vectors."""
        vectors = np.array([[0, 0, 0], [0, 0, 0]])
        matcher = OntologyMatcher(vectors)
        
        label, score = matcher.match(np.array([1, 0, 0]))
        assert label is None
        assert score == 0.0

    def test_to_dict(self):
        """Test to_dict method."""
        vectors = np.array([[1, 0], [0, 1]])
        matcher = OntologyMatcher(vectors, labels=["A", "B"])
        
        d = matcher.to_dict()
        assert "ontology_vectors" in d
        assert "labels" in d
        assert d["labels"] == ["A", "B"]


class TestQILMAdditional:
    """Additional tests for QILM to increase coverage."""

    def test_entangle_phase_invalid_vector_type(self):
        """Test that non-ndarray raises TypeError."""
        qilm = QILM()
        
        with pytest.raises(TypeError, match="event_vector must be a NumPy array"):
            qilm.entangle_phase([1, 2, 3], phase=0.5)

    def test_retrieve_negative_tolerance(self):
        """Test that negative tolerance raises ValueError."""
        qilm = QILM()
        
        with pytest.raises(ValueError, match="Tolerance must be non-negative"):
            qilm.retrieve(phase=0.5, tolerance=-0.1)

    def test_retrieve_random_phase(self):
        """Test entangle with None phase (random generation)."""
        qilm = QILM()
        vec = np.array([1.0, 2.0, 3.0])
        
        qilm.entangle_phase(vec, phase=None)
        assert len(qilm.phases) == 1
        assert isinstance(qilm.phases[0], float)

    def test_to_dict(self):
        """Test to_dict method."""
        qilm = QILM()
        vec = np.array([1.0, 2.0, 3.0])
        qilm.entangle_phase(vec, phase=0.5)
        
        d = qilm.to_dict()
        assert "memory" in d
        assert "phases" in d
        assert len(d["memory"]) == 1


class TestMultiLevelSynapticMemoryAdditional:
    """Additional tests for MultiLevelSynapticMemory to increase coverage."""

    def test_update_invalid_dimension(self):
        """Test that invalid dimension raises ValueError."""
        mlm = MultiLevelSynapticMemory(dimension=10)
        
        with pytest.raises(ValueError, match="Event vector must be a NumPy array of dimension"):
            mlm.update(np.array([1, 2, 3]))

    def test_update_non_ndarray(self):
        """Test that non-ndarray raises ValueError."""
        mlm = MultiLevelSynapticMemory(dimension=10)
        
        with pytest.raises(ValueError, match="Event vector must be a NumPy array of dimension"):
            mlm.update([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_reset_all(self):
        """Test reset_all method."""
        mlm = MultiLevelSynapticMemory(dimension=10)
        
        # Add some data
        mlm.update(np.ones(10))
        mlm.update(np.ones(10))
        
        # Reset
        mlm.reset_all()
        
        l1, l2, l3 = mlm.state()
        assert np.all(l1 == 0.0)
        assert np.all(l2 == 0.0)
        assert np.all(l3 == 0.0)

    def test_to_dict(self):
        """Test to_dict method."""
        mlm = MultiLevelSynapticMemory(dimension=5, lambda_l1=0.4, lambda_l2=0.15)
        
        d = mlm.to_dict()
        assert d["dimension"] == 5
        assert d["lambda_l1"] == 0.4
        assert d["lambda_l2"] == 0.15
        assert "state_L1" in d


class TestCognitiveRhythmAdditional:
    """Additional tests for CognitiveRhythm to increase coverage."""

    def test_initialization_invalid_wake_duration(self):
        """Test that invalid wake duration raises ValueError."""
        with pytest.raises(ValueError, match="Durations must be positive"):
            CognitiveRhythm(wake_duration=0, sleep_duration=3)

    def test_initialization_invalid_sleep_duration(self):
        """Test that invalid sleep duration raises ValueError."""
        with pytest.raises(ValueError, match="Durations must be positive"):
            CognitiveRhythm(wake_duration=3, sleep_duration=0)

    def test_to_dict(self):
        """Test to_dict method."""
        cr = CognitiveRhythm(wake_duration=5, sleep_duration=2)
        
        d = cr.to_dict()
        assert d["wake_duration"] == 5
        assert d["sleep_duration"] == 2
        assert d["phase"] == "wake"
        assert d["counter"] == 5


class TestMetricsCollectorAdditional:
    """Additional tests for MetricsCollector to increase coverage."""

    def test_stop_event_timer_without_start(self):
        """Test that stopping timer without starting doesn't crash."""
        mc = MetricsCollector()
        mc.stop_event_timer_and_record_latency()
        
        # Should not crash and should not add latency
        assert len(mc.metrics["latencies"]) == 0

    def test_entropy_empty_vector(self):
        """Test entropy calculation with empty vector."""
        entropy = MetricsCollector._entropy(np.array([]))
        assert entropy == 0.0

    def test_entropy_zero_sum(self):
        """Test entropy when sum is zero."""
        # Create vector where exp sum would be zero (very negative values)
        vec = np.array([-1000.0, -1000.0, -1000.0])
        entropy = MetricsCollector._entropy(vec)
        assert entropy >= 0.0

    def test_reset_metrics(self):
        """Test reset_metrics method."""
        mc = MetricsCollector()
        mc.add_accepted_event()
        mc.add_latent_event()
        
        mc.reset_metrics()
        
        metrics = mc.get_metrics()
        assert metrics["accepted_events_count"] == 0
        assert metrics["latent_events_count"] == 0
