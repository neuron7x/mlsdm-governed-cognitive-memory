"""
Comprehensive tests for coherence_safety_metrics module.
Tests edge cases and uncovered functionality to achieve ≥85% coverage.
"""

import numpy as np
import pytest

from mlsdm.utils.coherence_safety_metrics import (
    CoherenceMetrics,
    CoherenceSafetyAnalyzer,
    SafetyMetrics,
)


class TestCoherenceMetrics:
    """Test CoherenceMetrics dataclass"""

    def test_overall_score(self):
        """Test overall score calculation"""
        metrics = CoherenceMetrics(
            temporal_consistency=0.8,
            semantic_coherence=0.9,
            retrieval_stability=0.7,
            phase_separation=0.6
        )
        expected = (0.8 + 0.9 + 0.7 + 0.6) / 4.0
        assert abs(metrics.overall_score() - expected) < 1e-6


class TestSafetyMetrics:
    """Test SafetyMetrics dataclass"""

    def test_overall_score(self):
        """Test overall safety score calculation"""
        metrics = SafetyMetrics(
            toxic_rejection_rate=0.95,
            moral_drift=0.1,
            threshold_convergence=0.85,
            false_positive_rate=0.05
        )
        expected = (0.95 + (1.0 - 0.1) + 0.85 + (1.0 - 0.05)) / 4.0
        assert abs(metrics.overall_score() - expected) < 1e-6


class TestCoherenceSafetyAnalyzer:
    """Test CoherenceSafetyAnalyzer class"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return CoherenceSafetyAnalyzer()

    # ========== Temporal Consistency Tests ==========

    def test_measure_temporal_consistency_single_retrieval(self, analyzer):
        """Test temporal consistency with single retrieval"""
        retrieval_sequence = [
            [np.array([1.0, 0.0, 0.0])]
        ]
        score = analyzer.measure_temporal_consistency(retrieval_sequence)
        assert score == 1.0

    def test_measure_temporal_consistency_empty_retrievals(self, analyzer):
        """Test temporal consistency with empty retrievals in window"""
        retrieval_sequence = [
            [np.array([1.0, 0.0, 0.0])],
            [],  # Empty retrieval
            [np.array([1.0, 0.0, 0.0])]
        ]
        score = analyzer.measure_temporal_consistency(retrieval_sequence, window_size=2)
        assert 0.0 <= score <= 1.0

    def test_measure_temporal_consistency_no_consistencies(self, analyzer):
        """Test temporal consistency when no valid consistencies found"""
        retrieval_sequence = [
            [],  # All empty
            [],
            []
        ]
        score = analyzer.measure_temporal_consistency(retrieval_sequence)
        assert score == 1.0

    def test_measure_temporal_consistency_normal(self, analyzer):
        """Test temporal consistency with normal retrieval sequence"""
        retrieval_sequence = [
            [np.array([1.0, 0.0, 0.0])],
            [np.array([0.9, 0.1, 0.0])],
            [np.array([0.8, 0.2, 0.0])],
            [np.array([0.85, 0.15, 0.0])],
            [np.array([0.87, 0.13, 0.0])]
        ]
        score = analyzer.measure_temporal_consistency(retrieval_sequence)
        assert 0.0 <= score <= 1.0

    # ========== Semantic Coherence Tests ==========

    def test_measure_semantic_coherence_empty_inputs(self, analyzer):
        """Test semantic coherence with empty inputs"""
        score = analyzer.measure_semantic_coherence([], [])
        assert score == 0.0

    def test_measure_semantic_coherence_empty_retrievals(self, analyzer):
        """Test semantic coherence with empty retrievals"""
        query_vectors = [np.array([1.0, 0.0, 0.0])]
        retrieved_vectors = [[]]  # Empty retrieval
        score = analyzer.measure_semantic_coherence(query_vectors, retrieved_vectors)
        assert score == 0.0

    def test_measure_semantic_coherence_no_coherence_scores(self, analyzer):
        """Test semantic coherence when no valid coherence scores"""
        query_vectors = [np.array([1.0, 0.0, 0.0])]
        retrieved_vectors = [[]]  # All empty
        score = analyzer.measure_semantic_coherence(query_vectors, retrieved_vectors)
        assert score == 0.0

    def test_measure_semantic_coherence_normal(self, analyzer):
        """Test semantic coherence with normal inputs"""
        query_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        retrieved_vectors = [
            [np.array([0.9, 0.1, 0.0]), np.array([0.95, 0.05, 0.0])],
            [np.array([0.1, 0.9, 0.0]), np.array([0.05, 0.95, 0.0])]
        ]
        score = analyzer.measure_semantic_coherence(query_vectors, retrieved_vectors)
        assert 0.0 <= score <= 1.0

    # ========== Phase Separation Tests ==========

    def test_measure_phase_separation_empty_inputs(self, analyzer):
        """Test phase separation with empty inputs"""
        score = analyzer.measure_phase_separation([], [])
        assert score == 0.0

    def test_measure_phase_separation_normal(self, analyzer):
        """Test phase separation with normal inputs"""
        wake_retrievals = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0])
        ]
        sleep_retrievals = [
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.1, 0.9])
        ]
        score = analyzer.measure_phase_separation(wake_retrievals, sleep_retrievals)
        assert 0.0 <= score <= 1.0

    # ========== Retrieval Stability Tests ==========

    def test_measure_retrieval_stability_single_retrieval(self, analyzer):
        """Test retrieval stability with single retrieval"""
        retrievals = [[np.array([1.0, 0.0, 0.0])]]
        score = analyzer.measure_retrieval_stability(retrievals)
        assert score == 1.0

    def test_measure_retrieval_stability_empty_retrievals(self, analyzer):
        """Test retrieval stability with empty retrievals"""
        retrievals = [
            [np.array([1.0, 0.0, 0.0])],
            [],  # Empty
            [np.array([1.0, 0.0, 0.0])]
        ]
        score = analyzer.measure_retrieval_stability(retrievals)
        assert 0.0 <= score <= 1.0

    def test_measure_retrieval_stability_no_stability_scores(self, analyzer):
        """Test retrieval stability when no valid stability scores"""
        retrievals = [[], []]
        score = analyzer.measure_retrieval_stability(retrievals)
        assert score == 1.0

    def test_measure_retrieval_stability_normal(self, analyzer):
        """Test retrieval stability with normal retrievals"""
        retrievals = [
            [np.array([1.0, 0.0, 0.0]), np.array([0.9, 0.1, 0.0])],
            [np.array([0.95, 0.05, 0.0]), np.array([0.85, 0.15, 0.0])],
            [np.array([0.98, 0.02, 0.0]), np.array([0.88, 0.12, 0.0])]
        ]
        score = analyzer.measure_retrieval_stability(retrievals, top_k=2)
        assert 0.0 <= score <= 1.0

    # ========== Compute Coherence Metrics Tests ==========

    def test_compute_coherence_metrics(self, analyzer):
        """Test comprehensive coherence metrics computation"""
        wake_retrievals = [np.array([1.0, 0.0, 0.0])]
        sleep_retrievals = [np.array([0.0, 1.0, 0.0])]
        query_sequence = [np.array([1.0, 0.0, 0.0])]
        retrieval_sequence = [[np.array([0.9, 0.1, 0.0])]]

        metrics = analyzer.compute_coherence_metrics(
            wake_retrievals, sleep_retrievals, query_sequence, retrieval_sequence
        )

        assert isinstance(metrics, CoherenceMetrics)
        assert 0.0 <= metrics.temporal_consistency <= 1.0
        assert 0.0 <= metrics.semantic_coherence <= 1.0
        assert 0.0 <= metrics.retrieval_stability <= 1.0
        assert 0.0 <= metrics.phase_separation <= 1.0

    # ========== Toxic Rejection Rate Tests ==========

    def test_measure_toxic_rejection_rate_empty(self, analyzer):
        """Test toxic rejection rate with empty inputs"""
        score = analyzer.measure_toxic_rejection_rate([], [])
        assert score == 0.0

    def test_measure_toxic_rejection_rate_no_toxic(self, analyzer):
        """Test toxic rejection rate with no toxic content"""
        moral_values = [0.8, 0.9, 0.7]
        rejections = [False, False, False]
        score = analyzer.measure_toxic_rejection_rate(moral_values, rejections, toxic_threshold=0.4)
        assert score == 1.0

    def test_measure_toxic_rejection_rate_normal(self, analyzer):
        """Test toxic rejection rate with mixed content"""
        moral_values = [0.2, 0.3, 0.8, 0.1]
        rejections = [True, True, False, True]
        score = analyzer.measure_toxic_rejection_rate(moral_values, rejections, toxic_threshold=0.4)
        assert score == 1.0  # All 3 toxic items were rejected

    # ========== Moral Drift Tests ==========

    def test_measure_moral_drift_insufficient_history(self, analyzer):
        """Test moral drift with insufficient history"""
        threshold_history = [0.5]
        score = analyzer.measure_moral_drift(threshold_history)
        assert score == 0.0

    def test_measure_moral_drift_normal(self, analyzer):
        """Test moral drift with normal threshold history"""
        threshold_history = [0.5, 0.52, 0.48, 0.51, 0.49]
        score = analyzer.measure_moral_drift(threshold_history)
        assert 0.0 <= score <= 1.0

    # ========== Threshold Convergence Tests ==========

    def test_measure_threshold_convergence_insufficient_history(self, analyzer):
        """Test threshold convergence with insufficient history"""
        threshold_history = [0.5] * 40  # Less than window_size
        score = analyzer.measure_threshold_convergence(threshold_history, window_size=50)
        assert score == 0.0

    def test_measure_threshold_convergence_normal(self, analyzer):
        """Test threshold convergence with sufficient history"""
        threshold_history = [0.5] * 50
        score = analyzer.measure_threshold_convergence(threshold_history, target_threshold=0.5)
        assert score == 1.0

    # ========== False Positive Rate Tests ==========

    def test_measure_false_positive_rate_empty(self, analyzer):
        """Test false positive rate with empty inputs"""
        score = analyzer.measure_false_positive_rate([], [])
        assert score == 0.0

    def test_measure_false_positive_rate_no_safe_content(self, analyzer):
        """Test false positive rate with no safe content"""
        moral_values = [0.2, 0.3, 0.4]
        rejections = [True, True, True]
        score = analyzer.measure_false_positive_rate(moral_values, rejections, safe_threshold=0.6)
        assert score == 0.0

    def test_measure_false_positive_rate_normal(self, analyzer):
        """Test false positive rate with mixed content"""
        moral_values = [0.7, 0.8, 0.9, 0.3]
        rejections = [False, True, False, True]  # One false positive at 0.8
        score = analyzer.measure_false_positive_rate(moral_values, rejections, safe_threshold=0.6)
        expected = 1 / 3  # 1 false positive out of 3 safe items
        assert abs(score - expected) < 1e-6

    # ========== Compute Safety Metrics Tests ==========

    def test_compute_safety_metrics(self, analyzer):
        """Test comprehensive safety metrics computation"""
        moral_values = [0.2, 0.8, 0.3, 0.9]
        rejections = [True, False, True, False]
        threshold_history = [0.5] * 100

        metrics = analyzer.compute_safety_metrics(moral_values, rejections, threshold_history)

        assert isinstance(metrics, SafetyMetrics)
        assert 0.0 <= metrics.toxic_rejection_rate <= 1.0
        assert 0.0 <= metrics.moral_drift <= 1.0
        assert 0.0 <= metrics.threshold_convergence <= 1.0
        assert 0.0 <= metrics.false_positive_rate <= 1.0

    # ========== Comparative Analysis Tests ==========

    def test_compare_with_without_feature(self, analyzer):
        """Test comparative analysis"""
        with_metrics = {
            'accuracy': 0.95,
            'precision': 0.90,
            'recall': 0.85
        }
        without_metrics = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75
        }

        results = analyzer.compare_with_without_feature(with_metrics, without_metrics)

        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results

        # Check accuracy improvement
        assert results['accuracy']['improvement'] > 0
        assert results['accuracy']['significant'] is True
        assert results['accuracy']['with_feature'] == 0.95
        assert results['accuracy']['without_feature'] == 0.85

    def test_compare_with_without_feature_small_improvement(self, analyzer):
        """Test comparative analysis with insignificant improvement"""
        with_metrics = {'accuracy': 0.851}
        without_metrics = {'accuracy': 0.850}

        results = analyzer.compare_with_without_feature(with_metrics, without_metrics)

        assert results['accuracy']['significant'] is False

    # ========== Report Generation Tests ==========

    def test_generate_report(self, analyzer):
        """Test report generation"""
        coherence = CoherenceMetrics(
            temporal_consistency=0.8,
            semantic_coherence=0.9,
            retrieval_stability=0.7,
            phase_separation=0.6
        )
        safety = SafetyMetrics(
            toxic_rejection_rate=0.95,
            moral_drift=0.1,
            threshold_convergence=0.85,
            false_positive_rate=0.05
        )

        report = analyzer.generate_report(coherence, safety)

        assert "COHERENCE AND SAFETY METRICS REPORT" in report
        assert "COHERENCE METRICS:" in report
        assert "SAFETY METRICS:" in report
        assert "Temporal Consistency:" in report
        assert "Toxic Rejection Rate:" in report
        assert "0.8000" in report
        assert "0.9500" in report

    # ========== Reset Tests ==========

    def test_reset(self, analyzer):
        """Test reset functionality"""
        analyzer.wake_retrievals = [np.array([1.0, 0.0, 0.0])]
        analyzer.sleep_retrievals = [np.array([0.0, 1.0, 0.0])]
        analyzer.moral_history = [0.5, 0.6]
        analyzer.rejection_history = [True, False]
        analyzer.threshold_history = [0.5, 0.51]

        analyzer.reset()

        assert len(analyzer.wake_retrievals) == 0
        assert len(analyzer.sleep_retrievals) == 0
        assert len(analyzer.moral_history) == 0
        assert len(analyzer.rejection_history) == 0
        assert len(analyzer.threshold_history) == 0


class TestEdgeCases:
    """Test edge cases for QILM and other components"""

    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        """Set random seed for reproducible tests"""
        np.random.seed(42)

    def test_coherence_metrics_extreme_values(self):
        """Test coherence metrics with extreme values"""
        metrics = CoherenceMetrics(
            temporal_consistency=0.0,
            semantic_coherence=0.0,
            retrieval_stability=0.0,
            phase_separation=0.0
        )
        assert metrics.overall_score() == 0.0

        metrics = CoherenceMetrics(
            temporal_consistency=1.0,
            semantic_coherence=1.0,
            retrieval_stability=1.0,
            phase_separation=1.0
        )
        assert metrics.overall_score() == 1.0

    def test_safety_metrics_extreme_values(self):
        """Test safety metrics with extreme values"""
        metrics = SafetyMetrics(
            toxic_rejection_rate=0.0,
            moral_drift=1.0,
            threshold_convergence=0.0,
            false_positive_rate=1.0
        )
        assert metrics.overall_score() == 0.0

        metrics = SafetyMetrics(
            toxic_rejection_rate=1.0,
            moral_drift=0.0,
            threshold_convergence=1.0,
            false_positive_rate=0.0
        )
        assert metrics.overall_score() == 1.0

    def test_analyzer_with_large_datasets(self):
        """Test analyzer with large datasets"""
        analyzer = CoherenceSafetyAnalyzer()

        # Large retrieval sequence
        retrieval_sequence = [
            [np.random.randn(128) for _ in range(5)]
            for _ in range(100)
        ]
        score = analyzer.measure_temporal_consistency(retrieval_sequence)
        # Allow small numerical errors
        assert -0.01 <= score <= 1.01

        # Large moral values
        moral_values = list(np.random.uniform(0, 1, 1000))
        rejections = [v < 0.4 for v in moral_values]
        toxic_rate = analyzer.measure_toxic_rejection_rate(moral_values, rejections)
        assert 0.0 <= toxic_rate <= 1.0
