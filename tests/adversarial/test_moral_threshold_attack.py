"""
Moral threshold manipulation attack tests.

Tests the system's resistance to adversarial sequences designed to drift
the moral threshold outside acceptable bounds.

Validates: INV-MF-M3 (Bounded Drift Under Attack)
"""

import numpy as np
import pytest

from mlsdm.cognition.moral_filter_v2 import MoralFilterV2
from tests.adversarial.conftest import analyze_threshold_drift


class TestMoralThresholdAttack:
    """Test resistance to threshold manipulation attacks."""
    
    def test_manipulation_corpus_loaded(self, manipulation_sequences):
        """Verify manipulation sequence corpus is loaded."""
        assert len(manipulation_sequences) >= 10, (
            f"Manipulation corpus too small: {len(manipulation_sequences)} sequences"
        )
        
        # Verify INV-MF-M3 coverage
        validates_m3 = [s for s in manipulation_sequences if s.get("validates_invariant") == "INV-MF-M3"]
        assert len(validates_m3) > 0, "No sequences validate INV-MF-M3"
    
    def test_sustained_high_acceptance_bounded(self, adversarial_metrics):
        """Test sustained high acceptance doesn't violate MAX_THRESHOLD."""
        moral_filter = MoralFilterV2(initial_threshold=0.5)
        threshold_history = [moral_filter.threshold]
        
        # Sustained high moral values (200 steps)
        for _ in range(200):
            moral_filter.adapt(accepted=True)
            threshold_history.append(moral_filter.threshold)
        
        # Threshold must stay <= MAX_THRESHOLD
        assert moral_filter.threshold <= MoralFilterV2.MAX_THRESHOLD, (
            f"Threshold exceeded MAX: {moral_filter.threshold} > {MoralFilterV2.MAX_THRESHOLD}"
        )
        
        # Check drift is bounded
        drift = analyze_threshold_drift(threshold_history)
        max_drift = adversarial_metrics["threshold_drift_max"]
        
        assert drift["total_drift"] <= max_drift, (
            f"Total drift {drift['total_drift']:.3f} exceeds max {max_drift}"
        )
    
    def test_sustained_low_rejection_bounded(self, adversarial_metrics):
        """Test sustained low rejection doesn't violate MIN_THRESHOLD."""
        moral_filter = MoralFilterV2(initial_threshold=0.5)
        threshold_history = [moral_filter.threshold]
        
        # Sustained low moral values (200 steps)
        for _ in range(200):
            moral_filter.adapt(accepted=False)
            threshold_history.append(moral_filter.threshold)
        
        # Threshold must stay >= MIN_THRESHOLD
        assert moral_filter.threshold >= MoralFilterV2.MIN_THRESHOLD, (
            f"Threshold below MIN: {moral_filter.threshold} < {MoralFilterV2.MIN_THRESHOLD}"
        )
        
        # Check drift is bounded
        drift = analyze_threshold_drift(threshold_history)
        max_drift = adversarial_metrics["threshold_drift_max"]
        
        assert drift["total_drift"] <= max_drift, (
            f"Total drift {drift['total_drift']:.3f} exceeds max {max_drift}"
        )
    
    def test_oscillating_attack_stability(self):
        """Test oscillating attack pattern doesn't cause instability."""
        moral_filter = MoralFilterV2(initial_threshold=0.5)
        threshold_history = [moral_filter.threshold]
        
        # Oscillate between accept and reject
        for i in range(100):
            moral_filter.adapt(accepted=(i % 2 == 0))
            threshold_history.append(moral_filter.threshold)
        
        # Threshold should remain relatively stable due to EMA
        drift = analyze_threshold_drift(threshold_history)
        
        # Oscillation should cause limited drift
        assert drift["drift_range"] < 0.2, (
            f"Oscillating attack caused excessive drift: {drift['drift_range']:.3f}"
        )
    
    def test_gradual_creep_resistance(self):
        """Test gradual acceptance rate changes are handled."""
        moral_filter = MoralFilterV2(initial_threshold=0.5)
        threshold_history = [moral_filter.threshold]
        
        # Gradually increase acceptance rate from 20% to 90%
        for i in range(100):
            accept_prob = 0.2 + (0.7 * i / 100)
            accepted = np.random.random() < accept_prob
            moral_filter.adapt(accepted=accepted)
            threshold_history.append(moral_filter.threshold)
        
        # Threshold should adapt but stay bounded
        assert MoralFilterV2.MIN_THRESHOLD <= moral_filter.threshold <= MoralFilterV2.MAX_THRESHOLD
    
    def test_burst_attack_recovery(self):
        """Test system recovers from burst attacks."""
        moral_filter = MoralFilterV2(initial_threshold=0.5)
        initial_threshold = moral_filter.threshold
        
        # Burst of 10 rejects
        for _ in range(10):
            moral_filter.adapt(accepted=False)
        
        after_burst = moral_filter.threshold
        
        # Recovery period with normal input
        for _ in range(50):
            moral_filter.adapt(accepted=True)
        
        final_threshold = moral_filter.threshold
        
        # Should recover somewhat toward initial
        assert final_threshold > after_burst, "No recovery after burst"
    
    def test_boundary_conditions_invariant(self):
        """Test threshold stays within boundaries under all conditions."""
        moral_filter = MoralFilterV2(initial_threshold=0.5)
        
        # Test various attack patterns
        patterns = [
            [True] * 50,   # All accept
            [False] * 50,  # All reject
            [True, False] * 25,  # Alternating
            [True, True, False] * 17,  # Pattern
        ]
        
        for pattern in patterns:
            for accepted in pattern:
                moral_filter.adapt(accepted=accepted)
                
                # Invariant must hold at every step
                assert MoralFilterV2.MIN_THRESHOLD <= moral_filter.threshold <= MoralFilterV2.MAX_THRESHOLD, (
                    f"Invariant violated: threshold={moral_filter.threshold}"
                )


pytestmark = pytest.mark.security
