"""Unit tests for core modules."""
import numpy as np
import pytest
from src.core.memory import MultiLevelSynapticMemory, MemoryConfig
from src.core.moral import MoralFilter, MoralConfig
from src.core.qilm import QILM
from src.core.rhythm import CognitiveRhythm, RhythmConfig

class TestMultiLevelSynapticMemory:
    """Test MultiLevelSynapticMemory class."""
    
    def test_initialization(self):
        config = MemoryConfig(
            dimension=10,
            lambda_l1=0.5,
            lambda_l2=0.1,
            lambda_l3=0.01,
            theta_l1=1.2,
            theta_l2=2.5,
            gating12=0.45,
            gating23=0.30
        )
        memory = MultiLevelSynapticMemory(config)
        assert memory.dim == 10
        assert np.allclose(memory.l1, 0)
        assert np.allclose(memory.l2, 0)
        assert np.allclose(memory.l3, 0)
    
    def test_update_dimension_mismatch(self):
        config = MemoryConfig(
            dimension=10,
            lambda_l1=0.5,
            lambda_l2=0.1,
            lambda_l3=0.01,
            theta_l1=1.2,
            theta_l2=2.5,
            gating12=0.45,
            gating23=0.30
        )
        memory = MultiLevelSynapticMemory(config)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            memory.update(np.array([1.0, 2.0]))
    
    def test_state_returns_copy(self):
        config = MemoryConfig(
            dimension=5,
            lambda_l1=0.5,
            lambda_l2=0.1,
            lambda_l3=0.01,
            theta_l1=1.2,
            theta_l2=2.5,
            gating12=0.45,
            gating23=0.30
        )
        memory = MultiLevelSynapticMemory(config)
        l1, l2, l3 = memory.state()
        l1[0] = 999.0
        # Verify that modifying returned state doesn't affect internal state
        l1_new, _, _ = memory.state()
        assert l1_new[0] != 999.0
    
    def test_norms(self):
        config = MemoryConfig(
            dimension=3,
            lambda_l1=0.0,
            lambda_l2=0.0,
            lambda_l3=0.0,
            theta_l1=10.0,
            theta_l2=10.0,
            gating12=0.0,
            gating23=0.0
        )
        memory = MultiLevelSynapticMemory(config)
        event = np.array([3.0, 4.0, 0.0])
        memory.update(event)
        n1, n2, n3 = memory.norms()
        assert abs(n1 - 5.0) < 0.01

class TestMoralFilter:
    """Test MoralFilter class."""
    
    def test_initialization(self):
        config = MoralConfig(threshold=0.5, adapt_rate=0.05, min_threshold=0.3, max_threshold=0.9)
        moral = MoralFilter(config)
        assert moral.threshold == 0.5
    
    def test_evaluate_out_of_range(self):
        config = MoralConfig()
        moral = MoralFilter(config)
        with pytest.raises(ValueError, match="Moral value must be in"):
            moral.evaluate(1.5)
        with pytest.raises(ValueError, match="Moral value must be in"):
            moral.evaluate(-0.1)
    
    def test_adapt_increase(self):
        config = MoralConfig(threshold=0.5, adapt_rate=0.1)
        moral = MoralFilter(config)
        moral.adapt(0.8)  # High accept rate
        assert moral.threshold > 0.5
    
    def test_adapt_decrease(self):
        config = MoralConfig(threshold=0.5, adapt_rate=0.1)
        moral = MoralFilter(config)
        moral.adapt(0.2)  # Low accept rate
        assert moral.threshold < 0.5
    
    def test_adapt_bounds(self):
        config = MoralConfig(threshold=0.3, adapt_rate=0.5, min_threshold=0.3, max_threshold=0.9)
        moral = MoralFilter(config)
        
        # Try to go below min
        moral.adapt(0.0)
        assert moral.threshold == 0.3
        
        # Try to go above max
        moral.threshold = 0.9
        moral.adapt(1.0)
        assert moral.threshold == 0.9

class TestQILM:
    """Test QILM class."""
    
    def test_initialization(self):
        qilm = QILM(dimension=10)
        assert qilm.dim == 10
        assert len(qilm) == 0
    
    def test_entangle_dimension_mismatch(self):
        qilm = QILM(dimension=5)
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            qilm.entangle(np.array([1.0, 2.0]))
    
    def test_entangle_without_phase(self):
        qilm = QILM(dimension=3)
        vector = np.array([1.0, 2.0, 3.0])
        qilm.entangle(vector)
        assert len(qilm) == 1
        assert qilm.phases[0] is not None
    
    def test_entangle_with_phase(self):
        qilm = QILM(dimension=3)
        vector = np.array([1.0, 2.0, 3.0])
        phase = 0.75
        qilm.entangle(vector, phase)
        assert len(qilm) == 1
        assert qilm.phases[0] == phase
    
    def test_retrieve_empty(self):
        qilm = QILM(dimension=5)
        results = qilm.retrieve(0.5)
        assert len(results) == 0
    
    def test_retrieve_with_tolerance(self):
        qilm = QILM(dimension=2)
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        v3 = np.array([1.0, 1.0])
        
        qilm.entangle(v1, 0.1)
        qilm.entangle(v2, 0.5)
        qilm.entangle(v3, 0.15)
        
        results = qilm.retrieve(0.1, tolerance=0.1)
        assert len(results) == 2  # v1 and v3

class TestCognitiveRhythm:
    """Test CognitiveRhythm class."""
    
    def test_initialization(self):
        config = RhythmConfig(wake_duration=8, sleep_duration=3)
        rhythm = CognitiveRhythm(config)
        assert rhythm.phase == "wake"
        assert rhythm.counter == 8
    
    def test_is_wake(self):
        config = RhythmConfig(wake_duration=2, sleep_duration=1)
        rhythm = CognitiveRhythm(config)
        assert rhythm.is_wake()
        rhythm.phase = "sleep"
        assert not rhythm.is_wake()
    
    def test_step_transition_to_sleep(self):
        config = RhythmConfig(wake_duration=2, sleep_duration=1)
        rhythm = CognitiveRhythm(config)
        
        rhythm.step()
        assert rhythm.phase == "wake"
        assert rhythm.counter == 1
        
        rhythm.step()
        assert rhythm.phase == "sleep"
        assert rhythm.counter == 1
    
    def test_step_transition_to_wake(self):
        config = RhythmConfig(wake_duration=2, sleep_duration=1)
        rhythm = CognitiveRhythm(config)
        
        # Go to sleep
        rhythm.step()
        rhythm.step()
        assert rhythm.phase == "sleep"
        
        # Go back to wake
        rhythm.step()
        assert rhythm.phase == "wake"
        assert rhythm.counter == 2
    
    def test_full_cycle(self):
        config = RhythmConfig(wake_duration=3, sleep_duration=2)
        rhythm = CognitiveRhythm(config)
        
        phases = []
        for _ in range(10):
            phases.append(rhythm.phase)
            rhythm.step()
        
        # Should see wake-wake-wake-sleep-sleep pattern
        assert phases[:3] == ["wake", "wake", "wake"]
        assert phases[3:5] == ["sleep", "sleep"]
        assert phases[5:8] == ["wake", "wake", "wake"]
