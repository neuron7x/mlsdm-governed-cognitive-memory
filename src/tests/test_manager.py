"""Integration tests for CognitiveMemoryManager."""
import asyncio
import numpy as np
import pytest
from src.manager import CognitiveMemoryManager

@pytest.fixture
def manager_config():
    return {
        "dimension": 10,
        "strict_mode": False,
        "multi_level_memory": {
            "lambda_l1": 0.5,
            "lambda_l2": 0.1,
            "lambda_l3": 0.01,
            "theta_l1": 1.2,
            "theta_l2": 2.5,
            "gating12": 0.45,
            "gating23": 0.30,
        },
        "moral_filter": {
            "threshold": 0.5,
            "adapt_rate": 0.05,
            "min_threshold": 0.3,
            "max_threshold": 0.9,
        },
        "cognitive_rhythm": {
            "wake_duration": 8,
            "sleep_duration": 3,
        }
    }

@pytest.mark.asyncio
async def test_manager_process_event_accepted(manager_config):
    """Test that events with high moral values are accepted."""
    manager = CognitiveMemoryManager(manager_config)
    
    event = np.random.randn(10)
    moral_value = 0.8
    
    state = await manager.process_event(event, moral_value)
    
    assert state["metrics"]["total"] == 1
    assert state["metrics"]["accepted"] == 1
    assert state["qilm_size"] == 1

@pytest.mark.asyncio
async def test_manager_process_event_rejected_moral(manager_config):
    """Test that events with low moral values are rejected."""
    manager = CognitiveMemoryManager(manager_config)
    
    event = np.random.randn(10)
    moral_value = 0.2  # Below threshold
    
    state = await manager.process_event(event, moral_value)
    
    assert state["metrics"]["total"] == 1
    assert state["metrics"]["accepted"] == 0
    assert state["metrics"]["latent"] == 1
    assert state["qilm_size"] == 0

@pytest.mark.asyncio
async def test_manager_strict_mode_detection(manager_config):
    """Test that strict mode detects anomalous vectors."""
    manager_config["strict_mode"] = True
    manager = CognitiveMemoryManager(manager_config)
    
    # Anomalous vector (large norm)
    event = np.ones(10) * 25.0
    moral_value = 0.8
    
    with pytest.raises(ValueError, match="Sensitive or anomalous"):
        await manager.process_event(event, moral_value)

@pytest.mark.asyncio
async def test_manager_moral_adaptation(manager_config):
    """Test that moral filter adapts based on accept rate."""
    manager = CognitiveMemoryManager(manager_config)
    
    initial_threshold = manager.moral.threshold
    
    # Process 100 events with low accept rate (50 triggers adaptation, need more to see effect)
    for i in range(100):
        event = np.random.randn(10) * 0.1  # Small events
        moral_value = 0.3  # Low moral value
        await manager.process_event(event, moral_value)
    
    # Threshold should decrease (adaptation happens at step 50 and 100)
    assert manager.moral.threshold <= initial_threshold

@pytest.mark.asyncio
async def test_manager_rhythm_gating(manager_config):
    """Test that events are gated during sleep phase."""
    manager_config["cognitive_rhythm"]["wake_duration"] = 1
    manager = CognitiveMemoryManager(manager_config)
    
    event = np.random.randn(10)
    moral_value = 0.8
    
    # First event during wake
    state1 = await manager.process_event(event, moral_value)
    assert state1["metrics"]["accepted"] == 1
    
    # Second event during sleep (after rhythm stepped)
    state2 = await manager.process_event(event, moral_value)
    assert state2["metrics"]["latent"] == 1
    assert state2["metrics"]["accepted"] == 1  # Still 1 from before

@pytest.mark.asyncio
async def test_manager_state_consistency(manager_config):
    """Test that manager state is consistent across updates."""
    manager = CognitiveMemoryManager(manager_config)
    
    for i in range(10):
        event = np.random.randn(10)
        moral_value = np.random.rand()
        state = await manager.process_event(event, moral_value)
        
        # Verify state structure
        assert "norms" in state
        assert "phase" in state
        assert "moral_threshold" in state
        assert "qilm_size" in state
        assert "metrics" in state
        
        # Verify metrics consistency
        assert state["metrics"]["total"] == i + 1
        assert state["metrics"]["accepted"] + state["metrics"]["latent"] <= state["metrics"]["total"]
        
        # Verify norms are non-negative
        assert state["norms"]["L1"] >= 0
        assert state["norms"]["L2"] >= 0
        assert state["norms"]["L3"] >= 0
