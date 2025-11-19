"""Property-based tests for MLSDM Governed Cognitive Memory invariants."""
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.numpy import arrays
from src.core.memory import MultiLevelSynapticMemory, MemoryConfig
from src.core.moral import MoralFilter, MoralConfig
from src.core.qilm import QILM
from src.core.rhythm import CognitiveRhythm, RhythmConfig

# Strategy for generating valid memory configs
@st.composite
def memory_configs(draw):
    return MemoryConfig(
        dimension=draw(st.integers(min_value=2, max_value=100)),
        lambda_l1=draw(st.floats(min_value=0.0, max_value=0.99)),
        lambda_l2=draw(st.floats(min_value=0.0, max_value=0.99)),
        lambda_l3=draw(st.floats(min_value=0.0, max_value=0.99)),
        theta_l1=draw(st.floats(min_value=0.1, max_value=10.0)),
        theta_l2=draw(st.floats(min_value=0.1, max_value=10.0)),
        gating12=draw(st.floats(min_value=0.0, max_value=1.0)),
        gating23=draw(st.floats(min_value=0.0, max_value=1.0)),
    )

@given(config=memory_configs(), event=st.data())
@settings(max_examples=100)
def test_memory_energy_non_increasing(config, event):
    """Test that total memory mass is non-increasing beyond input energy."""
    memory = MultiLevelSynapticMemory(config)
    
    # Generate event with correct dimension
    event_vec = event.draw(arrays(
        dtype=np.float32,
        shape=(config.dimension,),
        elements=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False, 
                          width=32, allow_subnormal=False)
    ))
    
    # Get initial total
    l1_0, l2_0, l3_0 = memory.state()
    total_before = np.sum(np.abs(l1_0)) + np.sum(np.abs(l2_0)) + np.sum(np.abs(l3_0))
    
    # Update
    memory.update(event_vec)
    
    # Get final total
    l1_1, l2_1, l3_1 = memory.state()
    total_after = np.sum(np.abs(l1_1)) + np.sum(np.abs(l2_1)) + np.sum(np.abs(l3_1))
    
    # Invariant: total_after ≤ input + (1-λ₁)*total_before
    expected_max = np.sum(event_vec) + (1 - config.lambda_l1) * total_before
    
    # Allow small numerical tolerance for float32
    assert total_after <= expected_max + 1e-4, f"Energy increased beyond input: {total_after} > {expected_max}"

@given(config=memory_configs(), num_steps=st.integers(min_value=1, max_value=20))
@settings(max_examples=50)
def test_memory_non_negative_for_non_negative_inputs(config, num_steps):
    """Test that L1, L2, L3 remain non-negative for non-negative inputs."""
    memory = MultiLevelSynapticMemory(config)
    
    for _ in range(num_steps):
        event = np.random.rand(config.dimension).astype(np.float32) * 5.0
        memory.update(event)
        
        l1, l2, l3 = memory.state()
        assert np.all(l1 >= -1e-6), "L1 became negative"
        assert np.all(l2 >= -1e-6), "L2 became negative"
        assert np.all(l3 >= -1e-6), "L3 became negative"

@given(
    threshold=st.floats(min_value=0.3, max_value=0.9),
    moral_value=st.floats(min_value=0.0, max_value=1.0)
)
@settings(max_examples=100)
def test_moral_filter_threshold_invariant(threshold, moral_value):
    """Test that moral filter threshold stays within bounds."""
    config = MoralConfig(threshold=threshold)
    moral_filter = MoralFilter(config)
    
    # Test evaluation
    result = moral_filter.evaluate(moral_value)
    assert result == (moral_value >= threshold)
    
    # Test adaptation maintains bounds
    for accept_rate in [0.0, 0.3, 0.5, 0.7, 1.0]:
        moral_filter.adapt(accept_rate)
        assert config.min_threshold <= moral_filter.threshold <= config.max_threshold

@given(
    dimension=st.integers(min_value=2, max_value=100),
    num_entanglements=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=50)
def test_qilm_length_invariant(dimension, num_entanglements):
    """Test that QILM maintains equal lengths of memory and phases."""
    qilm = QILM(dimension)
    
    for _ in range(num_entanglements):
        vector = np.random.randn(dimension).astype(np.float32)
        phase = np.random.rand()
        qilm.entangle(vector, phase)
    
    # Invariant: |memory| = |phases|
    assert len(qilm.memory) == len(qilm.phases)
    assert len(qilm) == num_entanglements

@given(
    wake_duration=st.integers(min_value=1, max_value=20),
    sleep_duration=st.integers(min_value=1, max_value=20),
    num_steps=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=50)
def test_rhythm_counter_invariant(wake_duration, sleep_duration, num_steps):
    """Test that rhythm counter is always positive and phases alternate correctly."""
    config = RhythmConfig(wake_duration=wake_duration, sleep_duration=sleep_duration)
    rhythm = CognitiveRhythm(config)
    
    for _ in range(num_steps):
        # Invariant: counter > 0
        assert rhythm.counter > 0, "Counter became non-positive"
        
        # Check phase bounds
        if rhythm.phase == "wake":
            assert rhythm.counter <= wake_duration
        else:
            assert rhythm.counter <= sleep_duration
        
        rhythm.step()
    
    # Counter should still be valid
    assert rhythm.counter > 0

@given(data=st.data())
@settings(max_examples=50)
def test_qilm_retrieve_tolerance(data):
    """Test that QILM retrieve respects tolerance."""
    dimension = data.draw(st.integers(min_value=2, max_value=50))
    qilm = QILM(dimension)
    
    # Add some vectors with known phases
    test_phase = 0.5
    tolerance = 0.1
    
    for i in range(5):
        vector = np.random.randn(dimension).astype(np.float32)
        phase = test_phase + data.draw(st.floats(min_value=-0.3, max_value=0.3))
        qilm.entangle(vector, phase)
    
    # Retrieve
    results = qilm.retrieve(test_phase, tolerance)
    
    # All results should be within tolerance
    for vec in results:
        idx = next(i for i, v in enumerate(qilm.memory) if np.allclose(v, vec))
        assert abs(qilm.phases[idx] - test_phase) <= tolerance
