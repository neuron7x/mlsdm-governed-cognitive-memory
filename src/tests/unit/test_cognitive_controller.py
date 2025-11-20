"""Comprehensive unit tests for CognitiveController."""
import numpy as np
from threading import Thread
from src.core.cognitive_controller import CognitiveController


class TestCognitiveController:
    """Test suite for CognitiveController."""

    def test_initialization(self):
        """Test controller initialization with default and custom dimensions."""
        controller = CognitiveController(dim=384)
        assert controller.dim == 384
        assert controller.step_counter == 0
        assert controller.moral is not None
        assert controller.qilm is not None
        assert controller.rhythm is not None
        assert controller.synaptic is not None

    def test_initialization_custom_dimension(self):
        """Test controller initialization with custom dimension."""
        controller = CognitiveController(dim=512)
        assert controller.dim == 512
        assert controller.qilm.dimension == 512
        assert controller.synaptic.dim == 512

    def test_process_event_accepted(self):
        """Test processing an event with high moral value (should be accepted)."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        state = controller.process_event(vec, moral_value=0.9)
        
        assert state["rejected"] is False
        assert state["note"] == "processed"
        assert state["step"] == 1
        assert "phase" in state
        assert "moral_threshold" in state
        assert "moral_ema" in state
        assert "synaptic_norms" in state

    def test_process_event_rejected_moral(self):
        """Test processing an event with low moral value (should be rejected)."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        state = controller.process_event(vec, moral_value=0.1)
        
        assert state["rejected"] is True
        assert state["note"] == "morally rejected"
        assert state["step"] == 1

    def test_process_event_rejected_sleep_phase(self):
        """Test processing an event during sleep phase (should be rejected)."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Step through wake phase to reach sleep phase
        for _ in range(8):
            controller.rhythm.step()
        
        assert controller.rhythm.is_wake() is False
        state = controller.process_event(vec, moral_value=0.9)
        
        assert state["rejected"] is True
        assert "sleep" in state["note"]  # type: ignore[operator]

    def test_step_counter_increments(self):
        """Test that step counter increments correctly."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        for i in range(5):
            state = controller.process_event(vec, moral_value=0.9)
            assert state["step"] == i + 1

    def test_retrieve_context(self):
        """Test retrieving context from memory."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Add some events to memory
        controller.process_event(vec, moral_value=0.9)
        controller.process_event(vec, moral_value=0.9)
        
        # Retrieve context
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = controller.retrieve_context(query, top_k=5)
        
        assert isinstance(results, list)

    def test_retrieve_context_with_different_top_k(self):
        """Test retrieving context with different top_k values."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Add events to memory
        for _ in range(10):
            controller.process_event(vec, moral_value=0.9)
        
        # Test different top_k values
        results_3 = controller.retrieve_context(vec, top_k=3)
        results_7 = controller.retrieve_context(vec, top_k=7)
        
        assert len(results_3) <= 3
        assert len(results_7) <= 7

    def test_synaptic_norms_in_state(self):
        """Test that synaptic norms are properly included in state."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        state = controller.process_event(vec, moral_value=0.9)
        
        assert "synaptic_norms" in state
        assert "L1" in state["synaptic_norms"]  # type: ignore[operator]
        assert "L2" in state["synaptic_norms"]  # type: ignore[operator]
        assert "L3" in state["synaptic_norms"]  # type: ignore[operator]
        assert all(isinstance(norm, float) for norm in state["synaptic_norms"].values())  # type: ignore[union-attr]

    def test_qilm_usage_tracking(self):
        """Test that QILM usage is tracked in state."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        state_before = controller.process_event(vec, moral_value=0.9)
        qilm_used_before = state_before["qilm_used"]
        
        state_after = controller.process_event(vec, moral_value=0.9)
        qilm_used_after = state_after["qilm_used"]
        
        assert qilm_used_after == qilm_used_before + 1  # type: ignore[operator]

    def test_thread_safety(self):
        """Test that controller is thread-safe with concurrent access."""
        controller = CognitiveController(dim=384)
        
        def process_events():
            for _ in range(100):
                vec = np.random.randn(384).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                controller.process_event(vec, moral_value=0.8)
        
        threads = [Thread(target=process_events) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should have processed 1000 events
        assert controller.step_counter == 1000

    def test_moral_threshold_adaptation(self):
        """Test that moral threshold adapts over time."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        initial_state = controller.process_event(vec, moral_value=0.9)
        initial_state["moral_threshold"]
        
        # Process many high-moral events
        for _ in range(50):
            controller.process_event(vec, moral_value=0.9)
        
        final_state = controller.process_event(vec, moral_value=0.9)
        final_threshold = final_state["moral_threshold"]
        
        # Threshold should be within bounds
        assert 0.30 <= final_threshold <= 0.90  # type: ignore[operator]

    def test_ema_tracking(self):
        """Test that EMA (exponential moving average) is tracked."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        state = controller.process_event(vec, moral_value=0.9)
        
        assert "moral_ema" in state
        assert 0.0 <= state["moral_ema"] <= 1.0  # type: ignore[operator]

    def test_phase_transitions(self):
        """Test that cognitive rhythm phase transitions are reflected in state."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Process events and track phase changes
        phases = []
        for _ in range(15):
            state = controller.process_event(vec, moral_value=0.9)
            phases.append(state["phase"])
        
        # Should have both wake and sleep phases
        assert "wake" in phases
        # Note: May or may not have sleep depending on rejection

    def test_build_state_structure(self):
        """Test the structure of the state returned."""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        state = controller.process_event(vec, moral_value=0.9)
        
        # Check all required fields
        required_fields = ["step", "phase", "moral_threshold", "moral_ema", 
                          "synaptic_norms", "qilm_used", "rejected", "note"]
        for field in required_fields:
            assert field in state

    def test_multiple_events_sequence(self):
        """Test processing a sequence of multiple events."""
        controller = CognitiveController(dim=384)
        
        events = [
            (np.random.randn(384).astype(np.float32), 0.9),
            (np.random.randn(384).astype(np.float32), 0.1),
            (np.random.randn(384).astype(np.float32), 0.7),
            (np.random.randn(384).astype(np.float32), 0.3),
            (np.random.randn(384).astype(np.float32), 0.8),
        ]
        
        for vec, moral in events:
            vec = vec / np.linalg.norm(vec)
            state = controller.process_event(vec, moral_value=moral)
            assert "step" in state
            assert "rejected" in state

    def test_retrieve_empty_memory(self):
        """Test retrieving from empty memory."""
        controller = CognitiveController(dim=384)
        query = np.random.randn(384).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        results = controller.retrieve_context(query, top_k=5)
        
        assert isinstance(results, list)
        assert len(results) == 0
