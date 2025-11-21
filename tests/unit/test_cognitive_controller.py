"""
Unit Tests for Cognitive Controller

Tests memory monitoring, processing time limits, and emergency shutdown functionality.
"""

import gc
import time

import numpy as np
import pytest

from mlsdm.core.cognitive_controller import CognitiveController


class TestCognitiveControllerInitialization:
    """Test cognitive controller initialization."""

    def test_default_initialization(self):
        """Test controller can be initialized with defaults."""
        controller = CognitiveController()
        assert controller.dim == 384
        assert controller.memory_threshold_mb == 1024.0
        assert controller.max_processing_time_ms == 1000.0
        assert controller.emergency_shutdown is False
        assert controller.step_counter == 0

    def test_custom_initialization(self):
        """Test controller can be initialized with custom values."""
        controller = CognitiveController(
            dim=128,
            memory_threshold_mb=512.0,
            max_processing_time_ms=500.0
        )
        assert controller.dim == 128
        assert controller.memory_threshold_mb == 512.0
        assert controller.max_processing_time_ms == 500.0
        assert controller.emergency_shutdown is False


class TestCognitiveControllerMemoryMonitoring:
    """Test memory monitoring functionality."""

    def test_get_memory_usage(self):
        """Test memory usage can be retrieved."""
        controller = CognitiveController()
        memory_mb = controller.get_memory_usage()
        assert isinstance(memory_mb, float)
        assert memory_mb > 0, "Memory usage should be positive"
        # Sanity check: memory usage should be reasonable (< 10GB for this test)
        assert memory_mb < 10240, f"Memory usage seems unreasonable: {memory_mb} MB"

    def test_memory_threshold_exceeded_triggers_emergency_shutdown(self):
        """Test emergency shutdown is triggered when memory threshold is exceeded."""
        # Set a very low threshold to trigger emergency shutdown
        controller = CognitiveController(memory_threshold_mb=0.001)

        vector = np.random.randn(384).astype(np.float32)
        result = controller.process_event(vector, moral_value=0.8)

        assert controller.emergency_shutdown is True
        assert result["rejected"] is True
        assert "emergency shutdown" in result["note"]

    def test_emergency_shutdown_blocks_further_processing(self):
        """Test that once emergency shutdown is triggered, no further events are processed."""
        controller = CognitiveController(memory_threshold_mb=0.001)

        vector = np.random.randn(384).astype(np.float32)

        # First event triggers emergency shutdown
        result1 = controller.process_event(vector, moral_value=0.8)
        assert controller.emergency_shutdown is True
        assert result1["rejected"] is True

        # Second event should be rejected immediately
        result2 = controller.process_event(vector, moral_value=0.8)
        assert result2["rejected"] is True
        assert result2["note"] == "emergency shutdown"

    def test_reset_emergency_shutdown(self):
        """Test emergency shutdown can be reset."""
        controller = CognitiveController(memory_threshold_mb=0.001)

        vector = np.random.randn(384).astype(np.float32)

        # Trigger emergency shutdown
        controller.process_event(vector, moral_value=0.8)
        assert controller.emergency_shutdown is True

        # Reset emergency shutdown
        controller.reset_emergency_shutdown()
        assert controller.emergency_shutdown is False


class TestCognitiveControllerProcessingTime:
    """Test processing time limits."""

    def test_normal_processing_time(self):
        """Test events process within normal time limits."""
        # Use a reasonable time limit
        controller = CognitiveController(max_processing_time_ms=5000.0)

        vector = np.random.randn(384).astype(np.float32)
        result = controller.process_event(vector, moral_value=0.8)

        # Event should be processed normally (not rejected for time)
        if result["rejected"]:
            # Could be rejected for other reasons (sleep phase, moral)
            assert "processing time exceeded" not in result["note"]


class TestCognitiveControllerProcessEvent:
    """Test event processing functionality."""

    def test_process_accepted_event(self):
        """Test processing of an accepted event."""
        controller = CognitiveController()
        vector = np.random.randn(384).astype(np.float32)

        result = controller.process_event(vector, moral_value=0.8)

        assert isinstance(result, dict)
        assert "step" in result
        assert "rejected" in result
        assert "note" in result
        assert controller.step_counter > 0

    def test_process_rejected_moral_event(self):
        """Test processing of morally rejected event."""
        controller = CognitiveController()
        vector = np.random.randn(384).astype(np.float32)

        # Use low moral value to trigger rejection
        result = controller.process_event(vector, moral_value=0.1)

        assert result["rejected"] is True
        assert "morally rejected" in result["note"]

    def test_step_counter_increments(self):
        """Test step counter increments with each event."""
        controller = CognitiveController()
        vector = np.random.randn(384).astype(np.float32)

        initial_count = controller.step_counter
        controller.process_event(vector, moral_value=0.8)
        assert controller.step_counter == initial_count + 1

        controller.process_event(vector, moral_value=0.8)
        assert controller.step_counter == initial_count + 2


class TestCognitiveControllerMemoryLeak:
    """Test memory leak detection with high volume of events."""

    @pytest.mark.slow
    def test_no_memory_leak_10k_events(self):
        """Test that processing 10k events doesn't cause excessive memory growth."""
        controller = CognitiveController()

        # Get initial memory usage
        gc.collect()  # Force garbage collection before measuring
        time.sleep(0.1)  # Give GC time to complete
        initial_memory = controller.get_memory_usage()

        # Process 10k events
        num_events = 10_000
        for i in range(num_events):
            vector = np.random.randn(384).astype(np.float32)
            moral_value = 0.5 + (i % 10) * 0.05  # Vary moral values
            controller.process_event(vector, moral_value)

            # Periodically force garbage collection
            if i % 1000 == 0:
                gc.collect()

        # Force final garbage collection
        gc.collect()
        time.sleep(0.1)  # Give GC time to complete
        final_memory = controller.get_memory_usage()

        memory_growth = final_memory - initial_memory

        # Assert memory growth is reasonable (< 500 MB for 10k events)
        # This is a soft check - some growth is expected due to data structures
        assert memory_growth < 500, (
            f"Potential memory leak detected: memory grew by {memory_growth:.2f} MB "
            f"after processing {num_events} events. "
            f"Initial: {initial_memory:.2f} MB, Final: {final_memory:.2f} MB"
        )

        # Verify controller is still functional after high load
        vector = np.random.randn(384).astype(np.float32)
        result = controller.process_event(vector, moral_value=0.8)
        assert isinstance(result, dict)
        assert controller.step_counter == num_events + 1

    @pytest.mark.slow
    def test_memory_stays_stable_over_time(self):
        """Test memory usage stabilizes and doesn't continuously grow."""
        controller = CognitiveController()

        gc.collect()
        time.sleep(0.1)

        memory_samples = []

        # Process events in batches and measure memory
        for _ in range(5):
            for _ in range(1000):
                vector = np.random.randn(384).astype(np.float32)
                controller.process_event(vector, moral_value=0.7)

            gc.collect()
            time.sleep(0.1)
            memory_samples.append(controller.get_memory_usage())

        # Check that memory doesn't continuously grow between batches
        # Allow for initial growth but later samples should stabilize
        if len(memory_samples) >= 3:
            # Compare last 2 samples with first sample
            initial_growth = memory_samples[1] - memory_samples[0]
            later_growth = memory_samples[-1] - memory_samples[-2]

            # Later growth should be less than or similar to initial growth
            # This indicates stabilization rather than continuous leak
            assert later_growth <= initial_growth * 2, (
                f"Memory continues to grow significantly: "
                f"initial growth: {initial_growth:.2f} MB, "
                f"later growth: {later_growth:.2f} MB"
            )


class TestCognitiveControllerRetrieveContext:
    """Test context retrieval functionality."""

    def test_retrieve_context(self):
        """Test context retrieval works."""
        controller = CognitiveController()

        # Add some events first
        for _ in range(10):
            vector = np.random.randn(384).astype(np.float32)
            controller.process_event(vector, moral_value=0.8)

        # Retrieve context
        query = np.random.randn(384).astype(np.float32)
        results = controller.retrieve_context(query, top_k=5)

        assert isinstance(results, list)
        assert len(results) <= 5
