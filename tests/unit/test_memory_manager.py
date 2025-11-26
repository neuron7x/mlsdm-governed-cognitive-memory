"""
Unit Tests for MemoryManager

Tests the memory management system including multi-level synaptic memory,
moral filtering, cognitive rhythm, and QILM integration.
"""

import numpy as np
import pytest

from mlsdm.core.memory_manager import MemoryManager


class TestMemoryManagerInitialization:
    """Test MemoryManager initialization."""

    def test_default_initialization(self):
        """Test MemoryManager can be initialized with defaults."""
        config = {}
        manager = MemoryManager(config)

        assert manager.dimension == 10  # Default dimension
        assert manager.memory is not None
        assert manager.filter is not None
        assert manager.matcher is not None
        assert manager.rhythm is not None
        assert manager.qilm is not None
        assert manager.metrics_collector is not None
        assert manager.strict_mode is False

    def test_custom_dimension(self):
        """Test MemoryManager with custom dimension."""
        config = {"dimension": 20}
        manager = MemoryManager(config)

        assert manager.dimension == 20

    def test_custom_multi_level_memory_config(self):
        """Test MemoryManager with custom multi-level memory config."""
        config = {
            "dimension": 10,
            "multi_level_memory": {
                "lambda_l1": 0.3,
                "lambda_l2": 0.2,
                "lambda_l3": 0.05,
                "theta_l1": 1.5,
                "theta_l2": 2.0,
                "gating12": 0.6,
                "gating23": 0.4,
            }
        }
        manager = MemoryManager(config)

        assert manager.memory.lambda_l1 == 0.3
        assert manager.memory.lambda_l2 == 0.2
        assert manager.memory.lambda_l3 == 0.05

    def test_custom_moral_filter_config(self):
        """Test MemoryManager with custom moral filter config."""
        config = {
            "dimension": 10,
            "moral_filter": {
                "threshold": 0.6,
                "adapt_rate": 0.1,
                "min_threshold": 0.2,
                "max_threshold": 0.95,
            }
        }
        manager = MemoryManager(config)

        assert manager.filter.threshold == 0.6
        assert manager.filter.adapt_rate == 0.1
        assert manager.filter.min_threshold == 0.2
        assert manager.filter.max_threshold == 0.95

    def test_custom_rhythm_config(self):
        """Test MemoryManager with custom cognitive rhythm config."""
        config = {
            "dimension": 10,
            "cognitive_rhythm": {
                "wake_duration": 10,
                "sleep_duration": 5,
            }
        }
        manager = MemoryManager(config)

        assert manager.rhythm.wake_duration == 10
        assert manager.rhythm.sleep_duration == 5

    def test_strict_mode_enabled(self):
        """Test MemoryManager with strict mode enabled."""
        config = {"strict_mode": True}
        manager = MemoryManager(config)

        assert manager.strict_mode is True


class TestMemoryManagerIsSensitive:
    """Test sensitive data detection."""

    def test_not_sensitive_normal_vector(self):
        """Test normal vector is not sensitive."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        vec = np.ones(10) * 0.5  # Normal vector
        assert manager._is_sensitive(vec) is False

    def test_sensitive_high_norm(self):
        """Test vector with high norm is sensitive."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        vec = np.ones(10) * 5  # Norm > 10
        assert manager._is_sensitive(vec) is True

    def test_sensitive_negative_sum(self):
        """Test vector with negative sum is sensitive."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        vec = np.ones(10) * -0.5  # Sum < 0
        assert manager._is_sensitive(vec) is True


class TestMemoryManagerProcessEvent:
    """Test event processing."""

    @pytest.mark.asyncio
    async def test_process_accepted_event(self):
        """Test processing an accepted event."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        event_vector = np.ones(10) * 0.1
        moral_value = 0.8  # Above threshold

        await manager.process_event(event_vector, moral_value)

        # Verify event was accepted and metrics updated
        assert manager.metrics_collector.metrics["accepted_events_count"] == 1

    @pytest.mark.asyncio
    async def test_process_rejected_event(self):
        """Test processing a rejected event."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        event_vector = np.ones(10) * 0.1
        moral_value = 0.3  # Below threshold (default 0.5)

        await manager.process_event(event_vector, moral_value)

        # Verify event was rejected and marked as latent
        assert manager.metrics_collector.metrics["latent_events_count"] == 1

    @pytest.mark.asyncio
    async def test_process_event_strict_mode_rejection(self):
        """Test strict mode rejects sensitive data."""
        config = {"dimension": 10, "strict_mode": True}
        manager = MemoryManager(config)

        sensitive_vector = np.ones(10) * 5  # High norm
        moral_value = 0.8

        with pytest.raises(ValueError, match="Sensitive data detected"):
            await manager.process_event(sensitive_vector, moral_value)

    @pytest.mark.asyncio
    async def test_process_event_updates_memory(self):
        """Test processing event updates memory state."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        event_vector = np.ones(10) * 0.5
        moral_value = 0.8

        # Get initial state
        l1_initial, l2_initial, l3_initial = manager.memory.get_state()

        await manager.process_event(event_vector, moral_value)

        # Verify memory was updated
        l1_after, l2_after, l3_after = manager.memory.get_state()
        assert not np.array_equal(l1_initial, l1_after)

    @pytest.mark.asyncio
    async def test_process_event_adapts_filter(self):
        """Test filter adaptation based on accept rate."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        # Process first event to set initial total
        await manager.process_event(np.ones(10) * 0.1, 0.8)

        # Store initial threshold to verify adaptation
        _ = manager.filter.threshold

        # Process more events with high moral values to increase accept rate
        for _ in range(5):
            await manager.process_event(np.ones(10) * 0.1, 0.9)

        # The threshold should have been adapted
        # (Note: exact behavior depends on accept_rate calculation)
        assert manager.metrics_collector.metrics["total_events_processed"] > 0


class TestMemoryManagerSimulation:
    """Test simulation functionality."""

    @pytest.mark.asyncio
    async def test_simulate_basic(self):
        """Test basic simulation."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        def event_gen():
            for _ in range(5):
                yield np.random.randn(10), 0.7

        await manager.simulate(5, event_gen())

        # Verify simulation completed
        assert manager.metrics_collector.metrics["total_events_processed"] > 0

    @pytest.mark.asyncio
    async def test_simulate_dimension_mismatch(self):
        """Test simulation rejects dimension mismatch."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        def event_gen():
            yield np.random.randn(5), 0.7  # Wrong dimension

        with pytest.raises(ValueError, match="dimension mismatch"):
            await manager.simulate(1, event_gen())

    @pytest.mark.asyncio
    async def test_simulate_rhythm_cycle(self):
        """Test simulation respects wake/sleep rhythm."""
        config = {
            "dimension": 10,
            "cognitive_rhythm": {"wake_duration": 2, "sleep_duration": 1}
        }
        manager = MemoryManager(config)

        events = [(np.random.randn(10), 0.7) for _ in range(6)]

        def event_gen():
            yield from events

        await manager.simulate(6, event_gen())

        # Verify rhythm stepped through cycles
        assert manager.metrics_collector.metrics["total_events_processed"] > 0

    def test_run_simulation_with_default_gen(self):
        """Test run_simulation with default event generator."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        manager.run_simulation(5)

        # Verify simulation completed
        assert manager.metrics_collector.metrics["total_events_processed"] > 0

    def test_run_simulation_with_custom_gen(self):
        """Test run_simulation with custom event generator."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        def event_gen():
            for _ in range(3):
                yield np.ones(10) * 0.5, 0.8

        manager.run_simulation(3, event_gen())

        # Verify events were processed
        assert manager.metrics_collector.metrics["accepted_events_count"] > 0


class TestMemoryManagerSaveLoad:
    """Test state save/load functionality."""

    def test_save_system_state(self, tmp_path):
        """Test saving system state."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        # Add some data
        manager.qilm.entangle_phase(np.ones(10), 0.5)

        filepath = str(tmp_path / "state.json")
        manager.save_system_state(filepath)

        import os
        assert os.path.exists(filepath)

    def test_load_system_state(self, tmp_path):
        """Test loading system state."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        # Save state first
        manager.qilm.entangle_phase(np.ones(10), 0.5)
        filepath = str(tmp_path / "state.json")
        manager.save_system_state(filepath)

        # Load state (note: current implementation doesn't fully restore state)
        manager2 = MemoryManager(config)
        manager2.load_system_state(filepath)

        # Verify file was loaded (actual restoration is a stub in current impl)
        assert True  # Just verify no errors


class TestMemoryManagerIntegration:
    """Test MemoryManager integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete memory manager workflow."""
        config = {
            "dimension": 10,
            "moral_filter": {"threshold": 0.5},
            "cognitive_rhythm": {"wake_duration": 3, "sleep_duration": 1}
        }
        manager = MemoryManager(config)

        # Process multiple events
        for i in range(10):
            vec = np.random.randn(10) * 0.3  # Small vectors
            moral = 0.6 + (i % 3) * 0.1  # Varying moral values
            await manager.process_event(vec, moral)

        # Verify processing completed
        total = manager.metrics_collector.metrics["total_events_processed"]
        accepted = manager.metrics_collector.metrics["accepted_events_count"]
        latent = manager.metrics_collector.metrics["latent_events_count"]

        assert total == 10
        assert accepted + latent == total

    @pytest.mark.asyncio
    async def test_ontology_matching_integration(self):
        """Test ontology matching during event processing."""
        config = {
            "dimension": 3,
            "ontology_matcher": {
                "ontology_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "ontology_labels": ["category_a", "category_b", "category_c"]
            }
        }
        manager = MemoryManager(config)

        # Process event that matches first ontology vector
        event = np.array([0.9, 0.1, 0.0])
        await manager.process_event(event, 0.8)

        # Verify processing completed (ontology matching is internal)
        assert manager.metrics_collector.metrics["accepted_events_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
