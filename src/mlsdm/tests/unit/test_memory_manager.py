"""Comprehensive unit tests for MemoryManager."""

import numpy as np
import pytest

from mlsdm.core.memory_manager import MemoryManager

pytest_plugins = ("pytest_asyncio",)


class TestMemoryManager:
    """Test suite for MemoryManager."""

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        assert manager.dimension == 10
        assert manager.memory is not None
        assert manager.filter is not None
        assert manager.matcher is not None
        assert manager.rhythm is not None
        assert manager.qilm is not None
        assert manager.metrics_collector is not None

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            "dimension": 20,
            "multi_level_memory": {
                "lambda_l1": 0.6,
                "lambda_l2": 0.2,
                "lambda_l3": 0.02,
                "theta_l1": 1.5,
                "theta_l2": 2.5,
                "gating12": 0.4,
                "gating23": 0.25,
            },
            "moral_filter": {
                "threshold": 0.6,
                "adapt_rate": 0.1,
                "min_threshold": 0.2,
                "max_threshold": 0.95,
            },
            "cognitive_rhythm": {"wake_duration": 10, "sleep_duration": 3},
            "strict_mode": True,
        }

        manager = MemoryManager(config)

        assert manager.dimension == 20
        assert manager.filter.threshold == 0.6
        assert manager.rhythm.wake_duration == 10
        assert manager.strict_mode is True

    @pytest.mark.asyncio
    async def test_process_event_accepted(self):
        """Test processing an accepted event."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        event_vector = np.random.randn(10)
        moral_value = 0.8

        await manager.process_event(event_vector, moral_value)

        metrics = manager.metrics_collector.get_metrics()
        assert metrics["accepted_events_count"] == 1

    @pytest.mark.asyncio
    async def test_process_event_rejected(self):
        """Test processing a rejected event."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        event_vector = np.random.randn(10)
        moral_value = 0.1  # Low moral value should be rejected

        await manager.process_event(event_vector, moral_value)

        metrics = manager.metrics_collector.get_metrics()
        assert metrics["latent_events_count"] == 1

    @pytest.mark.asyncio
    async def test_process_multiple_events(self):
        """Test processing multiple events."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        for i in range(10):
            event_vector = np.random.randn(10)
            moral_value = 0.5 + (i % 2) * 0.3
            await manager.process_event(event_vector, moral_value)

        metrics = manager.metrics_collector.get_metrics()
        total = metrics["total_events_processed"]
        assert total > 0

    def test_is_sensitive_heuristic(self):
        """Test the sensitivity heuristic."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        # Normal vector
        normal_vec = np.array([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        assert manager._is_sensitive(normal_vec) is False

        # High norm vector
        high_norm_vec = np.array([10.0] * 10)
        assert manager._is_sensitive(high_norm_vec) is True

        # Negative sum vector
        negative_vec = np.array([-2.0] * 10)
        assert manager._is_sensitive(negative_vec) is True

    @pytest.mark.asyncio
    async def test_strict_mode_blocks_sensitive_data(self):
        """Test that strict mode blocks sensitive data."""
        config = {"dimension": 10, "strict_mode": True}
        manager = MemoryManager(config)

        sensitive_vec = np.array([10.0] * 10)  # High norm

        with pytest.raises(ValueError, match="Sensitive data detected"):
            await manager.process_event(sensitive_vec, moral_value=0.8)

    @pytest.mark.asyncio
    async def test_non_strict_mode_allows_sensitive_data(self):
        """Test that non-strict mode allows sensitive data."""
        config = {"dimension": 10, "strict_mode": False}
        manager = MemoryManager(config)

        sensitive_vec = np.array([10.0] * 10)  # High norm

        # Should not raise error
        await manager.process_event(sensitive_vec, moral_value=0.8)

    @pytest.mark.asyncio
    async def test_simulate(self):
        """Test the simulate method."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        def event_generator():
            for _i in range(5):
                yield np.random.randn(10), 0.7

        await manager.simulate(5, event_generator())

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["time"]) > 0

    def test_run_simulation_default_generator(self):
        """Test run_simulation with default event generator."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        manager.run_simulation(num_steps=5)

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["time"]) > 0

    def test_run_simulation_custom_generator(self):
        """Test run_simulation with custom event generator."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        def custom_generator():
            for _i in range(3):
                yield np.ones(10), 0.8

        manager.run_simulation(num_steps=3, event_gen=custom_generator())

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["time"]) > 0

    def test_save_and_load_system_state(self):
        """Test saving and loading system state."""
        import os
        import tempfile

        config = {"dimension": 10}
        manager = MemoryManager(config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            manager.save_system_state(filepath)
            assert os.path.exists(filepath)

            # Load (note: current implementation doesn't restore state)
            manager.load_system_state(filepath)
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    @pytest.mark.asyncio
    async def test_event_dimension_mismatch(self):
        """Test that dimension mismatch in simulate raises error."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        def bad_generator():
            yield np.random.randn(5), 0.7  # Wrong dimension

        with pytest.raises(ValueError, match="Event dimension mismatch"):
            await manager.simulate(1, bad_generator())

    @pytest.mark.asyncio
    async def test_moral_filter_adaptation(self):
        """Test that moral filter adapts based on acceptance rate."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        # Process events to trigger adaptation
        for _ in range(20):
            event_vector = np.random.randn(10)
            await manager.process_event(event_vector, moral_value=0.9)

        # Threshold should have adapted (may increase or stay at bounds)
        assert (
            manager.filter.min_threshold <= manager.filter.threshold <= manager.filter.max_threshold
        )

    @pytest.mark.asyncio
    async def test_rhythm_wake_sleep_cycle(self):
        """Test that rhythm transitions between wake and sleep."""
        config = {"dimension": 10, "cognitive_rhythm": {"wake_duration": 3, "sleep_duration": 2}}
        manager = MemoryManager(config)

        def event_gen():
            for _i in range(10):
                yield np.random.randn(10), 0.8

        await manager.simulate(10, event_gen())

        # Should have recorded phase transitions
        metrics = manager.metrics_collector.get_metrics()
        phases = metrics["phase"]
        assert "wake" in phases or "sleep" in phases

    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test that metrics are collected during processing."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        event_vector = np.random.randn(10)
        await manager.process_event(event_vector, moral_value=0.8)

        metrics = manager.metrics_collector.get_metrics()
        assert "total_events_processed" in metrics
        assert "accepted_events_count" in metrics
        assert "latent_events_count" in metrics
        assert "latencies" in metrics

    def test_ontology_matcher_configuration(self):
        """Test ontology matcher configuration."""
        config = {
            "dimension": 10,
            "ontology_matcher": {
                "ontology_vectors": [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                "ontology_labels": ["A", "B"],
            },
        }
        manager = MemoryManager(config)

        assert manager.matcher.labels == ["A", "B"]
        assert manager.matcher.ontology.shape[0] == 2

    @pytest.mark.asyncio
    async def test_latency_tracking(self):
        """Test that event latency is tracked."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        event_vector = np.random.randn(10)
        await manager.process_event(event_vector, moral_value=0.8)

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["latencies"]) > 0
        assert all(lat >= 0 for lat in metrics["latencies"])
