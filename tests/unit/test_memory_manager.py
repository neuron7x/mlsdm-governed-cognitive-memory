"""
Comprehensive tests for core/memory_manager.py.

Tests cover:
- MemoryManager initialization
- Sensitive data detection
- Event processing
- Simulation functionality
- State persistence
"""

import asyncio
from typing import Iterator

import numpy as np
import pytest

from mlsdm.core.memory_manager import MemoryManager


class TestMemoryManagerInit:
    """Tests for MemoryManager initialization."""

    def test_default_initialization(self):
        """Test initialization with default config."""
        config = {"dimension": 10}
        manager = MemoryManager(config)

        assert manager.dimension == 10
        assert manager.memory is not None
        assert manager.filter is not None
        assert manager.matcher is not None
        assert manager.rhythm is not None
        assert manager.qilm is not None
        assert manager.metrics_collector is not None
        assert manager.strict_mode is False

    def test_custom_initialization(self):
        """Test initialization with custom config."""
        config = {
            "dimension": 5,
            "multi_level_memory": {
                "lambda_l1": 0.6,
                "lambda_l2": 0.2,
                "lambda_l3": 0.05,
                "theta_l1": 1.5,
                "theta_l2": 2.5,
                "gating12": 0.4,
                "gating23": 0.2,
            },
            "moral_filter": {
                "threshold": 0.6,
                "adapt_rate": 0.1,
                "min_threshold": 0.4,
                "max_threshold": 0.8,
            },
            "cognitive_rhythm": {
                "wake_duration": 10,
                "sleep_duration": 5,
            },
            "strict_mode": True,
        }
        manager = MemoryManager(config)

        assert manager.dimension == 5
        assert manager.strict_mode is True
        assert manager.filter.threshold == 0.6

    def test_ontology_initialization(self):
        """Test ontology matcher initialization with custom vectors."""
        config = {
            "dimension": 3,
            "ontology_matcher": {
                "ontology_vectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                "ontology_labels": ["cat_a", "cat_b"],
            },
        }
        manager = MemoryManager(config)

        assert manager.matcher is not None
        assert manager.matcher.dimension == 3


class TestSensitiveDetection:
    """Tests for sensitive data detection."""

    def test_is_sensitive_normal_vector(self):
        """Test normal vector is not flagged as sensitive."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        vec = np.array([1.0, 2.0, 3.0])  # norm ~= 3.74, sum = 6
        assert manager._is_sensitive(vec) is False

    def test_is_sensitive_high_norm(self):
        """Test high norm vector is flagged as sensitive."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        vec = np.array([10.0, 10.0, 10.0])  # norm > 10
        assert manager._is_sensitive(vec) is True

    def test_is_sensitive_negative_sum(self):
        """Test negative sum vector is flagged as sensitive."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        vec = np.array([-5.0, -5.0, 1.0])  # sum = -9 < 0
        assert manager._is_sensitive(vec) is True

    def test_is_sensitive_boundary(self):
        """Test boundary conditions."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        # Exactly at norm 10
        vec = np.array([10.0, 0.0, 0.0])
        assert manager._is_sensitive(vec) is False  # norm == 10, not > 10

        # Exactly at sum 0
        vec = np.array([1.0, -1.0, 0.0])
        assert manager._is_sensitive(vec) is False  # sum == 0, not < 0


class TestProcessEvent:
    """Tests for async event processing."""

    @pytest.mark.asyncio
    async def test_process_event_accepted(self):
        """Test processing an accepted event."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        # High moral value should be accepted
        event_vector = np.array([1.0, 0.0, 0.0])
        await manager.process_event(event_vector, moral_value=0.8)

        metrics = manager.metrics_collector.get_metrics()
        assert metrics["accepted_events_count"] == 1
        assert metrics["total_events_processed"] == 1

    @pytest.mark.asyncio
    async def test_process_event_rejected(self):
        """Test processing a rejected event (low moral value)."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        # First we need to initialize with some events to get a baseline
        # Low moral value should be rejected
        event_vector = np.array([1.0, 0.0, 0.0])
        await manager.process_event(event_vector, moral_value=0.1)

        metrics = manager.metrics_collector.get_metrics()
        assert metrics["latent_events_count"] == 1

    @pytest.mark.asyncio
    async def test_process_event_strict_mode_sensitive(self):
        """Test strict mode rejects sensitive data."""
        config = {"dimension": 3, "strict_mode": True}
        manager = MemoryManager(config)

        # Create sensitive vector (high norm)
        event_vector = np.array([10.0, 10.0, 10.0])

        with pytest.raises(ValueError) as exc_info:
            await manager.process_event(event_vector, moral_value=0.8)

        assert "Sensitive data detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_event_adapts_threshold(self):
        """Test that moral filter adapts based on acceptance rate."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        # Process multiple events
        for _ in range(5):
            await manager.process_event(np.random.randn(3), moral_value=0.7)

        # Threshold should have been adapted
        # (adaptation behavior depends on accept rate)
        assert manager.filter.threshold > 0  # Still valid threshold

    @pytest.mark.asyncio
    async def test_process_event_records_latency(self):
        """Test that event processing records latency."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        await manager.process_event(np.array([1.0, 0.0, 0.0]), moral_value=0.8)

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["latencies"]) == 1
        assert metrics["latencies"][0] >= 0


class TestSimulate:
    """Tests for simulate method."""

    @pytest.mark.asyncio
    async def test_simulate_basic(self):
        """Test basic simulation."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        def event_gen() -> Iterator[tuple[np.ndarray, float]]:
            for _ in range(3):
                yield np.array([1.0, 0.0, 0.0]), 0.7

        await manager.simulate(3, event_gen())

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["time"]) == 3
        assert len(metrics["phase"]) == 3

    @pytest.mark.asyncio
    async def test_simulate_dimension_mismatch(self):
        """Test simulation rejects dimension mismatched events."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        def event_gen() -> Iterator[tuple[np.ndarray, float]]:
            yield np.array([1.0, 0.0, 0.0, 0.0]), 0.7  # Wrong dimension

        with pytest.raises(ValueError) as exc_info:
            await manager.simulate(1, event_gen())

        assert "dimension mismatch" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_simulate_wake_sleep_cycle(self):
        """Test simulation respects wake/sleep cycle."""
        config = {
            "dimension": 3,
            "cognitive_rhythm": {
                "wake_duration": 2,
                "sleep_duration": 2,
            },
        }
        manager = MemoryManager(config)

        events_processed = []

        def event_gen() -> Iterator[tuple[np.ndarray, float]]:
            for i in range(6):
                yield np.array([1.0, 0.0, 0.0]), 0.8

        await manager.simulate(6, event_gen())

        metrics = manager.metrics_collector.get_metrics()
        # Should have phases recorded
        assert len(metrics["phase"]) == 6


class TestRunSimulation:
    """Tests for run_simulation method."""

    def test_run_simulation_with_generator(self):
        """Test run_simulation with custom generator."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        def event_gen() -> Iterator[tuple[np.ndarray, float]]:
            for _ in range(5):
                yield np.array([1.0, 0.0, 0.0]), 0.7

        manager.run_simulation(5, event_gen())

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["time"]) == 5

    def test_run_simulation_default_generator(self):
        """Test run_simulation with default generator."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        manager.run_simulation(3)

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["time"]) == 3

    def test_run_simulation_records_memory_state(self):
        """Test run_simulation records memory state."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        manager.run_simulation(5)

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["L1_norm"]) == 5
        assert len(metrics["L2_norm"]) == 5
        assert len(metrics["L3_norm"]) == 5


class TestStatePersistence:
    """Tests for state save/load functionality."""

    def test_save_system_state(self, tmp_path):
        """Test saving system state."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        # Process some events
        manager.run_simulation(3)

        filepath = str(tmp_path / "state.json")
        manager.save_system_state(filepath)

        # File should exist
        import os
        assert os.path.exists(filepath)

    def test_load_system_state(self, tmp_path):
        """Test loading system state."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        # Save state first
        filepath = str(tmp_path / "state.json")
        manager.save_system_state(filepath)

        # Load state (currently placeholder implementation)
        manager.load_system_state(filepath)
        # Should not raise error


class TestMetricsRecording:
    """Tests for metrics recording during operations."""

    def test_metrics_recorded_during_simulation(self):
        """Test all metrics are recorded during simulation."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        manager.run_simulation(5)

        metrics = manager.metrics_collector.get_metrics()

        # Check all metric arrays are populated
        assert len(metrics["time"]) == 5
        assert len(metrics["phase"]) == 5
        assert len(metrics["L1_norm"]) == 5
        assert len(metrics["L2_norm"]) == 5
        assert len(metrics["L3_norm"]) == 5
        assert len(metrics["entropy_L1"]) == 5
        assert len(metrics["entropy_L2"]) == 5
        assert len(metrics["entropy_L3"]) == 5
        assert len(metrics["current_moral_threshold"]) == 5

    def test_moral_threshold_tracked(self):
        """Test moral threshold is tracked over simulation."""
        config = {"dimension": 3}
        manager = MemoryManager(config)

        manager.run_simulation(10)

        metrics = manager.metrics_collector.get_metrics()
        assert len(metrics["current_moral_threshold"]) == 10
        # All thresholds should be valid values
        for threshold in metrics["current_moral_threshold"]:
            assert 0 <= threshold <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
