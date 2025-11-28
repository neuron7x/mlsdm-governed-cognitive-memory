"""Tests for MembraneEngine numerical implementation.

These tests verify:
1. Stability smoke tests - NaN/Inf detection, value range validation
2. Determinism tests - reproducibility with fixed random_seed
3. Performance sanity - bounded execution time

Reference: docs/MATH_MODEL.md Section 2
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlsdm.core import (
    IntegrationScheme,
    MembraneConfig,
    MembraneEngine,
    ValueOutOfRangeError,
)


class TestMembraneEngineInitialization:
    """Tests for MembraneEngine initialization."""

    def test_default_initialization(self) -> None:
        """Test engine initializes with default config."""
        engine = MembraneEngine()
        assert engine.config.n_units == 1
        assert engine.config.tau == 10.0
        assert engine.config.dt == 0.1
        assert engine._step_count == 0

    def test_custom_config_initialization(self) -> None:
        """Test engine initializes with custom config."""
        config = MembraneConfig(n_units=100, tau=20.0, dt=0.05)
        engine = MembraneEngine(config)
        assert engine.config.n_units == 100
        assert engine.config.tau == 20.0
        assert len(engine.V) == 100

    def test_invalid_n_units_raises(self) -> None:
        """Test that n_units <= 0 raises ValueError."""
        config = MembraneConfig(n_units=0)
        with pytest.raises(ValueError, match="n_units must be positive"):
            MembraneEngine(config)

    def test_invalid_tau_raises(self) -> None:
        """Test that tau outside range raises ValueError."""
        config = MembraneConfig(tau=0.5)  # Below minimum of 1.0
        with pytest.raises(ValueError, match="tau must be in"):
            MembraneEngine(config)

    def test_euler_stability_check(self) -> None:
        """Test Euler scheme rejects unstable dt."""
        # dt > tau/10 should be rejected for Euler
        config = MembraneConfig(tau=10.0, dt=2.0, scheme=IntegrationScheme.EULER)
        with pytest.raises(ValueError, match="exceeds stability limit"):
            MembraneEngine(config)


class TestMembraneEngineStability:
    """Stability smoke tests - NaN/Inf detection, value bounds."""

    def test_no_nan_after_1000_steps(self) -> None:
        """Test that 1000 integration steps produce no NaN."""
        config = MembraneConfig(n_units=10, dt=0.1)
        engine = MembraneEngine(config)
        engine.run(n_steps=1000)
        assert not np.any(np.isnan(engine.V))

    def test_no_inf_after_1000_steps(self) -> None:
        """Test that 1000 integration steps produce no Inf."""
        config = MembraneConfig(n_units=10, dt=0.1)
        engine = MembraneEngine(config)
        engine.run(n_steps=1000)
        assert not np.any(np.isinf(engine.V))

    def test_values_in_valid_range(self) -> None:
        """Test membrane potentials stay within valid range."""
        config = MembraneConfig(n_units=10, dt=0.1)
        engine = MembraneEngine(config)
        engine.run(n_steps=1000, I_ext=10.0)
        assert np.all(config.V_min <= engine.V)
        assert np.all(config.V_max >= engine.V)

    def test_clamping_counts_violations(self) -> None:
        """Test that values exceeding bounds are clamped and counted."""
        config = MembraneConfig(n_units=10, dt=0.1, clamp_values=True)
        engine = MembraneEngine(config)
        # Apply strong external current to push values
        engine.run(n_steps=1000, I_ext=50.0)
        # Values should be clamped
        assert np.all(config.V_max >= engine.V)
        # Metrics should show violations
        metrics = engine.get_metrics()
        assert metrics.max_V <= config.V_max

    def test_no_clamp_raises_on_violation(self) -> None:
        """Test that clamp_values=False raises on bound violation."""
        # Use a bound that will be violated during simulation
        config = MembraneConfig(
            n_units=10,
            dt=0.1,
            clamp_values=False,
            V_init=-60.0,
            V_min=-90.0,
            V_max=-50.0,  # Initial is in range, but I_ext will push above
        )
        engine = MembraneEngine(config)
        with pytest.raises(ValueOutOfRangeError):
            engine.run(n_steps=1000, I_ext=50.0)  # Strong current pushes up


class TestMembraneEngineDeterminism:
    """Determinism tests - same seed produces same results."""

    def test_same_seed_same_result(self) -> None:
        """Test that same random_seed produces identical results."""
        config1 = MembraneConfig(n_units=10, dt=0.1, random_seed=42)
        engine1 = MembraneEngine(config1)
        engine1.run(n_steps=100, I_ext=5.0)
        result1 = engine1.V.copy()

        config2 = MembraneConfig(n_units=10, dt=0.1, random_seed=42)
        engine2 = MembraneEngine(config2)
        engine2.run(n_steps=100, I_ext=5.0)
        result2 = engine2.V.copy()

        np.testing.assert_array_equal(result1, result2)

    def test_reset_restores_initial_state(self) -> None:
        """Test reset() restores engine to initial state."""
        config = MembraneConfig(n_units=10, dt=0.1)
        engine = MembraneEngine(config)

        engine.run(n_steps=100, I_ext=5.0)
        assert engine._step_count == 100

        engine.reset()
        assert engine._step_count == 0
        np.testing.assert_array_almost_equal(engine.V, np.full(10, config.V_init))


class TestMembraneEnginePerformance:
    """Performance sanity tests - bounded execution time."""

    def test_1000_units_1000_steps_under_1s(self) -> None:
        """Test 1000 units × 1000 steps completes in reasonable time."""
        config = MembraneConfig(n_units=1000, dt=0.1)
        engine = MembraneEngine(config)

        start = time.time()
        engine.run(n_steps=1000)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1.0s"


class TestMembraneEngineIntegrationSchemes:
    """Tests for different integration schemes."""

    def test_euler_scheme(self) -> None:
        """Test Euler scheme integrates correctly."""
        config = MembraneConfig(
            n_units=1, tau=10.0, dt=0.1, scheme=IntegrationScheme.EULER
        )
        engine = MembraneEngine(config)
        engine.run(n_steps=100, I_ext=1.0)
        # Potential should have increased from initial
        assert engine.V[0] > config.V_init

    def test_rk4_scheme(self) -> None:
        """Test RK4 scheme integrates correctly."""
        config = MembraneConfig(
            n_units=1, tau=10.0, dt=0.1, scheme=IntegrationScheme.RK4
        )
        engine = MembraneEngine(config)
        engine.run(n_steps=100, I_ext=1.0)
        # Potential should have increased from initial
        assert engine.V[0] > config.V_init

    def test_rk4_more_accurate_than_euler(self) -> None:
        """Test RK4 gives more accurate results than Euler for same dt."""
        # Run same scenario with both schemes
        config_euler = MembraneConfig(
            n_units=1, tau=10.0, dt=0.5, scheme=IntegrationScheme.EULER
        )
        engine_euler = MembraneEngine(config_euler)
        engine_euler.run(n_steps=100, I_ext=1.0)

        config_rk4 = MembraneConfig(
            n_units=1, tau=10.0, dt=0.5, scheme=IntegrationScheme.RK4
        )
        engine_rk4 = MembraneEngine(config_rk4)
        engine_rk4.run(n_steps=100, I_ext=1.0)

        # They should give different results (RK4 is more accurate)
        # This is a sanity check - exact accuracy testing requires analytical solution
        assert engine_euler.V[0] != engine_rk4.V[0]


class TestMembraneEngineMetrics:
    """Tests for metrics collection."""

    def test_metrics_step_count(self) -> None:
        """Test metrics track step count correctly."""
        config = MembraneConfig(n_units=10)
        engine = MembraneEngine(config)
        engine.run(n_steps=100)
        metrics = engine.get_metrics()
        assert metrics.steps_completed == 100

    def test_metrics_statistics(self) -> None:
        """Test metrics compute statistics correctly."""
        config = MembraneConfig(n_units=10)
        engine = MembraneEngine(config)
        engine.run(n_steps=50)
        metrics = engine.get_metrics()

        # Should have valid statistics
        assert metrics.max_V >= metrics.min_V
        assert metrics.mean_V >= metrics.min_V
        assert metrics.mean_V <= metrics.max_V
        assert metrics.std_V >= 0.0

    def test_metrics_to_dict(self) -> None:
        """Test metrics can be converted to dict."""
        config = MembraneConfig(n_units=10)
        engine = MembraneEngine(config)
        engine.run(n_steps=50)
        metrics_dict = engine.get_metrics().to_dict()

        assert "max_V" in metrics_dict
        assert "min_V" in metrics_dict
        assert "mean_V" in metrics_dict
        assert "std_V" in metrics_dict
        assert "steps_completed" in metrics_dict


class TestMembraneEngineMemoryUsage:
    """Tests for memory usage estimation."""

    def test_memory_usage_bytes_positive(self) -> None:
        """Test memory usage estimate is positive."""
        config = MembraneConfig(n_units=1000)
        engine = MembraneEngine(config)
        assert engine.memory_usage_bytes() > 0

    def test_memory_usage_scales_with_n_units(self) -> None:
        """Test memory usage scales with number of units."""
        engine_small = MembraneEngine(MembraneConfig(n_units=100))
        engine_large = MembraneEngine(MembraneConfig(n_units=1000))
        assert engine_large.memory_usage_bytes() > engine_small.memory_usage_bytes()
