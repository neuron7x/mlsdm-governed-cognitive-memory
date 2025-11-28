"""Tests for ReactionDiffusionEngine numerical implementation.

These tests verify:
1. Stability smoke tests - NaN/Inf detection, value range validation, CFL condition
2. Determinism tests - reproducibility with fixed random_seed
3. Performance sanity - bounded execution time

Reference: docs/MATH_MODEL.md Section 3
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlsdm.core import (
    BoundaryCondition,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
)


class TestReactionDiffusionInitialization:
    """Tests for ReactionDiffusionEngine initialization."""

    def test_default_initialization(self) -> None:
        """Test engine initializes with default config."""
        engine = ReactionDiffusionEngine()
        assert engine.config.grid_size == 64
        assert engine.config.D == 0.1
        assert engine.config.dt == 0.1
        assert engine._step_count == 0

    def test_custom_config_initialization(self) -> None:
        """Test engine initializes with custom config."""
        config = ReactionDiffusionConfig(grid_size=32, D=0.2, dt=0.05)
        engine = ReactionDiffusionEngine(config)
        assert engine.config.grid_size == 32
        assert engine.u.shape == (32, 32)

    def test_invalid_grid_size_raises(self) -> None:
        """Test that grid_size outside range raises ValueError."""
        config = ReactionDiffusionConfig(grid_size=4)  # Too small
        with pytest.raises(ValueError, match="grid_size must be in"):
            ReactionDiffusionEngine(config)

    def test_cfl_violation_raises(self) -> None:
        """Test that CFL condition violation raises ValueError."""
        # CFL: dt <= h^2 / (4*D) = 1 / (4*0.5) = 0.5
        config = ReactionDiffusionConfig(D=0.5, h=1.0, dt=1.0)  # Violates CFL
        with pytest.raises(ValueError, match="exceeds CFL stability limit"):
            ReactionDiffusionEngine(config)


class TestReactionDiffusionStability:
    """Stability smoke tests - NaN/Inf detection, value bounds."""

    def test_no_nan_after_500_steps(self) -> None:
        """Test that 500 integration steps produce no NaN."""
        config = ReactionDiffusionConfig(grid_size=32, dt=0.1)
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=500)
        assert not np.any(np.isnan(engine.u))

    def test_no_inf_after_500_steps(self) -> None:
        """Test that 500 integration steps produce no Inf."""
        config = ReactionDiffusionConfig(grid_size=32, dt=0.1)
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=500)
        assert not np.any(np.isinf(engine.u))

    def test_values_in_valid_range(self) -> None:
        """Test concentrations stay within valid range."""
        config = ReactionDiffusionConfig(grid_size=32, dt=0.1)
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=500)
        assert np.all(engine.u >= config.u_min)
        assert np.all(engine.u <= config.u_max)

    def test_cfl_limit_property(self) -> None:
        """Test CFL limit is correctly computed."""
        config = ReactionDiffusionConfig(D=0.1, h=1.0)
        # CFL limit should be h^2 / (4*D) = 1 / 0.4 = 2.5
        assert abs(config.cfl_limit - 2.5) < 1e-10


class TestReactionDiffusionBoundaryConditions:
    """Tests for different boundary conditions."""

    def test_periodic_boundary(self) -> None:
        """Test periodic boundary maintains field structure."""
        config = ReactionDiffusionConfig(
            grid_size=32, boundary=BoundaryCondition.PERIODIC, dt=0.1
        )
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=100)
        # Field should remain bounded
        assert np.all(engine.u >= 0)
        assert np.all(engine.u <= 1)

    def test_neumann_boundary(self) -> None:
        """Test Neumann boundary (zero-flux) condition."""
        config = ReactionDiffusionConfig(
            grid_size=32, boundary=BoundaryCondition.NEUMANN, dt=0.1
        )
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=100)
        # Field should remain bounded
        assert np.all(engine.u >= 0)
        assert np.all(engine.u <= 1)

    def test_dirichlet_boundary(self) -> None:
        """Test Dirichlet boundary (fixed values) condition."""
        config = ReactionDiffusionConfig(
            grid_size=32,
            boundary=BoundaryCondition.DIRICHLET,
            dirichlet_value=0.3,
            dt=0.1,
        )
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=100)
        # Boundary should be at dirichlet_value
        u = engine.u
        np.testing.assert_almost_equal(u[0, :], 0.3)
        np.testing.assert_almost_equal(u[-1, :], 0.3)
        np.testing.assert_almost_equal(u[:, 0], 0.3)
        np.testing.assert_almost_equal(u[:, -1], 0.3)


class TestReactionDiffusionDeterminism:
    """Determinism tests - same seed produces same results."""

    def test_same_seed_same_result(self) -> None:
        """Test that same random_seed produces identical results."""
        config1 = ReactionDiffusionConfig(grid_size=32, random_seed=42)
        engine1 = ReactionDiffusionEngine(config1)
        engine1.run(n_steps=100)
        result1 = engine1.u.copy()

        config2 = ReactionDiffusionConfig(grid_size=32, random_seed=42)
        engine2 = ReactionDiffusionEngine(config2)
        engine2.run(n_steps=100)
        result2 = engine2.u.copy()

        np.testing.assert_array_equal(result1, result2)

    def test_different_seed_different_result(self) -> None:
        """Test that different seeds produce different results."""
        config1 = ReactionDiffusionConfig(grid_size=32, random_seed=42)
        engine1 = ReactionDiffusionEngine(config1)
        engine1.run(n_steps=100)

        config2 = ReactionDiffusionConfig(grid_size=32, random_seed=123)
        engine2 = ReactionDiffusionEngine(config2)
        engine2.run(n_steps=100)

        # Results should differ due to different initial noise
        assert not np.array_equal(engine1.u, engine2.u)

    def test_reset_restores_initial_state(self) -> None:
        """Test reset() restores engine to initial state."""
        config = ReactionDiffusionConfig(grid_size=32, random_seed=42)
        engine = ReactionDiffusionEngine(config)
        initial_u = engine.u.copy()

        engine.run(n_steps=100)
        assert engine._step_count == 100

        engine.reset()
        assert engine._step_count == 0
        np.testing.assert_array_equal(engine.u, initial_u)


class TestReactionDiffusionPerformance:
    """Performance sanity tests - bounded execution time."""

    def test_64x64_500_steps_under_1s(self) -> None:
        """Test 64×64 grid × 500 steps completes in reasonable time."""
        config = ReactionDiffusionConfig(grid_size=64, dt=0.1)
        engine = ReactionDiffusionEngine(config)

        start = time.time()
        engine.run(n_steps=500)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Took {elapsed:.2f}s, expected < 1.0s"


class TestReactionDiffusionMetrics:
    """Tests for metrics collection."""

    def test_metrics_step_count(self) -> None:
        """Test metrics track step count correctly."""
        config = ReactionDiffusionConfig(grid_size=32)
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=100)
        metrics = engine.get_metrics()
        assert metrics.steps_completed == 100

    def test_metrics_statistics(self) -> None:
        """Test metrics compute statistics correctly."""
        config = ReactionDiffusionConfig(grid_size=32)
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=50)
        metrics = engine.get_metrics()

        # Should have valid statistics
        assert metrics.max_u >= metrics.min_u
        assert metrics.mean_u >= metrics.min_u
        assert metrics.mean_u <= metrics.max_u
        assert metrics.std_u >= 0.0
        assert metrics.total_mass > 0.0  # Grid initialized with positive values

    def test_metrics_to_dict(self) -> None:
        """Test metrics can be converted to dict."""
        config = ReactionDiffusionConfig(grid_size=32)
        engine = ReactionDiffusionEngine(config)
        engine.run(n_steps=50)
        metrics_dict = engine.get_metrics().to_dict()

        assert "max_u" in metrics_dict
        assert "min_u" in metrics_dict
        assert "mean_u" in metrics_dict
        assert "std_u" in metrics_dict
        assert "total_mass" in metrics_dict


class TestReactionDiffusionMemoryUsage:
    """Tests for memory usage estimation."""

    def test_memory_usage_bytes_positive(self) -> None:
        """Test memory usage estimate is positive."""
        config = ReactionDiffusionConfig(grid_size=64)
        engine = ReactionDiffusionEngine(config)
        assert engine.memory_usage_bytes() > 0

    def test_memory_usage_scales_with_grid_size(self) -> None:
        """Test memory usage scales with grid size."""
        engine_small = ReactionDiffusionEngine(ReactionDiffusionConfig(grid_size=32))
        engine_large = ReactionDiffusionEngine(ReactionDiffusionConfig(grid_size=64))
        # 64^2 / 32^2 = 4x memory
        assert engine_large.memory_usage_bytes() > engine_small.memory_usage_bytes()


class TestReactionDiffusionPhysics:
    """Tests for physical behavior correctness."""

    def test_diffusion_spreads_concentration(self) -> None:
        """Test that diffusion spreads initial concentration peaks."""
        config = ReactionDiffusionConfig(
            grid_size=32, u_init=0.0, u_init_noise=0.0, dt=0.1, alpha=0.0, beta=0.0
        )
        engine = ReactionDiffusionEngine(config)

        # Create a central peak
        engine._u[15:17, 15:17] = 1.0
        initial_variance = np.var(engine.u)

        # Run diffusion-only simulation
        engine.run(n_steps=200)
        final_variance = np.var(engine.u)

        # Variance should decrease (spreading)
        assert final_variance < initial_variance

    def test_reaction_drives_growth(self) -> None:
        """Test that reaction term drives logistic growth."""
        config = ReactionDiffusionConfig(
            grid_size=32, u_init=0.3, u_init_noise=0.0, dt=0.05, D=0.001, alpha=0.5, beta=0.0
        )
        engine = ReactionDiffusionEngine(config)
        initial_mean = np.mean(engine.u)

        # Run with reaction
        engine.run(n_steps=100)
        final_mean = np.mean(engine.u)

        # Mean should increase toward carrying capacity (1.0)
        assert final_mean > initial_mean
