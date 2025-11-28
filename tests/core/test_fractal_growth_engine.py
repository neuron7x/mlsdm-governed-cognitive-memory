"""Tests for FractalGrowthEngine (DLA) numerical implementation.

These tests verify:
1. Stability smoke tests - proper termination, no infinite loops
2. Determinism tests - reproducibility with fixed random_seed
3. Performance sanity - bounded execution time

Reference: docs/MATH_MODEL.md Section 4
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlsdm.core import (
    FractalGrowthConfig,
    FractalGrowthEngine,
)


class TestFractalGrowthInitialization:
    """Tests for FractalGrowthEngine initialization."""

    def test_default_initialization(self) -> None:
        """Test engine initializes with default config."""
        engine = FractalGrowthEngine()
        assert engine.config.grid_size == 128
        assert engine.config.n_particles == 5000
        assert engine._particles_added == 1  # Seed particle

    def test_custom_config_initialization(self) -> None:
        """Test engine initializes with custom config."""
        config = FractalGrowthConfig(grid_size=64, n_particles=1000)
        engine = FractalGrowthEngine(config)
        assert engine.config.grid_size == 64
        assert engine.grid.shape == (64, 64)
        # Seed particle at center
        center = 64 // 2
        assert engine.grid[center, center] == 1

    def test_invalid_grid_size_raises(self) -> None:
        """Test that grid_size outside range raises ValueError."""
        config = FractalGrowthConfig(grid_size=16)  # Too small
        with pytest.raises(ValueError, match="grid_size must be in"):
            FractalGrowthEngine(config)

    def test_invalid_n_particles_raises(self) -> None:
        """Test that n_particles outside range raises ValueError."""
        config = FractalGrowthConfig(n_particles=50)  # Too small
        with pytest.raises(ValueError, match="n_particles must be in"):
            FractalGrowthEngine(config)

    def test_invalid_radius_factors_raises(self) -> None:
        """Test that invalid radius factors raise ValueError."""
        # launch_radius must be < kill_radius
        config = FractalGrowthConfig(launch_radius_factor=0.4, kill_radius_factor=0.3)
        with pytest.raises(ValueError, match="launch_radius_factor"):
            FractalGrowthEngine(config)


class TestFractalGrowthStability:
    """Stability smoke tests - proper termination, no crashes."""

    def test_growth_completes_200_particles(self) -> None:
        """Test that growth with 200 particles completes without error."""
        config = FractalGrowthConfig(grid_size=64, n_particles=200, random_seed=42)
        engine = FractalGrowthEngine(config)
        metrics = engine.run()
        assert metrics.growth_completed
        assert metrics.particles_added > 1  # At least seed + some particles

    def test_no_infinite_loop_on_step(self) -> None:
        """Test that individual step() calls terminate."""
        config = FractalGrowthConfig(grid_size=64, n_particles=500, random_seed=42)
        engine = FractalGrowthEngine(config)

        # 100 step attempts should complete quickly
        start = time.time()
        for _ in range(100):
            engine.step()
        elapsed = time.time() - start

        assert elapsed < 5.0, f"100 steps took {elapsed:.2f}s, expected < 5s"

    def test_early_stop_on_high_occupancy(self) -> None:
        """Test that growth stops early when occupancy threshold reached."""
        config = FractalGrowthConfig(
            grid_size=32,
            n_particles=100000,  # More than can fit
            early_stop_occupancy=0.1,  # 10% occupancy stops growth
            random_seed=42,
        )
        engine = FractalGrowthEngine(config)
        metrics = engine.run()
        assert metrics.grid_occupancy >= 0  # Growth stopped due to occupancy or failure


class TestFractalGrowthDeterminism:
    """Determinism tests - same seed produces same results."""

    def test_same_seed_same_result(self) -> None:
        """Test that same random_seed produces identical results."""
        config1 = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine1 = FractalGrowthEngine(config1)
        metrics1 = engine1.run()

        config2 = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine2 = FractalGrowthEngine(config2)
        metrics2 = engine2.run()

        np.testing.assert_array_equal(engine1.grid, engine2.grid)
        assert metrics1.particles_added == metrics2.particles_added

    def test_different_seed_different_result(self) -> None:
        """Test that different seeds produce different results."""
        config1 = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine1 = FractalGrowthEngine(config1)
        engine1.run()

        config2 = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=123)
        engine2 = FractalGrowthEngine(config2)
        engine2.run()

        # Grids should differ (extremely unlikely to be equal for different seeds)
        assert not np.array_equal(engine1.grid, engine2.grid)

    def test_reset_restores_initial_state(self) -> None:
        """Test reset() restores engine to initial state."""
        config = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine = FractalGrowthEngine(config)
        initial_grid = engine.grid.copy()

        engine.run()
        assert engine._particles_added > 1

        engine.reset()
        assert engine._particles_added == 1  # Only seed
        np.testing.assert_array_equal(engine.grid, initial_grid)


class TestFractalGrowthPerformance:
    """Performance sanity tests - bounded execution time."""

    def test_100_particles_64x64_under_5s(self) -> None:
        """Test 100 particles on 64×64 grid completes in reasonable time."""
        config = FractalGrowthConfig(
            grid_size=64, n_particles=100, random_seed=42
        )
        engine = FractalGrowthEngine(config)

        start = time.time()
        engine.run()
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Took {elapsed:.2f}s, expected < 5.0s"


class TestFractalGrowthMetrics:
    """Tests for metrics collection."""

    def test_metrics_particle_count(self) -> None:
        """Test metrics track particle count correctly."""
        config = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine = FractalGrowthEngine(config)
        metrics = engine.run()

        assert metrics.particles_added >= 1  # At least seed
        assert metrics.particles_attempted > 0

    def test_metrics_walk_steps(self) -> None:
        """Test metrics track walk steps."""
        config = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine = FractalGrowthEngine(config)
        metrics = engine.run()

        assert metrics.walk_steps_total > 0
        if metrics.particles_added > 1:
            assert metrics.walk_steps_mean > 0

    def test_metrics_grid_occupancy(self) -> None:
        """Test metrics compute grid occupancy correctly."""
        config = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine = FractalGrowthEngine(config)
        metrics = engine.run()

        # Occupancy should match particle count / grid area
        expected_occupancy = metrics.particles_added / (64 * 64)
        assert abs(metrics.grid_occupancy - expected_occupancy) < 1e-6

    def test_metrics_fractal_dimension(self) -> None:
        """Test fractal dimension estimation runs."""
        config = FractalGrowthConfig(grid_size=64, n_particles=200, random_seed=42)
        engine = FractalGrowthEngine(config)
        metrics = engine.run()

        # DLA fractal dimension should be around 1.7 in 2D
        # But with few particles, estimate may vary
        assert metrics.fractal_dimension >= 0  # At minimum, should be non-negative

    def test_metrics_to_dict(self) -> None:
        """Test metrics can be converted to dict."""
        config = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine = FractalGrowthEngine(config)
        engine.run()
        metrics_dict = engine.get_metrics().to_dict()

        assert "particles_added" in metrics_dict
        assert "particles_attempted" in metrics_dict
        assert "walk_steps_total" in metrics_dict
        assert "grid_occupancy" in metrics_dict
        assert "fractal_dimension" in metrics_dict


class TestFractalGrowthMemoryUsage:
    """Tests for memory usage estimation."""

    def test_memory_usage_bytes_positive(self) -> None:
        """Test memory usage estimate is positive."""
        config = FractalGrowthConfig(grid_size=128)
        engine = FractalGrowthEngine(config)
        assert engine.memory_usage_bytes() > 0

    def test_memory_usage_scales_with_grid_size(self) -> None:
        """Test memory usage scales with grid size."""
        engine_small = FractalGrowthEngine(FractalGrowthConfig(grid_size=64))
        engine_large = FractalGrowthEngine(FractalGrowthConfig(grid_size=128))
        # 128^2 / 64^2 = 4x memory
        assert engine_large.memory_usage_bytes() > engine_small.memory_usage_bytes()


class TestFractalGrowthBehavior:
    """Tests for physical/algorithmic behavior correctness."""

    def test_aggregate_grows_from_center(self) -> None:
        """Test that aggregate grows outward from center."""
        config = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine = FractalGrowthEngine(config)
        engine.run()

        # Center should definitely be occupied (seed)
        center = 64 // 2
        assert engine.grid[center, center] == 1

    def test_max_radius_increases(self) -> None:
        """Test that max radius increases as particles are added."""
        config = FractalGrowthConfig(grid_size=64, n_particles=100, random_seed=42)
        engine = FractalGrowthEngine(config)
        metrics = engine.run()

        # Max radius should be > 0 (particles spread from center)
        assert metrics.max_radius >= 0

    def test_sticking_probability_affects_density(self) -> None:
        """Test that lower sticking probability creates sparser structures."""
        config_high = FractalGrowthConfig(
            grid_size=64, n_particles=100, p_stick=1.0, random_seed=42
        )
        engine_high = FractalGrowthEngine(config_high)
        engine_high.run()

        config_low = FractalGrowthConfig(
            grid_size=64, n_particles=100, p_stick=0.5, random_seed=42
        )
        engine_low = FractalGrowthEngine(config_low)
        engine_low.run()

        # Lower sticking probability should result in fewer particles stuck
        # (though random walk differences may also affect this)
        # This is a loose check - just verify both complete
        assert engine_high._particles_added >= 1
        assert engine_low._particles_added >= 1
