"""Fractal Growth Engine for MyceliumFractalNet.

This module implements fractal structure generation using Diffusion-Limited
Aggregation (DLA) as described in docs/MATH_MODEL.md Section 4.

Algorithm:
    1. Initialize seed particle at center of grid
    2. Launch random walker from boundary
    3. Random walk until walker touches aggregate → stick
    4. Repeat for n_particles iterations

Stochastic Control:
    - All random operations use numpy.random.Generator with configurable seed
    - Deterministic reproduction guaranteed for same seed

Example:
    >>> from mlsdm.core.fractal_growth_engine import FractalGrowthEngine, FractalGrowthConfig
    >>> config = FractalGrowthConfig(grid_size=128, n_particles=1000)
    >>> engine = FractalGrowthEngine(config)
    >>> engine.run()  # Generate fractal structure
    >>> metrics = engine.get_metrics()
    >>> print(f"Fractal dimension estimate: {metrics.fractal_dimension:.3f}")

Reference:
    See docs/MATH_MODEL.md Section 4 for complete model specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mlsdm.core.numerical_exceptions import StabilityError


class GrowthMethod(Enum):
    """Fractal growth methods.

    DLA: Diffusion-Limited Aggregation.
    L_SYSTEM: Lindenmayer system (future implementation).
    """

    DLA = "dla"
    L_SYSTEM = "l_system"


@dataclass
class FractalGrowthConfig:
    """Configuration for FractalGrowthEngine.

    Attributes:
        grid_size: Grid dimension N (creates N×N grid). Range: [32, 1024]. Default: 128.
        n_particles: Number of particles to add. Range: [100, 100000]. Default: 5000.
        p_stick: Sticking probability when particle touches aggregate. Default: 1.0.
        max_walk_steps: Maximum steps per random walker. Default: 100000.
        launch_radius_factor: Factor for launch radius relative to grid. Default: 0.45.
        kill_radius_factor: Factor for kill radius (walker death). Default: 0.49.
        random_seed: RNG seed for reproducibility. Default: 42.
        method: Growth method (DLA or L_SYSTEM). Default: DLA.
        early_stop_occupancy: Stop if grid occupancy exceeds this. Default: 0.5.

    Reference:
        See docs/MATH_MODEL.md Section 4 for parameter definitions.
    """

    grid_size: int = 128
    n_particles: int = 5000
    p_stick: float = 1.0
    max_walk_steps: int = 100000
    launch_radius_factor: float = 0.45
    kill_radius_factor: float = 0.49
    random_seed: int = 42
    method: GrowthMethod = GrowthMethod.DLA
    early_stop_occupancy: float = 0.5

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not (32 <= self.grid_size <= 1024):
            raise ValueError(f"grid_size must be in [32, 1024], got {self.grid_size}")
        if not (100 <= self.n_particles <= 100000):
            raise ValueError(f"n_particles must be in [100, 100000], got {self.n_particles}")
        if not (0.1 <= self.p_stick <= 1.0):
            raise ValueError(f"p_stick must be in [0.1, 1.0], got {self.p_stick}")
        if self.max_walk_steps <= 0:
            raise ValueError(f"max_walk_steps must be positive, got {self.max_walk_steps}")
        if not (0.1 <= self.launch_radius_factor <= 0.49):
            raise ValueError(
                f"launch_radius_factor must be in [0.1, 0.49], got {self.launch_radius_factor}"
            )
        if not (0.1 <= self.kill_radius_factor <= 0.5):
            raise ValueError(
                f"kill_radius_factor must be in [0.1, 0.5], got {self.kill_radius_factor}"
            )
        if self.launch_radius_factor >= self.kill_radius_factor:
            raise ValueError(
                f"launch_radius_factor ({self.launch_radius_factor}) must be "
                f"< kill_radius_factor ({self.kill_radius_factor})"
            )
        if not (0.01 <= self.early_stop_occupancy <= 1.0):
            raise ValueError(
                f"early_stop_occupancy must be in [0.01, 1.0], got {self.early_stop_occupancy}"
            )


@dataclass
class FractalGrowthMetrics:
    """Metrics collected during fractal growth.

    Attributes:
        particles_added: Number of particles successfully added.
        particles_attempted: Number of particle launches attempted.
        walk_steps_total: Total random walk steps taken.
        walk_steps_mean: Mean walk steps per successful particle.
        grid_occupancy: Fraction of grid cells occupied.
        fractal_dimension: Estimated fractal dimension (box-counting).
        radius_of_gyration: Radius of gyration of aggregate.
        max_radius: Maximum radius reached by aggregate.
        growth_completed: Whether growth completed normally.
    """

    particles_added: int = 0
    particles_attempted: int = 0
    walk_steps_total: int = 0
    walk_steps_mean: float = 0.0
    grid_occupancy: float = 0.0
    fractal_dimension: float = 0.0
    radius_of_gyration: float = 0.0
    max_radius: float = 0.0
    growth_completed: bool = False

    def to_dict(self) -> dict[str, float | int | bool]:
        """Convert to dictionary for logging/serialization."""
        return {
            "particles_added": self.particles_added,
            "particles_attempted": self.particles_attempted,
            "walk_steps_total": self.walk_steps_total,
            "walk_steps_mean": self.walk_steps_mean,
            "grid_occupancy": self.grid_occupancy,
            "fractal_dimension": self.fractal_dimension,
            "radius_of_gyration": self.radius_of_gyration,
            "max_radius": self.max_radius,
            "growth_completed": self.growth_completed,
        }


class FractalGrowthEngine:
    """Engine for generating fractal structures via DLA.

    Implements Diffusion-Limited Aggregation with configurable parameters.
    Provides deterministic reproduction through controlled random seeding.

    Example:
        >>> config = FractalGrowthConfig(grid_size=64, n_particles=500)
        >>> engine = FractalGrowthEngine(config)
        >>> metrics = engine.run()
        >>> print(f"Added {metrics.particles_added} particles")

    Reference:
        See docs/MATH_MODEL.md Section 4 for model equations.
    """

    ENGINE_NAME = "FractalGrowthEngine"

    # 4-connected neighbors (up, down, left, right)
    NEIGHBORS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)

    def __init__(self, config: FractalGrowthConfig | None = None) -> None:
        """Initialize FractalGrowthEngine.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or FractalGrowthConfig()
        self.config.validate()

        # Initialize state
        self._rng = np.random.default_rng(self.config.random_seed)
        self._grid: NDArray[np.int8] = np.zeros(
            (self.config.grid_size, self.config.grid_size), dtype=np.int8
        )

        # Grid center
        self._center = self.config.grid_size // 2

        # Computed radii
        self._launch_radius = int(self.config.grid_size * self.config.launch_radius_factor)
        self._kill_radius = int(self.config.grid_size * self.config.kill_radius_factor)

        # Tracking
        self._particles_added = 0
        self._particles_attempted = 0
        self._total_walk_steps = 0
        self._walk_steps_per_particle: list[int] = []
        self._max_radius_reached = 0.0
        self._growth_completed = False

        # Initialize with seed particle at center
        self._grid[self._center, self._center] = 1
        self._particles_added = 1

    def reset(self) -> None:
        """Reset engine state to initial conditions."""
        self._rng = np.random.default_rng(self.config.random_seed)
        self._grid.fill(0)
        self._grid[self._center, self._center] = 1

        self._particles_added = 1
        self._particles_attempted = 0
        self._total_walk_steps = 0
        self._walk_steps_per_particle = []
        self._max_radius_reached = 0.0
        self._growth_completed = False

    @property
    def grid(self) -> np.ndarray:
        """Current grid state (read-only copy)."""
        copied: np.ndarray = self._grid.copy()
        return copied

    @property
    def state(self) -> dict[str, Any]:
        """Current engine state as dictionary."""
        return {
            "grid": self._grid.copy(),
            "particles_added": self._particles_added,
            "config": {
                "grid_size": self.config.grid_size,
                "n_particles": self.config.n_particles,
                "p_stick": self.config.p_stick,
                "random_seed": self.config.random_seed,
            },
        }

    def _launch_particle(self) -> tuple[int, int]:
        """Launch a random walker from the boundary circle.

        Returns:
            Initial (row, col) position of the walker.
        """
        # Random angle on circle
        theta = self._rng.uniform(0, 2 * np.pi)
        row = int(self._center + self._launch_radius * np.sin(theta))
        col = int(self._center + self._launch_radius * np.cos(theta))

        # Clamp to grid bounds
        row = max(0, min(row, self.config.grid_size - 1))
        col = max(0, min(col, self.config.grid_size - 1))

        return row, col

    def _is_adjacent_to_aggregate(self, row: int, col: int) -> bool:
        """Check if position is adjacent to existing aggregate.

        Args:
            row: Row position.
            col: Column position.

        Returns:
            True if adjacent to aggregate.
        """
        for dr, dc in self.NEIGHBORS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.config.grid_size and 0 <= nc < self.config.grid_size:
                if self._grid[nr, nc] == 1:
                    return True
        return False

    def _distance_from_center(self, row: int, col: int) -> float:
        """Calculate distance from grid center.

        Args:
            row: Row position.
            col: Column position.

        Returns:
            Euclidean distance from center.
        """
        return float(np.sqrt((row - self._center) ** 2 + (col - self._center) ** 2))

    def _random_walk_step(self, row: int, col: int) -> tuple[int, int]:
        """Take one random walk step.

        Args:
            row: Current row.
            col: Current column.

        Returns:
            New (row, col) position.
        """
        direction = self._rng.integers(0, 4)
        dr, dc = self.NEIGHBORS[direction]
        new_row = row + dr
        new_col = col + dc

        # Boundary reflection
        if new_row < 0:
            new_row = 0
        elif new_row >= self.config.grid_size:
            new_row = self.config.grid_size - 1

        if new_col < 0:
            new_col = 0
        elif new_col >= self.config.grid_size:
            new_col = self.config.grid_size - 1

        return new_row, new_col

    def _add_particle(self) -> bool:
        """Attempt to add one particle via random walk.

        Returns:
            True if particle was successfully added, False otherwise.
        """
        self._particles_attempted += 1

        row, col = self._launch_particle()
        walk_steps = 0

        for _ in range(self.config.max_walk_steps):
            walk_steps += 1
            self._total_walk_steps += 1

            # Check if on existing aggregate (shouldn't happen, but safety check)
            if self._grid[row, col] == 1:
                # Move away
                row, col = self._random_walk_step(row, col)
                continue

            # Check if adjacent to aggregate
            if self._is_adjacent_to_aggregate(row, col):
                # Sticking decision
                if self._rng.random() < self.config.p_stick:
                    self._grid[row, col] = 1
                    self._particles_added += 1
                    self._walk_steps_per_particle.append(walk_steps)

                    # Update max radius
                    r = self._distance_from_center(row, col)
                    self._max_radius_reached = max(self._max_radius_reached, r)

                    return True

            # Take random walk step
            row, col = self._random_walk_step(row, col)

            # Check if walker escaped (kill radius)
            if self._distance_from_center(row, col) > self._kill_radius:
                return False

        # Max steps exceeded
        return False

    def _check_early_stop(self) -> bool:
        """Check if early stopping condition is met.

        Returns:
            True if growth should stop early.
        """
        occupancy = float(np.sum(self._grid)) / (self.config.grid_size**2)
        return occupancy >= self.config.early_stop_occupancy

    def step(self) -> bool:
        """Attempt to add one particle.

        Returns:
            True if particle was added, False if failed.

        Raises:
            StabilityError: If too many consecutive failures.
        """
        return self._add_particle()

    def run(self) -> FractalGrowthMetrics:
        """Run complete DLA growth simulation.

        Returns:
            Metrics after growth completion.

        Raises:
            StabilityError: If growth fails to progress.
        """
        # Target number of particles (excluding seed)
        target = self.config.n_particles - 1
        consecutive_failures = 0
        max_consecutive_failures = 100

        for _ in range(target):
            if self._check_early_stop():
                break

            success = self._add_particle()

            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1

                if consecutive_failures >= max_consecutive_failures:
                    raise StabilityError(
                        f"Growth stalled: {max_consecutive_failures} consecutive failures",
                        step=self._particles_added,
                        engine=self.ENGINE_NAME,
                        state_snapshot={
                            "particles_added": self._particles_added,
                            "max_radius": self._max_radius_reached,
                            "launch_radius": self._launch_radius,
                        },
                    )

        self._growth_completed = True
        return self.get_metrics()

    def _estimate_fractal_dimension(self) -> float:
        """Estimate fractal dimension using box-counting method.

        Returns:
            Estimated fractal dimension (typically ~1.7 for DLA in 2D).
        """
        if self._particles_added < 10:
            return 0.0

        # Box-counting at different scales
        sizes = [2, 4, 8, 16, 32]
        counts = []

        for size in sizes:
            if size >= self.config.grid_size:
                break

            count = 0
            for i in range(0, self.config.grid_size, size):
                for j in range(0, self.config.grid_size, size):
                    box = self._grid[i : i + size, j : j + size]
                    if np.any(box > 0):
                        count += 1

            if count > 0:
                counts.append(count)

        if len(counts) < 2:
            return 0.0

        # Linear regression on log-log plot
        valid_sizes = sizes[: len(counts)]
        log_sizes = np.log(valid_sizes)
        log_counts = np.log(counts)

        # D = -slope of log(N) vs log(epsilon)
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return float(-slope)

    def _compute_radius_of_gyration(self) -> float:
        """Compute radius of gyration of the aggregate.

        Returns:
            Radius of gyration.
        """
        positions = np.argwhere(self._grid > 0)
        if len(positions) < 2:
            return 0.0

        center_of_mass = np.mean(positions, axis=0)
        distances_sq = np.sum((positions - center_of_mass) ** 2, axis=1)
        return float(np.sqrt(np.mean(distances_sq)))

    def get_metrics(self) -> FractalGrowthMetrics:
        """Get current growth metrics.

        Returns:
            Metrics with statistics about the growth process.
        """
        n_walk = len(self._walk_steps_per_particle)
        mean_steps = (
            sum(self._walk_steps_per_particle) / n_walk if n_walk > 0 else 0.0
        )

        occupancy = float(np.sum(self._grid)) / (self.config.grid_size**2)

        return FractalGrowthMetrics(
            particles_added=self._particles_added,
            particles_attempted=self._particles_attempted,
            walk_steps_total=self._total_walk_steps,
            walk_steps_mean=mean_steps,
            grid_occupancy=occupancy,
            fractal_dimension=self._estimate_fractal_dimension(),
            radius_of_gyration=self._compute_radius_of_gyration(),
            max_radius=self._max_radius_reached,
            growth_completed=self._growth_completed,
        )

    def memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Conservative memory estimate including grid and overhead.
        """
        # Grid array
        grid_bytes = self._grid.nbytes

        # Walk steps list (approximate)
        list_bytes = len(self._walk_steps_per_particle) * 8

        # Metadata overhead
        metadata_overhead = 512

        # 15% overhead for Python structures
        return int((grid_bytes + list_bytes + metadata_overhead) * 1.15)
