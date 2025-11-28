"""Reaction-Diffusion Engine for MyceliumFractalNet.

This module implements numerical integration for reaction-diffusion PDEs
as described in docs/MATH_MODEL.md Section 3.

The morphogen concentration field u(x, y, t) evolves according to:
    ∂u/∂t = D * ∇²u + R(u)

Where:
    - D is the diffusion coefficient
    - ∇² is the Laplacian (5-point stencil discretization)
    - R(u) = α * u * (1 - u) - β * u is the logistic reaction term

Numerical Scheme:
    - Forward Time Centered Space (FTCS) explicit scheme
    - CFL stability condition: dt ≤ h² / (4 * D)

Boundary Conditions:
    - Periodic (default): Toroidal topology
    - Neumann: Zero-flux at boundaries
    - Dirichlet: Fixed values at boundaries

Example:
    >>> from mlsdm.core.reaction_diffusion_engine import (
    ...     ReactionDiffusionEngine,
    ...     ReactionDiffusionConfig,
    ... )
    >>> config = ReactionDiffusionConfig(grid_size=64, D=0.1)
    >>> engine = ReactionDiffusionEngine(config)
    >>> engine.step()  # Advance one time step
    >>> metrics = engine.get_metrics()

Reference:
    See docs/MATH_MODEL.md Section 3 for complete model specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mlsdm.core.numerical_exceptions import StabilityError, ValueOutOfRangeError


class BoundaryCondition(Enum):
    """Boundary conditions for reaction-diffusion field.

    PERIODIC: Toroidal topology (u wraps around edges).
    NEUMANN: Zero-flux at boundaries (∂u/∂n = 0).
    DIRICHLET: Fixed values at boundaries.
    """

    PERIODIC = "periodic"
    NEUMANN = "neumann"
    DIRICHLET = "dirichlet"


@dataclass
class ReactionDiffusionConfig:
    """Configuration for ReactionDiffusionEngine.

    Attributes:
        grid_size: Grid dimension N (creates N×N grid). Default: 64.
        D: Diffusion coefficient. Range: [0.001, 10.0]. Default: 0.1.
        alpha: Logistic growth rate. Range: [0.0, 1.0]. Default: 0.1.
        beta: Decay rate. Range: [0.0, 1.0]. Default: 0.01.
        h: Grid spacing. Range: [0.1, 10.0]. Default: 1.0.
        dt: Time step. Must satisfy CFL condition. Default: 0.1.
        boundary: Boundary condition type. Default: PERIODIC.
        u_init: Initial field value. Range: [0.0, 1.0]. Default: 0.5.
        u_init_noise: Noise amplitude for initialization. Default: 0.1.
        u_min: Minimum allowed concentration. Default: 0.0.
        u_max: Maximum allowed concentration. Default: 1.0.
        dirichlet_value: Boundary value for Dirichlet BC. Default: 0.5.
        random_seed: RNG seed for reproducibility. Default: 42.
        clamp_values: Whether to clamp values to valid range. Default: True.

    Reference:
        See docs/MATH_MODEL.md Section 3 for parameter definitions and CFL condition.
    """

    grid_size: int = 64
    D: float = 0.1
    alpha: float = 0.1
    beta: float = 0.01
    h: float = 1.0
    dt: float = 0.1
    boundary: BoundaryCondition = BoundaryCondition.PERIODIC
    u_init: float = 0.5
    u_init_noise: float = 0.1
    u_min: float = 0.0
    u_max: float = 1.0
    dirichlet_value: float = 0.5
    random_seed: int = 42
    clamp_values: bool = True

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid or CFL condition violated.
        """
        if not (8 <= self.grid_size <= 512):
            raise ValueError(f"grid_size must be in [8, 512], got {self.grid_size}")
        if not (0.001 <= self.D <= 10.0):
            raise ValueError(f"D must be in [0.001, 10.0], got {self.D}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0.0, 1.0], got {self.alpha}")
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"beta must be in [0.0, 1.0], got {self.beta}")
        if not (0.1 <= self.h <= 10.0):
            raise ValueError(f"h must be in [0.1, 10.0], got {self.h}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not (0.0 <= self.u_init <= 1.0):
            raise ValueError(f"u_init must be in [0.0, 1.0], got {self.u_init}")
        if self.u_min >= self.u_max:
            raise ValueError(f"u_min ({self.u_min}) must be < u_max ({self.u_max})")

        # Check CFL stability condition
        max_stable_dt = self.h**2 / (4.0 * self.D)
        if self.dt > max_stable_dt:
            raise ValueError(
                f"dt={self.dt:.4f} exceeds CFL stability limit {max_stable_dt:.4f} "
                f"for D={self.D}, h={self.h}. "
                f"Use dt ≤ h²/(4D) = {max_stable_dt:.4f}"
            )

    @property
    def cfl_limit(self) -> float:
        """Maximum stable dt for current D and h values."""
        return self.h**2 / (4.0 * self.D)


@dataclass
class ReactionDiffusionMetrics:
    """Metrics collected during reaction-diffusion simulation.

    Attributes:
        max_u: Maximum concentration observed.
        min_u: Minimum concentration observed.
        mean_u: Mean concentration.
        std_u: Standard deviation of concentration.
        total_mass: Total integrated concentration (sum of field).
        steps_completed: Total integration steps completed.
        stability_violations: Number of times values were clamped/corrected.
    """

    max_u: float = float("-inf")
    min_u: float = float("inf")
    mean_u: float = 0.0
    std_u: float = 0.0
    total_mass: float = 0.0
    steps_completed: int = 0
    stability_violations: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for logging/serialization."""
        return {
            "max_u": self.max_u,
            "min_u": self.min_u,
            "mean_u": self.mean_u,
            "std_u": self.std_u,
            "total_mass": self.total_mass,
            "steps_completed": self.steps_completed,
            "stability_violations": self.stability_violations,
        }


class ReactionDiffusionEngine:
    """Engine for integrating reaction-diffusion PDEs.

    Implements the FTCS scheme for reaction-diffusion on a 2D grid.
    Provides stability guarantees through CFL condition enforcement
    and NaN/Inf detection.

    Example:
        >>> config = ReactionDiffusionConfig(grid_size=32, D=0.1)
        >>> engine = ReactionDiffusionEngine(config)
        >>> for _ in range(100):
        ...     engine.step()
        >>> print(engine.get_metrics())

    Reference:
        See docs/MATH_MODEL.md Section 3 for model equations.
    """

    ENGINE_NAME = "ReactionDiffusionEngine"

    def __init__(self, config: ReactionDiffusionConfig | None = None) -> None:
        """Initialize ReactionDiffusionEngine.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or ReactionDiffusionConfig()
        self.config.validate()

        # Initialize state
        self._rng = np.random.default_rng(self.config.random_seed)
        self._u = self._initialize_field()
        self._step_count = 0
        self._stability_violations = 0

        # Pre-compute constants for efficiency
        self._diffusion_factor = self.config.D * self.config.dt / (self.config.h**2)
        self._reaction_dt = self.config.dt

    def _initialize_field(self) -> NDArray[np.float64]:
        """Initialize concentration field with optional noise.

        Returns:
            2D array of initial concentrations.
        """
        n = self.config.grid_size
        base = self.config.u_init * np.ones((n, n), dtype=np.float64)

        if self.config.u_init_noise > 0:
            noise = self._rng.uniform(
                -self.config.u_init_noise,
                self.config.u_init_noise,
                size=(n, n),
            )
            base += noise
            # Clamp to valid range
            np.clip(base, self.config.u_min, self.config.u_max, out=base)

        return base

    def reset(self, u_init: float | None = None) -> None:
        """Reset engine state to initial conditions.

        Args:
            u_init: Initial concentration. Uses config default if None.
        """
        # Reset RNG first so _initialize_field uses fresh seed
        self._rng = np.random.default_rng(self.config.random_seed)

        if u_init is not None:
            old_init = self.config.u_init
            self.config.u_init = u_init
            self._u = self._initialize_field()
            self.config.u_init = old_init
        else:
            self._u = self._initialize_field()

        self._step_count = 0
        self._stability_violations = 0

    @property
    def u(self) -> np.ndarray:
        """Current concentration field (read-only copy)."""
        copied: np.ndarray = self._u.copy()
        return copied

    @property
    def state(self) -> dict[str, Any]:
        """Current engine state as dictionary."""
        return {
            "u": self._u.copy(),
            "step_count": self._step_count,
            "config": {
                "grid_size": self.config.grid_size,
                "D": self.config.D,
                "dt": self.config.dt,
                "boundary": self.config.boundary.value,
            },
        }

    def _apply_boundary_conditions(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply boundary conditions to the field.

        Args:
            u: Concentration field with boundary region.

        Returns:
            Field with boundary conditions applied.
        """
        if self.config.boundary == BoundaryCondition.PERIODIC:
            # No explicit action needed - handled in Laplacian computation
            pass
        elif self.config.boundary == BoundaryCondition.NEUMANN:
            # Zero-flux: copy adjacent interior values to boundary
            u[0, :] = u[1, :]
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
        elif self.config.boundary == BoundaryCondition.DIRICHLET:
            # Fixed boundary values
            u[0, :] = self.config.dirichlet_value
            u[-1, :] = self.config.dirichlet_value
            u[:, 0] = self.config.dirichlet_value
            u[:, -1] = self.config.dirichlet_value

        return u

    def _compute_laplacian(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute discrete Laplacian using 5-point stencil.

        ∇²u[i,j] ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h²

        Args:
            u: Concentration field.

        Returns:
            Laplacian field (same shape as u).
        """
        if self.config.boundary == BoundaryCondition.PERIODIC:
            # Use numpy roll for periodic boundary
            laplacian = (
                np.roll(u, 1, axis=0)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + np.roll(u, -1, axis=1)
                - 4.0 * u
            )
        else:
            # Interior points using slicing
            laplacian = np.zeros_like(u)
            laplacian[1:-1, 1:-1] = (
                u[2:, 1:-1]
                + u[:-2, 1:-1]
                + u[1:-1, 2:]
                + u[1:-1, :-2]
                - 4.0 * u[1:-1, 1:-1]
            )
            # Boundary Laplacian is zero for Neumann/Dirichlet (flux handled separately)

        return laplacian / (self.config.h**2)

    def _reaction_term(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute reaction term R(u) = α * u * (1 - u) - β * u.

        Args:
            u: Concentration field.

        Returns:
            Reaction term field (same shape as u).
        """
        return self.config.alpha * u * (1.0 - u) - self.config.beta * u

    def _validate_state(self) -> None:
        """Validate state after integration step.

        Raises:
            StabilityError: If NaN or Inf detected.
            ValueOutOfRangeError: If values exceed bounds (when clamp_values=False).
        """
        # Check for NaN/Inf
        has_nan = bool(np.any(np.isnan(self._u)))
        has_inf = bool(np.any(np.isinf(self._u)))

        if has_nan or has_inf:
            raise StabilityError(
                "NaN or Inf detected in concentration field",
                step=self._step_count,
                engine=self.ENGINE_NAME,
                state_snapshot={
                    "max_abs_u": float(np.max(np.abs(self._u[np.isfinite(self._u)])))
                    if np.any(np.isfinite(self._u))
                    else float("nan"),
                },
                has_nan=has_nan,
                has_inf=has_inf,
            )

        # Check bounds
        u_min = float(np.min(self._u))
        u_max = float(np.max(self._u))

        if u_min < self.config.u_min or u_max > self.config.u_max:
            if self.config.clamp_values:
                # Clamp to valid range
                np.clip(self._u, self.config.u_min, self.config.u_max, out=self._u)
                self._stability_violations += 1
            else:
                # Raise error
                if u_min < self.config.u_min:
                    raise ValueOutOfRangeError(
                        variable_name="u",
                        value=u_min,
                        min_value=self.config.u_min,
                        max_value=self.config.u_max,
                        step=self._step_count,
                        engine=self.ENGINE_NAME,
                    )
                else:
                    raise ValueOutOfRangeError(
                        variable_name="u",
                        value=u_max,
                        min_value=self.config.u_min,
                        max_value=self.config.u_max,
                        step=self._step_count,
                        engine=self.ENGINE_NAME,
                    )

    def step(self) -> None:
        """Advance simulation by one time step using FTCS scheme.

        u(t+dt) = u(t) + dt * (D * ∇²u + R(u))

        Raises:
            StabilityError: If numerical instability detected.
            ValueOutOfRangeError: If values exceed bounds (when clamp_values=False).
        """
        # Compute Laplacian (diffusion)
        laplacian = self._compute_laplacian(self._u)

        # Compute reaction term
        reaction = self._reaction_term(self._u)

        # FTCS update
        self._u += self.config.dt * (self.config.D * laplacian + reaction)

        # Apply boundary conditions
        self._u = self._apply_boundary_conditions(self._u)

        self._step_count += 1

        # Validate state
        self._validate_state()

    def run(self, n_steps: int) -> ReactionDiffusionMetrics:
        """Run simulation for multiple steps.

        Args:
            n_steps: Number of integration steps.

        Returns:
            Metrics collected during simulation.

        Raises:
            StabilityError: If numerical instability detected.
        """
        for _ in range(n_steps):
            self.step()

        return self.get_metrics()

    def get_metrics(self) -> ReactionDiffusionMetrics:
        """Get current simulation metrics.

        Returns:
            Metrics with statistics about the current state.
        """
        return ReactionDiffusionMetrics(
            max_u=float(np.max(self._u)),
            min_u=float(np.min(self._u)),
            mean_u=float(np.mean(self._u)),
            std_u=float(np.std(self._u)),
            total_mass=float(np.sum(self._u)),
            steps_completed=self._step_count,
            stability_violations=self._stability_violations,
        )

    def memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Conservative memory estimate including arrays and overhead.
        """
        # Field array
        array_bytes = self._u.nbytes

        # Temporary arrays during computation (Laplacian, reaction)
        temp_arrays = 2 * array_bytes

        # Metadata overhead
        metadata_overhead = 512

        # 15% overhead for Python structures
        return int((array_bytes + temp_arrays + metadata_overhead) * 1.15)
