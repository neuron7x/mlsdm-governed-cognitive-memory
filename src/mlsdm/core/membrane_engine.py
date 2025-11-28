"""Membrane Potential Engine for MyceliumFractalNet.

This module implements numerical integration for membrane potential dynamics
based on the Hodgkin-Huxley-inspired model described in docs/MATH_MODEL.md Section 2.

The membrane potential V evolves according to:
    dV/dt = (1/τ) * (-g_L * (V - E_L) + I_ext + I_syn)

Numerical Scheme:
    - Default: 4th-order Runge-Kutta (RK4)
    - Alternative: Explicit Euler (for performance-critical scenarios)

Stability Guarantees:
    - NaN/Inf detection after each integration step
    - Value clamping to physiological range [-90, 40] mV
    - Configurable time step satisfying CFL condition

Example:
    >>> from mlsdm.core.membrane_engine import MembraneEngine, MembraneConfig
    >>> config = MembraneConfig(n_units=100, dt=0.1)
    >>> engine = MembraneEngine(config)
    >>> engine.step(I_ext=0.5)  # Advance one time step
    >>> metrics = engine.get_metrics()

Reference:
    See docs/MATH_MODEL.md Section 2 for complete model specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mlsdm.core.numerical_exceptions import StabilityError, ValueOutOfRangeError


class IntegrationScheme(Enum):
    """Integration schemes for membrane potential dynamics.

    EULER: First-order explicit Euler. Fast but requires smaller dt.
    RK4: Fourth-order Runge-Kutta. More accurate, larger stability region.
    """

    EULER = "euler"
    RK4 = "rk4"


@dataclass
class MembraneConfig:
    """Configuration for MembraneEngine.

    Attributes:
        n_units: Number of membrane units to simulate.
        tau: Membrane time constant in ms (default: 10.0).
        g_L: Leak conductance in mS/cm² (default: 0.1).
        E_L: Leak reversal potential in mV (default: -65.0).
        V_init: Initial membrane potential in mV (default: -65.0).
        dt: Integration time step in ms (default: 0.1).
        scheme: Integration scheme (default: RK4).
        V_min: Minimum allowed potential in mV (default: -90.0).
        V_max: Maximum allowed potential in mV (default: 40.0).
        I_ext_min: Minimum external current in µA/cm² (default: -100.0).
        I_ext_max: Maximum external current in µA/cm² (default: 100.0).
        random_seed: RNG seed for reproducibility (default: 42).
        clamp_values: Whether to clamp values to valid range (default: True).

    Reference:
        See docs/MATH_MODEL.md Section 2 for parameter definitions and ranges.
    """

    n_units: int = 1
    tau: float = 10.0
    g_L: float = 0.1
    E_L: float = -65.0
    V_init: float = -65.0
    dt: float = 0.1
    scheme: IntegrationScheme = IntegrationScheme.RK4
    V_min: float = -90.0
    V_max: float = 40.0
    I_ext_min: float = -100.0
    I_ext_max: float = 100.0
    random_seed: int = 42
    clamp_values: bool = True

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.n_units <= 0:
            raise ValueError(f"n_units must be positive, got {self.n_units}")
        if not (1.0 <= self.tau <= 100.0):
            raise ValueError(f"tau must be in [1.0, 100.0], got {self.tau}")
        if not (0.01 <= self.g_L <= 1.0):
            raise ValueError(f"g_L must be in [0.01, 1.0], got {self.g_L}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.V_min >= self.V_max:
            raise ValueError(f"V_min ({self.V_min}) must be < V_max ({self.V_max})")

        # Check stability condition for Euler scheme
        if self.scheme == IntegrationScheme.EULER:
            max_stable_dt = self.tau / 10.0
            if self.dt > max_stable_dt:
                raise ValueError(
                    f"dt={self.dt} exceeds stability limit {max_stable_dt:.3f} "
                    f"for Euler scheme with tau={self.tau}"
                )


@dataclass
class MembraneMetrics:
    """Metrics collected during membrane simulation.

    Attributes:
        max_V: Maximum membrane potential observed.
        min_V: Minimum membrane potential observed.
        mean_V: Mean membrane potential.
        std_V: Standard deviation of membrane potential.
        steps_completed: Total integration steps completed.
        stability_violations: Number of times values were clamped/corrected.
    """

    max_V: float = float("-inf")
    min_V: float = float("inf")
    mean_V: float = 0.0
    std_V: float = 0.0
    steps_completed: int = 0
    stability_violations: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for logging/serialization."""
        return {
            "max_V": self.max_V,
            "min_V": self.min_V,
            "mean_V": self.mean_V,
            "std_V": self.std_V,
            "steps_completed": self.steps_completed,
            "stability_violations": self.stability_violations,
        }


class MembraneEngine:
    """Engine for integrating membrane potential dynamics.

    Implements the membrane potential ODE using configurable numerical schemes.
    Provides stability guarantees through NaN/Inf detection and value clamping.

    Example:
        >>> config = MembraneConfig(n_units=10, dt=0.1)
        >>> engine = MembraneEngine(config)
        >>> for _ in range(100):
        ...     engine.step(I_ext=0.5)
        >>> print(engine.get_metrics())

    Reference:
        See docs/MATH_MODEL.md Section 2 for model equations.
    """

    ENGINE_NAME = "MembraneEngine"

    def __init__(self, config: MembraneConfig | None = None) -> None:
        """Initialize MembraneEngine.

        Args:
            config: Configuration parameters. Uses defaults if None.
        """
        self.config = config or MembraneConfig()
        self.config.validate()

        # Initialize state
        self._rng = np.random.default_rng(self.config.random_seed)
        self._V: NDArray[np.float64] = np.full(
            self.config.n_units, self.config.V_init, dtype=np.float64
        )
        self._step_count = 0
        self._stability_violations = 0

        # Pre-compute constants for efficiency
        self._inv_tau = 1.0 / self.config.tau

    def reset(self, V_init: float | None = None) -> None:
        """Reset engine state to initial conditions.

        Args:
            V_init: Initial potential. Uses config default if None.
        """
        v_init = V_init if V_init is not None else self.config.V_init
        self._V.fill(v_init)
        self._step_count = 0
        self._stability_violations = 0
        self._rng = np.random.default_rng(self.config.random_seed)

    @property
    def V(self) -> np.ndarray:
        """Current membrane potential array (read-only copy)."""
        copied: np.ndarray = self._V.copy()
        return copied

    @property
    def state(self) -> dict[str, Any]:
        """Current engine state as dictionary."""
        return {
            "V": self._V.copy(),
            "step_count": self._step_count,
            "config": {
                "n_units": self.config.n_units,
                "tau": self.config.tau,
                "dt": self.config.dt,
            },
        }

    def _dV_dt(
        self,
        V: NDArray[np.float64],
        I_ext: float | NDArray[np.float64],
        I_syn: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute membrane potential derivative.

        Implements: dV/dt = (1/τ) * (-g_L * (V - E_L) + I_ext + I_syn)

        Args:
            V: Current membrane potential(s).
            I_ext: External current(s).
            I_syn: Synaptic current(s).

        Returns:
            Rate of change of membrane potential.
        """
        leak_current = -self.config.g_L * (V - self.config.E_L)
        return self._inv_tau * (leak_current + I_ext + I_syn)

    def _euler_step(
        self,
        I_ext: float | NDArray[np.float64],
        I_syn: float | NDArray[np.float64],
    ) -> None:
        """Perform one explicit Euler integration step.

        V(t+dt) = V(t) + dt * dV/dt

        Args:
            I_ext: External current(s).
            I_syn: Synaptic current(s).
        """
        dV = self._dV_dt(self._V, I_ext, I_syn)
        self._V += self.config.dt * dV

    def _rk4_step(
        self,
        I_ext: float | NDArray[np.float64],
        I_syn: float | NDArray[np.float64],
    ) -> None:
        """Perform one RK4 integration step.

        Uses classic 4th-order Runge-Kutta scheme for improved accuracy.

        Args:
            I_ext: External current(s).
            I_syn: Synaptic current(s).
        """
        dt = self.config.dt
        V = self._V

        k1 = self._dV_dt(V, I_ext, I_syn)
        k2 = self._dV_dt(V + 0.5 * dt * k1, I_ext, I_syn)
        k3 = self._dV_dt(V + 0.5 * dt * k2, I_ext, I_syn)
        k4 = self._dV_dt(V + dt * k3, I_ext, I_syn)

        self._V += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _validate_state(self) -> None:
        """Validate state after integration step.

        Raises:
            StabilityError: If NaN or Inf detected.
            ValueOutOfRangeError: If values exceed bounds (when clamp_values=False).
        """
        # Check for NaN/Inf
        has_nan = bool(np.any(np.isnan(self._V)))
        has_inf = bool(np.any(np.isinf(self._V)))

        if has_nan or has_inf:
            raise StabilityError(
                "NaN or Inf detected in membrane potential",
                step=self._step_count,
                engine=self.ENGINE_NAME,
                state_snapshot={
                    "max_abs_V": float(np.max(np.abs(self._V[np.isfinite(self._V)])))
                    if np.any(np.isfinite(self._V))
                    else float("nan"),
                },
                has_nan=has_nan,
                has_inf=has_inf,
            )

        # Check bounds
        v_min = float(np.min(self._V))
        v_max = float(np.max(self._V))

        if v_min < self.config.V_min or v_max > self.config.V_max:
            if self.config.clamp_values:
                # Clamp to valid range
                np.clip(self._V, self.config.V_min, self.config.V_max, out=self._V)
                self._stability_violations += 1
            else:
                # Raise error
                if v_min < self.config.V_min:
                    raise ValueOutOfRangeError(
                        variable_name="V",
                        value=v_min,
                        min_value=self.config.V_min,
                        max_value=self.config.V_max,
                        step=self._step_count,
                        engine=self.ENGINE_NAME,
                    )
                else:
                    raise ValueOutOfRangeError(
                        variable_name="V",
                        value=v_max,
                        min_value=self.config.V_min,
                        max_value=self.config.V_max,
                        step=self._step_count,
                        engine=self.ENGINE_NAME,
                    )

    def step(
        self,
        I_ext: float | NDArray[np.float64] = 0.0,
        I_syn: float | NDArray[np.float64] = 0.0,
    ) -> None:
        """Advance simulation by one time step.

        Args:
            I_ext: External current(s). Scalar or array of shape (n_units,).
            I_syn: Synaptic current(s). Scalar or array of shape (n_units,).

        Raises:
            StabilityError: If numerical instability detected.
            ValueOutOfRangeError: If values exceed bounds (when clamp_values=False).
        """
        # Validate and clip external current
        if isinstance(I_ext, np.ndarray):
            I_ext = np.clip(I_ext, self.config.I_ext_min, self.config.I_ext_max)
        else:
            I_ext = max(self.config.I_ext_min, min(I_ext, self.config.I_ext_max))

        # Perform integration step
        if self.config.scheme == IntegrationScheme.EULER:
            self._euler_step(I_ext, I_syn)
        else:  # RK4
            self._rk4_step(I_ext, I_syn)

        self._step_count += 1

        # Validate state
        self._validate_state()

    def run(
        self,
        n_steps: int,
        I_ext: float | NDArray[np.float64] = 0.0,
        I_syn: float | NDArray[np.float64] = 0.0,
    ) -> MembraneMetrics:
        """Run simulation for multiple steps.

        Args:
            n_steps: Number of integration steps.
            I_ext: External current(s).
            I_syn: Synaptic current(s).

        Returns:
            Metrics collected during simulation.

        Raises:
            StabilityError: If numerical instability detected.
        """
        for _ in range(n_steps):
            self.step(I_ext, I_syn)

        return self.get_metrics()

    def get_metrics(self) -> MembraneMetrics:
        """Get current simulation metrics.

        Returns:
            Metrics with statistics about the current state.
        """
        return MembraneMetrics(
            max_V=float(np.max(self._V)),
            min_V=float(np.min(self._V)),
            mean_V=float(np.mean(self._V)),
            std_V=float(np.std(self._V)),
            steps_completed=self._step_count,
            stability_violations=self._stability_violations,
        )

    def memory_usage_bytes(self) -> int:
        """Estimate memory usage in bytes.

        Returns:
            Conservative memory estimate including arrays and overhead.
        """
        # State array
        array_bytes = self._V.nbytes

        # Metadata overhead
        metadata_overhead = 512

        # 15% overhead for Python structures
        return int((array_bytes + metadata_overhead) * 1.15)
