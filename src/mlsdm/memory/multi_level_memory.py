from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from config.calibration import SynapticMemoryCalibration

# Import calibration defaults for consistent parameter values
# Type annotation uses Optional since module may not be available
_SYNAPTIC_MEMORY_DEFAULTS: Optional["SynapticMemoryCalibration"] = None
try:
    from config.calibration import SYNAPTIC_MEMORY_DEFAULTS as _IMPORTED_DEFAULTS
    _SYNAPTIC_MEMORY_DEFAULTS = _IMPORTED_DEFAULTS
except ImportError:
    # Fallback if calibration module is not available - already None
    pass


# Helper to get default value from calibration or fallback
def _get_default(attr: str, fallback: float) -> float:
    if _SYNAPTIC_MEMORY_DEFAULTS is not None:
        return getattr(_SYNAPTIC_MEMORY_DEFAULTS, attr, fallback)
    return fallback


class MultiLevelSynapticMemory:
    def __init__(
        self,
        dimension: int = 384,
        lambda_l1: float | None = None,
        lambda_l2: float | None = None,
        lambda_l3: float | None = None,
        theta_l1: float | None = None,
        theta_l2: float | None = None,
        gating12: float | None = None,
        gating23: float | None = None,
        *,
        config: "SynapticMemoryCalibration | None" = None,
    ) -> None:
        """Initialize MultiLevelSynapticMemory.

        Args:
            dimension: Vector dimension for memory arrays.
            lambda_l1: L1 decay rate. If None, uses SYNAPTIC_MEMORY_DEFAULTS.
            lambda_l2: L2 decay rate. If None, uses SYNAPTIC_MEMORY_DEFAULTS.
            lambda_l3: L3 decay rate. If None, uses SYNAPTIC_MEMORY_DEFAULTS.
            theta_l1: L1→L2 consolidation threshold. If None, uses SYNAPTIC_MEMORY_DEFAULTS.
            theta_l2: L2→L3 consolidation threshold. If None, uses SYNAPTIC_MEMORY_DEFAULTS.
            gating12: L1→L2 gating factor. If None, uses SYNAPTIC_MEMORY_DEFAULTS.
            gating23: L2→L3 gating factor. If None, uses SYNAPTIC_MEMORY_DEFAULTS.
            config: Optional SynapticMemoryCalibration instance. If provided,
                all λ/θ/gating values are taken from it (unless explicitly overridden).
        """
        # Resolve parameter source: explicit arg > config > SYNAPTIC_MEMORY_DEFAULTS
        if config is not None:
            _lambda_l1 = lambda_l1 if lambda_l1 is not None else config.lambda_l1
            _lambda_l2 = lambda_l2 if lambda_l2 is not None else config.lambda_l2
            _lambda_l3 = lambda_l3 if lambda_l3 is not None else config.lambda_l3
            _theta_l1 = theta_l1 if theta_l1 is not None else config.theta_l1
            _theta_l2 = theta_l2 if theta_l2 is not None else config.theta_l2
            _gating12 = gating12 if gating12 is not None else config.gating12
            _gating23 = gating23 if gating23 is not None else config.gating23
        else:
            _lambda_l1 = lambda_l1 if lambda_l1 is not None else _get_default("lambda_l1", 0.50)
            _lambda_l2 = lambda_l2 if lambda_l2 is not None else _get_default("lambda_l2", 0.10)
            _lambda_l3 = lambda_l3 if lambda_l3 is not None else _get_default("lambda_l3", 0.01)
            _theta_l1 = theta_l1 if theta_l1 is not None else _get_default("theta_l1", 1.2)
            _theta_l2 = theta_l2 if theta_l2 is not None else _get_default("theta_l2", 2.5)
            _gating12 = gating12 if gating12 is not None else _get_default("gating12", 0.45)
            _gating23 = gating23 if gating23 is not None else _get_default("gating23", 0.30)
        # Validate inputs
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if not (0 < _lambda_l1 <= 1.0):
            raise ValueError(f"lambda_l1 must be in (0, 1], got {_lambda_l1}")
        if not (0 < _lambda_l2 <= 1.0):
            raise ValueError(f"lambda_l2 must be in (0, 1], got {_lambda_l2}")
        if not (0 < _lambda_l3 <= 1.0):
            raise ValueError(f"lambda_l3 must be in (0, 1], got {_lambda_l3}")
        if _theta_l1 <= 0:
            raise ValueError(f"theta_l1 must be positive, got {_theta_l1}")
        if _theta_l2 <= 0:
            raise ValueError(f"theta_l2 must be positive, got {_theta_l2}")
        if not (0 <= _gating12 <= 1.0):
            raise ValueError(f"gating12 must be in [0, 1], got {_gating12}")
        if not (0 <= _gating23 <= 1.0):
            raise ValueError(f"gating23 must be in [0, 1], got {_gating23}")

        self.dim = int(dimension)
        self.lambda_l1 = float(_lambda_l1)
        self.lambda_l2 = float(_lambda_l2)
        self.lambda_l3 = float(_lambda_l3)
        self.theta_l1 = float(_theta_l1)
        self.theta_l2 = float(_theta_l2)
        self.gating12 = float(_gating12)
        self.gating23 = float(_gating23)

        self.l1 = np.zeros(self.dim, dtype=np.float32)
        self.l2 = np.zeros(self.dim, dtype=np.float32)
        self.l3 = np.zeros(self.dim, dtype=np.float32)

    def update(self, event: np.ndarray) -> None:
        if not isinstance(event, np.ndarray) or event.shape[0] != self.dim:
            raise ValueError(f"Event vector must be a NumPy array of dimension {self.dim}.")

        # Optimize: perform decay in-place to avoid temporary arrays
        self.l1 *= (1 - self.lambda_l1)
        self.l2 *= (1 - self.lambda_l2)
        self.l3 *= (1 - self.lambda_l3)

        # Optimize: avoid unnecessary astype if already float32
        if event.dtype != np.float32:
            self.l1 += event.astype(np.float32)
        else:
            self.l1 += event

        # Transfer logic maintains original behavior but avoids temp array creation
        transfer12 = (self.l1 > self.theta_l1) * self.l1 * self.gating12
        self.l1 -= transfer12
        self.l2 += transfer12
        transfer23 = (self.l2 > self.theta_l2) * self.l2 * self.gating23
        self.l2 -= transfer23
        self.l3 += transfer23

    def state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.l1.copy(), self.l2.copy(), self.l3.copy()

    def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.state()

    def reset_all(self) -> None:
        self.l1.fill(0.0)
        self.l2.fill(0.0)
        self.l3.fill(0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dim,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            "lambda_l3": self.lambda_l3,
            "theta_l1": self.theta_l1,
            "theta_l2": self.theta_l2,
            "gating12": self.gating12,
            "gating23": self.gating23,
            "state_L1": self.l1.tolist(),
            "state_L2": self.l2.tolist(),
            "state_L3": self.l3.tolist(),
        }
