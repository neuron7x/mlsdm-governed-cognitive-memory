import numpy as np
from typing import Dict, Tuple, Any


class MultiLevelSynapticMemory:
    def __init__(
        self,
        dimension: int = 384,
        lambda_l1: float = 0.50,
        lambda_l2: float = 0.10,
        lambda_l3: float = 0.01,
        theta_l1: float = 1.2,
        theta_l2: float = 2.5,
        gating12: float = 0.45,
        gating23: float = 0.30,
    ) -> None:
        # Validate inputs
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if not (0 < lambda_l1 <= 1.0):
            raise ValueError(f"lambda_l1 must be in (0, 1], got {lambda_l1}")
        if not (0 < lambda_l2 <= 1.0):
            raise ValueError(f"lambda_l2 must be in (0, 1], got {lambda_l2}")
        if not (0 < lambda_l3 <= 1.0):
            raise ValueError(f"lambda_l3 must be in (0, 1], got {lambda_l3}")
        if theta_l1 <= 0:
            raise ValueError(f"theta_l1 must be positive, got {theta_l1}")
        if theta_l2 <= 0:
            raise ValueError(f"theta_l2 must be positive, got {theta_l2}")
        if not (0 <= gating12 <= 1.0):
            raise ValueError(f"gating12 must be in [0, 1], got {gating12}")
        if not (0 <= gating23 <= 1.0):
            raise ValueError(f"gating23 must be in [0, 1], got {gating23}")
        
        self.dim = int(dimension)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)
        self.lambda_l3 = float(lambda_l3)
        self.theta_l1 = float(theta_l1)
        self.theta_l2 = float(theta_l2)
        self.gating12 = float(gating12)
        self.gating23 = float(gating23)

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

    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.l1.copy(), self.l2.copy(), self.l3.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.state()

    def reset_all(self) -> None:
        self.l1.fill(0.0)
        self.l2.fill(0.0)
        self.l3.fill(0.0)

    def to_dict(self) -> Dict[str, Any]:
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
