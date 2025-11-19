from __future__ import annotations
import numpy as np
from typing import Tuple
from pydantic import BaseModel

class MemoryConfig(BaseModel):
    dimension: int
    lambda_l1: float
    lambda_l2: float
    lambda_l3: float
    theta_l1: float
    theta_l2: float
    gating12: float
    gating23: float

class MultiLevelSynapticMemory:
    def __init__(self, config: MemoryConfig):
        self.dim = config.dimension
        self.l1 = np.zeros(self.dim, dtype=np.float32)
        self.l2 = np.zeros(self.dim, dtype=np.float32)
        self.l3 = np.zeros(self.dim, dtype=np.float32)
        self.cfg = config

    def update(self, event: np.ndarray) -> None:
        if event.shape != (self.dim,):
            raise ValueError("Dimension mismatch")

        # Decay
        self.l1 *= (1 - self.cfg.lambda_l1)
        self.l2 *= (1 - self.cfg.lambda_l2)
        self.l3 *= (1 - self.cfg.lambda_l3)

        # Add new event to L1
        self.l1 += event.astype(np.float32)

        # Gated transfer
        mask12 = self.l1 > self.cfg.theta_l1
        transfer12 = mask12 * self.l1 * self.cfg.gating12
        self.l1 -= transfer12
        self.l2 += transfer12

        mask23 = self.l2 > self.cfg.theta_l2
        transfer23 = mask23 * self.l2 * self.cfg.gating23
        self.l2 -= transfer23
        self.l3 += transfer23

    def state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.l1.copy(), self.l2.copy(), self.l3.copy()

    def norms(self) -> Tuple[float, float, float]:
        return float(np.linalg.norm(self.l1)), float(np.linalg.norm(self.l2)), float(np.linalg.norm(self.l3))
