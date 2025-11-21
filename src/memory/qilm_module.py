from typing import Any, Dict, List, Optional, Union

import numpy as np


class QILM:
    def __init__(self) -> None:
        self.memory: List[np.ndarray] = []
        self.phases: List[Union[float, Any]] = []

    def entangle_phase(self, event_vector: np.ndarray, phase: Optional[Union[float, Any]] = None) -> None:
        if not isinstance(event_vector, np.ndarray):
            raise TypeError("event_vector must be a NumPy array.")

        vec = event_vector.astype(float)
        self.memory.append(vec)
        if phase is None:
            phase = float(np.random.rand())
        self.phases.append(phase)

    def retrieve(self, phase: Union[float, Any], tolerance: float = 0.1) -> List[np.ndarray]:
        if tolerance < 0:
            raise ValueError("Tolerance must be non-negative.")
        results: List[np.ndarray] = []
        for v, ph in zip(self.memory, self.phases):
            if isinstance(ph, (float, int)) and isinstance(phase, (float, int)):
                if abs(float(ph) - float(phase)) <= tolerance:
                    results.append(v)
            elif ph == phase:
                results.append(v)
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {"memory": [m.tolist() for m in self.memory], "phases": self.phases}
