import numpy as np
from threading import Lock
from typing import Dict, Any, List
from ..cognition.moral_filter_v2 import MoralFilterV2
from ..memory.qilm_v2 import QILM_v2, MemoryRetrieval
from ..rhythm.cognitive_rhythm import CognitiveRhythm
from ..memory.multi_level_memory import MultiLevelSynapticMemory

class CognitiveController:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._lock = Lock()
        self.moral = MoralFilterV2(initial_threshold=0.50)
        self.qilm = QILM_v2(dimension=dim, capacity=20_000)
        self.rhythm = CognitiveRhythm(wake_duration=8, sleep_duration=3)
        self.synaptic = MultiLevelSynapticMemory(dimension=dim)
        self.step_counter = 0
        # Cache for phase values to avoid repeated computation
        self._phase_cache: Dict[str, float] = {"wake": 0.1, "sleep": 0.9}

    def process_event(self, vector: np.ndarray, moral_value: float) -> Dict[str, Any]:
        with self._lock:
            self.step_counter += 1
            accepted = self.moral.evaluate(moral_value)
            self.moral.adapt(accepted)
            if not accepted:
                return self._build_state(rejected=True, note="morally rejected")
            if not self.rhythm.is_wake():
                return self._build_state(rejected=True, note="sleep phase")
            self.synaptic.update(vector)
            # Optimize: use cached phase value
            phase_val = self._phase_cache[self.rhythm.phase]
            self.qilm.entangle(vector.tolist(), phase=phase_val)
            self.rhythm.step()
            return self._build_state(rejected=False, note="processed")

    def retrieve_context(self, query_vector: np.ndarray, top_k: int = 5) -> List[MemoryRetrieval]:
        with self._lock:
            # Optimize: use cached phase value
            phase_val = self._phase_cache[self.rhythm.phase]
            return self.qilm.retrieve(query_vector.tolist(), current_phase=phase_val, 
                                     phase_tolerance=0.15, top_k=top_k)

    def _build_state(self, rejected: bool, note: str) -> Dict[str, Any]:
        l1, l2, l3 = self.synaptic.state()
        return {
            "step": self.step_counter,
            "phase": self.rhythm.phase,
            "moral_threshold": round(self.moral.threshold, 4),
            "moral_ema": round(self.moral.ema_accept_rate, 4),
            "synaptic_norms": {
                "L1": float(np.linalg.norm(l1)),
                "L2": float(np.linalg.norm(l2)),
                "L3": float(np.linalg.norm(l3))
            },
            "qilm_used": self.qilm.get_state_stats()["used"],
            "rejected": rejected,
            "note": note
        }
