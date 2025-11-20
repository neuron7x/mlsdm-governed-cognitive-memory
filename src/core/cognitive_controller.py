"""Cognitive Controller: Main orchestration for cognitive memory system."""
from typing import Dict, Any, List, Union
import numpy as np
from numpy.typing import NDArray
from threading import Lock
from ..cognition.moral_filter_v2 import MoralFilterV2
from ..memory.qilm_v2 import QILM_v2
from ..rhythm.cognitive_rhythm import CognitiveRhythm
from ..memory.multi_level_memory import MultiLevelSynapticMemory


class CognitiveController:
    """
    Main controller for cognitive memory processing.
    
    Orchestrates moral filtering, memory storage, and circadian rhythm.
    Thread-safe for concurrent access.
    """
    
    def __init__(self, dim: int = 384) -> None:
        """
        Initialize cognitive controller.
        
        Args:
            dim: Vector dimension for memory storage (default: 384)
        """
        self.dim: int = dim
        self._lock: Lock = Lock()
        self.moral: MoralFilterV2 = MoralFilterV2(initial_threshold=0.50)
        self.qilm: QILM_v2 = QILM_v2(dimension=dim, capacity=20_000)
        self.rhythm: CognitiveRhythm = CognitiveRhythm(
            wake_duration=8,
            sleep_duration=3
        )
        self.synaptic: MultiLevelSynapticMemory = MultiLevelSynapticMemory(
            dimension=dim
        )
        self.step_counter: int = 0

    def process_event(
        self,
        vector: NDArray[np.float32],
        moral_value: float
    ) -> Dict[str, Union[int, str, float, bool, Dict[str, float]]]:
        """
        Process a cognitive event with moral filtering and memory storage.
        
        Args:
            vector: Input vector to process (shape: (dim,))
            moral_value: Moral evaluation score [0, 1]
            
        Returns:
            State dictionary with processing results
        """
        with self._lock:
            self.step_counter += 1
            accepted = self.moral.evaluate(moral_value)
            self.moral.adapt(accepted)
            
            if not accepted:
                return self._build_state(
                    rejected=True,
                    note="morally rejected"
                )
            
            if not self.rhythm.is_wake():
                return self._build_state(
                    rejected=True,
                    note="sleep phase"
                )
            
            self.synaptic.update(vector)
            phase_val = 0.1 if self.rhythm.phase == "wake" else 0.9
            self.qilm.entangle(vector.tolist(), phase=phase_val)
            self.rhythm.step()
            
            return self._build_state(rejected=False, note="processed")

    def retrieve_context(
        self,
        query_vector: NDArray[np.float32],
        top_k: int = 5
    ) -> List[Any]:
        """
        Retrieve relevant context from memory.
        
        Args:
            query_vector: Query vector (shape: (dim,))
            top_k: Number of results to retrieve (default: 5)
            
        Returns:
            List of memory retrieval results
        """
        with self._lock:
            phase_val = 0.1 if self.rhythm.is_wake() else 0.9
            return self.qilm.retrieve(
                query_vector.tolist(),
                current_phase=phase_val,
                phase_tolerance=0.15,
                top_k=top_k
            )

    def _build_state(
        self,
        rejected: bool,
        note: str
    ) -> Dict[str, Union[int, str, float, bool, Dict[str, float]]]:
        """
        Build state dictionary for response.
        
        Args:
            rejected: Whether event was rejected
            note: Description of processing result
            
        Returns:
            Complete state dictionary
        """
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
