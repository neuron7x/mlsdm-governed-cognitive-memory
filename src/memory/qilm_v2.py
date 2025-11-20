"""
QILM v2: Quantum-Inspired Localized Memory with Phase-Based Retrieval.

Implements neurobiologically-constrained memory with hard 20K vector limit
representing hippocampal CA3 capacity.
"""
from typing import List, Dict, Union
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from threading import Lock


@dataclass
class MemoryRetrieval:
    """Result from memory retrieval operation."""
    vector: NDArray[np.float32]
    phase: float
    resonance: float


class QILM_v2:
    """
    Phase-based memory storage with circular buffer and fixed capacity.
    
    Biological inspiration: Hippocampal CA3 region with ~20K place cells.
    Memory never exceeds capacity - oldest entries are overwritten.
    """
    def __init__(self, dimension: int = 384, capacity: int = 20000) -> None:
        """
        Initialize phase-based memory with fixed capacity.
        
        Args:
            dimension: Vector dimensionality
            capacity: Maximum number of vectors (hard limit)
        """
        self.dimension: int = dimension
        self.capacity: int = capacity
        self.pointer: int = 0
        self.size: int = 0
        self._lock: Lock = Lock()
        self.memory_bank: NDArray[np.float32] = np.zeros(
            (capacity, dimension), dtype=np.float32
        )
        self.phase_bank: NDArray[np.float32] = np.zeros(
            capacity, dtype=np.float32
        )
        self.norms: NDArray[np.float32] = np.zeros(capacity, dtype=np.float32)

    def entangle(self, vector: List[float], phase: float) -> int:
        """
        Store vector in memory with circular buffer overwrite.
        
        Args:
            vector: Input vector to store
            phase: Phase value for retrieval filtering
            
        Returns:
            Index where vector was stored
        """
        with self._lock:
            vec_np = np.array(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec_np) or 1e-9)
            idx = self.pointer
            self.memory_bank[idx] = vec_np
            self.phase_bank[idx] = phase
            self.norms[idx] = norm
            self.pointer = (self.pointer + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            return idx

    def retrieve(
        self,
        query_vector: List[float],
        current_phase: float,
        phase_tolerance: float = 0.15,
        top_k: int = 5
    ) -> List[MemoryRetrieval]:
        """
        Retrieve similar vectors filtered by phase.
        
        Args:
            query_vector: Query vector for similarity search
            current_phase: Current phase value
            phase_tolerance: Maximum phase difference to consider
            top_k: Number of results to return
            
        Returns:
            List of retrieval results sorted by resonance
        """
        with self._lock:
            if self.size == 0:
                return []
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec) or 1e-9)
            phase_diff = np.abs(self.phase_bank[:self.size] - current_phase)
            phase_mask = phase_diff <= phase_tolerance
            if not np.any(phase_mask):
                return []
            candidates_idx = np.nonzero(phase_mask)[0]
            candidate_vectors = self.memory_bank[candidates_idx]
            candidate_norms = self.norms[candidates_idx]
            dots = np.dot(candidate_vectors, q_vec)
            cosine_sims = dots / (candidate_norms * q_norm)
            if len(cosine_sims) > top_k:
                top_local = np.argpartition(cosine_sims, -top_k)[-top_k:]
                top_local = top_local[np.argsort(cosine_sims[top_local])[::-1]]
            else:
                top_local = np.argsort(cosine_sims)[::-1]
            results: List[MemoryRetrieval] = []
            for loc in top_local:
                glob = candidates_idx[loc]
                results.append(MemoryRetrieval(
                    vector=self.memory_bank[glob],
                    phase=self.phase_bank[glob],
                    resonance=float(cosine_sims[loc])
                ))
            return results

    def get_state_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with capacity, usage, and memory size
        """
        return {
            "capacity": self.capacity,
            "used": self.size,
            "memory_mb": round(
                (self.memory_bank.nbytes + self.phase_bank.nbytes) / 1024**2,
                2
            )
        }
