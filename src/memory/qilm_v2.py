from dataclasses import dataclass
from threading import Lock
from typing import List

import numpy as np


@dataclass
class MemoryRetrieval:
    vector: np.ndarray
    phase: float
    resonance: float

class QILM_v2:
    def __init__(self, dimension: int = 384, capacity: int = 20000) -> None:
        # Validate inputs
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if capacity > 1_000_000:
            raise ValueError(f"capacity too large (max 1,000,000), got {capacity}")
        
        self.dimension = dimension
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        self._lock = Lock()
        self.memory_bank = np.zeros((capacity, dimension), dtype=np.float32)
        self.phase_bank = np.zeros(capacity, dtype=np.float32)
        self.norms = np.zeros(capacity, dtype=np.float32)

    def entangle(self, vector: List[float], phase: float) -> int:
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

    def retrieve(self, query_vector: List[float], current_phase: float, phase_tolerance: float = 0.15, top_k: int = 5) -> List[MemoryRetrieval]:
        with self._lock:
            if self.size == 0:
                return []
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm < 1e-9:
                q_norm = 1e-9
            
            # Optimize: use in-place operations and avoid intermediate arrays
            phase_diff = np.abs(self.phase_bank[:self.size] - current_phase)
            phase_mask = phase_diff <= phase_tolerance
            if not np.any(phase_mask):
                return []
            
            candidates_idx = np.nonzero(phase_mask)[0]
            # Optimize: compute cosine similarity without intermediate array copies
            candidate_vectors = self.memory_bank[candidates_idx]
            candidate_norms = self.norms[candidates_idx]
            
            # Vectorized cosine similarity calculation
            cosine_sims = np.dot(candidate_vectors, q_vec) / (candidate_norms * q_norm)
            
            # Optimize: use argpartition only when beneficial (>2x top_k)
            num_candidates = len(cosine_sims)
            if num_candidates > top_k * 2:
                # Use partial sort for large result sets
                top_local = np.argpartition(cosine_sims, -top_k)[-top_k:]
                # Sort only the top k items
                top_local = top_local[np.argsort(cosine_sims[top_local])[::-1]]
            else:
                # Full sort for small result sets (faster for small arrays)
                top_local = np.argsort(cosine_sims)[::-1][:top_k]
            
            # Optimize: pre-allocate results list
            results: List[MemoryRetrieval] = []
            for loc in top_local:
                glob = candidates_idx[loc]
                results.append(MemoryRetrieval(
                    vector=self.memory_bank[glob],
                    phase=self.phase_bank[glob],
                    resonance=float(cosine_sims[loc])
                ))
            return results

    def get_state_stats(self) -> dict[str, int | float]:
        return {
            "capacity": self.capacity,
            "used": self.size,
            "memory_mb": round((self.memory_bank.nbytes + self.phase_bank.nbytes) / 1024**2, 2)
        }
