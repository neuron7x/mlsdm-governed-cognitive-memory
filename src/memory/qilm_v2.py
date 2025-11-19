import numpy as np
from typing import List
from dataclasses import dataclass
from threading import Lock

@dataclass
class MemoryRetrieval:
    vector: np.ndarray
    phase: float
    resonance: float

class QILM_v2:
    def __init__(self, dimension=384, capacity=20000):
        self.dimension = dimension
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        self._lock = Lock()
        self.memory_bank = np.zeros((capacity, dimension), dtype=np.float32)
        self.phase_bank = np.zeros(capacity, dtype=np.float32)
        self.norms = np.zeros(capacity, dtype=np.float32)

    def entangle(self, vector, phase):
        with self._lock:
            vec_np = np.array(vector, dtype=np.float32)
            norm = np.linalg.norm(vec_np) or 1e-9
            idx = self.pointer
            self.memory_bank[idx] = vec_np
            self.phase_bank[idx] = phase
            self.norms[idx] = norm
            self.pointer = (self.pointer + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            return idx

    def retrieve(self, query_vector, current_phase, phase_tolerance=0.15, top_k=5):
        with self._lock:
            if self.size == 0:
                return []
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec) or 1e-9
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
            results = []
            for loc in top_local:
                glob = candidates_idx[loc]
                results.append(MemoryRetrieval(
                    vector=self.memory_bank[glob],
                    phase=self.phase_bank[glob],
                    resonance=float(cosine_sims[loc])
                ))
            return results

    def get_state_stats(self):
        return {
            "capacity": self.capacity,
            "used": self.size,
            "memory_mb": round((self.memory_bank.nbytes + self.phase_bank.nbytes) / 1024**2, 2)
        }
