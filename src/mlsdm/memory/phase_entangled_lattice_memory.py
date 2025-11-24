import hashlib
from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass
class MemoryRetrieval:
    vector: np.ndarray
    phase: float
    resonance: float

class PhaseEntangledLatticeMemory:
    """
    Phase-Entangled Lattice Memory (PELM, formerly QILM_v2).

    A bounded phase-entangled lattice in embedding space that provides
    phase-based retrieval with geometric constraints. This memory system
    stores vectors with associated phase values and enables retrieval
    based on phase proximity and cosine similarity.

    Not related to quantum hardware - the design is mathematically inspired
    by quantum concepts but operates entirely in classical embedding space.
    """

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
        self._checksum = self._compute_checksum()

    def _ensure_integrity(self) -> None:
        """
        Ensure memory integrity, attempting recovery if corruption detected.
        Should only be called from within a lock context.

        Raises:
            RuntimeError: If corruption is detected and recovery fails.
        """
        if self._detect_corruption_unsafe():  # noqa: SIM102
            if not self._auto_recover_unsafe():
                raise RuntimeError("Memory corruption detected and recovery failed")

    def entangle(self, vector: list[float], phase: float) -> int:
        with self._lock:
            # Ensure integrity before operation
            self._ensure_integrity()

            vec_np = np.array(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec_np) or 1e-9)
            idx = self.pointer
            self.memory_bank[idx] = vec_np
            self.phase_bank[idx] = phase
            self.norms[idx] = norm

            # Update pointer with wraparound check
            new_pointer = self.pointer + 1
            if new_pointer >= self.capacity:
                new_pointer = 0  # Explicit wraparound
            self.pointer = new_pointer

            self.size = min(self.size + 1, self.capacity)

            # Update checksum after modification
            self._checksum = self._compute_checksum()

            return idx

    def retrieve(self, query_vector: list[float], current_phase: float, phase_tolerance: float = 0.15, top_k: int = 5) -> list[MemoryRetrieval]:
        with self._lock:
            # Ensure integrity before operation
            self._ensure_integrity()

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
            results: list[MemoryRetrieval] = []
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

    def _compute_checksum(self) -> str:
        """Compute checksum for memory bank integrity validation."""
        # Create a hash of the used portion of memory banks
        hasher = hashlib.sha256()
        hasher.update(self.memory_bank[:self.size].tobytes())
        hasher.update(self.phase_bank[:self.size].tobytes())
        hasher.update(self.norms[:self.size].tobytes())
        # Include metadata
        hasher.update(f"{self.pointer}:{self.size}:{self.capacity}".encode())
        return hasher.hexdigest()

    def _validate_pointer_bounds(self) -> bool:
        """Validate pointer is within acceptable bounds."""
        if self.pointer < 0 or self.pointer >= self.capacity:
            return False
        return not (self.size < 0 or self.size > self.capacity)

    def _detect_corruption_unsafe(self) -> bool:
        """
        Detect if memory bank has been corrupted (unsafe - no locking).
        Should only be called from within a lock context.

        Returns:
            True if corruption detected, False otherwise.
        """
        # Check pointer bounds
        if not self._validate_pointer_bounds():
            return True

        # Check checksum
        current_checksum = self._compute_checksum()
        return current_checksum != self._checksum

    def detect_corruption(self) -> bool:
        """
        Detect if memory bank has been corrupted.

        Returns:
            True if corruption detected, False otherwise.
        """
        with self._lock:
            return self._detect_corruption_unsafe()

    def _rebuild_index(self) -> None:
        """Rebuild the index by recomputing norms and metadata."""
        # Validate and fix size first before attempting to iterate
        if self.size < 0:
            self.size = 0
        elif self.size > self.capacity:
            self.size = self.capacity

        # Validate and fix pointer
        if self.pointer < 0 or self.pointer >= self.capacity:
            self.pointer = self.size % self.capacity if self.size > 0 else 0

        # Recompute norms for all stored vectors (now safe to iterate)
        for i in range(self.size):
            vec = self.memory_bank[i]
            self.norms[i] = float(np.linalg.norm(vec) or 1e-9)

    def _auto_recover_unsafe(self) -> bool:
        """
        Attempt to recover from corruption by rebuilding the index (unsafe - no locking).
        Should only be called from within a lock context.

        Returns:
            True if recovery successful, False otherwise.
        """
        if not self._detect_corruption_unsafe():
            return True  # No corruption detected

        # Attempt recovery
        try:
            self._rebuild_index()
            # Update checksum after rebuild
            self._checksum = self._compute_checksum()
            # Verify recovery
            return not self._detect_corruption_unsafe()
        except Exception:
            return False

    def auto_recover(self) -> bool:
        """
        Attempt to recover from corruption by rebuilding the index.

        Returns:
            True if recovery successful, False otherwise.
        """
        with self._lock:
            return self._auto_recover_unsafe()
