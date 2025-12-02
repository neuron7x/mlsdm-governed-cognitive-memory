from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mlsdm.config import PELMCalibration

# Import calibration defaults - these can be overridden via config
# Type hints use Optional to allow None when calibration module unavailable
PELM_DEFAULTS: PELMCalibration | None

try:
    from mlsdm.config import PELM_DEFAULTS
except ImportError:
    PELM_DEFAULTS = None

# Observability imports - gracefully handle missing module
try:
    from mlsdm.observability.memory_telemetry import (
        record_pelm_corruption,
        record_pelm_retrieve,
        record_pelm_store,
    )
    _OBSERVABILITY_AVAILABLE = True
except ImportError:
    _OBSERVABILITY_AVAILABLE = False


@dataclass
class MemoryRetrieval:
    """Result from a memory retrieval operation.

    Attributes:
        vector: The retrieved embedding vector
        phase: The phase value associated with this memory
        resonance: Cosine similarity score (higher = better match)
    """
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

    # Default values from calibration
    DEFAULT_CAPACITY = PELM_DEFAULTS.default_capacity if PELM_DEFAULTS else 20_000
    MAX_CAPACITY = PELM_DEFAULTS.max_capacity if PELM_DEFAULTS else 1_000_000
    DEFAULT_PHASE_TOLERANCE = PELM_DEFAULTS.phase_tolerance if PELM_DEFAULTS else 0.15
    DEFAULT_TOP_K = PELM_DEFAULTS.default_top_k if PELM_DEFAULTS else 5
    MIN_NORM_THRESHOLD = PELM_DEFAULTS.min_norm_threshold if PELM_DEFAULTS else 1e-9

    def __init__(self, dimension: int = 384, capacity: int | None = None) -> None:
        # Use calibration default if not specified
        if capacity is None:
            capacity = self.DEFAULT_CAPACITY

        # Validate inputs
        if dimension <= 0:
            raise ValueError(
                f"dimension must be positive, got {dimension}. "
                "Dimension determines the embedding vector size and must match the model's embedding dimension."
            )
        if capacity <= 0:
            raise ValueError(
                f"capacity must be positive, got {capacity}. "
                "Capacity determines the maximum number of vectors that can be stored in memory."
            )
        if capacity > self.MAX_CAPACITY:
            raise ValueError(
                f"capacity too large (max {self.MAX_CAPACITY:,}), got {capacity}. "
                "Large capacities may cause excessive memory usage. "
                f"Estimated memory: {capacity * dimension * 4 / (1024**2):.2f} MB"
            )

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
            recovered = self._auto_recover_unsafe()
            # Record corruption event
            if _OBSERVABILITY_AVAILABLE:
                record_pelm_corruption(
                    detected=True,
                    recovered=recovered,
                    pointer=self.pointer,
                    size=self.size,
                )
            if not recovered:
                raise RuntimeError(
                    "Memory corruption detected and recovery failed. "
                    f"Current state: pointer={self.pointer}, size={self.size}, capacity={self.capacity}. "
                    "This may indicate hardware issues, race conditions, or memory overwrites. "
                    "Consider restarting the system or reducing capacity."
                )

    def entangle(
        self,
        vector: list[float],
        phase: float,
        correlation_id: str | None = None,
    ) -> int:
        """Store a vector with associated phase in memory.

        Args:
            vector: Embedding vector to store (must match dimension)
            phase: Phase value in [0.0, 1.0] representing cognitive state
            correlation_id: Optional correlation ID for observability tracking

        Returns:
            Index where the vector was stored

        Raises:
            TypeError: If vector is not a list or phase is not numeric
            ValueError: If vector dimension doesn't match, phase out of range,
                       or vector contains NaN/inf values
        """
        start_time = time.perf_counter() if _OBSERVABILITY_AVAILABLE else None

        with self._lock:
            # Ensure integrity before operation
            self._ensure_integrity()

            # Validate vector type
            if not isinstance(vector, list):
                raise TypeError(f"vector must be a list, got {type(vector).__name__}")
            if len(vector) != self.dimension:
                raise ValueError(
                    f"vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )

            # Validate vector values (check for NaN/inf)
            for i, val in enumerate(vector):
                if not isinstance(val, (int, float)):
                    raise TypeError(
                        f"vector element at index {i} must be numeric, got {type(val).__name__}"
                    )
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"vector contains invalid value at index {i}: {val}. "
                        "NaN and infinity are not allowed in memory vectors."
                    )

            # Validate phase type and range
            if not isinstance(phase, (int, float)):
                raise TypeError(f"phase must be numeric, got {type(phase).__name__}")
            if math.isnan(phase) or math.isinf(phase):
                raise ValueError(
                    f"phase must be a finite number, got {phase}. "
                    "NaN and infinity are not allowed."
                )
            if not (0.0 <= phase <= 1.0):
                raise ValueError(
                    f"phase must be in [0.0, 1.0], got {phase}. "
                    "Phase values represent cognitive states (e.g., 0.1=wake, 0.9=sleep)."
                )

            vec_np = np.array(vector, dtype=np.float32)
            norm = float(np.linalg.norm(vec_np) or self.MIN_NORM_THRESHOLD)
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

            # Record observability metrics
            if _OBSERVABILITY_AVAILABLE and start_time is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                record_pelm_store(
                    index=idx,
                    phase=phase,
                    vector_norm=norm,
                    capacity_used=self.size,
                    capacity_total=self.capacity,
                    memory_bytes=self.memory_usage_bytes(),
                    latency_ms=latency_ms,
                    correlation_id=correlation_id,
                )

            return idx

    def retrieve(
        self,
        query_vector: list[float],
        current_phase: float,
        phase_tolerance: float | None = None,
        top_k: int | None = None,
        correlation_id: str | None = None,
    ) -> list[MemoryRetrieval]:
        # Use calibration defaults if not specified
        if phase_tolerance is None:
            phase_tolerance = self.DEFAULT_PHASE_TOLERANCE
        if top_k is None:
            top_k = self.DEFAULT_TOP_K

        start_time = time.perf_counter() if _OBSERVABILITY_AVAILABLE else None

        with self._lock:
            # Ensure integrity before operation
            self._ensure_integrity()

            if self.size == 0:
                # Record empty result
                if _OBSERVABILITY_AVAILABLE and start_time is not None:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    record_pelm_retrieve(
                        query_phase=current_phase,
                        phase_tolerance=phase_tolerance,
                        top_k=top_k,
                        results_count=0,
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                    )
                return []
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm < self.MIN_NORM_THRESHOLD:
                q_norm = self.MIN_NORM_THRESHOLD

            # Optimize: use in-place operations and avoid intermediate arrays
            phase_diff = np.abs(self.phase_bank[:self.size] - current_phase)
            phase_mask = phase_diff <= phase_tolerance
            if not np.any(phase_mask):
                # Record empty result due to phase mismatch
                if _OBSERVABILITY_AVAILABLE and start_time is not None:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    record_pelm_retrieve(
                        query_phase=current_phase,
                        phase_tolerance=phase_tolerance,
                        top_k=top_k,
                        results_count=0,
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                    )
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

            # Record successful retrieval
            if _OBSERVABILITY_AVAILABLE and start_time is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                record_pelm_retrieve(
                    query_phase=current_phase,
                    phase_tolerance=phase_tolerance,
                    top_k=top_k,
                    results_count=len(results),
                    latency_ms=latency_ms,
                    correlation_id=correlation_id,
                )

            return results

    def get_state_stats(self) -> dict[str, int | float]:
        return {
            "capacity": self.capacity,
            "used": self.size,
            "memory_mb": round((self.memory_bank.nbytes + self.phase_bank.nbytes) / 1024**2, 2)
        }

    def memory_usage_bytes(self) -> int:
        """Calculate conservative memory usage estimate in bytes.

        Returns:
            Estimated memory usage including all arrays and metadata overhead.

        Note:
            This is a conservative estimate (10-20% overhead) to ensure we
            never underestimate actual memory usage.
        """
        # Core numpy arrays
        memory_bank_bytes = self.memory_bank.nbytes  # capacity × dimension × float32
        phase_bank_bytes = self.phase_bank.nbytes    # capacity × float32
        norms_bytes = self.norms.nbytes              # capacity × float32

        # Subtotal for arrays
        array_bytes = memory_bank_bytes + phase_bank_bytes + norms_bytes

        # Metadata overhead (conservative estimate for Python object overhead)
        # Includes: dimension, capacity, pointer, size, checksum string,
        # Lock object, and Python object headers
        metadata_overhead = 1024  # ~1KB for metadata

        # Conservative 15% overhead for potential fragmentation and internal
        # Python structures
        conservative_multiplier = 1.15

        total_bytes = int((array_bytes + metadata_overhead) * conservative_multiplier)
        return total_bytes

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
