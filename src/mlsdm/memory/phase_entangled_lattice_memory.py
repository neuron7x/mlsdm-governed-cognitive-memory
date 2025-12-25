from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from mlsdm.memory.provenance import MemoryProvenance, MemorySource
from mlsdm.utils.math_constants import safe_norm

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
        provenance: Metadata about memory origin and confidence
        memory_id: Unique identifier for this memory
    """

    vector: np.ndarray
    phase: float
    resonance: float
    provenance: MemoryProvenance
    memory_id: str


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
        # Optimization: Pre-allocate query buffer to reduce allocations during retrieval
        self._query_buffer = np.zeros(dimension, dtype=np.float32)
        self._checksum = self._compute_checksum()

        # Provenance tracking for AI safety (TD-003)
        self._provenance: list[MemoryProvenance] = []
        self._memory_ids: list[str] = []
        self._confidence_threshold = 0.5  # Minimum confidence for storage

    def _ensure_integrity(self) -> None:
        """
        Ensure memory integrity, attempting recovery if corruption detected.
        Should only be called from within a lock context.

        Raises:
            RuntimeError: If corruption is detected and recovery fails.
        """
        if self._detect_corruption_unsafe():
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
        provenance: MemoryProvenance | None = None,
    ) -> int:
        """Store a vector with associated phase in memory.

        Args:
            vector: Embedding vector to store (must match dimension)
            phase: Phase value in [0.0, 1.0] representing cognitive state
            correlation_id: Optional correlation ID for observability tracking
            provenance: Optional provenance metadata (source, confidence, etc.)

        Returns:
            Index where the vector was stored, or -1 if rejected due to low confidence

        Raises:
            TypeError: If vector is not a list or phase is not numeric
            ValueError: If vector dimension doesn't match, phase out of range,
                       or vector contains NaN/inf values
        """
        start_time = time.perf_counter() if _OBSERVABILITY_AVAILABLE else None

        with self._lock:
            # Ensure integrity before operation
            self._ensure_integrity()

            # Generate unique memory ID
            memory_id = str(uuid.uuid4())

            # Create default provenance if not provided
            if provenance is None:
                provenance = MemoryProvenance(
                    source=MemorySource.SYSTEM_PROMPT, confidence=1.0, timestamp=datetime.now()
                )

            # Check confidence threshold - reject low-confidence memories
            if provenance.confidence < self._confidence_threshold:
                # Log rejection for observability
                if _OBSERVABILITY_AVAILABLE and start_time is not None:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    record_pelm_store(
                        index=-1,
                        phase=phase,
                        vector_norm=0.0,
                        capacity_used=self.size,
                        capacity_total=self.capacity,
                        memory_bytes=self.memory_usage_bytes(),
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                    )
                return -1  # Rejection sentinel value

            # Validate vector type
            if not isinstance(vector, list):
                raise TypeError(f"vector must be a list, got {type(vector).__name__}")
            if len(vector) != self.dimension:
                raise ValueError(
                    f"vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )

            # Validate vector values (check for NaN/inf)
            for i, val in enumerate(vector):
                if not isinstance(val, int | float):
                    raise TypeError(
                        f"vector element at index {i} must be numeric, got {type(val).__name__}"
                    )
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"vector contains invalid value at index {i}: {val}. "
                        "NaN and infinity are not allowed in memory vectors."
                    )

            # Validate phase type and range
            if not isinstance(phase, int | float):
                raise TypeError(f"phase must be numeric, got {type(phase).__name__}")
            if math.isnan(phase) or math.isinf(phase):
                raise ValueError(
                    f"phase must be a finite number, got {phase}. NaN and infinity are not allowed."
                )
            if not (0.0 <= phase <= 1.0):
                raise ValueError(
                    f"phase must be in [0.0, 1.0], got {phase}. "
                    "Phase values represent cognitive states (e.g., 0.1=wake, 0.9=sleep)."
                )

            vec_np = np.array(vector, dtype=np.float32)
            norm = max(safe_norm(vec_np), self.MIN_NORM_THRESHOLD)

            # Check capacity and evict if necessary
            if self.size >= self.capacity:
                self._evict_lowest_confidence()

            idx = self.pointer
            self.memory_bank[idx] = vec_np
            self.phase_bank[idx] = phase
            self.norms[idx] = norm

            # Store provenance metadata
            if idx < len(self._provenance):
                self._provenance[idx] = provenance
                self._memory_ids[idx] = memory_id
            else:
                self._provenance.append(provenance)
                self._memory_ids.append(memory_id)

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

    def entangle_batch(
        self,
        vectors: list[list[float]],
        phases: list[float],
        correlation_id: str | None = None,
        provenances: list[MemoryProvenance] | None = None,
    ) -> list[int]:
        """Store multiple vectors with associated phases in memory (batch operation).

        This is more efficient than calling entangle() multiple times because it:
        - Acquires the lock only once
        - Performs integrity check only once
        - Updates checksum only once at the end
        - Uses vectorized numpy operations where possible

        Args:
            vectors: List of embedding vectors to store (each must match dimension)
            phases: List of phase values in [0.0, 1.0] (must match vectors length)
            correlation_id: Optional correlation ID for observability tracking
            provenances: Optional list of provenance metadata (must match vectors length if provided)

        Returns:
            List of indices where the vectors were stored (-1 for rejected)

        Raises:
            TypeError: If vectors/phases are not lists or contain invalid types
            ValueError: If dimensions don't match, phases out of range,
                       or vectors contain NaN/inf values
        """
        start_time = time.perf_counter() if _OBSERVABILITY_AVAILABLE else None

        if not isinstance(vectors, list) or not isinstance(phases, list):
            raise TypeError("vectors and phases must be lists")

        if len(vectors) != len(phases):
            raise ValueError(
                f"vectors and phases must have same length: "
                f"{len(vectors)} vectors, {len(phases)} phases"
            )

        if provenances is not None and len(provenances) != len(vectors):
            raise ValueError(
                f"provenances must match vectors length: "
                f"{len(vectors)} vectors, {len(provenances)} provenances"
            )

        if len(vectors) == 0:
            return []

        with self._lock:
            # Ensure integrity before operation (only once for batch)
            self._ensure_integrity()

            indices: list[int] = []
            last_accepted: tuple[int, float, float] | None = None

            for i, (vector, phase) in enumerate(zip(vectors, phases, strict=True)):
                # Get or create provenance for this vector
                if provenances is not None:
                    provenance = provenances[i]
                else:
                    provenance = MemoryProvenance(
                        source=MemorySource.SYSTEM_PROMPT, confidence=1.0, timestamp=datetime.now()
                    )

                # Check confidence threshold
                if provenance.confidence < self._confidence_threshold:
                    indices.append(-1)  # Reject
                    continue

                # Generate unique memory ID
                memory_id = str(uuid.uuid4())
                # Validate vector type
                if not isinstance(vector, list):
                    raise TypeError(
                        f"vector at index {i} must be a list, got {type(vector).__name__}"
                    )
                if len(vector) != self.dimension:
                    raise ValueError(
                        f"vector at index {i} dimension mismatch: "
                        f"expected {self.dimension}, got {len(vector)}"
                    )

                # Validate phase type and range
                if not isinstance(phase, int | float):
                    raise TypeError(
                        f"phase at index {i} must be numeric, got {type(phase).__name__}"
                    )
                if math.isnan(phase) or math.isinf(phase):
                    raise ValueError(f"phase at index {i} must be a finite number, got {phase}")
                if not (0.0 <= phase <= 1.0):
                    raise ValueError(f"phase at index {i} must be in [0.0, 1.0], got {phase}")

                # Validate vector values using numpy (faster than element-by-element)
                vec_np = np.array(vector, dtype=np.float32)
                if not np.all(np.isfinite(vec_np)):
                    raise ValueError(f"vector at index {i} contains NaN or infinity values")

                norm = max(safe_norm(vec_np), self.MIN_NORM_THRESHOLD)

                # Check capacity and evict if necessary
                if self.size >= self.capacity:
                    self._evict_lowest_confidence()

                idx = self.pointer
                self.memory_bank[idx] = vec_np
                self.phase_bank[idx] = phase
                self.norms[idx] = norm

                # Store provenance metadata
                if idx < len(self._provenance):
                    self._provenance[idx] = provenance
                    self._memory_ids[idx] = memory_id
                else:
                    self._provenance.append(provenance)
                    self._memory_ids.append(memory_id)

                indices.append(idx)
                last_accepted = (idx, float(phase), float(norm))

                # Update pointer with wraparound check
                new_pointer = self.pointer + 1
                if new_pointer >= self.capacity:
                    new_pointer = 0
                self.pointer = new_pointer

            # Update size only once at the end (more efficient)
            # Count only accepted vectors (not -1)
            accepted_count = sum(1 for idx in indices if idx != -1)
            self.size = min(self.size + accepted_count, self.capacity)

            # Update checksum only once after all modifications
            self._checksum = self._compute_checksum()

            # Record observability metrics for batch operation
            if _OBSERVABILITY_AVAILABLE and start_time is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                # Record as single batch operation
                if last_accepted is not None:
                    last_index, last_phase, last_norm = last_accepted
                else:
                    last_index, last_phase, last_norm = 0, 0.0, 0.0
                record_pelm_store(
                    index=last_index,
                    phase=last_phase,
                    vector_norm=last_norm,
                    capacity_used=self.size,
                    capacity_total=self.capacity,
                    memory_bytes=self.memory_usage_bytes(),
                    latency_ms=latency_ms,
                    correlation_id=correlation_id,
                )

            return indices

    @overload
    def retrieve(
        self,
        query_vector: list[float],
        current_phase: float,
        phase_tolerance: float | None = None,
        top_k: int | None = None,
        correlation_id: str | None = None,
        min_confidence: float = 0.0,
        return_indices: Literal[False] = False,
    ) -> list[MemoryRetrieval]: ...

    @overload
    def retrieve(
        self,
        query_vector: list[float],
        current_phase: float,
        phase_tolerance: float | None = None,
        top_k: int | None = None,
        correlation_id: str | None = None,
        min_confidence: float = 0.0,
        return_indices: Literal[True] = True,
    ) -> tuple[list[MemoryRetrieval], list[int]]: ...

    def retrieve(
        self,
        query_vector: list[float],
        current_phase: float,
        phase_tolerance: float | None = None,
        top_k: int | None = None,
        correlation_id: str | None = None,
        min_confidence: float = 0.0,
        return_indices: bool = False,
    ) -> list[MemoryRetrieval] | tuple[list[MemoryRetrieval], list[int]]:
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
                        avg_resonance=None,
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                    )
                if return_indices:
                    return [], []
                return []

            # Optimization: Use pre-allocated buffer with numpy copy
            # Validate dimension first to avoid buffer overflow
            if len(query_vector) != self.dimension:
                raise ValueError(
                    f"query_vector dimension mismatch: expected {self.dimension}, "
                    f"got {len(query_vector)}"
                )
            self._query_buffer[:] = query_vector
            q_vec = self._query_buffer
            q_norm = safe_norm(q_vec)
            if q_norm < self.MIN_NORM_THRESHOLD:
                q_norm = self.MIN_NORM_THRESHOLD

            # Optimize: use in-place operations and avoid intermediate arrays
            phase_diff = np.abs(self.phase_bank[: self.size] - current_phase)
            phase_mask = phase_diff <= phase_tolerance

            # Add confidence filtering
            confidence_mask = np.empty(self.size, dtype=bool)
            provenance_size = len(self._provenance)
            for i in range(self.size):
                confidence_mask[i] = (
                    i < provenance_size and self._provenance[i].confidence >= min_confidence
                )

            # Combine phase and confidence masks
            valid_mask = phase_mask & confidence_mask

            if not np.any(valid_mask):
                # Record empty result due to phase mismatch
                if _OBSERVABILITY_AVAILABLE and start_time is not None:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    record_pelm_retrieve(
                        query_phase=current_phase,
                        phase_tolerance=phase_tolerance,
                        top_k=top_k,
                        results_count=0,
                        avg_resonance=None,
                        latency_ms=latency_ms,
                        correlation_id=correlation_id,
                    )
                if return_indices:
                    return [], []
                return []

            candidates_idx = np.nonzero(valid_mask)[0]
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
            indices: list[int] = []
            resonance_sum = 0.0
            for loc in top_local:
                glob = candidates_idx[loc]
                resonance_value = float(cosine_sims[loc])
                resonance_sum += resonance_value
                # Get provenance (use default if not available for backward compatibility)
                if glob < len(self._provenance):
                    prov = self._provenance[glob]
                    mem_id = self._memory_ids[glob]
                else:
                    # Fallback for memories created before provenance was added
                    prov = MemoryProvenance(
                        source=MemorySource.SYSTEM_PROMPT, confidence=1.0, timestamp=datetime.now()
                    )
                    mem_id = str(uuid.uuid4())

                results.append(
                    MemoryRetrieval(
                        vector=self.memory_bank[glob],
                        phase=self.phase_bank[glob],
                        resonance=resonance_value,
                        provenance=prov,
                        memory_id=mem_id,
                    )
                )
                indices.append(int(glob))

            avg_resonance = resonance_sum / len(results) if results else None

            # Record successful retrieval
            if _OBSERVABILITY_AVAILABLE and start_time is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                record_pelm_retrieve(
                    query_phase=current_phase,
                    phase_tolerance=phase_tolerance,
                    top_k=top_k,
                    results_count=len(results),
                    avg_resonance=avg_resonance,
                    latency_ms=latency_ms,
                    correlation_id=correlation_id,
                )

            if return_indices:
                return results, indices
            return results

    def get_state_stats(self) -> dict[str, int | float]:
        return {
            "capacity": self.capacity,
            "used": self.size,
            "memory_mb": round((self.memory_bank.nbytes + self.phase_bank.nbytes) / 1024**2, 2),
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
        phase_bank_bytes = self.phase_bank.nbytes  # capacity × float32
        norms_bytes = self.norms.nbytes  # capacity × float32

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
        hasher.update(self.memory_bank[: self.size].tobytes())
        hasher.update(self.phase_bank[: self.size].tobytes())
        hasher.update(self.norms[: self.size].tobytes())
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
            self.norms[i] = max(safe_norm(vec), 1e-9)

    def _evict_lowest_confidence(self) -> None:
        """Evict the memory with the lowest confidence score.

        This is called when the memory is at capacity and a new high-confidence
        memory needs to be stored. Sets the pointer to the lowest confidence
        memory slot so it will be overwritten.

        Should only be called from within a lock context.
        """
        if self.size == 0:
            return

        # Find the index with lowest confidence
        confidences = [
            self._provenance[i].confidence if i < len(self._provenance) else 0.0
            for i in range(self.size)
        ]
        min_idx = int(np.argmin(confidences))

        # Simply set pointer to overwrite the lowest confidence slot
        # The normal entangle logic will handle the replacement
        self.pointer = min_idx

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
