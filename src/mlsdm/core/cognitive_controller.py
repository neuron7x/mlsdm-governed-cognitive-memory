import time
from threading import Lock
from typing import Any

import numpy as np
import psutil

from ..cognition.moral_filter_v2 import MoralFilterV2
from ..memory.multi_level_memory import MultiLevelSynapticMemory
from ..memory.qilm_v2 import MemoryRetrieval, QILM_v2
from ..rhythm.cognitive_rhythm import CognitiveRhythm


class CognitiveController:
    def __init__(
        self,
        dim: int = 384,
        memory_threshold_mb: float = 1024.0,
        max_processing_time_ms: float = 1000.0
    ) -> None:
        self.dim = dim
        self._lock = Lock()
        self.moral = MoralFilterV2(initial_threshold=0.50)
        self.qilm = QILM_v2(dimension=dim, capacity=20_000)
        self.rhythm = CognitiveRhythm(wake_duration=8, sleep_duration=3)
        self.synaptic = MultiLevelSynapticMemory(dimension=dim)
        self.step_counter = 0
        # Optimization: Cache for phase values to avoid repeated computation
        self._phase_cache: dict[str, float] = {"wake": 0.1, "sleep": 0.9}
        # Optimization: Cache for frequently accessed state values
        self._state_cache: dict[str, Any] = {}
        self._state_cache_valid = False
        # Memory monitoring and limits
        self.memory_threshold_mb = memory_threshold_mb
        self.max_processing_time_ms = max_processing_time_ms
        self.emergency_shutdown = False
        self._process = psutil.Process()

    def process_event(self, vector: np.ndarray, moral_value: float) -> dict[str, Any]:
        with self._lock:
            # Check emergency shutdown
            if self.emergency_shutdown:
                return self._build_state(rejected=True, note="emergency shutdown")

            start_time = time.perf_counter()
            self.step_counter += 1
            # Optimization: Invalidate state cache when processing
            self._state_cache_valid = False

            # Check memory usage before processing
            memory_mb = self._check_memory_usage()
            if memory_mb > self.memory_threshold_mb:
                self.emergency_shutdown = True
                return self._build_state(rejected=True, note="emergency shutdown: memory exceeded")

            accepted = self.moral.evaluate(moral_value)
            self.moral.adapt(accepted)
            if not accepted:
                return self._build_state(rejected=True, note="morally rejected")
            if not self.rhythm.is_wake():
                return self._build_state(rejected=True, note="sleep phase")

            self.synaptic.update(vector)
            # Optimization: use cached phase value
            phase_val = self._phase_cache[self.rhythm.phase]
            self.qilm.entangle(vector.tolist(), phase=phase_val)
            self.rhythm.step()

            # Check processing time
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms > self.max_processing_time_ms:
                return self._build_state(rejected=True, note=f"processing time exceeded: {elapsed_ms:.2f}ms")

            return self._build_state(rejected=False, note="processed")

    def retrieve_context(self, query_vector: np.ndarray, top_k: int = 5) -> list[MemoryRetrieval]:
        with self._lock:
            # Optimize: use cached phase value
            phase_val = self._phase_cache[self.rhythm.phase]
            return self.qilm.retrieve(query_vector.tolist(), current_phase=phase_val,
                                     phase_tolerance=0.15, top_k=top_k)

    def _check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        memory_info = self._process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB

    def get_memory_usage(self) -> float:
        """Public method to get current memory usage in MB."""
        return self._check_memory_usage()

    def reset_emergency_shutdown(self) -> None:
        """Reset emergency shutdown flag (use with caution)."""
        self.emergency_shutdown = False

    def _build_state(self, rejected: bool, note: str) -> dict[str, Any]:
        # Optimization: Use cached norm calculations when state hasn't changed
        # Only cache when not rejected (rejected responses are cheap anyway)
        if not rejected and self._state_cache_valid and self._state_cache:
            # Use cached values but update step counter and note
            result = self._state_cache.copy()
            result["step"] = self.step_counter
            result["rejected"] = rejected
            result["accepted"] = not rejected
            result["note"] = note
            return result

        # Calculate fresh state
        l1, l2, l3 = self.synaptic.state()

        # Optimization: Compute norms in a single pass when possible
        # Pre-allocate result dict to avoid resizing
        result = {
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
            "accepted": not rejected,
            "note": note
        }

        # Cache result for accepted events
        if not rejected:
            self._state_cache = result.copy()
            self._state_cache_valid = True

        return result
