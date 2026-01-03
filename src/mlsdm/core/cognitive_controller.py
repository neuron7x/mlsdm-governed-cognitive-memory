import logging
import time
from collections.abc import Callable
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import psutil

from ..memory.phase_entangled_lattice_memory import MemoryRetrieval
from ..observability.metrics import get_metrics_exporter
from ..observability.tracing import get_tracer_manager
from .governance_kernel import GovernanceKernel, PelmRO

if TYPE_CHECKING:
    from mlsdm.config import SynapticMemoryCalibration

# Import recovery calibration parameters and synaptic memory defaults
# Type annotations use X | None since module may not be available
_SYNAPTIC_MEMORY_DEFAULTS: "SynapticMemoryCalibration | None" = None
_get_synaptic_memory_config: (
    Callable[[dict[str, Any] | None], "SynapticMemoryCalibration"] | None
) = None

# Default max memory bytes: 1.4 GB
_DEFAULT_MAX_MEMORY_BYTES = int(1.4 * 1024**3)

try:
    from mlsdm.config import (
        COGNITIVE_CONTROLLER_DEFAULTS,
    )
    from mlsdm.config import (
        SYNAPTIC_MEMORY_DEFAULTS as _IMPORTED_DEFAULTS,
    )
    from mlsdm.config import (
        get_synaptic_memory_config as _imported_get_config,
    )

    _CC_RECOVERY_COOLDOWN_STEPS = COGNITIVE_CONTROLLER_DEFAULTS.recovery_cooldown_steps
    _CC_RECOVERY_MEMORY_SAFETY_RATIO = COGNITIVE_CONTROLLER_DEFAULTS.recovery_memory_safety_ratio
    _CC_RECOVERY_MAX_ATTEMPTS = COGNITIVE_CONTROLLER_DEFAULTS.recovery_max_attempts
    _CC_MAX_MEMORY_BYTES = COGNITIVE_CONTROLLER_DEFAULTS.max_memory_bytes
    _CC_AUTO_RECOVERY_ENABLED = COGNITIVE_CONTROLLER_DEFAULTS.auto_recovery_enabled
    _CC_AUTO_RECOVERY_COOLDOWN_SECONDS = (
        COGNITIVE_CONTROLLER_DEFAULTS.auto_recovery_cooldown_seconds
    )
    _SYNAPTIC_MEMORY_DEFAULTS = _IMPORTED_DEFAULTS
    _get_synaptic_memory_config = _imported_get_config
except ImportError:
    # Fallback defaults if calibration module is not available
    _CC_RECOVERY_COOLDOWN_STEPS = 10
    _CC_RECOVERY_MEMORY_SAFETY_RATIO = 0.8
    _CC_RECOVERY_MAX_ATTEMPTS = 3
    _CC_MAX_MEMORY_BYTES = _DEFAULT_MAX_MEMORY_BYTES
    _CC_AUTO_RECOVERY_ENABLED = True
    _CC_AUTO_RECOVERY_COOLDOWN_SECONDS = 60.0

logger = logging.getLogger(__name__)


class CognitiveController:
    def __init__(
        self,
        dim: int = 384,
        memory_threshold_mb: float = 8192.0,
        max_processing_time_ms: float = 1000.0,
        *,
        max_memory_bytes: int | None = None,
        synaptic_config: "SynapticMemoryCalibration | None" = None,
        yaml_config: dict[str, Any] | None = None,
        auto_recovery_enabled: bool | None = None,
        auto_recovery_cooldown_seconds: float | None = None,
    ) -> None:
        """Initialize the CognitiveController.

        Args:
            dim: Vector dimension for embeddings.
            memory_threshold_mb: Memory threshold in MB before emergency shutdown.
            max_processing_time_ms: Maximum processing time in ms per event.
            max_memory_bytes: Global memory bound in bytes for cognitive circuit
                (PELM + SynapticMemory + controller buffers). Defaults to 1.4 GB.
                This is the hard limit from CORE-04 specification.
            synaptic_config: Optional SynapticMemoryCalibration for synaptic memory.
                If provided, uses these parameters for MultiLevelSynapticMemory.
            yaml_config: Optional YAML config dictionary. If provided and
                synaptic_config is None, loads synaptic memory config from
                'multi_level_memory' section merged with SYNAPTIC_MEMORY_DEFAULTS.
            auto_recovery_enabled: Enable time-based auto-recovery after emergency
                shutdown. When True, controller will attempt recovery after
                auto_recovery_cooldown_seconds have passed. Defaults to True.
            auto_recovery_cooldown_seconds: Time in seconds to wait before attempting
             automatic recovery after emergency shutdown. Defaults to 60.0.
        """
        self.dim = dim
        self._lock = Lock()

        # Resolve synaptic memory configuration:
        # Priority: synaptic_config > yaml_config > SYNAPTIC_MEMORY_DEFAULTS
        resolved_config: SynapticMemoryCalibration | None = synaptic_config
        if resolved_config is None and yaml_config is not None:
            if _get_synaptic_memory_config is not None:
                resolved_config = _get_synaptic_memory_config(yaml_config)
        if resolved_config is None:
            resolved_config = _SYNAPTIC_MEMORY_DEFAULTS

        self._kernel = GovernanceKernel(
            dim=dim,
            capacity=20_000,
            wake_duration=8,
            sleep_duration=3,
            initial_moral_threshold=0.50,
            synaptic_config=resolved_config,
        )
        self._bind_kernel_views()
        self.step_counter = 0
        # Optimization: Cache for phase values to avoid repeated computation
        self._phase_cache: dict[str, float] = {"wake": 0.1, "sleep": 0.9}
        # Optimization: Cache for frequently accessed state values
        self._state_cache: dict[str, Any] = {}
        self._state_cache_valid = False
        # Memory monitoring and limits
        self.memory_threshold_mb = memory_threshold_mb
        self.max_processing_time_ms = max_processing_time_ms
        # Global memory bound (CORE-04): PELM + Synaptic + controller buffers
        self.max_memory_bytes = (
            max_memory_bytes if max_memory_bytes is not None else _CC_MAX_MEMORY_BYTES
        )
        self.emergency_shutdown = False
        self._emergency_reason: str | None = None
        self._process = psutil.Process()
        # Auto-recovery state tracking
        self._last_emergency_step: int = 0
        self._recovery_attempts: int = 0
        # Time-based auto-recovery (REL-001)
        self._last_emergency_time: float = 0.0
        self.auto_recovery_enabled = (
            auto_recovery_enabled
            if auto_recovery_enabled is not None
            else _CC_AUTO_RECOVERY_ENABLED
        )
        self.auto_recovery_cooldown_seconds = (
            auto_recovery_cooldown_seconds
            if auto_recovery_cooldown_seconds is not None
            else _CC_AUTO_RECOVERY_COOLDOWN_SECONDS
        )

    def _bind_kernel_views(self) -> None:
        """Expose read-only proxies from the governance kernel."""
        self.moral = self._kernel.moral_ro
        self.synaptic = self._kernel.synaptic_ro
        self.pelm = self._kernel.pelm_ro
        self.rhythm = self._kernel.rhythm_ro

    def rhythm_step(self) -> None:
        """Advance rhythm via governance kernel."""
        self._kernel.rhythm_step()

    def moral_adapt(self, accepted: bool) -> None:
        """Adapt moral filter via governance kernel."""
        self._kernel.moral_adapt(accepted)

    def memory_commit(
        self, vector: np.ndarray, phase: float, *, provenance: Any | None = None
    ) -> None:
        """Commit memory via governance kernel."""
        self._kernel.memory_commit(vector, phase, provenance=provenance)

    @property
    def qilm(self) -> PelmRO:
        """Backward compatibility alias for pelm (deprecated, use self.pelm instead).

        This property will be removed in v2.0.0. Migrate to using self.pelm directly.
        """
        return self.pelm

    def memory_usage_bytes(self) -> int:
        """Calculate total memory usage for cognitive circuit in bytes.

        Aggregates memory usage from:
        - PELM (Phase-Entangled Lattice Memory)
        - MultiLevelSynapticMemory (L1/L2/L3)
        - Controller internal buffers and overhead

        Returns:
            Total estimated memory usage in bytes (conservative estimate).

        Note:
            This method is thread-safe and can be called from outside the lock.
            Used for enforcing the global memory bound (CORE-04 invariant).
        """
        pelm_bytes = self.pelm.memory_usage_bytes()
        synaptic_bytes = self.synaptic.memory_usage_bytes()

        # Controller internal overhead (caches, state, locks, etc.)
        # Estimate: phase_cache dict, state_cache dict, misc Python object overhead
        controller_overhead = 4096  # ~4KB for internal structures

        return pelm_bytes + synaptic_bytes + controller_overhead

    def get_phase(self) -> str:
        """Get the current cognitive phase from rhythm.

        Read-only method for introspection - no side effects.

        Returns:
            Current phase as string (e.g., "wake", "sleep").
        """
        return self.rhythm.phase

    def get_step_counter(self) -> int:
        """Get the current step counter.

        Read-only method for introspection - no side effects.

        Returns:
            Current step count (number of processed events).
        """
        return self.step_counter

    def is_emergency_shutdown(self) -> bool:
        """Check if the controller is in emergency shutdown state.

        Read-only method for introspection - no side effects.

        Returns:
            True if emergency shutdown is active, False otherwise.
        """
        return self.emergency_shutdown

    def process_event(self, vector: np.ndarray, moral_value: float) -> dict[str, Any]:
        """Process a cognitive event with full observability tracing.

        This method wraps event processing with OpenTelemetry spans for
        visibility into the cognitive pipeline.

        Args:
            vector: Input embedding vector
            moral_value: Moral score for this interaction (0.0-1.0)

        Returns:
            Dictionary with processing state and results
        """
        # Get tracer manager for spans (graceful fallback if tracing disabled)
        tracer_manager = get_tracer_manager()

        with self._lock:  # noqa: SIM117 - Lock must be held for entire operation
            # Create span for the entire process_event operation
            with tracer_manager.start_span(
                "cognitive_controller.process_event",
                attributes={
                    "mlsdm.step": self.step_counter + 1,
                    "mlsdm.moral_value": moral_value,
                    "mlsdm.emergency_shutdown": self.emergency_shutdown,
                },
            ) as event_span:
                # Check emergency shutdown and attempt auto-recovery if applicable
                if self.emergency_shutdown:
                    steps_since_emergency = self.step_counter - self._last_emergency_step
                    time_since_emergency = (
                        time.time() - self._last_emergency_time
                        if self._last_emergency_time > 0
                        else 0.0
                    )
                    logger.info(
                        "Emergency shutdown active; evaluating auto-recovery "
                        f"(reason={self._emergency_reason}, steps_since={steps_since_emergency}, "
                        f"time_since={time_since_emergency:.1f}s, "
                        f"recovery_attempts={self._recovery_attempts})"
                    )
                    if self._try_auto_recovery():
                        logger.info(
                            "auto-recovery succeeded after emergency_shutdown "
                            f"(cooldown_steps={self.step_counter - self._last_emergency_step}, "
                            f"recovery_attempt={self._recovery_attempts})"
                        )
                        event_span.set_attribute("mlsdm.auto_recovery", True)
                    else:
                        logger.debug(
                            "Auto-recovery conditions not met; rejecting event "
                            f"(reason={self._emergency_reason}, steps_since={steps_since_emergency}, "
                            f"time_since={time_since_emergency:.1f}s)"
                        )
                        event_span.set_attribute("mlsdm.rejected", True)
                        event_span.set_attribute("mlsdm.rejected_reason", "emergency_shutdown")
                        return self._build_state(rejected=True, note="emergency shutdown")

                start_time = time.perf_counter()
                self.step_counter += 1
                # Optimization: Invalidate state cache when processing
                self._state_cache_valid = False

                # Check memory usage before processing (psutil-based, legacy)
                memory_mb = self._check_memory_usage()
                if memory_mb > self.memory_threshold_mb:
                    logger.info(
                        "Entering emergency shutdown due to process memory threshold "
                        f"(memory_mb={memory_mb:.2f}, threshold_mb={self.memory_threshold_mb:.2f})"
                    )
                    self._enter_emergency_shutdown("process_memory_exceeded")
                    event_span.set_attribute("mlsdm.rejected", True)
                    event_span.set_attribute("mlsdm.rejected_reason", "memory_exceeded")
                    event_span.set_attribute("mlsdm.emergency_shutdown", True)
                    return self._build_state(
                        rejected=True, note="emergency shutdown: memory exceeded"
                    )

                # Moral evaluation with tracing
                with tracer_manager.start_span(
                    "cognitive_controller.moral_filter",
                    attributes={
                        "mlsdm.moral_value": moral_value,
                    },
                ) as moral_span:
                    accepted, threshold_used = self._kernel.evaluate_moral(moral_value)
                    moral_span.set_attribute("mlsdm.moral.accepted", accepted)
                    moral_span.set_attribute("mlsdm.moral_threshold", threshold_used)

                    if not accepted:
                        event_span.set_attribute("mlsdm.rejected", True)
                        event_span.set_attribute("mlsdm.rejected_reason", "morally_rejected")
                        return self._build_state(rejected=True, note="morally rejected")

                # Check cognitive phase
                if not self.rhythm.is_wake():
                    event_span.set_attribute("mlsdm.rejected", True)
                    event_span.set_attribute("mlsdm.rejected_reason", "sleep_phase")
                    event_span.set_attribute("mlsdm.phase", "sleep")
                    return self._build_state(rejected=True, note="sleep phase")

                event_span.set_attribute("mlsdm.phase", "wake")

                # Memory update with tracing
                with tracer_manager.start_span(
                    "cognitive_controller.memory_update",
                    attributes={
                        "mlsdm.phase": self.rhythm.phase,
                    },
                ) as memory_span:
                    # Optimization: use cached phase value
                    phase_val = self._phase_cache[self.rhythm.phase]
                    self.memory_commit(vector, phase_val)
                    memory_span.set_attribute(
                        "mlsdm.pelm_used", self.pelm.get_state_stats()["used"]
                    )

                self.rhythm_step()

                # Check global memory bound (CORE-04) after memory-modifying operations
                current_memory_bytes = self.memory_usage_bytes()
                if current_memory_bytes > self.max_memory_bytes:
                    logger.info(
                        "Entering emergency shutdown due to global memory limit "
                        f"(current_bytes={current_memory_bytes}, max_bytes={self.max_memory_bytes})"
                    )
                    self._enter_emergency_shutdown("memory_limit_exceeded")
                    logger.warning(
                        f"Global memory limit exceeded: {current_memory_bytes} > {self.max_memory_bytes} bytes. "
                        "Emergency shutdown triggered."
                    )
                    event_span.set_attribute("mlsdm.rejected", True)
                    event_span.set_attribute("mlsdm.rejected_reason", "memory_limit_exceeded")
                    event_span.set_attribute("mlsdm.emergency_shutdown", True)
                    return self._build_state(
                        rejected=True, note="emergency shutdown: global memory limit exceeded"
                    )

                # Check processing time
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                event_span.set_attribute("mlsdm.processing_time_ms", elapsed_ms)

                if elapsed_ms > self.max_processing_time_ms:
                    event_span.set_attribute("mlsdm.rejected", True)
                    event_span.set_attribute("mlsdm.rejected_reason", "processing_timeout")
                    return self._build_state(
                        rejected=True, note=f"processing time exceeded: {elapsed_ms:.2f}ms"
                    )

                # Success
                event_span.set_attribute("mlsdm.accepted", True)
                return self._build_state(rejected=False, note="processed")

    def retrieve_context(self, query_vector: np.ndarray, top_k: int = 5) -> list[MemoryRetrieval]:
        """Retrieve context from memory with tracing.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to retrieve

        Returns:
            List of MemoryRetrieval objects
        """
        tracer_manager = get_tracer_manager()

        with self._lock:  # noqa: SIM117 - Lock must be held for entire operation
            with tracer_manager.start_span(
                "cognitive_controller.retrieve_context",
                attributes={
                    "mlsdm.top_k": top_k,
                    "mlsdm.phase": self.rhythm.phase,
                },
            ) as span:
                # Optimize: use cached phase value
                phase_val = self._phase_cache[self.rhythm.phase]
                results = cast(
                    "list[MemoryRetrieval]",
                    self.pelm.retrieve(
                        query_vector.tolist(),
                        current_phase=phase_val,
                        phase_tolerance=0.15,
                        top_k=top_k,
                    ),
                )
                span.set_attribute("mlsdm.results_count", len(results))
                return results

    def _check_memory_usage(self) -> float:
        """Check current memory usage in MB."""
        memory_info = self._process.memory_info()
        return float(memory_info.rss / (1024 * 1024))  # Convert bytes to MB

    def get_memory_usage(self) -> float:
        """Public method to get current memory usage in MB."""
        return self._check_memory_usage()

    def reset_emergency_shutdown(self) -> None:
        """Reset emergency shutdown flag (use with caution).

        This also resets recovery attempt counter and time tracking,
        allowing auto-recovery to function again if the controller
        enters emergency state again.
        """
        self.emergency_shutdown = False
        self._emergency_reason = None
        self._recovery_attempts = 0
        self._last_emergency_time = 0.0
        logger.info("Emergency shutdown reset manually")
        try:
            metrics_exporter = get_metrics_exporter()
        except Exception:
            logger.debug(
                "Failed to initialize metrics exporter for manual emergency reset",
                exc_info=True,
            )
        else:
            metrics_exporter.set_emergency_shutdown_active(False)

    def _enter_emergency_shutdown(self, reason: str = "unknown") -> None:
        """Enter emergency shutdown state and record the step and time.

        Args:
            reason: The reason for emergency shutdown (e.g., 'memory_limit_exceeded').
        """
        self.emergency_shutdown = True
        self._emergency_reason = reason
        self._last_emergency_step = self.step_counter
        self._last_emergency_time = time.time()
        self._recovery_attempts += 1
        logger.warning(f"Emergency shutdown entered: reason={reason}, step={self.step_counter}")
        try:
            metrics_exporter = get_metrics_exporter()
        except Exception:
            logger.debug(
                "Failed to initialize metrics exporter for emergency shutdown",
                exc_info=True,
            )
        else:
            metrics_exporter.increment_emergency_shutdown(reason)
            metrics_exporter.set_emergency_shutdown_active(True)

    def _try_auto_recovery(self) -> bool:
        """Attempt automatic recovery from emergency shutdown.

        Returns:
            True if recovery succeeded and emergency_shutdown was cleared,
            False if recovery conditions are not met.

        Recovery requires:
        1. Either step-based cooldown OR time-based cooldown has passed
           - Step-based: step_counter - _last_emergency_step >= cooldown_steps
           - Time-based: time.time() - _last_emergency_time >= cooldown_seconds
             (only if auto_recovery_enabled is True)
        2. Memory usage is below safety threshold
        3. Recovery attempts have not exceeded the maximum limit
        """
        # Guard: check if max recovery attempts exceeded
        if self._recovery_attempts >= _CC_RECOVERY_MAX_ATTEMPTS:
            self._record_auto_recovery("failure", "max_attempts_exceeded")
            return False

        # Check cooldown period (step-based OR time-based)
        steps_since_emergency = self.step_counter - self._last_emergency_step
        step_cooldown_passed = steps_since_emergency >= _CC_RECOVERY_COOLDOWN_STEPS

        time_cooldown_passed = False
        if self.auto_recovery_enabled and self._last_emergency_time > 0:
            time_since_emergency = time.time() - self._last_emergency_time
            time_cooldown_passed = time_since_emergency >= self.auto_recovery_cooldown_seconds

        if not (step_cooldown_passed or time_cooldown_passed):
            self._record_auto_recovery("failure", "cooldown_pending")
            return False

        # Health check: verify memory is within safe limits
        memory_mb = self._check_memory_usage()
        memory_safety_threshold = self.memory_threshold_mb * _CC_RECOVERY_MEMORY_SAFETY_RATIO
        if memory_mb > memory_safety_threshold:
            self._record_auto_recovery("failure", "memory_above_safety_threshold")
            return False

        # All conditions met - perform recovery
        self.emergency_shutdown = False
        recovery_mode = "time" if time_cooldown_passed and not step_cooldown_passed else "step"
        logger.info(
            f"Auto-recovery succeeded via {recovery_mode}-based cooldown "
            f"(steps_since={steps_since_emergency}, time_since={time.time() - self._last_emergency_time:.1f}s)"
        )
        self._record_auto_recovery("success", "recovered")
        return True

    def _record_auto_recovery(self, result: str, reason: str) -> None:
        """Record observability for auto-recovery attempts."""
        logger.debug("Auto-recovery result=%s reason=%s", result, reason)
        try:
            metrics_exporter = get_metrics_exporter()
        except Exception:
            logger.debug("Failed to initialize metrics exporter for auto-recovery", exc_info=True)
            return
        metrics_exporter.increment_auto_recovery(result)
        metrics_exporter.set_emergency_shutdown_active(result != "success")

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
                "L3": float(np.linalg.norm(l3)),
            },
            "pelm_used": self.pelm.get_state_stats()["used"],
            # Backward compatibility (deprecated)
            "qilm_used": self.pelm.get_state_stats()["used"],
            "rejected": rejected,
            "accepted": not rejected,
            "note": note,
        }

        # Cache result for accepted events
        if not rejected:
            self._state_cache = result.copy()
            self._state_cache_valid = True

        return result
