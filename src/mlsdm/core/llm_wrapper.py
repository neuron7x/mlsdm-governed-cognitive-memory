"""
Universal LLM Wrapper with Cognitive Memory Governance

This module provides a production-ready wrapper around any LLM that enforces:
1. Hard memory limits (20k vectors, ≤1.4 GB RAM)
2. Adaptive moral homeostasis
3. Circadian rhythm with wake/sleep cycles
4. Multi-level synaptic memory
5. Phase-entangling retrieval

Prevents LLM degradation, memory bloat, toxicity, and identity loss.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable  # noqa: TC003 - used at runtime in type hints
from enum import Enum
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..cognition.moral_filter_v2 import MoralFilterV2
from ..memory.multi_level_memory import MultiLevelSynapticMemory
from ..memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory
from ..observability.tracing import get_tracer_manager
from ..rhythm.cognitive_rhythm import CognitiveRhythm
from ..speech.governance import (  # noqa: TC001 - used at runtime in function signatures
    SpeechGovernanceResult,
    SpeechGovernor,
)
from .cognitive_state import CognitiveState

if TYPE_CHECKING:
    from config.calibration import (
        CognitiveRhythmCalibration,
        PELMCalibration,
        ReliabilityCalibration,
    )

# Import calibration defaults - these can be overridden via config
# Type hints use Optional to allow None when calibration module unavailable
COGNITIVE_RHYTHM_DEFAULTS: CognitiveRhythmCalibration | None
PELM_DEFAULTS: PELMCalibration | None
RELIABILITY_DEFAULTS: ReliabilityCalibration | None

try:
    from config.calibration import (
        COGNITIVE_RHYTHM_DEFAULTS,
        PELM_DEFAULTS,
        RELIABILITY_DEFAULTS,
    )
except ImportError:
    COGNITIVE_RHYTHM_DEFAULTS = None
    PELM_DEFAULTS = None
    RELIABILITY_DEFAULTS = None

_logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for embedding function."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Too many failures, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for embedding function failures."""

    # Default values from calibration
    DEFAULT_FAILURE_THRESHOLD = (
        RELIABILITY_DEFAULTS.circuit_breaker_failure_threshold
        if RELIABILITY_DEFAULTS
        else 5
    )
    DEFAULT_RECOVERY_TIMEOUT = (
        RELIABILITY_DEFAULTS.circuit_breaker_recovery_timeout
        if RELIABILITY_DEFAULTS
        else 60.0
    )
    DEFAULT_SUCCESS_THRESHOLD = (
        RELIABILITY_DEFAULTS.circuit_breaker_success_threshold
        if RELIABILITY_DEFAULTS
        else 2
    )

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
        success_threshold: int | None = None,
    ):
        self.failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else self.DEFAULT_FAILURE_THRESHOLD
        )
        self.recovery_timeout = (
            recovery_timeout
            if recovery_timeout is not None
            else self.DEFAULT_RECOVERY_TIMEOUT
        )
        self.success_threshold = (
            success_threshold
            if success_threshold is not None
            else self.DEFAULT_SUCCESS_THRESHOLD
        )

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerState.CLOSED
        self._lock = Lock()

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise RuntimeError("Circuit breaker is OPEN - too many embedding failures")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = 0
            return result
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
            raise e

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0.0


class LLMWrapper:
    """
    Production-ready wrapper for any LLM with cognitive memory governance.

    This wrapper sits between the user and the LLM, enforcing biological constraints
    and maintaining memory coherence across thousands of interactions.

    Key Features:
    - Zero-allocation memory after initialization (20k vector capacity)
    - Adaptive moral filtering without RLHF
    - Circadian rhythm with forced short responses during sleep
    - Context retrieval from phase-entangled memory
    - Thread-safe for concurrent requests
    - Retry logic with exponential backoff for LLM calls
    - Circuit breaker for embedding failures
    - Graceful degradation to stateless mode on QILM failures

    Usage:
        wrapper = LLMWrapper(
            llm_generate_fn=my_llm.generate,
            embedding_fn=my_embedder.encode,
            dim=384
        )

        response = wrapper.generate(
            prompt="Hello, how are you?",
            moral_value=0.8
        )
    """

    # Default values from calibration
    MAX_WAKE_TOKENS = (
        COGNITIVE_RHYTHM_DEFAULTS.max_wake_tokens
        if COGNITIVE_RHYTHM_DEFAULTS
        else 2048
    )
    MAX_SLEEP_TOKENS = (
        COGNITIVE_RHYTHM_DEFAULTS.max_sleep_tokens
        if COGNITIVE_RHYTHM_DEFAULTS
        else 150
    )  # Forced short responses during sleep
    DEFAULT_LLM_TIMEOUT = (
        RELIABILITY_DEFAULTS.llm_timeout if RELIABILITY_DEFAULTS else 30.0
    )
    DEFAULT_LLM_RETRY_ATTEMPTS = (
        RELIABILITY_DEFAULTS.llm_retry_attempts if RELIABILITY_DEFAULTS else 3
    )
    DEFAULT_PELM_FAILURE_THRESHOLD = (
        RELIABILITY_DEFAULTS.pelm_failure_threshold if RELIABILITY_DEFAULTS else 3
    )
    DEFAULT_PHASE_TOLERANCE = (
        PELM_DEFAULTS.phase_tolerance if PELM_DEFAULTS else 0.15
    )
    WAKE_PHASE = PELM_DEFAULTS.wake_phase if PELM_DEFAULTS else 0.1
    SLEEP_PHASE = PELM_DEFAULTS.sleep_phase if PELM_DEFAULTS else 0.9

    def __init__(
        self,
        llm_generate_fn: Callable[[str, int], str],
        embedding_fn: Callable[[str], np.ndarray],
        dim: int = 384,
        capacity: int | None = None,
        wake_duration: int | None = None,
        sleep_duration: int | None = None,
        initial_moral_threshold: float | None = None,
        llm_timeout: float | None = None,
        llm_retry_attempts: int | None = None,
        speech_governor: SpeechGovernor | None = None,
    ):
        """
        Initialize the LLM wrapper with cognitive governance.

        Args:
            llm_generate_fn: Function that takes (prompt, max_tokens) and returns response
            embedding_fn: Function that takes text and returns embedding vector
            dim: Embedding dimension (default 384)
            capacity: Maximum memory vectors (default from calibration)
            wake_duration: Wake cycle duration in steps (default from calibration)
            sleep_duration: Sleep cycle duration in steps (default from calibration)
            initial_moral_threshold: Starting moral threshold (default from calibration)
            llm_timeout: Timeout for LLM calls in seconds (default from calibration)
            llm_retry_attempts: Number of retry attempts for LLM calls (default from calibration)
            speech_governor: Optional speech governance policy (default None)
        """
        # Apply calibration defaults
        defaults = self._apply_calibration_defaults(
            capacity, wake_duration, sleep_duration, llm_timeout, llm_retry_attempts
        )
        capacity, wake_duration, sleep_duration, llm_timeout, llm_retry_attempts = defaults
        # Initialize core parameters
        self._init_core_params(dim, llm_timeout, llm_retry_attempts)
        # Initialize core components
        self._init_core_components(
            llm_generate_fn, embedding_fn, dim, capacity,
            wake_duration, sleep_duration, initial_moral_threshold, speech_governor
        )
        # Initialize reliability components
        self._init_reliability()
        # Initialize state tracking
        self._init_state_tracking()

    def _apply_calibration_defaults(
        self,
        capacity: int | None,
        wake_duration: int | None,
        sleep_duration: int | None,
        llm_timeout: float | None,
        llm_retry_attempts: int | None,
    ) -> tuple[int, int, int, float, int]:
        """Apply calibration defaults where parameters not specified."""
        if capacity is None:
            capacity = PELM_DEFAULTS.default_capacity if PELM_DEFAULTS else 20_000
        if wake_duration is None:
            wake_duration = (
                COGNITIVE_RHYTHM_DEFAULTS.wake_duration
                if COGNITIVE_RHYTHM_DEFAULTS
                else 8
            )
        if sleep_duration is None:
            sleep_duration = (
                COGNITIVE_RHYTHM_DEFAULTS.sleep_duration
                if COGNITIVE_RHYTHM_DEFAULTS
                else 3
            )
        if llm_timeout is None:
            llm_timeout = self.DEFAULT_LLM_TIMEOUT
        if llm_retry_attempts is None:
            llm_retry_attempts = self.DEFAULT_LLM_RETRY_ATTEMPTS
        return capacity, wake_duration, sleep_duration, llm_timeout, llm_retry_attempts

    def _init_core_params(
        self,
        dim: int,
        llm_timeout: float,
        llm_retry_attempts: int,
    ) -> None:
        """Initialize core parameters and lock."""
        self.dim = dim
        self._lock = Lock()
        self.llm_timeout = llm_timeout
        self.llm_retry_attempts = llm_retry_attempts

    def _init_core_components(
        self,
        llm_generate_fn: Callable[[str, int], str],
        embedding_fn: Callable[[str], np.ndarray],
        dim: int,
        capacity: int,
        wake_duration: int,
        sleep_duration: int,
        initial_moral_threshold: float | None,
        speech_governor: SpeechGovernor | None,
    ) -> None:
        """Initialize core components: LLM, embedding, moral filter, memory, rhythm."""
        self.llm_generate = llm_generate_fn
        self.embed = embedding_fn
        self.moral = MoralFilterV2(initial_threshold=initial_moral_threshold)
        self.pelm = PhaseEntangledLatticeMemory(dimension=dim, capacity=capacity)
        self.rhythm = CognitiveRhythm(wake_duration=wake_duration, sleep_duration=sleep_duration)
        self.synaptic = MultiLevelSynapticMemory(dimension=dim)
        self._speech_governor = speech_governor

    def _init_reliability(self) -> None:
        """Initialize reliability components (circuit breaker, stateless mode flag)."""
        self.embedding_circuit_breaker = CircuitBreaker()
        self.stateless_mode = False

    def _init_state_tracking(self) -> None:
        """Initialize state tracking counters and buffers."""
        self.step_counter = 0
        self.rejected_count = 0
        self.accepted_count = 0
        self.consolidation_buffer: list[np.ndarray] = []
        self.pelm_failure_count = 0
        self.embedding_failure_count = 0
        self.llm_failure_count = 0

    @property
    def qilm_failure_count(self) -> int:
        """Backward compatibility alias for pelm_failure_count (deprecated, use pelm_failure_count instead).

        This property will be removed in v2.0.0. Migrate to using pelm_failure_count directly.
        """
        return self.pelm_failure_count

    @qilm_failure_count.setter
    def qilm_failure_count(self, value: int) -> None:
        """Backward compatibility setter for pelm_failure_count."""
        self.pelm_failure_count = value

    def _llm_generate_with_retry(self, prompt: str, max_tokens: int) -> str:
        """
        Generate LLM response with retry logic and exponential backoff.

        Uses tenacity to retry on common transient errors with exponential backoff.

        Note: Timeout detection occurs after the LLM call completes. This is a pragmatic
        approach that works for most LLM APIs which have their own internal timeouts.
        For true preemptive timeout, the LLM client should implement timeout in the
        llm_generate_fn itself (e.g., using requests timeout, asyncio timeout, etc.).
        """
        @retry(
            retry=retry_if_exception_type((TimeoutError, ConnectionError, RuntimeError)),
            stop=stop_after_attempt(self.llm_retry_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True
        )
        def _generate_with_timeout() -> str:
            start_time = time.time()
            result = self.llm_generate(prompt, max_tokens)
            elapsed = time.time() - start_time

            # Post-call timeout detection for monitoring and retry trigger
            if elapsed > self.llm_timeout:
                raise TimeoutError(f"LLM call exceeded timeout: {elapsed:.2f}s > {self.llm_timeout}s")

            return result

        try:
            return _generate_with_timeout()
        except Exception as e:
            self.llm_failure_count += 1
            raise e

    def _embed_with_circuit_breaker(self, text: str) -> np.ndarray:
        """
        Embed text with circuit breaker protection.

        Circuit breaker prevents cascading failures from embedding service.
        """
        try:
            # Circuit breaker returns Any; we validate it's ndarray or convert
            result: Any = self.embedding_circuit_breaker.call(self.embed, text)
            if not isinstance(result, np.ndarray):
                result = np.array(result, dtype=np.float32)
            # Cast to proper return type after validation
            return cast("np.ndarray", result.astype(np.float32))
        except Exception as e:
            self.embedding_failure_count += 1
            raise e

    def _safe_pelm_operation(
        self,
        operation: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute PELM operation with graceful degradation.

        If PELM fails repeatedly, switches to stateless mode.
        Returns empty/default values in stateless mode to allow processing to continue.
        """
        # Sentinel value for failed entangle operations
        ENTANGLE_FAILED = -1

        if self.stateless_mode:
            # In stateless mode, return empty results
            if operation == "retrieve":
                return []
            elif operation == "entangle":
                return ENTANGLE_FAILED
            return None

        try:
            if operation == "retrieve":
                return self.pelm.retrieve(*args, **kwargs)
            elif operation == "entangle":
                return self.pelm.entangle(*args, **kwargs)
            else:
                raise ValueError(f"Unknown PELM operation: {operation}")
        except (MemoryError, RuntimeError) as e:
            self.pelm_failure_count += 1
            if self.pelm_failure_count >= self.DEFAULT_PELM_FAILURE_THRESHOLD:
                # Switch to stateless mode after repeated failures
                self.stateless_mode = True
            raise e

    def generate(
        self,
        prompt: str,
        moral_value: float,
        max_tokens: int | None = None,
        context_top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Generate LLM response with cognitive governance and reliability features.

        This method:
        1. Embeds the prompt (with circuit breaker)
        2. Checks moral acceptability
        3. Retrieves relevant context from memory (with graceful degradation)
        4. Generates response with retry and timeout handling
        5. Updates memory and cognitive state

        Args:
            prompt: User input text
            moral_value: Moral score for this interaction (0.0-1.0)
            max_tokens: Optional max tokens override
            context_top_k: Number of memory items to retrieve for context

        Returns:
            Dictionary with:
                - response: Generated text
                - accepted: Whether morally accepted
                - phase: Current cognitive phase
                - step: Current step counter
                - note: Processing note
                - stateless_mode: Whether running in degraded mode
        """
        # Get tracer manager for spans (fallback to no-op if tracing disabled)
        tracer_manager = get_tracer_manager()

        with self._lock:
            self.step_counter += 1

            # Step 1: Moral evaluation and phase check
            with tracer_manager.start_span(
                "llm_wrapper.moral_filter",
                attributes={
                    "mlsdm.moral_value": moral_value,
                    "mlsdm.moral_threshold": self.moral.threshold,
                },
            ) as moral_span:
                rejection = self._check_moral_and_phase(moral_value)
                if rejection is not None:
                    moral_span.set_attribute("mlsdm.moral.accepted", False)
                    moral_span.set_attribute("mlsdm.moral.rejection_reason", rejection.get("note", ""))
                    return rejection
                moral_span.set_attribute("mlsdm.moral.accepted", True)

            # Step 2: Embed prompt
            embed_result = self._embed_and_validate_prompt(prompt)
            if isinstance(embed_result, dict):
                return embed_result
            prompt_vector = embed_result

            # Step 3: Retrieve context and build enhanced prompt
            is_wake = self.rhythm.is_wake()
            phase_val = self.WAKE_PHASE if is_wake else self.SLEEP_PHASE
            with tracer_manager.start_span(
                "llm_wrapper.memory_retrieval",
                attributes={
                    "mlsdm.phase": "wake" if is_wake else "sleep",
                    "mlsdm.context_top_k": context_top_k or 5,
                },
            ) as memory_span:
                memories, enhanced_prompt = self._retrieve_and_build_context(
                    prompt, prompt_vector, phase_val, context_top_k
                )
                memory_span.set_attribute("mlsdm.context_items_retrieved", len(memories))
                memory_span.set_attribute("mlsdm.stateless_mode", self.stateless_mode)

            # Step 4: Determine max tokens
            max_tokens = self._determine_max_tokens(max_tokens, is_wake)

            # Step 5: Generate and govern response
            with tracer_manager.start_span(
                "llm_wrapper.llm_call",
                attributes={
                    "mlsdm.prompt_length": len(enhanced_prompt),
                    "mlsdm.max_tokens": max_tokens,
                },
            ) as llm_span:
                gen_result = self._generate_and_govern(prompt, enhanced_prompt, max_tokens)
                if self._is_error_response(gen_result):
                    llm_span.set_attribute("mlsdm.llm_call.error", True)
                    return cast("dict[str, Any]", gen_result)
                # At this point gen_result must be a tuple
                response_text, governed_metadata = cast(
                    "tuple[str, dict[str, Any] | None]", gen_result
                )
                llm_span.set_attribute("mlsdm.response_length", len(response_text))
                llm_span.set_attribute("mlsdm.llm_call.error", False)

            # Step 6: Update memory state
            with tracer_manager.start_span(
                "llm_wrapper.memory_update",
                attributes={
                    "mlsdm.phase": "wake" if is_wake else "sleep",
                    "mlsdm.stateless_mode": self.stateless_mode,
                },
            ):
                self._update_memory_after_generate(prompt_vector, phase_val)

            # Step 7: Advance rhythm and consolidate
            self._advance_rhythm_and_consolidate()

            # Step 8: Build final response
            return self._build_success_response(
                response_text, memories, max_tokens, governed_metadata
            )

    def _check_moral_and_phase(self, moral_value: float) -> dict[str, Any] | None:
        """Check moral acceptability and cognitive phase. Returns rejection response or None."""
        accepted = self.moral.evaluate(moral_value)
        self.moral.adapt(accepted)

        if not accepted:
            self.rejected_count += 1
            return self._build_rejection_response("morally rejected")

        is_wake = self.rhythm.is_wake()
        if not is_wake:
            return self._build_rejection_response("sleep phase - consolidating")

        return None

    def _embed_and_validate_prompt(self, prompt: str) -> np.ndarray | dict[str, Any]:
        """Embed prompt with circuit breaker and validate. Returns vector or error response."""
        try:
            prompt_vector = self._embed_with_circuit_breaker(prompt)

            if prompt_vector.size == 0:
                raise ValueError("Corrupted embedding vector: empty vector")
            if not np.isfinite(prompt_vector).all():
                raise ValueError("Corrupted embedding vector: contains NaN or Inf values")

            norm = np.linalg.norm(prompt_vector)
            if norm > 1e-9:
                prompt_vector = cast("np.ndarray", prompt_vector / norm)
            else:
                raise ValueError("Zero-norm embedding vector")

            return prompt_vector

        except RuntimeError as e:
            if "Circuit breaker is OPEN" in str(e):
                return self._build_error_response("embedding service unavailable (circuit breaker open)")
            return self._build_error_response(f"embedding failed: {str(e)}")
        except Exception as e:
            return self._build_error_response(f"embedding failed: {str(e)}")

    def _retrieve_and_build_context(
        self,
        prompt: str,
        prompt_vector: np.ndarray,
        phase_val: float,
        context_top_k: int | None,
    ) -> tuple[list[Any], str]:
        """Retrieve context from memory and build enhanced prompt."""
        memories: list[Any] = []
        if context_top_k is None:
            context_top_k = PELM_DEFAULTS.default_top_k if PELM_DEFAULTS else 5
        try:
            memories = self._safe_pelm_operation(
                "retrieve",
                query_vector=prompt_vector.tolist(),
                current_phase=phase_val,
                phase_tolerance=self.DEFAULT_PHASE_TOLERANCE,
                top_k=context_top_k,
            )
        except Exception:
            if not self.stateless_mode:
                memories = []

        context_text = self._build_context_from_memories(memories)
        enhanced_prompt = self._enhance_prompt(prompt, context_text)
        return memories, enhanced_prompt

    def _determine_max_tokens(self, max_tokens: int | None, is_wake: bool) -> int:
        """Determine max tokens based on phase and user input."""
        if max_tokens is None:
            return self.MAX_WAKE_TOKENS if is_wake else self.MAX_SLEEP_TOKENS
        if not is_wake:
            return min(max_tokens, self.MAX_SLEEP_TOKENS)
        return max_tokens

    def _generate_and_govern(
        self,
        prompt: str,
        enhanced_prompt: str,
        max_tokens: int,
    ) -> tuple[str, dict[str, Any] | None] | dict[str, Any]:
        """Generate response with retry and apply speech governance."""
        try:
            base_text = self._llm_generate_with_retry(enhanced_prompt, max_tokens)
        except Exception as e:
            return self._build_error_response(f"generation failed: {str(e)}")

        governed_metadata = None
        response_text = base_text

        if self._speech_governor is not None:
            gov_result: SpeechGovernanceResult = self._speech_governor(
                prompt=prompt,
                draft=base_text,
                max_tokens=max_tokens,
            )
            response_text = gov_result.final_text
            governed_metadata = {
                "raw_text": gov_result.raw_text,
                "metadata": gov_result.metadata,
            }

        return response_text, governed_metadata

    def _update_memory_after_generate(
        self,
        prompt_vector: np.ndarray,
        phase_val: float,
    ) -> None:
        """Update memory (skip if in stateless mode)."""
        if not self.stateless_mode:
            try:
                self.synaptic.update(prompt_vector)
                self._safe_pelm_operation("entangle", prompt_vector.tolist(), phase=phase_val)
                self.consolidation_buffer.append(prompt_vector)
            except Exception as mem_err:
                _logger.debug(
                    "Memory update failed (graceful degradation): %s",
                    mem_err
                )

        self.accepted_count += 1

    def _advance_rhythm_and_consolidate(self) -> None:
        """Advance cognitive rhythm and perform consolidation if entering sleep."""
        self.rhythm.step()

        if self.rhythm.is_sleep() and len(self.consolidation_buffer) > 0 and not self.stateless_mode:
            try:
                self._consolidate_memories()
            except Exception as consol_err:
                _logger.debug(
                    "Memory consolidation failed (non-critical): %s",
                    consol_err
                )

    def _build_success_response(
        self,
        response_text: str,
        memories: list[Any],
        max_tokens: int,
        governed_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the successful response dictionary."""
        result = {
            "response": response_text,
            "accepted": True,
            "phase": self.rhythm.phase,
            "step": self.step_counter,
            "note": "processed (stateless mode)" if self.stateless_mode else "processed",
            "moral_threshold": round(self.moral.threshold, 4),
            "context_items": len(memories),
            "max_tokens_used": max_tokens,
            "stateless_mode": self.stateless_mode
        }

        if governed_metadata is not None:
            result["speech_governance"] = governed_metadata

        return result

    def _consolidate_memories(self) -> None:
        """
        Consolidate buffered memories during sleep phase.

        This simulates biological memory consolidation where recent experiences
        are integrated into long-term storage with appropriate phase encoding.
        """
        if len(self.consolidation_buffer) == 0:
            return

        # During sleep, re-encode memories with sleep phase
        for vector in self.consolidation_buffer:
            # Re-entangle with sleep phase for long-term storage
            self.pelm.entangle(vector.tolist(), phase=self.SLEEP_PHASE)

        # Clear buffer
        self.consolidation_buffer.clear()

    def _build_context_from_memories(self, memories: list[Any]) -> str:
        """Build context text from retrieved memories."""
        if not memories:
            return ""

        # Simple context building - could be enhanced with more sophisticated approaches
        context_parts = []
        for i, mem in enumerate(memories[:3]):  # Use top 3
            resonance = getattr(mem, 'resonance', 0.0)
            context_parts.append(f"[Context {i+1}, relevance: {resonance:.2f}]")

        return " ".join(context_parts) if context_parts else ""

    def _enhance_prompt(self, prompt: str, context: str) -> str:
        """Enhance prompt with memory context."""
        if not context:
            return prompt

        return f"{context}\n\nUser: {prompt}"

    def _build_rejection_response(self, reason: str) -> dict[str, Any]:
        """Build response for rejected requests."""
        return {
            "response": "",
            "accepted": False,
            "phase": self.rhythm.phase,
            "step": self.step_counter,
            "note": reason,
            "moral_threshold": round(self.moral.threshold, 4),
            "context_items": 0,
            "max_tokens_used": 0
        }

    def _build_error_response(self, error: str) -> dict[str, Any]:
        """Build response for errors."""
        return {
            "response": "",
            "accepted": False,
            "phase": self.rhythm.phase,
            "step": self.step_counter,
            "note": f"error: {error}",
            "moral_threshold": round(self.moral.threshold, 4),
            "context_items": 0,
            "max_tokens_used": 0
        }

    def _is_error_response(self, result: Any) -> bool:
        """Check if a result is an error response dict."""
        return isinstance(result, dict) and "note" in result and "error" in result.get("note", "")

    def get_state(self) -> dict[str, Any]:
        """
        Get current cognitive state with reliability metrics.

        Returns:
            Dictionary with system state including memory usage, phase, and statistics.
        """
        with self._lock:
            l1, l2, l3 = self.synaptic.state()
            return {
                "step": self.step_counter,
                "phase": self.rhythm.phase,
                "phase_counter": self.rhythm.counter,
                "moral_threshold": round(self.moral.threshold, 4),
                "moral_ema": round(self.moral.ema_accept_rate, 4),
                "accepted_count": self.accepted_count,
                "rejected_count": self.rejected_count,
                "synaptic_norms": {
                    "L1": float(np.linalg.norm(l1)),
                    "L2": float(np.linalg.norm(l2)),
                    "L3": float(np.linalg.norm(l3))
                },
                "pelm_stats": self.pelm.get_state_stats(),
                # Backward compatibility: also expose as qilm_stats (deprecated)
                "qilm_stats": self.pelm.get_state_stats(),
                "consolidation_buffer_size": len(self.consolidation_buffer),
                "reliability": {
                    "stateless_mode": self.stateless_mode,
                    "circuit_breaker_state": self.embedding_circuit_breaker.state.value,
                    "pelm_failure_count": self.pelm_failure_count,
                    # Backward compatibility: also expose as qilm_failure_count (deprecated, use pelm_failure_count instead)
                    "qilm_failure_count": self.pelm_failure_count,
                    "embedding_failure_count": self.embedding_failure_count,
                    "llm_failure_count": self.llm_failure_count
                }
            }

    def get_cognitive_state(self) -> CognitiveState:
        """
        Return a snapshot of the current cognitive state of MLSDM core.

        Safe for observability/health-checks:
        - read-only
        - O(1) over main components
        - thread-safe

        Returns:
            CognitiveState dataclass with aggregated state from all core components.
        """
        with self._lock:
            # Aggregate memory usage from PELM and synaptic memory
            pelm_bytes = self.pelm.memory_usage_bytes()
            synaptic_bytes = self.synaptic.memory_usage_bytes()
            total_memory_bytes = pelm_bytes + synaptic_bytes

            return CognitiveState(
                phase=self.rhythm.phase,
                stateless_mode=self.stateless_mode,
                memory_used_bytes=total_memory_bytes,
                moral_threshold=self.moral.get_current_threshold(),
                moral_ema=self.moral.get_ema_value(),
                rhythm_state=self.rhythm.get_state_label(),
                step_counter=self.step_counter,
                emergency_shutdown=False,  # LLMWrapper doesn't have controller-level shutdown
                aphasia_flags=None,  # Reserved for future use
                extra={},
            )

    def reset(self) -> None:
        """Reset the wrapper to initial state (for testing)."""
        with self._lock:
            self.step_counter = 0
            self.rejected_count = 0
            self.accepted_count = 0
            self.consolidation_buffer.clear()
            self.moral = MoralFilterV2(initial_threshold=0.50)
            self.pelm = PhaseEntangledLatticeMemory(dimension=self.dim, capacity=self.pelm.capacity)
            self.rhythm = CognitiveRhythm(
                wake_duration=self.rhythm.wake_duration,
                sleep_duration=self.rhythm.sleep_duration
            )
            self.synaptic.reset_all()

            # Reset reliability components
            self.embedding_circuit_breaker.reset()
            self.stateless_mode = False
            self.pelm_failure_count = 0
            self.embedding_failure_count = 0
            self.llm_failure_count = 0
