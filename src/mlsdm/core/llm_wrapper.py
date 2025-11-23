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

import time
from collections.abc import Callable
from enum import Enum
from threading import Lock
from typing import Any

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..cognition.moral_filter_v2 import MoralFilterV2
from ..memory.multi_level_memory import MultiLevelSynapticMemory
from ..memory.qilm_v2 import QILM_v2
from ..rhythm.cognitive_rhythm import CognitiveRhythm
from ..speech.governance import SpeechGovernanceResult, SpeechGovernor


class CircuitBreakerState(Enum):
    """Circuit breaker states for embedding function."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Too many failures, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for embedding function failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

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

    MAX_WAKE_TOKENS = 2048
    MAX_SLEEP_TOKENS = 150  # Forced short responses during sleep

    def __init__(
        self,
        llm_generate_fn: Callable[[str, int], str],
        embedding_fn: Callable[[str], np.ndarray],
        dim: int = 384,
        capacity: int = 20_000,
        wake_duration: int = 8,
        sleep_duration: int = 3,
        initial_moral_threshold: float = 0.50,
        llm_timeout: float = 30.0,
        llm_retry_attempts: int = 3,
        speech_governor: SpeechGovernor | None = None
    ):
        """
        Initialize the LLM wrapper with cognitive governance.

        Args:
            llm_generate_fn: Function that takes (prompt, max_tokens) and returns response
            embedding_fn: Function that takes text and returns embedding vector
            dim: Embedding dimension (default 384)
            capacity: Maximum memory vectors (default 20,000)
            wake_duration: Wake cycle duration in steps (default 8)
            sleep_duration: Sleep cycle duration in steps (default 3)
            initial_moral_threshold: Starting moral threshold (default 0.50)
            llm_timeout: Timeout for LLM calls in seconds (default 30.0)
            llm_retry_attempts: Number of retry attempts for LLM calls (default 3)
            speech_governor: Optional speech governance policy (default None)
        """
        self.dim = dim
        self._lock = Lock()
        self.llm_timeout = llm_timeout
        self.llm_retry_attempts = llm_retry_attempts

        # Core components
        self.llm_generate = llm_generate_fn
        self.embed = embedding_fn
        self.moral = MoralFilterV2(initial_threshold=initial_moral_threshold)
        self.qilm = QILM_v2(dimension=dim, capacity=capacity)
        self.rhythm = CognitiveRhythm(wake_duration=wake_duration, sleep_duration=sleep_duration)
        self.synaptic = MultiLevelSynapticMemory(dimension=dim)

        # Speech governance
        self._speech_governor = speech_governor

        # Reliability components
        self.embedding_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            success_threshold=2
        )
        self.stateless_mode = False  # Graceful degradation flag

        # State tracking
        self.step_counter = 0
        self.rejected_count = 0
        self.accepted_count = 0
        self.consolidation_buffer: list[np.ndarray] = []
        self.qilm_failure_count = 0
        self.embedding_failure_count = 0
        self.llm_failure_count = 0

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
            result = self.embedding_circuit_breaker.call(self.embed, text)
            if not isinstance(result, np.ndarray):
                result = np.array(result, dtype=np.float32)
            return result.astype(np.float32)
        except Exception as e:
            self.embedding_failure_count += 1
            raise e

    def _safe_qilm_operation(
        self,
        operation: str,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute QILM operation with graceful degradation.

        If QILM fails repeatedly, switches to stateless mode.
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
                return self.qilm.retrieve(*args, **kwargs)
            elif operation == "entangle":
                return self.qilm.entangle(*args, **kwargs)
            else:
                raise ValueError(f"Unknown QILM operation: {operation}")
        except (MemoryError, RuntimeError) as e:
            self.qilm_failure_count += 1
            if self.qilm_failure_count >= 3:
                # Switch to stateless mode after repeated failures
                self.stateless_mode = True
            raise e

    def generate(
        self,
        prompt: str,
        moral_value: float,
        max_tokens: int | None = None,
        context_top_k: int = 5
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
        with self._lock:
            self.step_counter += 1

            # Step 1: Moral evaluation
            accepted = self.moral.evaluate(moral_value)
            self.moral.adapt(accepted)

            if not accepted:
                self.rejected_count += 1
                return self._build_rejection_response("morally rejected")

            # Step 2: Check cognitive phase
            is_wake = self.rhythm.is_wake()
            if not is_wake:
                # During sleep, reject new processing but allow consolidation
                return self._build_rejection_response("sleep phase - consolidating")

            # Step 3: Embed prompt with circuit breaker
            try:
                prompt_vector = self._embed_with_circuit_breaker(prompt)

                # Validate and normalize
                if prompt_vector.size == 0:
                    raise ValueError("Corrupted embedding vector: empty vector")
                if not np.isfinite(prompt_vector).all():
                    raise ValueError("Corrupted embedding vector: contains NaN or Inf values")

                norm = np.linalg.norm(prompt_vector)
                if norm > 1e-9:
                    prompt_vector = prompt_vector / norm
                else:
                    raise ValueError("Zero-norm embedding vector")

            except RuntimeError as e:
                if "Circuit breaker is OPEN" in str(e):
                    return self._build_error_response("embedding service unavailable (circuit breaker open)")
                return self._build_error_response(f"embedding failed: {str(e)}")
            except Exception as e:
                return self._build_error_response(f"embedding failed: {str(e)}")

            # Step 4: Retrieve context from memory with graceful degradation
            phase_val = 0.1 if is_wake else 0.9
            memories = []
            try:
                memories = self._safe_qilm_operation(
                    "retrieve",
                    query_vector=prompt_vector.tolist(),
                    current_phase=phase_val,
                    phase_tolerance=0.15,
                    top_k=context_top_k
                )
            except Exception:
                # Continue in stateless mode if QILM fails
                if not self.stateless_mode:
                    memories = []

            # Step 5: Build context-aware prompt
            context_text = self._build_context_from_memories(memories)
            enhanced_prompt = self._enhance_prompt(prompt, context_text)

            # Step 6: Determine max tokens based on phase
            if max_tokens is None:
                max_tokens = self.MAX_WAKE_TOKENS if is_wake else self.MAX_SLEEP_TOKENS
            else:
                # During sleep, enforce short responses
                if not is_wake:
                    max_tokens = min(max_tokens, self.MAX_SLEEP_TOKENS)

            # Step 7: Generate response with retry and timeout
            try:
                base_text = self._llm_generate_with_retry(enhanced_prompt, max_tokens)
            except Exception as e:
                return self._build_error_response(f"generation failed: {str(e)}")

            # Step 7a: Apply speech governance if configured
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

            # Step 8: Update memory (skip if in stateless mode)
            if not self.stateless_mode:
                try:
                    self.synaptic.update(prompt_vector)
                    self._safe_qilm_operation("entangle", prompt_vector.tolist(), phase=phase_val)
                    self.consolidation_buffer.append(prompt_vector)
                except Exception:
                    # Continue even if memory update fails
                    pass

            self.accepted_count += 1

            # Step 9: Advance cognitive rhythm
            self.rhythm.step()

            # Step 10: Perform consolidation if entering sleep
            if self.rhythm.is_sleep() and len(self.consolidation_buffer) > 0 and not self.stateless_mode:
                try:
                    self._consolidate_memories()
                except Exception:
                    # Consolidation failure is non-critical
                    pass

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

        # During sleep, re-encode memories with sleep phase (0.9)
        sleep_phase = 0.9
        for vector in self.consolidation_buffer:
            # Re-entangle with sleep phase for long-term storage
            self.qilm.entangle(vector.tolist(), phase=sleep_phase)

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
                "qilm_stats": self.qilm.get_state_stats(),
                "consolidation_buffer_size": len(self.consolidation_buffer),
                "reliability": {
                    "stateless_mode": self.stateless_mode,
                    "circuit_breaker_state": self.embedding_circuit_breaker.state.value,
                    "qilm_failure_count": self.qilm_failure_count,
                    "embedding_failure_count": self.embedding_failure_count,
                    "llm_failure_count": self.llm_failure_count
                }
            }

    def reset(self) -> None:
        """Reset the wrapper to initial state (for testing)."""
        with self._lock:
            self.step_counter = 0
            self.rejected_count = 0
            self.accepted_count = 0
            self.consolidation_buffer.clear()
            self.moral = MoralFilterV2(initial_threshold=0.50)
            self.qilm = QILM_v2(dimension=self.dim, capacity=self.qilm.capacity)
            self.rhythm = CognitiveRhythm(
                wake_duration=self.rhythm.wake_duration,
                sleep_duration=self.rhythm.sleep_duration
            )
            self.synaptic.reset_all()

            # Reset reliability components
            self.embedding_circuit_breaker.reset()
            self.stateless_mode = False
            self.qilm_failure_count = 0
            self.embedding_failure_count = 0
            self.llm_failure_count = 0
