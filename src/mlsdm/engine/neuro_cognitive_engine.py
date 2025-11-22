"""
NeuroCognitiveEngine: integrated MLSDM + FSLGS orchestration layer.

Архітектура:
- MLSDM = Cognitive Substrate (єдина пам'ять, мораль, ритм, резилієнтність)
- FSLGS = Language Governance Layer (валидація / нагляд, без власної пам'яті)

Пайплайн:
    USER
      ↓
  PRE-FLIGHT:
    • moral pre-check (MLSDM.moral)
    • grammar pre-check (FSLGS.grammar, якщо є)
      ↓
  FSLGS (якщо увімкнено):
    • режими (rest/action), dual-stream, UG-констрейнти
      ↓
  MLSDM:
    • moral/rhythm/memory + LLM
      ↓
  FSLGS post-validation:
    • coherence, binding, grammar post-check
      ↓
  RESPONSE (+ timing, validation_steps, error)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from mlsdm.core.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

try:
    # FSLGS як optional dependency
    from fslgs import FSLGSWrapper
except Exception:  # pragma: no cover - optional
    FSLGSWrapper = None


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MLSDMRejectionError(Exception):
    """MLSDM відхилив запит (мораль, ритм, резилієнтність)."""
    pass


class EmptyResponseError(Exception):
    """LLM/MLSDM повернули порожню відповідь."""
    pass


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


class TimingContext:
    """Простий контекстний менеджер для вимірювання часу (мс)."""

    def __init__(self, metrics: dict[str, float], key: str) -> None:
        self._metrics = metrics
        self._key = key
        self._start: float | None = None

    def __enter__(self) -> TimingContext:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            elapsed = (time.perf_counter() - self._start) * 1000.0
            self._metrics[self._key] = elapsed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NeuroEngineConfig:
    """Конфіг для NeuroCognitiveEngine.

    FSLGS не має власної пам'яті: весь контекст зберігається в MLSDM (single
    source of truth). FSLGS виступає чистим governance-layer над MLSDM.
    """

    # MLSDM / memory layer
    dim: int = 384
    capacity: int = 20_000
    wake_duration: int = 8
    sleep_duration: int = 3
    initial_moral_threshold: float = 0.50
    llm_timeout: float = 30.0
    llm_retry_attempts: int = 3

    # FSLGS / language governance layer
    enable_fslgs: bool = True
    enable_universal_grammar: bool = True
    grammar_strictness: float = 0.9
    association_threshold: float = 0.65
    enable_monitoring: bool = True
    stress_threshold: float = 0.7

    # Пам'ять FSLGS вимкнена (використовує MLSDM як єдине джерело правди)
    # Ці параметри зберігаємо лише як конфіг, але в конструктор FSLGS
    # передаємо жорстко memory_capacity=0, fractal_levels=None.
    fslgs_memory_capacity: int = 0
    fslgs_fractal_levels: list[str] | None = None

    enable_entity_tracking: bool = True
    enable_temporal_validation: bool = True
    enable_causal_checking: bool = True

    # Runtime defaults
    default_moral_value: float = 0.5
    default_context_top_k: int = 5
    default_cognitive_load: float = 0.5
    default_user_intent: str = "conversational"

    # Observability / Metrics
    enable_metrics: bool = False

    # Phase 7: Cost tracking and efficiency
    enable_cost_tracking: bool = True
    pricing: dict[str, float] | None = None  # Optional pricing config

    # Phase 7: Semantic cache
    enable_semantic_cache: bool = True
    cache_max_entries: int = 1000
    cache_similarity_threshold: float = 0.85
    cache_moral_tolerance: float = 0.1

    # Phase 7: Adaptive context management
    min_context_top_k: int = 3
    max_context_top_k: int = 10
    target_latency_ms: float = 1000.0
    max_memory_tokens: int = 100_000  # Trigger summarization

    # Phase 7: QoS and graceful degradation
    priority_tier: Literal["low", "normal", "high"] = "normal"
    degradation_policy: dict[str, Any] = field(default_factory=lambda: {
        "disable_fslgs": True,
        "limit_max_tokens": 256,
        "min_context_top_k_under_load": 2,
    })


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class NeuroCognitiveEngine:
    """High-level orchestration of MLSDM + FSLGS.

    Приклад:
    --------
    >>> engine = NeuroCognitiveEngine(
    ...     llm_generate_fn=my_llm_call,
    ...     embedding_fn=my_embedding_call,
    ... )
    >>> result = engine.generate("Hello", max_tokens=128)
    >>> print(result["response"])
    """

    def __init__(
        self,
        llm_generate_fn: Callable[[str, int], str],
        embedding_fn: Callable[[str], np.ndarray],
        config: NeuroEngineConfig | None = None,
    ) -> None:
        self.config = config or NeuroEngineConfig()
        self._embedding_fn = embedding_fn

        # MLSDM: єдина пам'ять + мораль + ритм + резилієнтність
        self._mlsdm = LLMWrapper(
            llm_generate_fn=llm_generate_fn,
            embedding_fn=embedding_fn,
            dim=self.config.dim,
            capacity=self.config.capacity,
            wake_duration=self.config.wake_duration,
            sleep_duration=self.config.sleep_duration,
            initial_moral_threshold=self.config.initial_moral_threshold,
            llm_timeout=self.config.llm_timeout,
            llm_retry_attempts=self.config.llm_retry_attempts,
        )

        self._last_mlsdm_state: dict[str, Any] | None = None

        # Runtime параметри, які має бачити MLSDM всередині governed_llm
        self._runtime_moral_value: float = self.config.default_moral_value
        self._runtime_context_top_k: int = self.config.default_context_top_k

        # Опційний FSLGS (суто governance, без пам'яті)
        self._fslgs: Any | None = None
        if self.config.enable_fslgs and FSLGSWrapper is not None:
            self._fslgs = self._build_fslgs_wrapper()

        # Опційна система метрик
        self._metrics: Any | None = None
        if self.config.enable_metrics:
            # Import lazily to avoid circular dependencies
            # This is safe as metrics module has no dependencies on engine
            from mlsdm.observability.metrics import MetricsRegistry

            self._metrics = MetricsRegistry()

        # Phase 7: Cost tracking
        self._cost_tracker: Any | None = None
        if self.config.enable_cost_tracking:
            from mlsdm.observability.cost import CostTracker
            self._cost_tracker = CostTracker()

        # Phase 7: Semantic cache
        self._semantic_cache: Any | None = None
        if self.config.enable_semantic_cache:
            from mlsdm.memory.semantic_cache import SemanticResponseCache
            self._semantic_cache = SemanticResponseCache(
                max_entries=self.config.cache_max_entries,
                similarity_threshold=self.config.cache_similarity_threshold,
                moral_tolerance=self.config.cache_moral_tolerance,
            )

        # Phase 7: Adaptive context - track recent latencies
        self._recent_latencies: deque[float] = deque(maxlen=10)
        self._under_load = False

    # ------------------------------------------------------------------ #
    # Internal builders                                                   #
    # ------------------------------------------------------------------ #

    def _build_fslgs_wrapper(self) -> Any:
        """Підключення FSLGS поверх MLSDM без дублювання пам'яті.

        FSLGSWrapper бачить:
        - governed_llm(): делегує в MLSDM.LLMWrapper.generate(...)
        - embedding_fn(): той самий, що й MLSDM
        """

        def governed_llm(prompt: str, max_tokens: int) -> str:
            """Адаптер: FSLGS → MLSDM.

            Використовує runtime-параметри moral_value/context_top_k, які
            виставляються у generate().
            """
            state = self._mlsdm.generate(
                prompt=prompt,
                moral_value=self._runtime_moral_value,
                max_tokens=max_tokens,
                context_top_k=self._runtime_context_top_k,
            )
            self._last_mlsdm_state = state

            # Очікуємо флаг accepted в MLSDM-стані
            if not state.get("accepted", True):
                note = state.get("note", "rejected")
                raise MLSDMRejectionError(f"MLSDM rejected: {note}")

            response = state.get("response", "")
            if not isinstance(response, str) or not response.strip():
                raise EmptyResponseError("MLSDM returned empty response")

            return response

        # FSLGS без власної пам'яті: використовує тільки MLSDM
        return FSLGSWrapper(
            llm_generate_fn=governed_llm,
            embedding_fn=self._embedding_fn,
            dim=self.config.dim,
            enable_universal_grammar=self.config.enable_universal_grammar,
            grammar_strictness=self.config.grammar_strictness,
            association_threshold=self.config.association_threshold,
            enable_monitoring=self.config.enable_monitoring,
            stress_threshold=self.config.stress_threshold,
            fractal_levels=None,       # FIX-001: no internal memory/fractals
            memory_capacity=0,         # FIX-001: single source of truth = MLSDM
            enable_entity_tracking=self.config.enable_entity_tracking,
            enable_temporal_validation=self.config.enable_temporal_validation,
            enable_causal_checking=self.config.enable_causal_checking,
        )

    def _adapt_context_parameters(
        self,
        timing: dict[str, float],
        cognitive_load: float
    ) -> int:
        """Adapt context_top_k based on recent latency and cognitive load.

        Phase 7: Adaptive context management
        - If recent requests are slow (> target_latency_ms), reduce context_top_k
        - If fast and cognitive_load is high, increase up to max_context_top_k
        - Never go below min_context_top_k

        Args:
            timing: Timing dict from recent request
            cognitive_load: Current cognitive load (0.0-1.0)

        Returns:
            Adapted context_top_k value
        """
        # Track recent latency
        if "total" in timing:
            self._recent_latencies.append(timing["total"])

        # Calculate average recent latency
        if not self._recent_latencies:
            return self.config.default_context_top_k

        avg_latency = sum(self._recent_latencies) / len(self._recent_latencies)

        # Determine if under load
        self._under_load = avg_latency > self.config.target_latency_ms

        # Adapt context_top_k
        current_k = self._runtime_context_top_k

        if self._under_load:
            # Reduce context under load
            new_k = max(
                self.config.min_context_top_k,
                current_k - 1
            )
        elif cognitive_load > 0.7 and avg_latency < self.config.target_latency_ms * 0.7:
            # Increase context when fast and high cognitive load
            new_k = min(
                self.config.max_context_top_k,
                current_k + 1
            )
        else:
            # Keep current
            new_k = current_k

        return new_k

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        user_intent: str | None = None,
        cognitive_load: float | None = None,
        moral_value: float | None = None,
        context_top_k: int | None = None,
        enable_diagnostics: bool = True,
    ) -> dict[str, Any]:
        """Запустити повний пайплайн з pre-flight, MLSDM і FSLGS.

        Повертає:
        - response: текст для користувача (може бути "")
        - governance: сирий вихід FSLGS (або None)
        - mlsdm: сирий стан MLSDM (або None, якщо відмовлено на pre-flight)
        - timing: dict з мілісекундами по етапах
        - validation_steps: список кроків перевірки
        - error: None або {type, message, ...}
        - rejected_at: None або "pre_flight"/"generation"
        """

        timing: dict[str, float] = {}
        validation_steps: list[dict[str, Any]] = []

        # Increment requests counter
        if self._metrics is not None:
            self._metrics.increment_requests_total()

        # Заповнюємо рантайм за замовчуванням
        user_intent = user_intent or self.config.default_user_intent
        cognitive_load = (
            cognitive_load
            if cognitive_load is not None
            else self.config.default_cognitive_load
        )
        moral_value = (
            moral_value
            if moral_value is not None
            else self.config.default_moral_value
        )
        context_top_k = context_top_k or self.config.default_context_top_k

        mlsdm_state: dict[str, Any] | None = None
        fslgs_result: dict[str, Any] | None = None

        # Phase 7: QoS - Apply degradation if under load
        effective_fslgs = self.config.enable_fslgs
        effective_max_tokens = max_tokens

        if self._under_load:
            if self.config.priority_tier == "low":
                # Low priority: apply aggressive degradation
                if self.config.degradation_policy.get("disable_fslgs", True):
                    effective_fslgs = False
                limit_tokens = self.config.degradation_policy.get("limit_max_tokens", 256)
                effective_max_tokens = min(max_tokens, limit_tokens)
                min_k = self.config.degradation_policy.get("min_context_top_k_under_load", 2)
                context_top_k = max(min_k, self.config.min_context_top_k)
            elif self.config.priority_tier == "normal":
                # Normal priority: moderate degradation
                context_top_k = max(self.config.min_context_top_k, context_top_k - 1)
            # High priority: no degradation, but log over_budget

        with TimingContext(timing, "total"):
            # Phase 7: Semantic cache lookup
            cached_response: str | None = None
            query_embedding: Any = None

            if self._semantic_cache is not None:
                with TimingContext(timing, "cache_lookup"):
                    try:
                        query_embedding = self._embedding_fn(prompt)
                        cached_response = self._semantic_cache.lookup(
                            query_embedding, moral_value, user_intent
                        )
                    except Exception:
                        # Cache lookup failure shouldn't break the request
                        cached_response = None

                    if cached_response is not None:
                        # Cache hit - return immediately
                        return {
                            "response": cached_response,
                            "governance": None,
                            "mlsdm": None,
                            "timing": timing,
                            "validation_steps": validation_steps,
                            "error": None,
                            "rejected_at": None,
                            "from_cache": True,
                            "cost": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0,
                                "estimated_cost_usd": 0.0,
                            },
                        }
            # -----------------------
            # PRE-FLIGHT: moral check
            # -----------------------
            with TimingContext(timing, "moral_precheck"):
                prompt_moral = None
                moral_filter = getattr(self._mlsdm, "moral", None)

                if moral_filter is not None and hasattr(
                    moral_filter, "compute_moral_value"
                ):
                    prompt_moral = moral_filter.compute_moral_value(prompt)
                    passed = prompt_moral >= moral_value
                    validation_steps.append(
                        {
                            "step": "moral_precheck",
                            "passed": passed,
                            "score": prompt_moral,
                            "threshold": moral_value,
                        }
                    )
                    if not passed:
                        # Швидка відмова: не вантажимо FSLGS/LLM
                        # Record metrics
                        if self._metrics is not None:
                            self._metrics.increment_rejections_total("pre_flight")
                            self._metrics.increment_errors_total("moral_precheck")
                            if "moral_precheck" in timing:
                                self._metrics.record_latency_pre_flight(timing["moral_precheck"])
                            if "total" in timing:
                                self._metrics.record_latency_total(timing["total"])

                        return {
                            "response": "",
                            "governance": None,
                            "mlsdm": None,
                            "timing": timing,
                            "validation_steps": validation_steps,
                            "error": {
                                "type": "moral_precheck",
                                "score": prompt_moral,
                                "threshold": moral_value,
                            },
                            "rejected_at": "pre_flight",
                        }
                else:
                    # Якщо моральний фільтр недоступний — позначаємо як пропущений крок.
                    validation_steps.append(
                        {
                            "step": "moral_precheck",
                            "passed": True,
                            "skipped": True,
                            "reason": "moral_filter_not_available",
                        }
                    )

            # --------------------------
            # PRE-FLIGHT: grammar check
            # --------------------------
            if self._fslgs is not None and getattr(
                self._fslgs, "grammar", None
            ) is not None:
                with TimingContext(timing, "grammar_precheck"):
                    grammar = self._fslgs.grammar
                    if hasattr(grammar, "validate_input_structure"):
                        passed = bool(grammar.validate_input_structure(prompt))
                        validation_steps.append(
                            {
                                "step": "grammar_precheck",
                                "passed": passed,
                            }
                        )
                        if not passed:
                            # Record metrics
                            if self._metrics is not None:
                                self._metrics.increment_rejections_total("pre_flight")
                                self._metrics.increment_errors_total("grammar_precheck")
                                if "grammar_precheck" in timing:
                                    self._metrics.record_latency_pre_flight(timing["grammar_precheck"])
                                if "total" in timing:
                                    self._metrics.record_latency_total(timing["total"])

                            return {
                                "response": "",
                                "governance": None,
                                "mlsdm": None,
                                "timing": timing,
                                "validation_steps": validation_steps,
                                "error": {
                                    "type": "grammar_precheck",
                                    "message": "invalid_structure",
                                },
                                "rejected_at": "pre_flight",
                            }
                    else:
                        validation_steps.append(
                            {
                                "step": "grammar_precheck",
                                "passed": True,
                                "skipped": True,
                                "reason": "validate_input_structure_not_available",
                            }
                        )

            # -----------------------
            # MAIN PIPELINE
            # -----------------------
            # Оновлюємо runtime-параметри для governed_llm (FIX-002)
            self._runtime_moral_value = moral_value
            self._runtime_context_top_k = context_top_k

            try:
                with TimingContext(timing, "generation"):
                    if self._fslgs is not None and effective_fslgs:
                        # Повний governance-пайплайн
                        fslgs_result = self._fslgs.generate(
                            prompt=prompt,
                            cognitive_load=cognitive_load,
                            max_tokens=effective_max_tokens,
                            user_intent=user_intent,
                            enable_diagnostics=enable_diagnostics,
                        )
                        response_text = fslgs_result.get("response", "")
                        mlsdm_state = self._last_mlsdm_state
                    else:
                        # Фолбек: лише MLSDM без FSLGS
                        mlsdm_state = self._mlsdm.generate(
                            prompt=prompt,
                            moral_value=moral_value,
                            max_tokens=effective_max_tokens,
                            context_top_k=context_top_k,
                        )
                        self._last_mlsdm_state = mlsdm_state

                        if not mlsdm_state.get("accepted", True):
                            note = mlsdm_state.get("note", "rejected")
                            raise MLSDMRejectionError(
                                f"MLSDM rejected: {note}"
                            )

                        response_text = mlsdm_state.get("response", "")
                        if not isinstance(response_text, str) or not response_text.strip():
                            raise EmptyResponseError(
                                "MLSDM returned empty response"
                            )

            except MLSDMRejectionError as e:
                # Record metrics
                if self._metrics is not None:
                    self._metrics.increment_rejections_total("generation")
                    self._metrics.increment_errors_total("mlsdm_rejection")
                    if "generation" in timing:
                        self._metrics.record_latency_generation(timing["generation"])
                    if "total" in timing:
                        self._metrics.record_latency_total(timing["total"])

                return {
                    "response": "",
                    "governance": None,
                    "mlsdm": mlsdm_state,
                    "timing": timing,
                    "validation_steps": validation_steps,
                    "error": {
                        "type": "mlsdm_rejection",
                        "message": str(e),
                    },
                    "rejected_at": "generation",
                }
            except EmptyResponseError as e:
                # Record metrics
                if self._metrics is not None:
                    self._metrics.increment_rejections_total("generation")
                    self._metrics.increment_errors_total("empty_response")
                    if "generation" in timing:
                        self._metrics.record_latency_generation(timing["generation"])
                    if "total" in timing:
                        self._metrics.record_latency_total(timing["total"])

                return {
                    "response": "",
                    "governance": fslgs_result,
                    "mlsdm": mlsdm_state,
                    "timing": timing,
                    "validation_steps": validation_steps,
                    "error": {
                        "type": "empty_response",
                        "message": str(e),
                    },
                    "rejected_at": "generation",
                }

        # Успішний шлях
        # Record metrics for successful generation
        if self._metrics is not None:
            if "moral_precheck" in timing or "grammar_precheck" in timing:
                pre_flight_time = timing.get("moral_precheck", 0) + timing.get("grammar_precheck", 0)
                if pre_flight_time > 0:
                    self._metrics.record_latency_pre_flight(pre_flight_time)
            if "generation" in timing:
                self._metrics.record_latency_generation(timing["generation"])
            if "total" in timing:
                self._metrics.record_latency_total(timing["total"])

        # Phase 7: Cost tracking
        cost_info = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }

        if self._cost_tracker is not None:
            from mlsdm.observability.cost import estimate_tokens

            prompt_tokens = estimate_tokens(prompt)
            completion_tokens = estimate_tokens(response_text)

            cost_info["prompt_tokens"] = prompt_tokens
            cost_info["completion_tokens"] = completion_tokens
            cost_info["total_tokens"] = prompt_tokens + completion_tokens

            # Update tracker with pricing if configured
            self._cost_tracker.update(prompt, response_text, self.config.pricing)
            cost_info["estimated_cost_usd"] = self._cost_tracker.estimated_cost_usd

        # Phase 7: Store in semantic cache
        if self._semantic_cache is not None and query_embedding is not None:
            try:
                self._semantic_cache.store(
                    query_embedding, moral_value, user_intent, response_text
                )
            except Exception:
                # Cache store failure shouldn't break the response
                pass

        # Phase 7: Adaptive context management
        new_context_k = self._adapt_context_parameters(timing, cognitive_load)
        self._runtime_context_top_k = new_context_k

        return {
            "response": response_text,
            "governance": fslgs_result,
            "mlsdm": mlsdm_state,
            "timing": timing,
            "validation_steps": validation_steps,
            "error": None,
            "rejected_at": None,
            "from_cache": False,
            "cost": cost_info,
        }

    # ------------------------------------------------------------------ #
    # Diagnostics                                                         #
    # ------------------------------------------------------------------ #

    def get_last_states(self) -> dict[str, Any]:
        """Повертає останній MLSDM-стан і факт наявності FSLGS."""
        return {
            "mlsdm": self._last_mlsdm_state,
            "has_fslgs": self._fslgs is not None,
        }

    def get_metrics(self) -> Any | None:
        """Get MetricsRegistry instance if metrics are enabled.

        Returns:
            MetricsRegistry instance or None if metrics are disabled
        """
        return self._metrics

    def get_cache_stats(self) -> dict[str, Any] | None:
        """Get semantic cache statistics.

        Returns:
            Cache statistics dict or None if cache is disabled
        """
        if self._semantic_cache is not None:
            return self._semantic_cache.get_stats()
        return None

    def get_cost_summary(self) -> dict[str, Any] | None:
        """Get cost tracking summary.

        Returns:
            Cost summary dict or None if cost tracking is disabled
        """
        if self._cost_tracker is not None:
            return self._cost_tracker.to_dict()
        return None

    def clear_cache(self) -> None:
        """Clear the semantic cache."""
        if self._semantic_cache is not None:
            self._semantic_cache.clear()
