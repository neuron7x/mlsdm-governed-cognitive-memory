"""
NeuroCognitiveEngine: integrated MLSDM + FSLGS orchestration layer.

Цей модуль дає єдину точку входу, яка складає:
- MLSDM LLMWrapper (пам'ять, ритм, мораль, резилієнтність)
- FSLGSWrapper (dual-stream мова, anti-schizo, Universal Grammar)

Його можна під'єднати до будь-якого бекенд-LLM через llm_generate_fn
та embedding_fn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mlsdm.core.llm_wrapper import LLMWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

try:
    # FSLGS як optional dependency
    from fslgs import FSLGSWrapper
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional
    FSLGSWrapper = None


@dataclass
class NeuroEngineConfig:
    """Конфіг для NeuroCognitiveEngine.

    Це тонкий шар над дефолтами LLMWrapper + FSLGSWrapper.
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
    fractal_levels: list[str] | None = None
    memory_capacity: int = 2048
    enable_entity_tracking: bool = True
    enable_temporal_validation: bool = True
    enable_causal_checking: bool = True

    # Рантайм-дефолти
    default_moral_value: float = 0.5
    default_context_top_k: int = 5
    default_cognitive_load: float = 0.5
    default_user_intent: str = "conversational"


class NeuroCognitiveEngine:
    """High-level orchestration of MLSDM + FSLGS.

    Приклад використання
    --------------------
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

        # 1) Базовий MLSDM-обгортник (memory + rhythm + moral + resiliency)
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

        # 2) Опційна інтеграція FSLGS (dual-stream + UG + anti-schizo)
        # Note: _fslgs stores FSLGSWrapper instance when available, otherwise None
        # Using Any to avoid hard dependency on fslgs package types
        self._fslgs: Any | None = None
        if self.config.enable_fslgs and FSLGSWrapper is not None:
            self._fslgs = self._build_fslgs_wrapper()

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    def _build_fslgs_wrapper(self) -> Any:
        """Підключення FSLGS поверх LLMWrapper.

        FSLGSWrapper очікує:
        - llm_generate_fn(prompt: str, max_tokens: int) -> str
        - embedding_fn(text: str) -> np.ndarray

        Адаптер:
        - викликає MLSDM.LLMWrapper.generate(...)
        - зберігає повний стан MLSDM для діагностики
        - повертає тільки текст відповіді для FSLGS
        """

        def governed_llm(prompt: str, max_tokens: int) -> str:
            state = self._mlsdm.generate(
                prompt=prompt,
                moral_value=self.config.default_moral_value,
                max_tokens=max_tokens,
                context_top_k=self.config.default_context_top_k,
            )
            # Зберігаємо стан для спостереження
            self._last_mlsdm_state = state
            response: str = state.get("response", "")
            return response

        return FSLGSWrapper(
            llm_generate_fn=governed_llm,
            embedding_fn=self._embedding_fn,
            dim=self.config.dim,
            enable_universal_grammar=self.config.enable_universal_grammar,
            grammar_strictness=self.config.grammar_strictness,
            association_threshold=self.config.association_threshold,
            enable_monitoring=self.config.enable_monitoring,
            stress_threshold=self.config.stress_threshold,
            fractal_levels=self.config.fractal_levels,
            memory_capacity=self.config.memory_capacity,
            enable_entity_tracking=self.config.enable_entity_tracking,
            enable_temporal_validation=self.config.enable_temporal_validation,
            enable_causal_checking=self.config.enable_causal_checking,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        # High-level cognitive controls
        user_intent: str | None = None,
        cognitive_load: float | None = None,
        moral_value: float | None = None,
        context_top_k: int | None = None,
        enable_diagnostics: bool = True,
    ) -> dict[str, Any]:
        """Згенерувати відповідь з повним когнітивним наглядом.

        Повертає об'єднаний словник з трьома рівнями:
        - "response": фінальний текст
        - "governance": вихід FSLGS (якщо увімкнено)
        - "mlsdm": повний стан MLSDM з останнього кроку
        """

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

        if self._fslgs is not None:
            # Повний пайплайн: FSLGS поверх MLSDM
            fslgs_result = self._fslgs.generate(
                prompt=prompt,
                cognitive_load=cognitive_load,
                max_tokens=max_tokens,
                user_intent=user_intent,
                enable_diagnostics=enable_diagnostics,
            )
            response_text = fslgs_result.get("response", "")
            mlsdm_state = self._last_mlsdm_state
            return {
                "response": response_text,
                "governance": fslgs_result,
                "mlsdm": mlsdm_state,
            }

        # Фолбек: тільки MLSDM без мовного нагляду
        mlsdm_state = self._mlsdm.generate(
            prompt=prompt,
            moral_value=moral_value,
            max_tokens=max_tokens,
            context_top_k=context_top_k,
        )
        self._last_mlsdm_state = mlsdm_state
        return {
            "response": mlsdm_state.get("response", ""),
            "governance": None,
            "mlsdm": mlsdm_state,
        }

    def get_last_states(self) -> dict[str, Any]:
        """Повертає останні стани для дашбордів / логів."""
        return {
            "mlsdm": self._last_mlsdm_state,
            "has_fslgs": self._fslgs is not None,
        }
