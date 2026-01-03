import numpy as np

from mlsdm.core.cognitive_controller import CognitiveController
from mlsdm.core.governance_kernel import GovernanceKernel
from mlsdm.core.llm_wrapper import LLMWrapper


def _dummy_llm(prompt: str, max_tokens: int) -> str:
    return "ok"


def _dummy_embed(text: str) -> np.ndarray:
    vec = np.ones(8, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def test_evaluate_moral_adapts_threshold() -> None:
    kernel = GovernanceKernel(
        dim=4,
        capacity=8,
        wake_duration=2,
        sleep_duration=1,
        initial_moral_threshold=0.5,
    )
    before = kernel.moral_ro.get_state()

    for _ in range(5):
        accepted, used_threshold = kernel.evaluate_moral(0.95)
        assert accepted is True
        assert used_threshold >= before["min_threshold"]
        assert used_threshold <= before["max_threshold"]

    after = kernel.moral_ro.get_state()
    assert after["threshold"] >= before["threshold"]
    assert after["threshold"] <= after["max_threshold"]
    assert after["threshold"] >= after["min_threshold"]


def test_cognitive_controller_uses_kernel_moral_evaluation(monkeypatch) -> None:
    controller = CognitiveController(
        dim=4,
        memory_threshold_mb=1024.0,
        max_processing_time_ms=1000.0,
    )
    calls: list[float] = []

    def fake_eval(moral_value: float) -> tuple[bool, float]:
        calls.append(moral_value)
        return False, 0.5

    monkeypatch.setattr(controller._kernel, "evaluate_moral", fake_eval)  # type: ignore[attr-defined]

    result = controller.process_event(np.ones(4, dtype=np.float32), moral_value=0.8)

    assert calls == [0.8]
    assert result["rejected"] is True
    assert result["note"] == "morally rejected"


def test_llm_wrapper_uses_kernel_moral_evaluation(monkeypatch) -> None:
    wrapper = LLMWrapper(
        llm_generate_fn=_dummy_llm,
        embedding_fn=_dummy_embed,
        dim=8,
        capacity=16,
        wake_duration=2,
        sleep_duration=1,
    )
    calls: list[float] = []

    def fake_eval(moral_value: float) -> tuple[bool, float]:
        calls.append(moral_value)
        return False, 0.5

    monkeypatch.setattr(wrapper._kernel, "evaluate_moral", fake_eval)  # type: ignore[attr-defined]

    result = wrapper.generate(prompt="hello", moral_value=0.9, max_tokens=16)

    assert calls == [0.9]
    assert result["accepted"] is False
    assert result["note"] == "morally rejected"
