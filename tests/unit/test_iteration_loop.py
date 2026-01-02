from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


def _load_iteration_loop_module():
    module_path = Path(__file__).resolve().parents[2] / "src" / "mlsdm" / "core" / "iteration_loop.py"
    spec = importlib.util.spec_from_file_location("iteration_loop", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load iteration_loop module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["iteration_loop"] = module
    spec.loader.exec_module(module)
    return module


iteration_loop = _load_iteration_loop_module()

ActionProposal = iteration_loop.ActionProposal
EnvironmentAdapter = iteration_loop.EnvironmentAdapter
IterationContext = iteration_loop.IterationContext
IterationLoop = iteration_loop.IterationLoop
IterationState = iteration_loop.IterationState
ObservationBundle = iteration_loop.ObservationBundle
PredictionBundle = iteration_loop.PredictionBundle
Regime = iteration_loop.Regime
RegimeController = iteration_loop.RegimeController
PredictionError = iteration_loop.PredictionError
SafetyDecision = iteration_loop.SafetyDecision


class ToyEnvironment(EnvironmentAdapter):
    def __init__(self, target: float = 1.0, outcomes: list[float] | None = None) -> None:
        self.target = target
        self.outcomes = outcomes or []
        self.index = 0

    def reset(self, seed: int | None = None) -> ObservationBundle:
        self.index = 0
        return ObservationBundle(observed_outcome=[self.target], reward=0.0, terminal=False)

    def step(self, action_payload: Any) -> ObservationBundle:
        _ = action_payload
        if self.outcomes:
            value = self.outcomes[min(self.index, len(self.outcomes) - 1)]
        else:
            value = self.target
        self.index += 1
        return ObservationBundle(observed_outcome=[float(value)], reward=None, terminal=False)


def _ctx(step: int, threat: float = 0.0, risk: float = 0.0) -> IterationContext:
    return IterationContext(dt=1.0, timestamp=float(step), seed=42, threat=threat, risk=risk)


def test_disabled_loop_does_not_apply_updates() -> None:
    loop = IterationLoop(enabled=False)
    state = IterationState(parameter=0.0)
    env = ToyEnvironment(target=1.0)
    new_state, trace, safety = loop.step(state, env, _ctx(0))

    assert new_state.parameter == pytest.approx(state.parameter)
    assert trace["update"]["applied"] is False
    assert safety.allow_next is True


def test_delta_learning_reduces_error() -> None:
    loop = IterationLoop(enabled=True, delta_max=1.0)
    env = ToyEnvironment(target=1.0)
    state = IterationState(parameter=0.0, learning_rate=0.2)

    errors: list[float] = []
    for i in range(6):
        state, trace, _ = loop.step(state, env, _ctx(i))
        errors.append(abs(trace["prediction_error"]["delta"][0]))

    assert errors[0] > errors[-1]
    assert sum(errors[-3:]) / 3 < sum(errors[:3]) / 3


def test_threat_switches_regime_and_scales_dynamics() -> None:
    loop = IterationLoop(enabled=True)
    env = ToyEnvironment(target=0.5)
    base_state = IterationState(parameter=0.0, learning_rate=0.2, inhibition_gain=1.0, tau=0.5)

    new_state, trace, _ = loop.step(base_state, env, _ctx(0, threat=0.9, risk=0.8))

    assert new_state.regime == Regime.DEFENSIVE
    assert new_state.last_effective_lr < base_state.learning_rate
    assert new_state.learning_rate == base_state.learning_rate
    assert new_state.inhibition_gain > base_state.inhibition_gain
    assert new_state.tau > base_state.tau
    assert trace["regime"] == Regime.DEFENSIVE.value


def test_regime_controller_cooldown_transitions() -> None:
    controller = RegimeController(caution=0.4, defensive=0.7, cooldown=2)
    state = IterationState(regime=Regime.NORMAL, cooldown_steps=0)
    ctx = IterationContext(dt=1.0, timestamp=0.0, seed=0, threat=0.8, risk=0.0)

    regime, _, _, _, cooldown = controller.update(state, ctx)
    assert regime == Regime.DEFENSIVE
    assert cooldown == 2

    # Drop to caution; remain defensive while cooldown counts down
    ctx_caution = IterationContext(dt=1.0, timestamp=1.0, seed=0, threat=0.5, risk=0.0)
    state = replace(state, regime=regime, cooldown_steps=cooldown)
    regime, _, _, _, cooldown = controller.update(state, ctx_caution)
    assert regime == Regime.DEFENSIVE
    assert cooldown == 1

    # Back to normal threat; cooldown reaches zero then NORMAL
    ctx_normal = IterationContext(dt=1.0, timestamp=2.0, seed=0, threat=0.0, risk=0.0)
    state = replace(state, regime=regime, cooldown_steps=cooldown)
    regime, _, _, _, cooldown = controller.update(state, ctx_normal)
    assert regime == Regime.DEFENSIVE
    state = replace(state, regime=regime, cooldown_steps=cooldown)
    regime, _, _, _, cooldown = controller.update(state, ctx_normal)
    assert regime == Regime.NORMAL


def test_safety_gate_blocks_runaway_deltas() -> None:
    loop = IterationLoop(enabled=True, delta_max=1.0)
    env = ToyEnvironment(target=0.0, outcomes=[5.0, -5.0])
    state = IterationState(parameter=0.0, learning_rate=0.3)

    new_state, trace, safety = loop.step(state, env, _ctx(0, threat=0.2, risk=0.2))

    assert safety.allow_next is False
    assert trace["update"]["bounded"] is True
    assert abs(new_state.parameter) <= 1.0


def test_compute_prediction_error_mismatch_raises() -> None:
    loop = IterationLoop(enabled=True)
    prediction = PredictionBundle(predicted_outcome=[1.0], predicted_value=None)
    observation = ObservationBundle(observed_outcome=[1.0, 0.0])

    with pytest.raises(ValueError):
        loop.compute_prediction_error(prediction, observation, _ctx(0))


def test_apply_updates_clamps_parameter_bounds() -> None:
    loop = IterationLoop(enabled=True, clamp_bounds=(-0.2, 0.2), delta_max=1.0, alpha_max=0.5)
    state = IterationState(parameter=0.0, learning_rate=0.5, inhibition_gain=1.0, tau=1.0)
    ctx = _ctx(0)

    # Force upward clamp
    pe_high = PredictionError(delta=[-10.0], abs_delta=10.0, clipped_delta=[-1.0])
    new_state, update_result, _ = loop.apply_updates(state, pe_high, ctx)
    assert update_result.bounded
    assert new_state.parameter == 0.2

    # Force downward clamp
    pe_low = PredictionError(delta=[10.0], abs_delta=10.0, clipped_delta=[1.0])
    new_state, update_result, _ = loop.apply_updates(state, pe_low, ctx)
    assert update_result.bounded
    assert new_state.parameter == -0.2


def test_stability_envelope_triggers_fail_safe_on_oscillation() -> None:
    loop = IterationLoop(
        enabled=True,
        delta_max=0.5,
        max_regime_flip_rate=0.3,
        max_oscillation_index=0.4,
    )
    env = ToyEnvironment(outcomes=[1.0, -1.0] * 6)
    state = IterationState(parameter=0.0, learning_rate=0.3)

    frozen_seen = False
    safety: SafetyDecision | None = None
    for i in range(12):
        state, trace, safety = loop.step(state, env, _ctx(i, threat=0.9 if i % 2 == 0 else 0.1, risk=0.6))
        if state.frozen:
            frozen_seen = True
            assert trace["regime"] == Regime.DEFENSIVE.value
            assert trace["update"]["applied"] is False
            assert safety.allow_next is False
            assert safety.reason == "stability_envelope_breach"
            assert "oscillation_index" in safety.stability_metrics
            break

    assert frozen_seen
    assert safety is not None


def test_long_sequence_converges_or_halts_safely() -> None:
    loop = IterationLoop(
        enabled=True,
        delta_max=1.0,
        max_regime_flip_rate=0.6,
        max_oscillation_index=0.7,
        convergence_tol=0.3,
    )
    env = ToyEnvironment(outcomes=[0.8 + ((-1) ** i) * 0.05 for i in range(20)])
    state = IterationState(parameter=0.0, learning_rate=0.2)

    final_safety: SafetyDecision | None = None
    for i in range(20):
        state, _, final_safety = loop.step(state, env, _ctx(i, threat=0.3, risk=0.2))
        if state.frozen:
            break

    assert final_safety is not None
    assert (
        not state.frozen and abs(state.last_delta) <= loop.delta_max * loop.convergence_tol
    ) or state.frozen
    assert "convergence_time" in final_safety.stability_metrics
