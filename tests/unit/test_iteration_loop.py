from __future__ import annotations

import importlib.util
import sys
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


def test_safety_gate_blocks_runaway_deltas() -> None:
    loop = IterationLoop(enabled=True, delta_max=1.0)
    env = ToyEnvironment(target=0.0, outcomes=[5.0, -5.0])
    state = IterationState(parameter=0.0, learning_rate=0.3)

    new_state, trace, safety = loop.step(state, env, _ctx(0, threat=0.2, risk=0.2))

    assert safety.allow_next is False
    assert trace["update"]["bounded"] is True
    assert abs(new_state.parameter) <= 1.0
