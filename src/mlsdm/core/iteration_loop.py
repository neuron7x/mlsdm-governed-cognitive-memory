from __future__ import annotations

import json
from collections.abc import Sequence  # noqa: TC003 - needed at runtime for type guard in _to_vector
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pathlib


class Regime(str, Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    DEFENSIVE = "defensive"


@dataclass
class IterationContext:
    dt: float
    timestamp: float
    seed: int
    threat: float = 0.0
    risk: float = 0.0
    regime: Regime = Regime.NORMAL
    mode: str | None = None


@dataclass
class PredictionBundle:
    predicted_outcome: list[float]
    predicted_value: float | None = None
    predicted_uncertainty: float | None = None


@dataclass
class ObservationBundle:
    observed_outcome: list[float]
    reward: float | None = None
    terminal: bool = False


@dataclass
class PredictionError:
    delta: list[float]
    abs_delta: float
    clipped_delta: list[float]
    components: list[float] = field(default_factory=list)


@dataclass
class ActionProposal:
    action_id: str
    action_payload: Any
    scores: list[float] | None = None
    confidence: float | None = None


@dataclass
class UpdateResult:
    parameter_deltas: dict[str, float]
    bounded: bool
    applied: bool


@dataclass
class SafetyDecision:
    allow_next: bool
    reason: str
    stability_metrics: dict[str, float]
    risk_metrics: dict[str, float]
    regime: Regime


@dataclass
class IterationState:
    parameter: float = 0.0
    regime: Regime = Regime.NORMAL
    learning_rate: float = 0.1
    last_effective_lr: float = 0.0
    inhibition_gain: float = 1.0
    tau: float = 1.0
    cooldown_steps: int = 0
    last_delta: float = 0.0
    learning_enabled: bool = True
    kill_switch_active: bool = False
    stability_cooldown: int = 0
    instability_events_count: int = 0
    max_abs_delta: float = 0.0
    time_to_kill_switch: int | None = None
    recovered: bool = False
    step_index: int = 0
    delta_history: list[float] = field(default_factory=list)
    regime_history: list[Regime] = field(default_factory=list)


class EnvironmentAdapter(Protocol):
    def reset(self, seed: int | None = None) -> ObservationBundle: ...

    def step(self, action_payload: Any) -> ObservationBundle: ...


def _to_vector(value: float | Sequence[float]) -> list[float]:
    if not isinstance(value, (int, float)):
        return [float(v) for v in value]
    return [float(value)]


class RegimeController:
    def __init__(self, caution: float = 0.4, defensive: float = 0.7, cooldown: int = 2) -> None:
        self.caution = caution
        self.defensive = defensive
        self.cooldown = cooldown

    def update(self, state: IterationState, ctx: IterationContext) -> tuple[Regime, float, float, float, int]:
        cooldown = state.cooldown_steps
        regime = state.regime
        if ctx.threat >= self.defensive or ctx.risk >= self.defensive:
            regime = Regime.DEFENSIVE
            cooldown = self.cooldown
        elif ctx.threat >= self.caution or ctx.risk >= self.caution:
            if regime == Regime.NORMAL:
                regime = Regime.CAUTION
                cooldown = self.cooldown
            elif regime == Regime.DEFENSIVE and cooldown > 0:
                cooldown -= 1
            else:
                regime = Regime.CAUTION
        else:
            if cooldown > 0:
                cooldown -= 1
                regime = state.regime
            else:
                regime = Regime.NORMAL

        if regime == Regime.NORMAL:
            return regime, 1.0, 1.0, 1.0, cooldown
        if regime == Regime.CAUTION:
            return regime, 0.7, 1.1, 1.2, cooldown
        return regime, 0.4, 1.3, 1.4, cooldown


class IterationLoop:
    STABILITY_WINDOW = 6
    ABS_DELTA_LIMIT = 2.0
    SIGN_FLIP_RATE_LIMIT = 0.6
    REGIME_FLIP_RATE_LIMIT = 0.5
    STABILITY_COOLDOWN_STEPS = 3

    def __init__(
        self,
        *,
        enabled: bool = False,
        delta_max: float = 1.0,
        alpha_min: float = 0.01,
        alpha_max: float = 0.5,
        clamp_bounds: tuple[float, float] = (-1.0, 1.0),
        regime_controller: RegimeController | None = None,
        risk_scale: float = 0.5,
        safety_multiplier: float = 1.5,
        tau_decay: float = 0.9,
        tau_max: float = 5.0,
        metrics_emitter: IterationMetricsEmitter | None = None,
    ) -> None:
        self.enabled = enabled
        self.delta_max = delta_max
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.clamp_bounds = clamp_bounds
        self.regime_controller = regime_controller or RegimeController()
        self.risk_scale = risk_scale
        self.safety_multiplier = safety_multiplier
        self.tau_decay = tau_decay
        self.tau_max = tau_max
        self.metrics_emitter = metrics_emitter

    def propose_action(self, state: IterationState, ctx: IterationContext) -> tuple[ActionProposal, PredictionBundle]:
        predicted = _to_vector(state.parameter)
        proposal = ActionProposal(
            action_id="predict",
            action_payload=state.parameter,
            scores=[state.parameter],
            confidence=1.0,
        )
        bundle = PredictionBundle(predicted_outcome=predicted, predicted_value=state.parameter)
        return proposal, bundle

    def execute_action(self, env: EnvironmentAdapter, proposal: ActionProposal, ctx: IterationContext) -> ObservationBundle:
        _ = ctx  # ctx reserved for future environment modulation
        return env.step(proposal.action_payload)

    def compute_prediction_error(
        self, prediction: PredictionBundle, observation: ObservationBundle, ctx: IterationContext
    ) -> PredictionError:
        _ = ctx
        predicted = prediction.predicted_outcome
        observed = observation.observed_outcome
        if len(predicted) != len(observed):
            raise ValueError("Prediction and observation dimensions must match")
        delta = [p - o for p, o in zip(predicted, observed, strict=True)]
        abs_delta = sum(abs(d) for d in delta) / len(delta) if delta else 0.0
        clipped = [max(-self.delta_max, min(self.delta_max, d)) for d in delta]
        components = [abs(d) for d in delta]
        return PredictionError(delta=delta, abs_delta=abs_delta, clipped_delta=clipped, components=components)

    def apply_updates(
        self,
        state: IterationState,
        pe: PredictionError,
        ctx: IterationContext,
    ) -> tuple[IterationState, UpdateResult, dict[str, float]]:
        if not self.enabled or not state.learning_enabled:
            return state, UpdateResult(parameter_deltas={}, bounded=False, applied=False), {"effective_lr": 0.0}

        regime, lr_scale, inhibition_scale, tau_scale, cooldown = self.regime_controller.update(state, ctx)
        risk_adjusted_lr = state.learning_rate * lr_scale * (1 - ctx.risk * self.risk_scale)
        base_lr = max(self.alpha_min, min(self.alpha_max, risk_adjusted_lr))
        delta_mean = sum(pe.clipped_delta) / len(pe.clipped_delta) if pe.clipped_delta else 0.0
        new_param = state.parameter - base_lr * delta_mean
        bounded = any(abs(d) > self.delta_max for d in pe.delta)

        low, high = self.clamp_bounds
        if new_param < low:
            new_param = low
            bounded = True
        if new_param > high:
            new_param = high
            bounded = True

        smoothed_tau = state.tau * self.tau_decay + ctx.dt * tau_scale * (1 - self.tau_decay)
        bounded_tau = min(self.tau_max, smoothed_tau)

        new_state = replace(
            state,
            parameter=new_param,
            regime=regime,
            learning_rate=state.learning_rate,
            last_effective_lr=base_lr,
            inhibition_gain=state.inhibition_gain * inhibition_scale,
            tau=bounded_tau,
            last_delta=delta_mean,
            cooldown_steps=cooldown,
        )
        return new_state, UpdateResult(parameter_deltas={"parameter": new_param - state.parameter}, bounded=bounded, applied=True), {
            "effective_lr": base_lr,
            "inhibition_scale": inhibition_scale,
            "tau_scale": tau_scale,
        }

    def _update_stability_guard(
        self,
        state: IterationState,
        pe: PredictionError,
    ) -> tuple[IterationState, dict[str, float | int | bool | None]]:
        if not self.enabled:
            guard_metrics = {
                "instability_events_count": state.instability_events_count,
                "max_abs_delta": state.max_abs_delta,
                "time_to_kill_switch": state.time_to_kill_switch,
                "recovered": state.recovered,
                "kill_switch_active": state.kill_switch_active,
                "learning_enabled": state.learning_enabled,
                "cooldown_remaining": state.stability_cooldown,
                "sign_flip_rate": 0.0,
                "regime_flip_rate": 0.0,
            }
            return state, guard_metrics

        delta_signal = sum(pe.delta) / len(pe.delta) if pe.delta else 0.0
        delta_history = (state.delta_history + [delta_signal])[-self.STABILITY_WINDOW :]
        sign_flip_rate = _compute_sign_flip_rate(delta_history)

        regime_history = state.regime_history or [state.regime]
        regime_flip_rate = _compute_regime_flip_rate(regime_history)

        max_abs_delta = max(state.max_abs_delta, pe.abs_delta)
        unstable = (
            pe.abs_delta > self.ABS_DELTA_LIMIT
            or sign_flip_rate > self.SIGN_FLIP_RATE_LIMIT
            or regime_flip_rate > self.REGIME_FLIP_RATE_LIMIT
        )

        step_index = state.step_index + 1
        kill_switch_active = state.kill_switch_active
        learning_enabled = state.learning_enabled
        cooldown = state.stability_cooldown
        instability_events_count = state.instability_events_count
        time_to_kill_switch = state.time_to_kill_switch
        recovered = state.recovered

        if unstable:
            if not kill_switch_active:
                instability_events_count += 1
                if time_to_kill_switch is None:
                    time_to_kill_switch = step_index
            kill_switch_active = True
            learning_enabled = False
            cooldown = self.STABILITY_COOLDOWN_STEPS
            recovered = False
        elif kill_switch_active:
            if cooldown > 0:
                cooldown -= 1
            else:
                kill_switch_active = False
                learning_enabled = True
                recovered = True

        updated_state = replace(
            state,
            delta_history=delta_history,
            max_abs_delta=max_abs_delta,
            kill_switch_active=kill_switch_active,
            learning_enabled=learning_enabled,
            stability_cooldown=cooldown,
            instability_events_count=instability_events_count,
            time_to_kill_switch=time_to_kill_switch,
            recovered=recovered,
            step_index=step_index,
        )
        guard_metrics = {
            "instability_events_count": instability_events_count,
            "max_abs_delta": max_abs_delta,
            "time_to_kill_switch": time_to_kill_switch,
            "recovered": recovered,
            "kill_switch_active": kill_switch_active,
            "learning_enabled": learning_enabled,
            "cooldown_remaining": cooldown,
            "sign_flip_rate": sign_flip_rate,
            "regime_flip_rate": regime_flip_rate,
        }
        return updated_state, guard_metrics

    def evaluate_safety(self, state: IterationState, pe: PredictionError, ctx: IterationContext) -> SafetyDecision:
        abs_max = max((abs(d) for d in pe.delta), default=0.0)
        allow = abs_max <= self.delta_max * self.safety_multiplier
        reason = "stable"
        if state.regime == Regime.DEFENSIVE and abs_max > self.delta_max:
            allow = False
            reason = "defensive_clamp"
        if not allow and reason == "stable":
            reason = "delta_exceeds_bounds"

        stability = {
            "abs_delta": pe.abs_delta,
            "abs_delta_max": abs_max,
            "tau": state.tau,
            "inhibition_gain": state.inhibition_gain,
        }
        risks = {"threat": ctx.threat, "risk": ctx.risk}
        return SafetyDecision(
            allow_next=allow,
            reason=reason,
            stability_metrics=stability,
            risk_metrics=risks,
            regime=state.regime,
        )

    def step(
        self,
        state: IterationState,
        env: EnvironmentAdapter,
        ctx: IterationContext,
    ) -> tuple[IterationState, dict[str, Any], SafetyDecision]:
        proposal, prediction = self.propose_action(state, ctx)
        observation = self.execute_action(env, proposal, ctx)
        pe = self.compute_prediction_error(prediction, observation, ctx)
        if not self.enabled:
            new_state, update_result, dynamics = self.apply_updates(state, pe, ctx)
            guard_metrics = {
                "instability_events_count": state.instability_events_count,
                "max_abs_delta": state.max_abs_delta,
                "time_to_kill_switch": state.time_to_kill_switch,
                "recovered": state.recovered,
                "kill_switch_active": state.kill_switch_active,
                "learning_enabled": state.learning_enabled,
                "cooldown_remaining": state.stability_cooldown,
                "sign_flip_rate": 0.0,
                "regime_flip_rate": 0.0,
            }
            safety = self.evaluate_safety(new_state, pe, ctx)
            trace = {
                "action": {
                    "id": proposal.action_id,
                    "payload": proposal.action_payload,
                    "scores": proposal.scores,
                    "confidence": proposal.confidence,
                },
                "prediction": prediction.predicted_outcome,
                "observation": observation.observed_outcome,
                "prediction_error": {
                    "delta": pe.delta,
                    "abs_delta": pe.abs_delta,
                    "clipped_delta": pe.clipped_delta,
                },
                "regime": new_state.regime.value,
                "dynamics": {
                    "learning_rate": new_state.learning_rate,
                    "effective_learning_rate": new_state.last_effective_lr,
                    "inhibition_gain": new_state.inhibition_gain,
                    "tau": new_state.tau,
                    **dynamics,
                },
                "update": {
                    "parameter_deltas": update_result.parameter_deltas,
                    "bounded": update_result.bounded,
                    "applied": update_result.applied,
                },
                "safety": {
                    "allow_next": safety.allow_next,
                    "reason": safety.reason,
                    "stability_metrics": safety.stability_metrics,
                    "risk_metrics": safety.risk_metrics,
                    "regime": safety.regime.value,
                },
                "stability_guard": guard_metrics,
            }
            if self.metrics_emitter and self.metrics_emitter._should_emit():
                self.metrics_emitter.emit(ctx, trace)
            return new_state, trace, safety

        guard_state, guard_metrics = self._update_stability_guard(state, pe)

        if guard_state.kill_switch_active:
            new_state = replace(
                guard_state,
                regime=Regime.DEFENSIVE,
                last_effective_lr=0.0,
            )
            bounded = any(abs(d) > self.delta_max for d in pe.delta)
            update_result = UpdateResult(parameter_deltas={}, bounded=bounded, applied=False)
            dynamics = {"effective_lr": 0.0, "inhibition_scale": 1.0, "tau_scale": 1.0}
        else:
            new_state, update_result, dynamics = self.apply_updates(guard_state, pe, ctx)

        new_regime_history = (new_state.regime_history + [new_state.regime])[-self.STABILITY_WINDOW :]
        new_state = replace(new_state, regime_history=new_regime_history)
        safety = self.evaluate_safety(new_state, pe, ctx)

        trace = {
            "action": {
                "id": proposal.action_id,
                "payload": proposal.action_payload,
                "scores": proposal.scores,
                "confidence": proposal.confidence,
            },
            "prediction": prediction.predicted_outcome,
            "observation": observation.observed_outcome,
            "prediction_error": {
                "delta": pe.delta,
                "abs_delta": pe.abs_delta,
                "clipped_delta": pe.clipped_delta,
            },
            "regime": new_state.regime.value,
            "dynamics": {
                "learning_rate": new_state.learning_rate,
                "effective_learning_rate": new_state.last_effective_lr,
                "inhibition_gain": new_state.inhibition_gain,
                "tau": new_state.tau,
                **dynamics,
            },
            "update": {
                "parameter_deltas": update_result.parameter_deltas,
                "bounded": update_result.bounded,
                "applied": update_result.applied,
            },
            "safety": {
                "allow_next": safety.allow_next,
                "reason": safety.reason,
                "stability_metrics": safety.stability_metrics,
                "risk_metrics": safety.risk_metrics,
                "regime": safety.regime.value,
            },
            "stability_guard": guard_metrics,
        }
        if self.metrics_emitter and self.metrics_emitter._should_emit():
            self.metrics_emitter.emit(ctx, trace)
        return new_state, trace, safety


@dataclass
class IterationMetricsEmitter:
    enabled: bool = False
    output_path: pathlib.Path | None = None
    _prepared: bool = False

    def emit(self, ctx: IterationContext, trace: dict[str, Any]) -> None:
        if not self._should_emit():
            return
        if self.output_path is None:
            return
        if not self._prepared:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._prepared = True
        record = {
            "timestamp": ctx.timestamp,
            "dt": ctx.dt,
            "seed": ctx.seed,
            "threat": ctx.threat,
            "risk": ctx.risk,
            "regime": trace.get("regime"),
            "prediction_error": trace.get("prediction_error"),
            "action": trace.get("action", {}),
            "dynamics": trace.get("dynamics", {}),
            "safety": trace.get("safety", {}),
        }
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _should_emit(self) -> bool:
        return self.enabled and self.output_path is not None


def _compute_sign_flip_rate(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    signs = [_sign(value) for value in values]
    flips = 0
    for prev, current in zip(signs, signs[1:], strict=True):
        if prev == 0 or current == 0:
            continue
        if prev != current:
            flips += 1
    return flips / max(1, len(signs) - 1)


def _compute_regime_flip_rate(values: list[Regime]) -> float:
    if len(values) < 2:
        return 0.0
    flips = sum(1 for prev, current in zip(values, values[1:], strict=True) if prev != current)
    return flips / max(1, len(values) - 1)


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0
