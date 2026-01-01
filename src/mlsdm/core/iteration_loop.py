from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


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


class EnvironmentAdapter(Protocol):
    def reset(self, seed: int | None = None) -> ObservationBundle: ...

    def step(self, action_payload: Any) -> ObservationBundle: ...


def _to_vector(value: float | Sequence[float]) -> list[float]:
    if isinstance(value, (list, tuple)):
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
        if not self.enabled:
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
        new_state, update_result, dynamics = self.apply_updates(state, pe, ctx)
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
        }
        return new_state, trace, safety
