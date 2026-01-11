from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegimeThresholds:
    caution_threshold: float = 0.55
    defensive_threshold: float = 0.8
    hysteresis: float = 0.08
    cooldown: int = 2


@dataclass(frozen=True)
class RegimeScales:
    normal_lr_scale: float = 1.0
    normal_inhibition_scale: float = 1.0
    normal_tau_scale: float = 1.0
    caution_lr_scale: float = 0.7
    caution_inhibition_scale: float = 1.1
    caution_tau_scale: float = 1.2
    defensive_lr_scale: float = 0.4
    defensive_inhibition_scale: float = 1.3
    defensive_tau_scale: float = 1.4


@dataclass(frozen=True)
class IterationGuardConfig:
    guard_window: int = 10
    max_abs_delta: float = 1.5
    max_sign_flip_rate: float = 0.6
    max_regime_flip_rate: float = 0.5
    cooldown_steps: int = 2


@dataclass(frozen=True)
class NeuroRegimeTuning:
    """
    Tuning parameters for regime dynamics and update scaling.

    Values are bounded to preserve legacy behavior: NORMAL tracks the previous
    dynamics, CAUTION increases inhibition and slightly shortens τ, DEFENSIVE
    applies the strongest inhibition with aggressive τ shortening.
    """

    min_update_scale: float = 0.2
    max_update_scale: float = 2.0
    normal_exploration_base: float = 0.25
    normal_inhibition_slope: float = 0.1
    normal_tau_scale: float = 1.0
    caution_inhibition_base: float = 1.15
    caution_inhibition_slope: float = 0.35
    caution_exploration_base: float = 0.22
    caution_exploration_min: float = 0.12
    caution_tau_min: float = 0.75
    caution_tau_slope: float = 0.25
    defensive_inhibition_base: float = 1.35
    defensive_inhibition_slope: float = 0.45
    defensive_exploration_base: float = 0.18
    defensive_exploration_min: float = 0.08
    defensive_tau_min: float = 0.6
    defensive_tau_slope: float = 0.4


DEFAULT_REGIME_THRESHOLDS = RegimeThresholds()
DEFAULT_REGIME_SCALES = RegimeScales()
DEFAULT_ITERATION_GUARD = IterationGuardConfig()
DEFAULT_NEURO_TUNING = NeuroRegimeTuning()
