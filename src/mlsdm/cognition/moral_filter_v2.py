from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from config.calibration import MoralFilterCalibration

# Import calibration defaults - these can be overridden via config
# Type hints use Optional to allow None when calibration module unavailable
MORAL_FILTER_DEFAULTS: MoralFilterCalibration | None

try:
    from config.calibration import MORAL_FILTER_DEFAULTS
except ImportError:
    MORAL_FILTER_DEFAULTS = None


class MoralFilterV2:
    # Default class-level constants (overridden by calibration if available)
    MIN_THRESHOLD = MORAL_FILTER_DEFAULTS.min_threshold if MORAL_FILTER_DEFAULTS else 0.30
    MAX_THRESHOLD = MORAL_FILTER_DEFAULTS.max_threshold if MORAL_FILTER_DEFAULTS else 0.90
    DEAD_BAND = MORAL_FILTER_DEFAULTS.dead_band if MORAL_FILTER_DEFAULTS else 0.05
    EMA_ALPHA = MORAL_FILTER_DEFAULTS.ema_alpha if MORAL_FILTER_DEFAULTS else 0.1

    def __init__(self, initial_threshold: float | None = None) -> None:
        # Use calibration default if not specified
        if initial_threshold is None:
            initial_threshold = MORAL_FILTER_DEFAULTS.threshold if MORAL_FILTER_DEFAULTS else 0.50

        # Validate input
        if not isinstance(initial_threshold, (int, float)):
            raise TypeError(
                f"initial_threshold must be a number, got {type(initial_threshold).__name__}"
            )

        self.threshold = np.clip(initial_threshold, self.MIN_THRESHOLD, self.MAX_THRESHOLD)
        self.ema_accept_rate = 0.5

    def evaluate(self, moral_value: float) -> bool:
        # Optimize: fast-path for clear accept/reject cases
        if moral_value >= self.MAX_THRESHOLD:
            return True
        if moral_value < self.MIN_THRESHOLD:
            return False
        return bool(moral_value >= self.threshold)

    def adapt(self, accepted: bool) -> None:
        signal = 1.0 if accepted else 0.0
        self.ema_accept_rate = (
            self.EMA_ALPHA * signal + (1.0 - self.EMA_ALPHA) * self.ema_accept_rate
        )
        error = self.ema_accept_rate - 0.5
        if abs(error) > self.DEAD_BAND:
            delta = 0.05 * np.sign(error)
            self.threshold = np.clip(self.threshold + delta, self.MIN_THRESHOLD, self.MAX_THRESHOLD)

    def get_state(self) -> dict[str, float]:
        return {"threshold": float(self.threshold), "ema": float(self.ema_accept_rate)}

    def get_current_threshold(self) -> float:
        """Get the current moral threshold value.

        Read-only method for introspection - no side effects.

        Returns:
            Current threshold value (0.0-1.0).
        """
        return float(self.threshold)

    def get_ema_value(self) -> float:
        """Get the current EMA (exponential moving average) of acceptance rate.

        Read-only method for introspection - no side effects.

        Returns:
            Current EMA value (0.0-1.0).
        """
        return float(self.ema_accept_rate)
