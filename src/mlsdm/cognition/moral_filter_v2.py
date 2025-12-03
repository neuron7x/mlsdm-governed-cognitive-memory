from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlsdm.config import MoralFilterCalibration

# Import calibration defaults - these can be overridden via config
# Type hints use Optional to allow None when calibration module unavailable
MORAL_FILTER_DEFAULTS: MoralFilterCalibration | None

try:
    from mlsdm.config import MORAL_FILTER_DEFAULTS
except ImportError:
    MORAL_FILTER_DEFAULTS = None


class MoralFilterV2:
    """Moral filter with optimized threshold adaptation.

    Optimization: Uses pure Python operations instead of numpy for
    scalar operations which is faster for single values.
    """

    # Default class-level constants (overridden by calibration if available)
    MIN_THRESHOLD = MORAL_FILTER_DEFAULTS.min_threshold if MORAL_FILTER_DEFAULTS else 0.30
    MAX_THRESHOLD = MORAL_FILTER_DEFAULTS.max_threshold if MORAL_FILTER_DEFAULTS else 0.90
    DEAD_BAND = MORAL_FILTER_DEFAULTS.dead_band if MORAL_FILTER_DEFAULTS else 0.05
    EMA_ALPHA = MORAL_FILTER_DEFAULTS.ema_alpha if MORAL_FILTER_DEFAULTS else 0.1
    # Pre-computed constants for optimization
    _ONE_MINUS_ALPHA = 1.0 - EMA_ALPHA
    _ADAPT_DELTA = 0.05

    def __init__(self, initial_threshold: float | None = None) -> None:
        # Use calibration default if not specified
        if initial_threshold is None:
            initial_threshold = (
                MORAL_FILTER_DEFAULTS.threshold if MORAL_FILTER_DEFAULTS else 0.50
            )

        # Validate input
        if not isinstance(initial_threshold, (int, float)):
            raise TypeError(f"initial_threshold must be a number, got {type(initial_threshold).__name__}")

        # Optimization: Use pure Python min/max instead of np.clip for scalar
        self.threshold = max(self.MIN_THRESHOLD, min(float(initial_threshold), self.MAX_THRESHOLD))
        self.ema_accept_rate = 0.5

    def evaluate(self, moral_value: float) -> bool:
        # Optimize: fast-path for clear accept/reject cases
        if moral_value >= self.MAX_THRESHOLD:
            return True
        if moral_value < self.MIN_THRESHOLD:
            return False
        return moral_value >= self.threshold

    def adapt(self, accepted: bool) -> None:
        # Optimization: Use pre-computed constant and avoid np.sign for scalar
        signal = 1.0 if accepted else 0.0
        self.ema_accept_rate = self.EMA_ALPHA * signal + self._ONE_MINUS_ALPHA * self.ema_accept_rate
        error = self.ema_accept_rate - 0.5
        if error > self.DEAD_BAND:
            # Positive error - increase threshold
            new_threshold = self.threshold + self._ADAPT_DELTA
            self.threshold = min(new_threshold, self.MAX_THRESHOLD)
        elif error < -self.DEAD_BAND:
            # Negative error - decrease threshold
            new_threshold = self.threshold - self._ADAPT_DELTA
            self.threshold = max(new_threshold, self.MIN_THRESHOLD)

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
