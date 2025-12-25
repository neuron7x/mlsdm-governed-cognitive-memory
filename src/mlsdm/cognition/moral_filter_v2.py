from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlsdm.config import MoralFilterCalibration

# Import drift telemetry
from mlsdm.observability.policy_drift_telemetry import record_threshold_change

logger = logging.getLogger(__name__)

# Import calibration defaults - these can be overridden via config
# Type hints use Optional to allow None when calibration module unavailable
MORAL_FILTER_DEFAULTS: MoralFilterCalibration | None

try:
    from mlsdm.config import MORAL_FILTER_DEFAULTS
except ImportError:
    MORAL_FILTER_DEFAULTS = None

# Pre-compiled regex patterns for word boundary matching (module-level for performance)
# These patterns match whole words only to avoid false positives like "harm" in "pharmacy"
_HARMFUL_PATTERNS = [
    "hate",
    "violence",
    "attack",
    "kill",
    "destroy",
    "harm",
    "abuse",
    "exploit",
    "discriminate",
    "racist",
    "sexist",
    "terrorist",
    "weapon",
    "bomb",
    "murder",
]
_POSITIVE_PATTERNS = [
    "help",
    "support",
    "care",
    "love",
    "kind",
    "respect",
    "ethical",
    "fair",
    "honest",
    "trust",
    "safe",
    "protect",
    "collaborate",
    "peace",
    "understanding",
]
# Compile single regex patterns for O(n) matching instead of O(n*m)
_HARMFUL_REGEX = re.compile(r"\b(" + "|".join(_HARMFUL_PATTERNS) + r")\b", re.IGNORECASE)
_POSITIVE_REGEX = re.compile(r"\b(" + "|".join(_POSITIVE_PATTERNS) + r")\b", re.IGNORECASE)


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
    _BOUNDARY_EPS = 0.01

    def __init__(
        self, initial_threshold: float | None = None, filter_id: str = "default"
    ) -> None:
        # Use calibration default if not specified
        if initial_threshold is None:
            initial_threshold = (
                MORAL_FILTER_DEFAULTS.threshold if MORAL_FILTER_DEFAULTS else 0.50
            )

        # Validate input
        if not isinstance(initial_threshold, int | float):
            raise TypeError(
                f"initial_threshold must be a number, got {type(initial_threshold).__name__}"
            )

        # Optimization: Use pure Python min/max instead of np.clip for scalar
        self.threshold = max(
            self.MIN_THRESHOLD, min(float(initial_threshold), self.MAX_THRESHOLD)
        )
        self.ema_accept_rate = 0.5

        # NEW: Drift detection
        self._filter_id = filter_id
        self._drift_history: list[float] = []
        self._max_history = 100  # Keep last 100 changes

        # Initialize metrics
        record_threshold_change(
            filter_id=self._filter_id,
            old_threshold=self.threshold,
            new_threshold=self.threshold,
            ema_value=self.ema_accept_rate,
        )

    def evaluate(self, moral_value: float) -> bool:
        if logger.isEnabledFor(logging.DEBUG):
            self._log_boundary_cases(moral_value)

        # Optimize: fast-path for clear accept/reject cases
        if moral_value >= self.MAX_THRESHOLD:
            return True
        if moral_value < self.MIN_THRESHOLD:
            return False
        return moral_value >= self.threshold

    def adapt(self, accepted: bool) -> None:
        """Adapt threshold with drift detection."""
        # Store old value for drift calculation
        old_threshold = self.threshold

        # Existing adaptation logic
        signal = 1.0 if accepted else 0.0
        self.ema_accept_rate = (
            self.EMA_ALPHA * signal + self._ONE_MINUS_ALPHA * self.ema_accept_rate
        )
        error = self.ema_accept_rate - 0.5

        if error > self.DEAD_BAND:
            # Positive error - increase threshold
            new_threshold = self.threshold + self._ADAPT_DELTA
            self.threshold = min(new_threshold, self.MAX_THRESHOLD)
        elif error < -self.DEAD_BAND:
            # Negative error - decrease threshold
            new_threshold = self.threshold - self._ADAPT_DELTA
            self.threshold = max(new_threshold, self.MIN_THRESHOLD)

        # NEW: Record drift if threshold changed
        if self.threshold != old_threshold:
            logger.debug(
                "MoralFilterV2 threshold adapted: %.3f -> %.3f (ema=%.3f error=%.3f)",
                old_threshold,
                self.threshold,
                self.ema_accept_rate,
                error,
            )
            self._record_drift(old_threshold, self.threshold)

    def get_state(self) -> dict[str, float]:
        return {
            "threshold": float(self.threshold),
            "ema": float(self.ema_accept_rate),
            "min_threshold": float(self.MIN_THRESHOLD),
            "max_threshold": float(self.MAX_THRESHOLD),
            "dead_band": float(self.DEAD_BAND),
            "ema_alpha": float(self.EMA_ALPHA),
        }

    def _log_boundary_cases(self, moral_value: float) -> None:
        """Log boundary cases for moral evaluation at DEBUG level."""
        is_near_min = abs(moral_value - self.MIN_THRESHOLD) <= self._BOUNDARY_EPS
        is_near_max = abs(moral_value - self.MAX_THRESHOLD) <= self._BOUNDARY_EPS
        is_near_threshold = abs(moral_value - self.threshold) <= self._BOUNDARY_EPS

        if is_near_min or is_near_max or is_near_threshold:
            logger.debug(
                "MoralFilterV2 boundary case: value=%.3f threshold=%.3f min=%.3f max=%.3f",
                moral_value,
                self.threshold,
                self.MIN_THRESHOLD,
                self.MAX_THRESHOLD,
            )

    def _record_drift(self, old: float, new: float) -> None:
        """Record and analyze threshold drift.

        Args:
            old: Previous threshold value
            new: New threshold value

        Side Effects:
            - Updates drift history
            - Records Prometheus metrics
            - Logs warnings/errors for significant drift
        """
        # Update drift history
        self._drift_history.append(new)
        if len(self._drift_history) > self._max_history:
            self._drift_history.pop(0)

        # Record metrics
        record_threshold_change(
            filter_id=self._filter_id,
            old_threshold=old,
            new_threshold=new,
            ema_value=self.ema_accept_rate,
        )

        # Analyze for anomalous drift
        drift_magnitude = abs(new - old)

        # Note: These are absolute threshold differences, not percentages
        # since thresholds are in [0.3, 0.9] range, 0.1 absolute = ~17% relative change
        if drift_magnitude > 0.1:  # >0.1 absolute change (large single jump)
            logger.error(
                f"CRITICAL DRIFT: threshold changed {drift_magnitude:.3f} "
                f"({old:.3f} → {new:.3f}) for filter '{self._filter_id}'"
            )
        elif drift_magnitude > 0.05:  # >0.05 absolute change (moderate jump)
            logger.warning(
                f"Significant drift: threshold changed {drift_magnitude:.3f} "
                f"({old:.3f} → {new:.3f}) for filter '{self._filter_id}'"
            )

        # Check for sustained drift (trend over history)
        if len(self._drift_history) >= 10:
            recent_drift = self._drift_history[-1] - self._drift_history[-10]
            if abs(recent_drift) > 0.15:  # >0.15 absolute change over 10 operations
                logger.error(
                    f"SUSTAINED DRIFT: threshold drifted {recent_drift:.3f} "
                    f"over last 10 operations for filter '{self._filter_id}'"
                )

    def get_drift_stats(self) -> dict[str, float]:
        """Get drift statistics.

        Returns:
            Dictionary with drift statistics including:
            - total_changes: Number of threshold changes recorded
            - drift_range: Range of threshold values seen
            - min_threshold: Minimum threshold in history
            - max_threshold: Maximum threshold in history
            - current_threshold: Current threshold value
            - ema_acceptance: Current EMA acceptance rate
        """
        if len(self._drift_history) < 2:
            return {
                "total_changes": 0,
                "drift_range": 0.0,
                "current_threshold": self.threshold,
            }

        return {
            "total_changes": len(self._drift_history),
            "drift_range": max(self._drift_history) - min(self._drift_history),
            "min_threshold": min(self._drift_history),
            "max_threshold": max(self._drift_history),
            "current_threshold": self.threshold,
            "ema_acceptance": self.ema_accept_rate,
        }

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

    def compute_moral_value(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Compute a moral value score for the given text.

        This is a heuristic-based scoring method that analyzes text for
        potentially harmful patterns. The approach is "innocent until proven
        guilty" - text is considered acceptable (high score) unless harmful
        patterns are detected.

        Uses pre-compiled regex patterns with word boundary matching to avoid
        false positives (e.g., "harm" won't match "pharmacy" or "harmless").
        O(n) complexity for text length n due to single-pass regex matching.

        Args:
            text: Input text to analyze for moral content.

        Returns:
            Moral value score in [0.0, 1.0] where higher is more acceptable.
            - 0.8: Neutral/normal text (no harmful patterns)
            - 0.3-0.7: Text with some harmful patterns
            - <0.3: Text with multiple harmful patterns

        Note:
            This implementation uses simple pattern matching. For production
            use with higher accuracy, consider integrating with toxicity
            detection APIs (e.g., Perspective API) or fine-tuned classifiers.
        """
        if not text or not text.strip():
            return 0.8  # Assume empty text is acceptable

        # Use pre-compiled regex patterns with word boundary matching
        # This is O(n) for text length and avoids false positives
        harmful_matches = _HARMFUL_REGEX.findall(text)
        positive_matches = _POSITIVE_REGEX.findall(text)

        harmful_count = len(harmful_matches)
        positive_count = len(positive_matches)
        if metadata is not None:
            metadata["harmful_count"] = harmful_count
            metadata["positive_count"] = positive_count
        if context is not None:
            context_metadata = context.setdefault("metadata", {})
            if isinstance(context_metadata, dict):
                context_metadata["harmful_count"] = harmful_count
                context_metadata["positive_count"] = positive_count

        # Base score is high (0.8) - "innocent until proven guilty"
        # This ensures neutral text passes normal moral thresholds
        base_score = 0.8

        # Adjust score based on pattern matches
        # Each harmful pattern reduces score by 0.15 (more aggressive penalty)
        # Each positive pattern increases score by 0.05 (max 1.0)
        adjusted_score = base_score - (harmful_count * 0.15) + (positive_count * 0.05)

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted_score))
