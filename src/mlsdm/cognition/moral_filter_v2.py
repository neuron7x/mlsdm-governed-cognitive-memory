from __future__ import annotations

import re
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

    def __init__(self, initial_threshold: float | None = None) -> None:
        # Use calibration default if not specified
        if initial_threshold is None:
            initial_threshold = MORAL_FILTER_DEFAULTS.threshold if MORAL_FILTER_DEFAULTS else 0.50

        # Validate input
        if not isinstance(initial_threshold, int | float):
            raise TypeError(
                f"initial_threshold must be a number, got {type(initial_threshold).__name__}"
            )

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

    def compute_moral_value(self, text: str) -> float:
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

        # Base score is high (0.8) - "innocent until proven guilty"
        # This ensures neutral text passes normal moral thresholds
        base_score = 0.8

        # Adjust score based on pattern matches
        # Each harmful pattern reduces score by 0.15 (more aggressive penalty)
        # Each positive pattern increases score by 0.05 (max 1.0)
        adjusted_score = base_score - (harmful_count * 0.15) + (positive_count * 0.05)

        # Clamp to valid range
        return max(0.0, min(1.0, adjusted_score))
