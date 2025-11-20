"""
Moral Filter V2: Adaptive moral threshold with EMA-based homeostasis.

Implements biologically-inspired prefrontal regulation with bounded thresholds
to ensure stable cognitive governance.
"""
from typing import Dict
import numpy as np


class MoralFilterV2:
    """
    Adaptive moral filter with homeostatic bounds.
    
    Maintains moral threshold in [0.30, 0.90] range using exponential moving
    average (EMA) to track acceptance rate and adjust accordingly.
    
    Biological inspiration: Prefrontal cortex value-based decision making.
    """
    
    MIN_THRESHOLD: float = 0.30
    MAX_THRESHOLD: float = 0.90
    DEAD_BAND: float = 0.05
    EMA_ALPHA: float = 0.1

    def __init__(self, initial_threshold: float = 0.50) -> None:
        """
        Initialize moral filter with bounded threshold.
        
        Args:
            initial_threshold: Starting threshold value.
                              Automatically clamped to [MIN_THRESHOLD, MAX_THRESHOLD].
        """
        self.threshold: float = float(
            np.clip(initial_threshold, self.MIN_THRESHOLD, self.MAX_THRESHOLD)
        )
        self.ema_accept_rate: float = 0.5

    def evaluate(self, moral_value: float) -> bool:
        """
        Evaluate if content passes moral threshold.
        
        Args:
            moral_value: Moral score in range [0, 1]
            
        Returns:
            True if content passes threshold, False otherwise
        """
        return bool(moral_value >= self.threshold)

    def adapt(self, accepted: bool) -> None:
        """
        Adapt threshold based on acceptance outcome.
        
        Uses EMA to track acceptance rate and adjusts threshold to maintain
        homeostatic balance around 50% acceptance rate.
        
        Args:
            accepted: Whether the content was accepted
        """
        signal = 1.0 if accepted else 0.0
        self.ema_accept_rate = (
            self.EMA_ALPHA * signal +
            (1.0 - self.EMA_ALPHA) * self.ema_accept_rate
        )
        
        # Clamp EMA to valid range to prevent drift
        self.ema_accept_rate = float(np.clip(self.ema_accept_rate, 0.0, 1.0))
        
        error = self.ema_accept_rate - 0.5
        if abs(error) > self.DEAD_BAND:
            delta = 0.05 * np.sign(error)
            self.threshold = float(
                np.clip(
                    self.threshold + delta,
                    self.MIN_THRESHOLD,
                    self.MAX_THRESHOLD
                )
            )

    def get_state(self) -> Dict[str, float]:
        """
        Get current filter state.
        
        Returns:
            Dictionary with threshold and EMA values
        """
        return {
            "threshold": float(self.threshold),
            "ema": float(self.ema_accept_rate)
        }
