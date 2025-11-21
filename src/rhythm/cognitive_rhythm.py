"""Cognitive rhythm for wake/sleep phase cycling."""
from typing import Any, Dict


class CognitiveRhythm:
    """Manages wake/sleep phase cycling for cognitive rhythm.

    Implements a simple counter-based phase cycling system that alternates
    between wake and sleep phases.
    """

    def __init__(self, wake_duration: int = 8, sleep_duration: int = 3) -> None:
        """Initialize cognitive rhythm.

        Args:
            wake_duration: Number of steps in wake phase.
            sleep_duration: Number of steps in sleep phase.
        """
        if wake_duration <= 0 or sleep_duration <= 0:
            raise ValueError("Durations must be positive.")
        self.wake_duration = int(wake_duration)
        self.sleep_duration = int(sleep_duration)
        self.phase = "wake"
        self.counter = self.wake_duration

    def step(self) -> None:
        """Advance one step in the rhythm cycle."""
        self.counter -= 1
        if self.counter <= 0:
            if self.phase == "wake":
                self.phase = "sleep"
                self.counter = self.sleep_duration
            else:
                self.phase = "wake"
                self.counter = self.wake_duration

    def is_wake(self) -> bool:
        """Check if currently in wake phase.

        Returns:
            True if in wake phase, False otherwise.
        """
        return self.phase == "wake"

    def is_sleep(self) -> bool:
        """Check if currently in sleep phase.

        Returns:
            True if in sleep phase, False otherwise.
        """
        return self.phase == "sleep"

    def get_current_phase(self) -> str:
        """Get current phase name.

        Returns:
            Current phase name ('wake' or 'sleep').
        """
        return self.phase

    def to_dict(self) -> Dict[str, Any]:
        """Convert rhythm state to dictionary.

        Returns:
            Dictionary with rhythm configuration and state.
        """
        return {
            "wake_duration": self.wake_duration,
            "sleep_duration": self.sleep_duration,
            "phase": self.phase,
            "counter": self.counter,
        }
