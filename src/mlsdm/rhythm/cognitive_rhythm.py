from typing import Any


class CognitiveRhythm:
    """Cognitive rhythm manager with optimized phase tracking.

    Optimization: Uses boolean flag for fast phase checks in hot path,
    while maintaining string phase for compatibility and observability.
    """

    # Phase constants for avoiding repeated string comparisons
    _PHASE_WAKE = "wake"
    _PHASE_SLEEP = "sleep"

    def __init__(self, wake_duration: int = 8, sleep_duration: int = 3) -> None:
        if wake_duration <= 0 or sleep_duration <= 0:
            raise ValueError("Durations must be positive.")
        self.wake_duration = int(wake_duration)
        self.sleep_duration = int(sleep_duration)
        self.phase = self._PHASE_WAKE
        self.counter = self.wake_duration
        # Optimization: Boolean flag for fast phase checks
        self._is_wake = True

    def step(self) -> None:
        self.counter -= 1
        if self.counter <= 0:
            if self._is_wake:
                self.phase = self._PHASE_SLEEP
                self._is_wake = False
                self.counter = self.sleep_duration
            else:
                self.phase = self._PHASE_WAKE
                self._is_wake = True
                self.counter = self.wake_duration

    def is_wake(self) -> bool:
        # Optimization: Direct boolean check instead of string comparison
        return self._is_wake

    def is_sleep(self) -> bool:
        # Optimization: Direct boolean check instead of string comparison
        return not self._is_wake

    def get_current_phase(self) -> str:
        return self.phase

    def to_dict(self) -> dict[str, Any]:
        return {
            "wake_duration": self.wake_duration,
            "sleep_duration": self.sleep_duration,
            "phase": self.phase,
            "counter": self.counter,
        }

    def get_state_label(self) -> str:
        """Get a short label describing the current rhythm state.

        Read-only method for introspection - no side effects.

        Returns:
            State label: "wake", "sleep", or "unknown" if in an unexpected state.
        """
        return self.phase if self.phase in (self._PHASE_WAKE, self._PHASE_SLEEP) else "unknown"
