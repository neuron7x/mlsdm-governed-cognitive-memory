from typing import Any


class CognitiveRhythm:
    def __init__(self, wake_duration: int = 8, sleep_duration: int = 3) -> None:
        if wake_duration <= 0 or sleep_duration <= 0:
            raise ValueError("Durations must be positive.")
        self.wake_duration = int(wake_duration)
        self.sleep_duration = int(sleep_duration)
        self.phase = "wake"
        self.counter = self.wake_duration

    def step(self) -> None:
        self.counter -= 1
        if self.counter <= 0:
            if self.phase == "wake":
                self.phase = "sleep"
                self.counter = self.sleep_duration
            else:
                self.phase = "wake"
                self.counter = self.wake_duration

    def is_wake(self) -> bool:
        return self.phase == "wake"

    def is_sleep(self) -> bool:
        return self.phase == "sleep"

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
        return self.phase if self.phase in ("wake", "sleep") else "unknown"
