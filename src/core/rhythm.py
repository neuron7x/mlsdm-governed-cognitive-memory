from __future__ import annotations
from pydantic import BaseModel

class RhythmConfig(BaseModel):
    wake_duration: int = 8
    sleep_duration: int = 3

class CognitiveRhythm:
    def __init__(self, config: RhythmConfig):
        self.cfg = config
        self.counter = config.wake_duration
        self.phase: str = "wake"  # wake → sleep → wake

    def step(self) -> None:
        self.counter -= 1
        if self.counter <= 0:
            self.phase = "sleep" if self.phase == "wake" else "wake"
            self.counter = self.cfg.sleep_duration if self.phase == "sleep" else self.cfg.wake_duration

    def is_wake(self) -> bool:
        return self.phase == "wake"
