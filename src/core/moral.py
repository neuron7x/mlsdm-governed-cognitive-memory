from __future__ import annotations
from pydantic import BaseModel

class MoralConfig(BaseModel):
    threshold: float = 0.5
    adapt_rate: float = 0.05
    min_threshold: float = 0.30
    max_threshold: float = 0.90

class MoralFilter:
    def __init__(self, config: MoralConfig):
        self.cfg = config
        self.threshold = config.threshold

    def evaluate(self, value: float) -> bool:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Moral value must be in [0,1]")
        return value >= self.threshold

    def adapt(self, accept_rate: float) -> None:
        if accept_rate < 0.5:
            self.threshold = max(self.cfg.min_threshold, self.threshold - self.cfg.adapt_rate)
        else:
            self.threshold = min(self.cfg.max_threshold, self.threshold + self.cfg.adapt_rate)
