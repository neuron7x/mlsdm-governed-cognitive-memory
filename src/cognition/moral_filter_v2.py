import numpy as np

class MoralFilterV2:
    MIN_THRESHOLD = 0.30
    MAX_THRESHOLD = 0.90
    DEAD_BAND = 0.05
    EMA_ALPHA = 0.1

    def __init__(self, initial_threshold=0.50):
        self.threshold = np.clip(initial_threshold, self.MIN_THRESHOLD, self.MAX_THRESHOLD)
        self.ema_accept_rate = 0.5

    def evaluate(self, moral_value):
        return moral_value >= self.threshold

    def adapt(self, accepted):
        signal = 1.0 if accepted else 0.0
        self.ema_accept_rate = self.EMA_ALPHA * signal + (1.0 - self.EMA_ALPHA) * self.ema_accept_rate
        error = self.ema_accept_rate - 0.5
        if abs(error) > self.DEAD_BAND:
            delta = 0.05 * np.sign(error)
            self.threshold = np.clip(self.threshold + delta, self.MIN_THRESHOLD, self.MAX_THRESHOLD)

    def get_state(self):
        return {"threshold": self.threshold, "ema": self.ema_accept_rate}
