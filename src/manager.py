from __future__ import annotations
import asyncio
from typing import Dict, Any
import numpy as np
from src.core.memory import MultiLevelSynapticMemory, MemoryConfig
from src.core.moral import MoralFilter, MoralConfig
from src.core.qilm import QILM
from src.core.rhythm import CognitiveRhythm, RhythmConfig
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class CognitiveMemoryManager:
    def __init__(self, config_dict: Dict[str, Any]):
        self.dim = config_dict['dimension']
        self.strict_mode = config_dict.get('strict_mode', False)

        mem_config = config_dict['multi_level_memory']
        mem_config['dimension'] = self.dim
        self.memory = MultiLevelSynapticMemory(MemoryConfig(**mem_config))
        self.moral = MoralFilter(MoralConfig(**config_dict['moral_filter']))
        self.qilm = QILM(self.dim)
        self.rhythm = CognitiveRhythm(RhythmConfig(**config_dict['cognitive_rhythm']))

        self.metrics = {
            "total": 0, "accepted": 0, "latent": 0
        }

    async def process_event(self, event: np.ndarray, moral_value: float) -> Dict[str, Any]:
        with tracer.start_as_current_span("process_event"):
            self.metrics["total"] += 1

            # Strict mode sensitive detection
            if self.strict_mode and (np.linalg.norm(event) > 20.0 or np.any(event < -5.0)):
                raise ValueError("Sensitive or anomalous vector detected")

            # Moral gating
            if not self.moral.evaluate(moral_value):
                self.metrics["latent"] += 1
                self.rhythm.step()
                return self._state()

            # Rhythm gating
            if not self.rhythm.is_wake():
                self.metrics["latent"] += 1
                self.rhythm.step()
                return self._state()

            # Core update
            self.memory.update(event)
            self.qilm.entangle(event, phase=hash(self.rhythm.phase) % 1000 / 1000)

            self.metrics["accepted"] += 1

            # Adapt moral filter
            if self.metrics["total"] % 50 == 0:
                rate = self.metrics["accepted"] / self.metrics["total"]
                self.moral.adapt(rate)

            self.rhythm.step()
            return self._state()

    def _state(self) -> Dict[str, Any]:
        l1, l2, l3 = self.memory.state()
        n1, n2, n3 = self.memory.norms()
        return {
            "norms": {"L1": n1, "L2": n2, "L3": n3},
            "phase": self.rhythm.phase,
            "moral_threshold": self.moral.threshold,
            "qilm_size": len(self.qilm),
            "metrics": self.metrics.copy()
        }
