import asyncio
import logging
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from src.cognition.moral_filter import MoralFilter
from src.cognition.ontology_matcher import OntologyMatcher
from src.memory.multi_level_memory import MultiLevelSynapticMemory
from src.memory.qilm_module import QILM
from src.rhythm.cognitive_rhythm import CognitiveRhythm
from src.utils.data_serializer import DataSerializer
from src.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.dimension = int(config.get("dimension", 10))

        mem_cfg = config.get("multi_level_memory", {})
        self.memory = MultiLevelSynapticMemory(
            dimension=self.dimension,
            lambda_l1=mem_cfg.get("lambda_l1", 0.5),
            lambda_l2=mem_cfg.get("lambda_l2", 0.1),
            lambda_l3=mem_cfg.get("lambda_l3", 0.01),
            theta_l1=mem_cfg.get("theta_l1", 1.0),
            theta_l2=mem_cfg.get("theta_l2", 2.0),
            gating12=mem_cfg.get("gating12", 0.5),
            gating23=mem_cfg.get("gating23", 0.3),
        )

        filt_cfg = config.get("moral_filter", {})
        self.filter = MoralFilter(
            threshold=filt_cfg.get("threshold", 0.5),
            adapt_rate=filt_cfg.get("adapt_rate", 0.05),
            min_threshold=filt_cfg.get("min_threshold", 0.3),
            max_threshold=filt_cfg.get("max_threshold", 0.9),
        )

        onto_cfg = config.get("ontology_matcher", {})
        ontology_vectors = np.array(onto_cfg.get("ontology_vectors", np.eye(self.dimension).tolist()))
        ontology_labels = onto_cfg.get("ontology_labels")
        self.matcher = OntologyMatcher(ontology_vectors, labels=ontology_labels)

        rhythm_cfg = config.get("cognitive_rhythm", {})
        self.rhythm = CognitiveRhythm(
            wake_duration=rhythm_cfg.get("wake_duration", 5),
            sleep_duration=rhythm_cfg.get("sleep_duration", 2),
        )

        self.qilm = QILM()
        self.metrics_collector = MetricsCollector()
        self.strict_mode = bool(config.get("strict_mode", False))

    def _is_sensitive(self, vec: np.ndarray) -> bool:
        if np.linalg.norm(vec) > 10 or np.sum(vec) < 0:
            logger.warning("Sensitive vector detected by heuristic.")
            return True
        return False

    async def process_event(self, event_vector: np.ndarray, moral_value: float) -> None:
        if self.strict_mode and self._is_sensitive(event_vector):
            raise ValueError("Sensitive data detected in strict mode.")

        self.metrics_collector.start_event_timer()

        total = self.metrics_collector.metrics["total_events_processed"]
        if total > 0:
            accept_rate = (
                self.metrics_collector.metrics["accepted_events_count"] / float(total)
            )
            self.filter.adapt(accept_rate)

        if not self.filter.evaluate(moral_value):
            self.metrics_collector.add_latent_event()
            self.metrics_collector.stop_event_timer_and_record_latency()
            return

        self.memory.update(event_vector)
        self.matcher.match(event_vector, metric="cosine")
        self.qilm.entangle_phase(event_vector, phase=self.rhythm.get_current_phase())

        self.metrics_collector.add_accepted_event()
        self.metrics_collector.stop_event_timer_and_record_latency()

    async def simulate(self, num_steps: int, event_gen: Iterator[Tuple[np.ndarray, float]]) -> None:
        for step, (ev, mv) in enumerate(event_gen):
            if ev.shape[0] != self.dimension:
                raise ValueError("Event dimension mismatch.")
            L1, L2, L3 = self.memory.get_state()
            self.metrics_collector.record_memory_state(step, L1, L2, L3, self.rhythm.get_current_phase())
            self.metrics_collector.record_moral_threshold(self.filter.threshold)

            if self.rhythm.is_wake():
                await self.process_event(ev, mv)

            self.rhythm.step()
            await asyncio.sleep(0)

    def run_simulation(self, num_steps: int, event_gen: Optional[Iterator[Tuple[np.ndarray, float]]] = None) -> None:
        if event_gen is None:
            def default_gen() -> Iterator[Tuple[np.ndarray, float]]:
                for _ in range(num_steps):
                    ev = np.random.randn(self.dimension)
                    mv = float(np.clip(np.random.rand(), 0.0, 1.0))
                    yield ev, mv

            event_gen = default_gen()

        asyncio.run(self.simulate(num_steps, event_gen))

    def save_system_state(self, filepath: str) -> None:
        L1, L2, L3 = self.memory.get_state()
        data = {
            "memory_state": self.memory.to_dict(),
            "qilm": self.qilm.to_dict(),
        }
        DataSerializer.save(data, filepath)

    def load_system_state(self, filepath: str) -> None:
        data = DataSerializer.load(filepath)
        _ = data  # In a full version, restore memory and qilm.
