import numpy as np

from src.memory.multi_level_memory import MultiLevelSynapticMemory
from src.cognition.moral_filter import MoralFilter
from src.cognition.ontology_matcher import OntologyMatcher
from src.memory.qilm_module import QILM
from src.rhythm.cognitive_rhythm import CognitiveRhythm
from src.utils.metrics import MetricsCollector


def test_memory_update_and_get_state() -> None:
    mlm = MultiLevelSynapticMemory(dimension=3)
    event = np.array([1.0, 2.0, 3.0])
    mlm.update(event)
    L1, L2, L3 = mlm.get_state()
    assert L1.shape == (3,)
    assert L3.shape == (3,)


def test_moral_filter_basic() -> None:
    mf = MoralFilter()
    assert mf.evaluate(0.6)
    assert not mf.evaluate(0.4)
    mf.adapt(0.4)
    assert 0.3 <= mf.threshold <= 0.9


def test_ontology_matcher() -> None:
    om = OntologyMatcher(np.array([[1, 0, 0], [0, 1, 0]]), labels=["A", "B"])
    label, score = om.match(np.array([1, 0, 0], dtype=float))
    assert label == "A"
    assert score > 0.9


def test_qilm_entangle_and_retrieve() -> None:
    q = QILM()
    v = np.array([1.0, 2.0])
    q.entangle_phase(v, phase=1.0)
    res = q.retrieve(1.0, tolerance=0.0)
    assert len(res) == 1


def test_cognitive_rhythm_cycles() -> None:
    cr = CognitiveRhythm(wake_duration=2, sleep_duration=1)
    assert cr.is_wake()
    cr.step()
    assert cr.is_wake()
    cr.step()
    assert cr.is_sleep()


def test_metrics_collector_entropy_positive() -> None:
    mc = MetricsCollector()
    L1 = np.array([1.0, 0.0])
    L2 = np.array([0.0, 1.0])
    L3 = np.zeros(2)
    mc.record_memory_state(0, L1, L2, L3, "wake")
    m = mc.get_metrics()
    assert m["L1_norm"][0] == np.linalg.norm(L1)
    assert m["entropy_L1"][0] >= 0.0
