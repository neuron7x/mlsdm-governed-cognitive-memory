"""MLSDM Governed Cognitive Memory.

Production-ready neurobiologically-grounded cognitive architecture.

This package provides a universal wrapper for any LLM with hard biological
constraints, including moral governance, phase-based memory, and circadian rhythm.
"""

__version__ = "1.0.0"

# Core components - Universal LLM Wrapper (recommended entry point)
from src.cognition.moral_filter_v2 import MoralFilterV2
from src.core.cognitive_controller import CognitiveController
from src.core.llm_wrapper import LLMWrapper
from src.memory.multi_level_memory import MultiLevelSynapticMemory
from src.memory.qilm_v2 import QILM_v2
from src.rhythm.cognitive_rhythm import CognitiveRhythm

__all__ = [
    "__version__",
    "LLMWrapper",
    "CognitiveController",
    "MoralFilterV2",
    "QILM_v2",
    "MultiLevelSynapticMemory",
    "CognitiveRhythm",
]
