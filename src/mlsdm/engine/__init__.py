"""
High-level orchestration layer for MLSDM + FSLGS.

Exposes NeuroCognitiveEngine as a single entry point that composes:
- MLSDM LLMWrapper (memory, rhythm, moral governance, reliability)
- FSLGSWrapper (dual-stream language, anti-schizophrenia, UG constraints)
"""

from .neuro_cognitive_engine import (
    EmptyResponseError,
    MLSDMRejectionError,
    NeuroCognitiveEngine,
    NeuroEngineConfig,
)

__all__ = [
    "NeuroCognitiveEngine",
    "NeuroEngineConfig",
    "MLSDMRejectionError",
    "EmptyResponseError",
]
