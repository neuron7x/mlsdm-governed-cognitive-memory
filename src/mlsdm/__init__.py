"""
MLSDM Governed Cognitive Memory.

NeuroCognitiveEngine with moral governance, FSLGS integration, and production-ready features.
"""

from .speech.governance import SpeechGovernanceResult, SpeechGovernor

__version__ = "1.2.0"

__all__ = [
    "__version__",
    "SpeechGovernanceResult",
    "SpeechGovernor",
]
