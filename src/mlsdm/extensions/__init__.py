"""
Extension modules for MLSDM (NeuroLang, Aphasia-Broca, etc.).
"""

from mlsdm.speech.aphasia_detector import AphasiaBrocaDetector

from .neuro_lang_extension import NeuroLangWrapper

__all__ = [
    "NeuroLangWrapper",
    "AphasiaBrocaDetector",
]
