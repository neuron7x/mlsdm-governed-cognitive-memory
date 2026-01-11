"""Aphasia-Broca detection primitives for speech governance."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from mlsdm.config import AphasiaDetectorCalibration


APHASIA_DEFAULTS: AphasiaDetectorCalibration | None

try:
    from mlsdm.config import APHASIA_DEFAULTS
except ImportError:
    APHASIA_DEFAULTS = None


class AphasiaBrocaDetector:
    """Detect telegraphic speech patterns associated with Broca-like aphasia."""

    # Default values from calibration
    DEFAULT_MIN_SENTENCE_LEN = APHASIA_DEFAULTS.min_sentence_len if APHASIA_DEFAULTS else 6.0
    DEFAULT_MIN_FUNCTION_WORD_RATIO = (
        APHASIA_DEFAULTS.min_function_word_ratio if APHASIA_DEFAULTS else 0.15
    )
    DEFAULT_MAX_FRAGMENT_RATIO = APHASIA_DEFAULTS.max_fragment_ratio if APHASIA_DEFAULTS else 0.5
    DEFAULT_FRAGMENT_LENGTH_THRESHOLD = (
        APHASIA_DEFAULTS.fragment_length_threshold if APHASIA_DEFAULTS else 4
    )

    def __init__(
        self,
        min_sentence_len: float | None = None,
        min_function_word_ratio: float | None = None,
        max_fragment_ratio: float | None = None,
    ) -> None:
        # Use calibration defaults if not specified
        if min_sentence_len is None:
            min_sentence_len = self.DEFAULT_MIN_SENTENCE_LEN
        if min_function_word_ratio is None:
            min_function_word_ratio = self.DEFAULT_MIN_FUNCTION_WORD_RATIO
        if max_fragment_ratio is None:
            max_fragment_ratio = self.DEFAULT_MAX_FRAGMENT_RATIO

        self.function_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "that",
            "which",
            "who",
            "to",
            "of",
            "in",
            "on",
            "at",
            "for",
            "with",
            "by",
            "as",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }
        self.min_sentence_len = float(min_sentence_len)
        self.min_function_word_ratio = float(min_function_word_ratio)
        self.max_fragment_ratio = float(max_fragment_ratio)
        self.fragment_length_threshold = self.DEFAULT_FRAGMENT_LENGTH_THRESHOLD

    def analyze(self, text: str) -> AphasiaReport:
        cleaned = text.strip()
        if not cleaned:
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["empty_text"],
            }
        sentences = re.split(r"[.!?]+", cleaned)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return {
                "is_aphasic": True,
                "severity": 1.0,
                "avg_sentence_len": 0.0,
                "function_word_ratio": 0.0,
                "fragment_ratio": 1.0,
                "flags": ["no_sentences"],
            }
        word_counts = [len(s.split()) for s in sentences]
        avg_sentence_len = sum(word_counts) / max(len(word_counts), 1)

        total_words = sum(word_counts)
        function_word_count = sum(
            1 for s in sentences for w in s.split() if w.lower() in self.function_words
        )
        function_word_ratio = function_word_count / max(total_words, 1)

        fragment_count = sum(1 for count in word_counts if count < self.fragment_length_threshold)
        fragment_ratio = fragment_count / max(len(word_counts), 1)

        flags: list[str] = []
        if avg_sentence_len < self.min_sentence_len:
            flags.append("short_sentences")
        if function_word_ratio < self.min_function_word_ratio:
            flags.append("low_function_words")
        if fragment_ratio > self.max_fragment_ratio:
            flags.append("high_fragment_ratio")

        is_aphasic = bool(flags)
        severity = 0.0
        if flags:
            severity = min(
                1.0,
                (
                    max(0.0, self.min_sentence_len - avg_sentence_len) / self.min_sentence_len
                    + max(0.0, self.min_function_word_ratio - function_word_ratio)
                    / max(self.min_function_word_ratio, 1e-6)
                    + max(0.0, fragment_ratio - self.max_fragment_ratio)
                    / max(self.max_fragment_ratio, 1e-6)
                )
                / 3.0,
            )
        return {
            "is_aphasic": is_aphasic,
            "severity": float(severity),
            "avg_sentence_len": float(avg_sentence_len),
            "function_word_ratio": float(function_word_ratio),
            "fragment_ratio": float(fragment_ratio),
            "flags": flags,
        }


class AphasiaReport(TypedDict):
    """Structured report for aphasia detection results."""

    is_aphasic: bool
    severity: float
    avg_sentence_len: float
    function_word_ratio: float
    fragment_ratio: float
    flags: list[str]
