"""
Core speech governance abstractions.

This module defines the contract for speech governance policies that can be
plugged into the LLM wrapper to control or modify LLM outputs according to
various linguistic, safety, or quality policies.

Example policies:
- Aphasia-Broca repair (telegraphic speech detection and correction)
- Content filtering or censorship
- Style enforcement (formal/informal, technical/simple)
- Language correction (grammar, spelling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SpeechGovernanceResult:
    """
    Result of speech governance processing.

    Attributes:
        final_text: The final text after governance processing
        raw_text: The original unprocessed text
        metadata: Additional information about the processing (policy-specific)
    """

    final_text: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SpeechGovernor(Protocol):
    """
    Protocol for speech governance policies.

    A speech governor is a callable that takes an LLM draft response and
    applies some processing or validation policy to it.

    The protocol is intentionally simple to allow maximum flexibility in
    implementation while maintaining a clean contract.
    """

    def __call__(
        self, *, prompt: str, draft: str, max_tokens: int
    ) -> SpeechGovernanceResult:
        """
        Apply speech governance to a draft LLM response.

        Args:
            prompt: The original user prompt
            draft: The raw LLM-generated text
            max_tokens: Maximum tokens requested for generation

        Returns:
            SpeechGovernanceResult with final text and metadata
        """
        ...
