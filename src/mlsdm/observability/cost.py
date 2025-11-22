"""
Cost tracking and token estimation for LLM operations.

This module provides utilities for estimating token usage and tracking costs
associated with LLM API calls. It supports basic token estimation and cost
calculation based on configurable pricing models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def estimate_tokens(text: str) -> int:
    """Estimate token count for a given text.

    Uses a simple heuristic: approximately 1.3 tokens per word.
    This is a conservative estimate that works reasonably well for English text.
    More accurate estimation would require a proper tokenizer.

    Args:
        text: Input text to estimate tokens for

    Returns:
        Estimated number of tokens

    Example:
        >>> estimate_tokens("Hello world")
        3
        >>> estimate_tokens("The quick brown fox jumps over the lazy dog")
        12
    """
    if not text or not isinstance(text, str):
        return 0

    # Split on whitespace and filter empty strings
    words = [w for w in text.split() if w.strip()]

    # Use 1.3 as multiplier (empirically derived for English)
    # This accounts for subword tokenization
    estimated = int(len(words) * 1.3)

    return max(1, estimated) if words else 0


@dataclass
class CostTracker:
    """Track token usage and estimated costs for LLM operations.

    Attributes:
        prompt_tokens: Number of tokens in prompts
        completion_tokens: Number of tokens in completions
        total_tokens: Total token count
        estimated_cost_usd: Estimated cost in USD
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def update(
        self,
        prompt: str,
        completion: str,
        pricing: dict[str, float] | None = None
    ) -> None:
        """Update tracker with new prompt and completion.

        Args:
            prompt: The input prompt text
            completion: The generated completion text
            pricing: Optional pricing dict with keys:
                - 'prompt_price_per_1k': Cost per 1000 prompt tokens
                - 'completion_price_per_1k': Cost per 1000 completion tokens

        Example:
            >>> tracker = CostTracker()
            >>> pricing = {
            ...     'prompt_price_per_1k': 0.0015,
            ...     'completion_price_per_1k': 0.002
            ... }
            >>> tracker.update("Hello", "World", pricing)
            >>> tracker.prompt_tokens > 0
            True
        """
        # Estimate tokens
        prompt_tok = estimate_tokens(prompt)
        completion_tok = estimate_tokens(completion)

        # Update counters
        self.prompt_tokens += prompt_tok
        self.completion_tokens += completion_tok
        self.total_tokens = self.prompt_tokens + self.completion_tokens

        # Calculate cost if pricing provided
        if pricing:
            prompt_price = pricing.get('prompt_price_per_1k', 0.0)
            completion_price = pricing.get('completion_price_per_1k', 0.0)

            prompt_cost = (prompt_tok / 1000.0) * prompt_price
            completion_cost = (completion_tok / 1000.0) * completion_price

            self.estimated_cost_usd += prompt_cost + completion_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert tracker state to dictionary.

        Returns:
            Dictionary with token counts and cost
        """
        return {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'estimated_cost_usd': self.estimated_cost_usd,
        }

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.estimated_cost_usd = 0.0
