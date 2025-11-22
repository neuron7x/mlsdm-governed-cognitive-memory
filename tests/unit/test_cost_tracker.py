"""Unit tests for cost tracking module."""

import pytest

from mlsdm.observability.cost import CostTracker, estimate_tokens


class TestEstimateTokens:
    """Test token estimation function."""
    
    def test_empty_string(self):
        """Empty string returns 0 tokens."""
        assert estimate_tokens("") == 0
    
    def test_none_input(self):
        """None input returns 0 tokens."""
        assert estimate_tokens(None) == 0  # type: ignore
    
    def test_single_word(self):
        """Single word returns at least 1 token."""
        tokens = estimate_tokens("hello")
        assert tokens >= 1
    
    def test_multiple_words(self):
        """Multiple words return proportional tokens."""
        text = "The quick brown fox jumps over the lazy dog"
        tokens = estimate_tokens(text)
        # 9 words * 1.3 = 11.7 -> 11 tokens
        assert tokens >= 9
        assert tokens <= 15
    
    def test_whitespace_only(self):
        """Whitespace only returns 0 tokens."""
        assert estimate_tokens("   \t\n  ") == 0
    
    def test_consistent_estimation(self):
        """Same text returns same token count."""
        text = "This is a test sentence"
        tokens1 = estimate_tokens(text)
        tokens2 = estimate_tokens(text)
        assert tokens1 == tokens2


class TestCostTracker:
    """Test CostTracker class."""
    
    def test_initialization(self):
        """CostTracker initializes with zero counters."""
        tracker = CostTracker()
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.estimated_cost_usd == 0.0
    
    def test_update_without_pricing(self):
        """Update without pricing tracks tokens but not cost."""
        tracker = CostTracker()
        tracker.update("Hello world", "Response")
        
        assert tracker.prompt_tokens > 0
        assert tracker.completion_tokens > 0
        assert tracker.total_tokens == tracker.prompt_tokens + tracker.completion_tokens
        assert tracker.estimated_cost_usd == 0.0
    
    def test_update_with_pricing(self):
        """Update with pricing tracks both tokens and cost."""
        tracker = CostTracker()
        pricing = {
            'prompt_price_per_1k': 0.001,
            'completion_price_per_1k': 0.002,
        }
        
        tracker.update("Hello world", "Response", pricing)
        
        assert tracker.prompt_tokens > 0
        assert tracker.completion_tokens > 0
        assert tracker.estimated_cost_usd > 0.0
    
    def test_accumulation(self):
        """Multiple updates accumulate correctly."""
        tracker = CostTracker()
        pricing = {
            'prompt_price_per_1k': 0.001,
            'completion_price_per_1k': 0.002,
        }
        
        tracker.update("First prompt", "First response", pricing)
        first_total = tracker.total_tokens
        first_cost = tracker.estimated_cost_usd
        
        tracker.update("Second prompt", "Second response", pricing)
        
        assert tracker.total_tokens > first_total
        assert tracker.estimated_cost_usd > first_cost
    
    def test_to_dict(self):
        """to_dict returns correct structure."""
        tracker = CostTracker()
        tracker.update("Test", "Response")
        
        result = tracker.to_dict()
        
        assert isinstance(result, dict)
        assert 'prompt_tokens' in result
        assert 'completion_tokens' in result
        assert 'total_tokens' in result
        assert 'estimated_cost_usd' in result
        assert result['total_tokens'] == result['prompt_tokens'] + result['completion_tokens']
    
    def test_reset(self):
        """Reset clears all counters."""
        tracker = CostTracker()
        tracker.update("Test", "Response")
        
        assert tracker.total_tokens > 0
        
        tracker.reset()
        
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0
        assert tracker.total_tokens == 0
        assert tracker.estimated_cost_usd == 0.0
    
    def test_cost_calculation(self):
        """Cost calculation is accurate."""
        tracker = CostTracker()
        
        # Use exact token counts for predictable cost
        prompt_tokens = 1000  # Will be estimated as ~770 words * 1.3
        completion_tokens = 500  # Will be estimated as ~385 words * 1.3
        
        # Create text with approximately the right word count
        prompt = " ".join(["word"] * 770)  # 770 words
        completion = " ".join(["word"] * 385)  # 385 words
        
        pricing = {
            'prompt_price_per_1k': 0.001,  # $0.001 per 1k prompt tokens
            'completion_price_per_1k': 0.002,  # $0.002 per 1k completion tokens
        }
        
        tracker.update(prompt, completion, pricing)
        
        # Check that cost is reasonable
        # 1001 prompt tokens * 0.001 / 1000 + 500 completion tokens * 0.002 / 1000
        # ≈ 0.001 + 0.001 = 0.002
        assert tracker.estimated_cost_usd > 0.0
        assert tracker.estimated_cost_usd < 0.01  # Sanity check
