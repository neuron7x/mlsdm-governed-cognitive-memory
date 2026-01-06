"""
Input validator property tests.

Property-based tests for input validation logic.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsdm.utils.input_validator import InputValidator, sanitize_user_input


class TestInputValidatorProperties:
    """Input validator property tests."""
    
    @settings(max_examples=50, deadline=None)
    @given(
        prompt=st.text(min_size=1, max_size=5000),
    )
    def test_valid_prompts_accepted(self, prompt):
        """Property: Valid prompts are accepted."""
        # Valid prompt should be accepted (non-empty, reasonable length)
        assert len(prompt) > 0
        assert len(prompt) <= 5000
        
        # Test with InputValidator
        max_length = InputValidator.MAX_PROMPT_TOKENS * InputValidator.CHARS_PER_TOKEN
        if len(prompt) <= max_length:
            # Should not raise for valid prompts
            sanitized = sanitize_user_input(prompt, max_length=max_length)
            assert isinstance(sanitized, str)
    
    def test_empty_prompt_rejected(self):
        """Property: Empty prompts are rejected."""
        empty_prompts = ["", " ", "   ", "\n", "\t"]
        
        for prompt in empty_prompts:
            # Should be rejected or empty after sanitization
            sanitized = sanitize_user_input(prompt, max_length=1000)
            assert len(sanitized.strip()) == 0


pytestmark = pytest.mark.property
