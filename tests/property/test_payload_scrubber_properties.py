"""
Payload scrubber property tests.

Property-based tests for PayloadScrubber using Hypothesis to validate:
- All PII patterns are removed
- All secret patterns are removed
- Scrubbing never raises exceptions
- Nested structures are fully scrubbed
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsdm.security.payload_scrubber import scrub_text, scrub_request_payload


class TestPayloadScrubberProperties:
    """Property-based tests for payload scrubber invariants."""
    
    @settings(max_examples=100, deadline=None)
    @given(
        api_key=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=20,
            max_size=50,
        )
    )
    def test_api_keys_always_scrubbed(self, api_key):
        """
        Property: API keys are always scrubbed from text.
        
        ∀ text containing api_key: scrub(text) does not contain api_key
        """
        text = f"API_KEY={api_key}"
        scrubbed = scrub_text(text)
        
        # API key should not appear in scrubbed text
        assert api_key not in scrubbed, (
            f"API key not scrubbed: {api_key} still in {scrubbed}"
        )
        
        # Should contain redaction marker
        assert "REDACTED" in scrubbed
    
    @settings(max_examples=50, deadline=None)
    @given(
        password=st.text(min_size=8, max_size=30),
    )
    def test_passwords_always_scrubbed(self, password):
        """Property: Passwords are always scrubbed."""
        text = f"password={password}"
        scrubbed = scrub_text(text)
        
        # Password should not appear in scrubbed text
        assert password not in scrubbed or len(password) < 8, (
            f"Password not scrubbed: {password}"
        )
    
    @settings(max_examples=50, deadline=None)
    @given(
        email=st.emails(),
    )
    def test_email_field_scrubbed_in_dict(self, email):
        """Property: Email fields in dicts are scrubbed."""
        payload = {"email": email}
        scrubbed = scrub_request_payload(payload)
        
        # Email should be scrubbed
        assert scrubbed.get("email") == "***REDACTED***", (
            f"Email not scrubbed: {scrubbed}"
        )
        
        # Original email should not appear
        scrubbed_str = str(scrubbed)
        assert email not in scrubbed_str
    
    @settings(max_examples=50, deadline=None)
    @given(
        nested_depth=st.integers(min_value=1, max_value=5),
    )
    def test_nested_structure_fully_scrubbed(self, nested_depth):
        """Property: PII in nested structures is scrubbed at all levels."""
        # Build nested dict with email at bottom
        payload = {"email": "test@example.com"}
        for _ in range(nested_depth):
            payload = {"nested": payload}
        
        scrubbed = scrub_request_payload(payload)
        
        # Navigate to bottom and verify scrubbing
        current = scrubbed
        for _ in range(nested_depth):
            current = current["nested"]
        
        assert current["email"] == "***REDACTED***", (
            f"Email not scrubbed at depth {nested_depth}"
        )
    
    @settings(max_examples=50, deadline=None)
    @given(
        text=st.text(min_size=0, max_size=200),
    )
    def test_scrubbing_never_raises(self, text):
        """
        Property: Scrubbing never raises exceptions.
        
        ∀ text: scrub_text(text) returns string without raising
        """
        try:
            result = scrub_text(text)
            assert isinstance(result, str), "Result must be string"
        except Exception as e:
            pytest.fail(f"scrub_text raised exception: {e}")
    
    @settings(max_examples=50, deadline=None)
    @given(
        payload=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
                st.none(),
            ),
            min_size=0,
            max_size=10,
        )
    )
    def test_dict_scrubbing_never_raises(self, payload):
        """Property: Dict scrubbing never raises exceptions."""
        try:
            result = scrub_request_payload(payload)
            assert isinstance(result, dict), "Result must be dict"
        except Exception as e:
            pytest.fail(f"scrub_payload raised exception: {e}")
    
    def test_known_pii_fields_scrubbed(self):
        """Property: All known PII field names are scrubbed."""
        pii_fields = [
            "email",
            "ssn",
            "social_security_number",
            "phone",
            "phone_number",
            "address",
            "credit_card",
        ]
        
        for field in pii_fields:
            payload = {field: "sensitive_data"}
            scrubbed = scrub_request_payload(payload)
            
            assert scrubbed[field] == "***REDACTED***", (
                f"PII field '{field}' not scrubbed"
            )
    
    @settings(max_examples=30, deadline=None)
    @given(
        bearer_token=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=20,
            max_size=50,
        )
    )
    def test_bearer_tokens_scrubbed(self, bearer_token):
        """Property: Bearer tokens are always scrubbed."""
        text = f"Authorization: Bearer {bearer_token}"
        scrubbed = scrub_text(text)
        
        # Token should not appear
        assert bearer_token not in scrubbed, (
            f"Bearer token not scrubbed: {bearer_token}"
        )
    
    @settings(max_examples=30, deadline=None)
    @given(
        list_length=st.integers(min_value=1, max_value=10),
    )
    def test_lists_fully_scrubbed(self, list_length):
        """Property: PII in lists is scrubbed."""
        payload = {
            "users": [
                {"email": f"user{i}@example.com"}
                for i in range(list_length)
            ]
        }
        
        scrubbed = scrub_request_payload(payload)
        
        # All emails should be scrubbed
        for user in scrubbed["users"]:
            assert user["email"] == "***REDACTED***", (
                f"Email in list not scrubbed: {user}"
            )
    
    def test_credit_card_numbers_scrubbed(self):
        """Property: Credit card numbers are scrubbed."""
        test_cases = [
            "4111 1111 1111 1111",
            "4111-1111-1111-1111",
            "4111111111111111",
        ]
        
        for cc_number in test_cases:
            text = f"Credit card: {cc_number}"
            scrubbed = scrub_text(text)
            
            # Original number should not appear
            assert cc_number not in scrubbed, (
                f"Credit card not scrubbed: {cc_number}"
            )
            
            # Should contain masked version
            assert "****" in scrubbed
    
    @settings(max_examples=30, deadline=None)
    @given(
        safe_text=st.text(
            alphabet=st.characters(blacklist_characters="@"),
            min_size=1,
            max_size=100,
        )
    )
    def test_safe_text_preserved(self, safe_text):
        """Property: Text without secrets/PII is preserved."""
        # Skip if text accidentally contains PII-like patterns
        if any(word in safe_text.lower() for word in ["email", "password", "token", "key"]):
            return
        
        scrubbed = scrub_text(safe_text)
        
        # Safe text should be mostly preserved (or exactly preserved)
        # Allow for case where it gets scrubbed if it matches a pattern
        if "REDACTED" not in scrubbed:
            assert safe_text in scrubbed or scrubbed in safe_text


# Mark as property test
pytestmark = pytest.mark.property
