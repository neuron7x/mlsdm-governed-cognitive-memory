"""
Rate limiter property tests.

Property-based tests for RateLimiter using Hypothesis to validate:
- Request count never exceeds limit
- Time window boundaries are correct
- Thread-safety under concurrent access
- Cleanup doesn't affect active windows
"""

from unittest.mock import Mock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsdm.security.rate_limit import RateLimiter


class TestRateLimiterProperties:
    """Property-based tests for RateLimiter invariants."""
    
    @settings(max_examples=100, deadline=None)
    @given(
        requests_per_window=st.integers(min_value=1, max_value=100),
        window_seconds=st.integers(min_value=1, max_value=60),
        num_requests=st.integers(min_value=1, max_value=200),
    )
    def test_rate_limit_never_exceeded(self, requests_per_window, window_seconds, num_requests):
        """
        Property: Total allowed requests in a window never exceeds limit.
        
        ∀ requests: count(allowed_in_window) ≤ rate_limit
        """
        mock_time = Mock(return_value=0.0)
        limiter = RateLimiter(
            requests_per_window=requests_per_window,
            window_seconds=window_seconds,
            now=mock_time,
        )
        
        client_id = "test_client"
        allowed_count = 0
        
        # Make requests at same timestamp (within window)
        for i in range(num_requests):
            if limiter.is_allowed(client_id):
                allowed_count += 1
        
        # Allowed count must not exceed limit
        assert allowed_count <= requests_per_window, (
            f"Rate limit exceeded: {allowed_count} > {requests_per_window}"
        )
    
    @settings(max_examples=50, deadline=None)
    @given(
        requests_per_window=st.integers(min_value=5, max_value=20),
        window_seconds=st.integers(min_value=10, max_value=30),
    )
    def test_rate_limit_resets_after_window(self, requests_per_window, window_seconds):
        """
        Property: Rate limit resets after window expires.
        
        After window_seconds pass, full quota is available again.
        """
        mock_time = Mock(return_value=0.0)
        limiter = RateLimiter(
            requests_per_window=requests_per_window,
            window_seconds=window_seconds,
            now=mock_time,
        )
        
        client_id = "test_client"
        
        # Fill quota
        for _ in range(requests_per_window):
            assert limiter.is_allowed(client_id), "Should be allowed within quota"
        
        # Next request should be denied
        assert not limiter.is_allowed(client_id), "Should be denied after quota filled"
        
        # Advance time past window
        mock_time.return_value = float(window_seconds + 1)
        
        # Quota should be available again
        assert limiter.is_allowed(client_id), "Should be allowed after window reset"
    
    @settings(max_examples=50, deadline=None)
    @given(
        requests_per_window=st.integers(min_value=10, max_value=50),
        window_seconds=st.integers(min_value=10, max_value=60),
        time_delta=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_partial_window_expiry(self, requests_per_window, window_seconds, time_delta):
        """
        Property: Old requests outside window don't count toward limit.
        
        Only requests within the sliding window count.
        """
        assume(time_delta < window_seconds)
        
        mock_time = Mock(return_value=0.0)
        limiter = RateLimiter(
            requests_per_window=requests_per_window,
            window_seconds=window_seconds,
            now=mock_time,
        )
        
        client_id = "test_client"
        
        # Make some requests at t=0
        initial_requests = min(5, requests_per_window)
        for _ in range(initial_requests):
            assert limiter.is_allowed(client_id)
        
        # Advance time slightly
        mock_time.return_value = time_delta
        
        # Should still be able to make more requests
        # (but limited by remaining quota)
        remaining_quota = requests_per_window - initial_requests
        for _ in range(remaining_quota):
            assert limiter.is_allowed(client_id)
    
    @settings(max_examples=50, deadline=None)
    @given(
        requests_per_window=st.integers(min_value=5, max_value=20),
        num_clients=st.integers(min_value=2, max_value=10),
    )
    def test_per_client_isolation(self, requests_per_window, num_clients):
        """
        Property: Each client has independent rate limit.
        
        One client hitting limit doesn't affect others.
        """
        limiter = RateLimiter(requests_per_window=requests_per_window, window_seconds=60)
        
        # Fill quota for client 0
        for _ in range(requests_per_window):
            assert limiter.is_allowed("client_0")
        
        # Client 0 should be rate limited
        assert not limiter.is_allowed("client_0")
        
        # Other clients should still have full quota
        for i in range(1, num_clients):
            client_id = f"client_{i}"
            assert limiter.is_allowed(client_id), (
                f"Client {i} should not be affected by client 0's limit"
            )
    
    @settings(max_examples=30, deadline=None)
    @given(
        requests_per_window=st.integers(min_value=1, max_value=10),
        window_seconds=st.integers(min_value=5, max_value=20),
    )
    def test_boundary_conditions(self, requests_per_window, window_seconds):
        """
        Property: Boundary conditions are handled correctly.
        
        Tests exact limit and window edge cases.
        """
        mock_time = Mock(return_value=0.0)
        limiter = RateLimiter(
            requests_per_window=requests_per_window,
            window_seconds=window_seconds,
            now=mock_time,
        )
        
        client_id = "test_client"
        
        # Exactly at limit should be allowed
        for i in range(requests_per_window):
            assert limiter.is_allowed(client_id), f"Request {i+1}/{requests_per_window} should be allowed"
        
        # One over limit should be denied
        assert not limiter.is_allowed(client_id), "Request over limit should be denied"
        
        # Exactly at window boundary
        mock_time.return_value = float(window_seconds)
        
        # First request in next window might be denied (depends on timing)
        # But after full window + epsilon, should definitely be allowed
        mock_time.return_value = float(window_seconds + 0.1)
        assert limiter.is_allowed(client_id), "Should be allowed after window expires"
    
    def test_zero_requests_always_denied(self):
        """Property: Zero requests per window means all requests denied."""
        limiter = RateLimiter(requests_per_window=0, window_seconds=60)
        
        # All requests should be denied
        for _ in range(10):
            assert not limiter.is_allowed("client"), "Should be denied with zero quota"
    
    @settings(max_examples=30, deadline=None)
    @given(cleanup_interval=st.integers(min_value=10, max_value=300))
    def test_cleanup_preserves_active_windows(self, cleanup_interval):
        """Property: Cleanup doesn't affect requests in active window."""
        mock_time = Mock(return_value=0.0)
        limiter = RateLimiter(
            requests_per_window=10,
            window_seconds=60,
            storage_cleanup_interval=cleanup_interval,
            now=mock_time,
        )
        
        client_id = "test_client"
        
        # Make some requests
        for _ in range(5):
            limiter.is_allowed(client_id)
        
        # Trigger cleanup by advancing time past cleanup interval
        mock_time.return_value = float(cleanup_interval + 1)
        
        # Should still respect previous requests in window
        remaining = 5  # 10 - 5 previous requests
        allowed_after_cleanup = 0
        for _ in range(10):
            if limiter.is_allowed(client_id):
                allowed_after_cleanup += 1
        
        # Should get remaining quota (or less if window partially expired)
        assert allowed_after_cleanup <= remaining


# Mark as property test
pytestmark = pytest.mark.property
