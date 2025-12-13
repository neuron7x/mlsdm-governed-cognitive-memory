"""
Tests for webhook client integration.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from mlsdm.integrations import WebhookClient, WebhookEvent, WebhookEventType


class TestWebhookClient:
    """Test webhook client integration."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = WebhookClient(
            webhook_url="https://example.com/webhook",
            secret="webhook_secret",
            timeout=10,
            max_retries=3,
        )

        assert client.webhook_url == "https://example.com/webhook"
        assert client.secret == "webhook_secret"
        assert client.timeout == 10
        assert client.max_retries == 3

    def test_send_event_success(self) -> None:
        """Test successful webhook event delivery."""
        client = WebhookClient(
            webhook_url="https://example.com/webhook", timeout=5
        )

        event = WebhookEvent(
            event_type=WebhookEventType.MORAL_FILTER_REJECT.value,
            timestamp=time.time(),
            data={"prompt": "test", "moral_value": 0.3},
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status = MagicMock()

            result = client.send_event(event)
            assert result is True

    def test_send_event_with_signature(self) -> None:
        """Test webhook with HMAC signature."""
        client = WebhookClient(
            webhook_url="https://example.com/webhook",
            secret="test_secret",
        )

        event = WebhookEvent(
            event_type=WebhookEventType.EMERGENCY_SHUTDOWN.value,
            timestamp=time.time(),
            data={"reason": "memory_threshold"},
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status = MagicMock()

            result = client.send_event(event)

            # Verify signature header was included
            call_args = mock_post.call_args
            headers = call_args[1]["headers"]
            assert "X-Webhook-Signature" in headers
            assert headers["X-Webhook-Signature"].startswith("sha256=")

    def test_register_and_emit_event(self) -> None:
        """Test local event handler registration and emission."""
        client = WebhookClient(webhook_url="https://example.com/webhook")

        handler_called = []

        def test_handler(event: WebhookEvent) -> None:
            handler_called.append(event.event_type)

        client.register_handler(WebhookEventType.REQUEST_PROCESSED, test_handler)

        event = WebhookEvent(
            event_type=WebhookEventType.REQUEST_PROCESSED.value,
            timestamp=time.time(),
            data={"status": "success"},
        )

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status = MagicMock()

            client.emit_event(event)

        assert len(handler_called) == 1
        assert handler_called[0] == WebhookEventType.REQUEST_PROCESSED.value

    def test_verify_signature(self) -> None:
        """Test signature verification."""
        secret = "test_secret"
        payload = '{"test": "data"}'

        # Generate valid signature
        import hashlib
        import hmac

        expected_sig = hmac.new(
            secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()

        signature = f"sha256={expected_sig}"

        # Test valid signature
        assert WebhookClient.verify_signature(payload, signature, secret) is True

        # Test invalid signature
        assert (
            WebhookClient.verify_signature(payload, "sha256=invalid", secret) is False
        )

    def test_webhook_event_to_dict(self) -> None:
        """Test webhook event serialization."""
        event = WebhookEvent(
            event_type=WebhookEventType.CUSTOM.value,
            timestamp=12345.6,
            data={"key": "value"},
            source="test",
            event_id="evt_123",
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "custom.event"
        assert event_dict["timestamp"] == 12345.6
        assert event_dict["data"] == {"key": "value"}
        assert event_dict["source"] == "test"
        assert event_dict["event_id"] == "evt_123"
