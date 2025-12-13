"""
Webhook Event Integration

Provides webhook support for asynchronous event notifications and callbacks.
"""

import hashlib
import hmac
import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import requests


class WebhookEventType(Enum):
    """Types of webhook events."""

    MORAL_FILTER_REJECT = "moral_filter.reject"
    MORAL_FILTER_ACCEPT = "moral_filter.accept"
    EMERGENCY_SHUTDOWN = "system.emergency_shutdown"
    RECOVERY_TRIGGERED = "system.recovery_triggered"
    MEMORY_THRESHOLD = "memory.threshold_exceeded"
    REQUEST_PROCESSED = "request.processed"
    CUSTOM = "custom.event"


@dataclass
class WebhookEvent:
    """Webhook event payload."""

    event_type: str
    timestamp: float
    data: Dict[str, Any]
    source: str = "mlsdm"
    event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class WebhookClient:
    """
    Webhook client for sending event notifications.

    Supports HMAC signature verification, retry logic, and event batching.

    Example:
        >>> client = WebhookClient(
        ...     webhook_url="https://example.com/webhook",
        ...     secret="webhook_secret"
        ... )
        >>> event = WebhookEvent(
        ...     event_type=WebhookEventType.MORAL_FILTER_REJECT.value,
        ...     timestamp=time.time(),
        ...     data={"prompt": "test", "moral_value": 0.3}
        ... )
        >>> client.send_event(event)
    """

    def __init__(
        self,
        webhook_url: str,
        secret: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ) -> None:
        """
        Initialize webhook client.

        Args:
            webhook_url: Destination webhook URL
            secret: Secret for HMAC signature
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            verify_ssl: Verify SSL certificates
        """
        self.webhook_url = webhook_url
        self.secret = secret
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self.logger = logging.getLogger(__name__)

        self._event_handlers: Dict[str, List[Callable[[WebhookEvent], None]]] = {}

    def send_event(self, event: WebhookEvent) -> bool:
        """
        Send webhook event.

        Args:
            event: Event to send

        Returns:
            True if successfully delivered
        """
        payload = event.to_dict()
        payload_json = json.dumps(payload)

        # Generate signature if secret is configured
        headers = {"Content-Type": "application/json"}
        if self.secret:
            signature = self._generate_signature(payload_json)
            headers["X-Webhook-Signature"] = signature

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.webhook_url,
                    data=payload_json,
                    headers=headers,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )
                response.raise_for_status()

                self.logger.info(
                    f"Webhook delivered successfully: {event.event_type} (attempt {attempt + 1})"
                )
                return True

            except requests.RequestException as e:
                self.logger.warning(
                    f"Webhook delivery failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        self.logger.error(f"Webhook delivery failed after {self.max_retries} attempts")
        return False

    def _generate_signature(self, payload: str) -> str:
        """Generate HMAC signature for payload."""
        if not self.secret:
            return ""

        signature = hmac.new(
            self.secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return f"sha256={signature}"

    def register_handler(
        self, event_type: WebhookEventType, handler: Callable[[WebhookEvent], None]
    ) -> None:
        """
        Register event handler for local processing.

        Args:
            event_type: Event type to handle
            handler: Handler function
        """
        event_key = event_type.value
        if event_key not in self._event_handlers:
            self._event_handlers[event_key] = []
        self._event_handlers[event_key].append(handler)

    def emit_event(self, event: WebhookEvent) -> None:
        """
        Emit event to local handlers and remote webhook.

        Args:
            event: Event to emit
        """
        # Call local handlers
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Event handler failed: {e}")

        # Send to remote webhook
        self.send_event(event)

    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Request payload
            signature: Provided signature
            secret: Webhook secret

        Returns:
            True if signature is valid
        """
        if not signature.startswith("sha256="):
            return False

        expected_sig = hmac.new(
            secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        provided_sig = signature[7:]  # Remove "sha256=" prefix

        return hmac.compare_digest(expected_sig, provided_sig)
