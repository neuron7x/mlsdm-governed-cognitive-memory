"""
Integration tests for all 7 critical integrations.
"""

from mlsdm.integrations import (
    CIHealthMonitor,
    DistributedTracer,
    EmbeddingProvider,
    EmbeddingServiceClient,
    LLMProvider,
    LLMProviderClient,
    SecretProvider,
    SecretsManager,
    WebhookClient,
    WebhookEventType,
)


class TestIntegrationsImports:
    """Test that all integrations can be imported successfully."""

    def test_ci_health_monitor_import(self) -> None:
        """Test CI health monitor import."""
        assert CIHealthMonitor is not None

    def test_embedding_service_import(self) -> None:
        """Test embedding service import."""
        assert EmbeddingServiceClient is not None
        assert EmbeddingProvider is not None

    def test_secrets_manager_import(self) -> None:
        """Test secrets manager import."""
        assert SecretsManager is not None
        assert SecretProvider is not None

    def test_webhook_client_import(self) -> None:
        """Test webhook client import."""
        assert WebhookClient is not None
        assert WebhookEventType is not None

    def test_llm_provider_import(self) -> None:
        """Test LLM provider import."""
        assert LLMProviderClient is not None
        assert LLMProvider is not None

    def test_distributed_tracer_import(self) -> None:
        """Test distributed tracer import."""
        assert DistributedTracer is not None


class TestIntegrationsBasicFunctionality:
    """Test basic functionality of each integration."""

    def test_ci_monitor_creation(self) -> None:
        """Test CI monitor can be created."""
        monitor = CIHealthMonitor()
        assert monitor is not None

    def test_embedding_client_creation(self) -> None:
        """Test embedding client can be created."""
        client = EmbeddingServiceClient()
        assert client is not None

    def test_secrets_manager_creation(self) -> None:
        """Test secrets manager can be created."""
        manager = SecretsManager()
        assert manager is not None

    def test_webhook_client_creation(self) -> None:
        """Test webhook client can be created."""
        client = WebhookClient(webhook_url="https://example.com/webhook")
        assert client is not None

    def test_llm_provider_creation(self) -> None:
        """Test LLM provider client can be created."""
        client = LLMProviderClient()
        assert client is not None

    def test_distributed_tracer_creation(self) -> None:
        """Test distributed tracer can be created."""
        tracer = DistributedTracer()
        assert tracer is not None
