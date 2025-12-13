"""
MLSDM Integrations Module

Provides integration layers for external services and systems.

This module contains 7 critical integrations for MLSDM system stabilization:
1. CI Health Monitoring - Auto-recovery integration for CI pipelines
2. Embedding Service - External embedding API integration (OpenAI, Cohere, HF)
3. Secrets Management - HashiCorp Vault, AWS Secrets Manager integration
4. Redis Cache - Distributed caching for multi-instance deployments
5. Webhook Events - Async event notifications and callbacks
6. LLM Provider - Unified interface for multiple LLM providers
7. Distributed Tracing - Complete OpenTelemetry integration
"""

from mlsdm.integrations.ci_health_monitor import CIHealthMonitor, CIStatus
from mlsdm.integrations.distributed_tracing import DistributedTracer
from mlsdm.integrations.embedding_service import (
    EmbeddingProvider,
    EmbeddingServiceClient,
)
from mlsdm.integrations.llm_provider import LLMProvider, LLMProviderClient
from mlsdm.integrations.redis_cache import RedisCache
from mlsdm.integrations.secrets_manager import SecretProvider, SecretsManager
from mlsdm.integrations.webhook_client import WebhookClient, WebhookEvent, WebhookEventType

__all__ = [
    # CI Health Monitoring
    "CIHealthMonitor",
    "CIStatus",
    # Embedding Service
    "EmbeddingServiceClient",
    "EmbeddingProvider",
    # Secrets Management
    "SecretsManager",
    "SecretProvider",
    # Redis Cache
    "RedisCache",
    # Webhook Events
    "WebhookClient",
    "WebhookEvent",
    "WebhookEventType",
    # LLM Provider
    "LLMProvider",
    "LLMProviderClient",
    # Distributed Tracing
    "DistributedTracer",
]
