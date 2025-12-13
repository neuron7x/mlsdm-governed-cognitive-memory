# MLSDM Critical Integrations Guide

**Version**: 1.2.0+  
**Last Updated**: December 2025  
**Status**: Production Ready

This guide documents the 7 critically necessary integrations for MLSDM system stabilization.

---

## Overview

The MLSDM integrations module provides enterprise-grade integration layers for external services and systems, enabling:

- **Production Deployment**: CI/CD health monitoring with auto-recovery
- **Scalability**: Distributed caching and tracing for multi-instance deployments
- **Flexibility**: Pluggable LLM and embedding providers
- **Security**: External secret management integration
- **Observability**: Event webhooks and distributed tracing

---

## 7 Critical Integrations

### 1. CI Health Monitoring (`CIHealthMonitor`)

**Purpose**: Monitor CI pipeline health and integrate with auto-recovery mechanisms.

**Features**:
- GitHub Actions workflow status monitoring
- Consecutive failure tracking
- Automatic recovery workflow dispatch
- Customizable failure thresholds and cooldown periods

**Example**:
```python
from mlsdm.integrations import CIHealthMonitor

# Initialize monitor
monitor = CIHealthMonitor(
    github_token="ghp_xxxxx",
    repository="neuron7x/mlsdm",
    failure_threshold=3,
    recovery_cooldown_seconds=300,
    enable_auto_recovery=True
)

# Register recovery callback
def on_recovery():
    print("Triggering system recovery...")
    # Restart services, clear caches, etc.

monitor.register_recovery_callback(on_recovery)

# Check health periodically
health = monitor.check_health()
print(f"Consecutive failures: {health['consecutive_failures']}")
print(f"Should recover: {health['should_recover']}")
```

---

### 2. External Embedding Service (`EmbeddingServiceClient`)

**Purpose**: Unified interface for external embedding APIs.

**Supported Providers**:
- OpenAI (text-embedding-ada-002, text-embedding-3-small, etc.)
- Cohere (embed-english-v3.0, embed-multilingual-v3.0)
- HuggingFace Inference API
- Local models (fallback)

**Example**:
```python
from mlsdm.integrations import EmbeddingServiceClient, EmbeddingProvider

# OpenAI embeddings
client = EmbeddingServiceClient(
    provider=EmbeddingProvider.OPENAI,
    api_key="sk-xxxxx",
    model="text-embedding-ada-002",
    dimension=1536
)

embedding = client.embed("Hello, world!")
print(embedding.shape)  # (1536,)

# Batch processing
texts = ["Hello", "World", "MLSDM"]
embeddings = client.embed_batch(texts)
print(embeddings.shape)  # (3, 1536)
```

---

### 3. Secret Management (`SecretsManager`)

**Purpose**: Integrate with external secret management systems.

**Supported Providers**:
- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- Environment variables (fallback)

**Example**:
```python
from mlsdm.integrations import SecretsManager, SecretProvider

# HashiCorp Vault
manager = SecretsManager(
    provider=SecretProvider.VAULT,
    vault_addr="https://vault.example.com",
    vault_token="s.xxxxx",
    cache_ttl=300
)

api_key = manager.get_secret("mlsdm/openai_api_key")

# AWS Secrets Manager
aws_manager = SecretsManager(
    provider=SecretProvider.AWS_SECRETS,
    aws_region="us-east-1"
)

db_password = aws_manager.get_secret("mlsdm/database/password")
```

---

### 4. Distributed Cache (`RedisCache`)

**Purpose**: Redis-based distributed caching for multi-instance deployments.

**Features**:
- Automatic serialization (pickle)
- TTL management
- Embedding cache helpers
- Cache statistics and hit rates

**Example**:
```python
from mlsdm.integrations import RedisCache

# Initialize cache
cache = RedisCache(
    host="localhost",
    port=6379,
    password="secret",
    ttl=3600,
    prefix="mlsdm:"
)

# Cache embeddings
import hashlib
text = "Hello, world!"
text_hash = hashlib.sha256(text.encode()).hexdigest()

cache.set_embedding(text_hash, embedding_vector)

# Retrieve cached embedding
cached = cache.get_embedding(text_hash)

# Get cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

---

### 5. Webhook Events (`WebhookClient`)

**Purpose**: Asynchronous event notifications and callbacks.

**Features**:
- HMAC signature verification
- Automatic retry with exponential backoff
- Local event handlers
- Standard event types

**Event Types**:
- `MORAL_FILTER_REJECT` - Moral filter rejection
- `MORAL_FILTER_ACCEPT` - Moral filter acceptance
- `EMERGENCY_SHUTDOWN` - System emergency shutdown
- `RECOVERY_TRIGGERED` - Recovery workflow triggered
- `MEMORY_THRESHOLD` - Memory threshold exceeded
- `REQUEST_PROCESSED` - Request processed successfully
- `CUSTOM` - Custom events

**Example**:
```python
from mlsdm.integrations import WebhookClient, WebhookEvent, WebhookEventType
import time

# Initialize client
client = WebhookClient(
    webhook_url="https://example.com/webhook",
    secret="webhook_secret",
    max_retries=3
)

# Send event
event = WebhookEvent(
    event_type=WebhookEventType.MORAL_FILTER_REJECT.value,
    timestamp=time.time(),
    data={"prompt": "test", "moral_value": 0.3}
)

success = client.send_event(event)

# Register local handler
def handle_emergency(event: WebhookEvent):
    print(f"Emergency: {event.data}")

client.register_handler(WebhookEventType.EMERGENCY_SHUTDOWN, handle_emergency)
```

---

### 6. LLM Provider Abstraction (`LLMProviderClient`)

**Purpose**: Unified interface for multiple LLM providers.

**Supported Providers**:
- OpenAI (GPT-4, GPT-3.5-turbo, etc.)
- Anthropic (Claude 2, Claude 3)
- Cohere (Command, Command-R)
- HuggingFace models
- Local models (fallback)

**Example**:
```python
from mlsdm.integrations import LLMProviderClient, LLMProvider

# OpenAI
client = LLMProviderClient(
    provider=LLMProvider.OPENAI,
    api_key="sk-xxxxx",
    model="gpt-4",
    temperature=0.7
)

response = client.generate("Explain quantum computing", max_tokens=256)

# Anthropic
claude_client = LLMProviderClient(
    provider=LLMProvider.ANTHROPIC,
    api_key="sk-ant-xxxxx",
    model="claude-3-opus-20240229"
)

response = claude_client.generate("Write a poem", max_tokens=100, temperature=0.9)

# Get provider info
info = client.get_provider_info()
print(info)  # {'provider': 'openai', 'model': 'gpt-4', ...}
```

---

### 7. Distributed Tracing (`DistributedTracer`)

**Purpose**: Complete OpenTelemetry integration for distributed tracing.

**Features**:
- Automatic span creation and context propagation
- OTLP, Jaeger, and console exporters
- Graceful degradation when OpenTelemetry unavailable
- Span attributes and events

**Example**:
```python
from mlsdm.integrations import DistributedTracer

# Initialize tracer
tracer = DistributedTracer(
    service_name="mlsdm-engine",
    exporter_endpoint="http://localhost:4317",
    enable_console=False
)

# Create spans
with tracer.start_span("generate_request") as span:
    span.set_attribute("prompt_length", 100)
    span.set_attribute("moral_value", 0.8)
    
    # Perform work
    result = engine.generate(prompt)
    
    span.set_attribute("response_length", len(result))
    span.add_event("generation_complete")

# Add event to current span
tracer.add_event("cache_hit", {"cache_key": "abc123"})
```

---

## Installation

All integrations are included in the main MLSDM package:

```bash
pip install -e .
```

Optional dependencies for specific integrations:

```bash
# Redis cache
pip install redis

# AWS Secrets Manager
pip install boto3

# Azure Key Vault
pip install azure-keyvault-secrets azure-identity

# OpenTelemetry tracing (if not already installed)
pip install opentelemetry-api opentelemetry-sdk
```

---

## Testing

Run integration tests:

```bash
# All integration tests
pytest tests/integrations/ -v

# Specific integration
pytest tests/integrations/test_ci_health_monitor.py -v
pytest tests/integrations/test_embedding_service.py -v
pytest tests/integrations/test_secrets_manager.py -v
pytest tests/integrations/test_webhook_client.py -v
pytest tests/integrations/test_llm_provider.py -v
pytest tests/integrations/test_distributed_tracing.py -v
```

---

## Configuration

### Environment Variables

```bash
# CI Health Monitor
export GITHUB_TOKEN="ghp_xxxxx"
export GITHUB_REPOSITORY="neuron7x/mlsdm"

# Embedding Service
export OPENAI_API_KEY="sk-xxxxx"
export COHERE_API_KEY="xxxxx"
export HUGGINGFACE_API_KEY="hf_xxxxx"

# Secrets Management
export VAULT_ADDR="https://vault.example.com"
export VAULT_TOKEN="s.xxxxx"
export AWS_REGION="us-east-1"
export AZURE_VAULT_URL="https://myvault.vault.azure.net/"

# Redis Cache
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_PASSWORD="secret"

# Webhook
export WEBHOOK_URL="https://example.com/webhook"
export WEBHOOK_SECRET="webhook_secret"

# Distributed Tracing
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

---

## Integration Patterns

### Pattern 1: Full Stack Integration

```python
from mlsdm.integrations import (
    CIHealthMonitor,
    EmbeddingServiceClient,
    SecretsManager,
    WebhookClient,
    LLMProviderClient,
    DistributedTracer,
)

# Initialize all integrations
secrets = SecretsManager(provider=SecretProvider.VAULT, ...)
api_key = secrets.get_secret("mlsdm/openai_api_key")

embeddings = EmbeddingServiceClient(
    provider=EmbeddingProvider.OPENAI,
    api_key=api_key
)

llm = LLMProviderClient(
    provider=LLMProvider.OPENAI,
    api_key=api_key
)

webhooks = WebhookClient(webhook_url="...")
tracer = DistributedTracer(service_name="mlsdm")
ci_monitor = CIHealthMonitor(enable_auto_recovery=True)

# Use together
with tracer.start_span("process_request"):
    embedding = embeddings.embed(prompt)
    response = llm.generate(prompt, max_tokens=256)
    webhooks.emit_event(event)
```

### Pattern 2: LLM Wrapper Integration

```python
from mlsdm.core import create_llm_wrapper
from mlsdm.integrations import LLMProviderClient, EmbeddingServiceClient

# Create clients
llm_client = LLMProviderClient(provider=LLMProvider.OPENAI, api_key="...")
embed_client = EmbeddingServiceClient(provider=EmbeddingProvider.OPENAI, api_key="...")

# Wrap with MLSDM governance
wrapper = create_llm_wrapper(
    llm_generate_fn=llm_client.generate,
    embedding_fn=embed_client.embed,
    dim=1536
)

result = wrapper.generate("Hello", moral_value=0.8)
```

---

## Best Practices

1. **Use Environment Variables**: Never hardcode API keys or secrets
2. **Enable Caching**: Use Redis cache for production deployments
3. **Monitor CI Health**: Set up auto-recovery for high availability
4. **Emit Webhooks**: Send events for critical system state changes
5. **Trace Everything**: Add spans to all significant operations
6. **Rotate Secrets**: Use secret managers with automatic rotation
7. **Test Integrations**: Run integration tests before deployment

---

## Troubleshooting

### CI Monitor Not Fetching Status

**Problem**: `get_latest_workflow_status()` returns `UNKNOWN`

**Solution**:
- Verify GitHub token has `repo` and `actions:read` scopes
- Check repository name format: `owner/repo`
- Ensure workflow file name is correct

### Embedding Service Errors

**Problem**: API calls failing or timing out

**Solution**:
- Verify API key is valid
- Check rate limits on provider dashboard
- Increase `timeout` parameter
- Use fallback to local provider

### Secret Manager Connection Issues

**Problem**: Cannot retrieve secrets from Vault/AWS/Azure

**Solution**:
- Verify credentials and permissions
- Check network connectivity
- Ensure vault is unsealed (HashiCorp Vault)
- Use environment variable fallback for testing

### Redis Cache Connection Failed

**Problem**: Cannot connect to Redis

**Solution**:
- Verify Redis is running: `redis-cli ping`
- Check host, port, password configuration
- Ensure Redis allows remote connections if needed
- Use in-memory cache fallback

---

## Performance Considerations

| Integration | Overhead | Caching | Async Support |
|-------------|----------|---------|---------------|
| CI Monitor | Low | N/A | Yes (callbacks) |
| Embeddings | Medium | Yes (Redis) | Future |
| Secrets | Low | Yes (TTL cache) | No |
| Redis Cache | Low | N/A | Future |
| Webhooks | Low | N/A | Yes (fire-and-forget) |
| LLM Provider | Medium-High | No (use MLSDM memory) | Future |
| Tracing | Low | N/A | Yes (batch export) |

---

## Security Notes

- **API Keys**: Always use secret managers in production
- **Webhook Signatures**: Enable HMAC verification for webhooks
- **Redis Security**: Use password authentication and TLS
- **Vault Tokens**: Rotate regularly and use limited-scope tokens
- **Tracing Data**: Scrub PII before sending spans to collectors

---

## Related Documentation

- [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - General integration guide
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Production deployment
- [SECURITY_POLICY.md](SECURITY_POLICY.md) - Security guidelines
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation

---

**Version**: 1.2.0+  
**Updated**: December 2025  
**Status**: Production Ready ✅
