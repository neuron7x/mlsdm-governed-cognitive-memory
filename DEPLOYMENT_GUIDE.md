# Deployment Guide

**Document Version:** 1.2.0  
**Project Version:** 1.2.0  
**Last Updated:** December 2025  
**Status:** Production

Production deployment guide for MLSDM Governed Cognitive Memory v1.2.0.

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Requirements](#requirements)
- [Deployment Patterns](#deployment-patterns)
- [Configuration](#configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Security Considerations](#security-considerations)
- [Scaling & Performance](#scaling--performance)
- [Troubleshooting](#troubleshooting)
- [Production Checklist](#production-checklist)

---

## Deployment Overview

MLSDM can be deployed in several configurations:

1. **Standalone Python Application**: Direct integration into existing services
2. **FastAPI Microservice**: REST API with HTTP/JSON interface
3. **Docker Container**: Containerized deployment
4. **Kubernetes**: Scalable cloud deployment
5. **Serverless**: AWS Lambda, Google Cloud Functions (with considerations)

---

## Requirements

### System Requirements

- **CPU**: 2+ cores recommended
- **RAM**: 512 MB minimum, 1 GB recommended
- **Storage**: 100 MB for application, additional for logs
- **OS**: Linux, macOS, or Windows with Python 3.12+

### Python Requirements

```bash
Python >= 3.12
numpy >= 2.0.0
sentence-transformers >= 3.0.0  # Optional, for embeddings
fastapi >= 0.110.0  # If using API
uvicorn >= 0.29.0  # If using API
```

### Network Requirements

- **Outbound**: Access to LLM APIs (OpenAI, Anthropic, etc.)
- **Inbound**: Port 8000 (default FastAPI) or custom port

---

## Deployment Patterns

### Pattern 1: Standalone Integration

Simplest deployment - integrate directly into your Python application.

```python
# app.py
from mlsdm.core.llm_wrapper import LLMWrapper
import numpy as np

# Initialize once at startup
def create_wrapper():
    def my_llm(prompt: str, max_tokens: int) -> str:
        # Your LLM integration
        return call_your_llm(prompt, max_tokens)
    
    def my_embed(text: str) -> np.ndarray:
        # Your embedding integration
        return get_embeddings(text)
    
    return LLMWrapper(
        llm_generate_fn=my_llm,
        embedding_fn=my_embed,
        dim=384,
        capacity=20000
    )

# Global wrapper instance
wrapper = create_wrapper()

# Use in your application
def handle_request(user_input: str, moral_score: float) -> str:
    result = wrapper.generate(user_input, moral_score)
    if result["accepted"]:
        return result["response"]
    else:
        return f"Request rejected: {result['note']}"
```

**Pros:**
- Simple integration
- Low overhead
- Full control

**Cons:**
- Tied to application lifecycle
- Single process only
- No built-in API

---

### Pattern 2: FastAPI Microservice

Production-ready REST API with async support.

```python
# api_server.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from mlsdm.core.llm_wrapper import LLMWrapper
import numpy as np
from typing import Optional

app = FastAPI(title="MLSDM Cognitive API", version="1.0.0")

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt")
    moral_value: float = Field(..., ge=0.0, le=1.0, description="Moral score")
    max_tokens: Optional[int] = Field(None, description="Max tokens override")
    context_top_k: int = Field(5, ge=1, le=20, description="Context items")

class GenerateResponse(BaseModel):
    response: str
    accepted: bool
    phase: str
    step: int
    note: str
    moral_threshold: float
    context_items: int

# Initialize wrapper
def get_wrapper():
    # Configure your LLM and embeddings
    def my_llm(prompt: str, max_tokens: int) -> str:
        # Implementation
        pass
    
    def my_embed(text: str) -> np.ndarray:
        # Implementation
        pass
    
    return LLMWrapper(
        llm_generate_fn=my_llm,
        embedding_fn=my_embed,
        dim=384
    )

wrapper = get_wrapper()

# Endpoints
@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text with cognitive governance."""
    try:
        result = wrapper.generate(
            prompt=request.prompt,
            moral_value=request.moral_value,
            max_tokens=request.max_tokens,
            context_top_k=request.context_top_k
        )
        return GenerateResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal error")

@app.get("/v1/state")
async def get_state():
    """Get system state."""
    return wrapper.get_state()

@app.get("/health")
async def health():
    """Health check endpoint."""
    state = wrapper.get_state()
    return {
        "status": "healthy",
        "step": state["step"],
        "phase": state["phase"],
        "memory_used": state["qilm_stats"]["used"],
        "memory_capacity": state["qilm_stats"]["capacity"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run:**
```bash
# Development
uvicorn api_server:app --reload

# Production
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

**Pros:**
- Standard REST API
- Built-in docs (Swagger UI)
- Easy to scale
- Language-agnostic clients

**Cons:**
- Additional complexity
- Network overhead

---

### Pattern 3: Docker Deployment

Containerized deployment for consistency and portability.

#### Dockerfile

```dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ src/
COPY api_server.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  mlsdm-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '1.0'
          memory: 512M
```

**Build and Run:**
```bash
# Build image
docker build -t mlsdm-cognitive:1.0.0 .

# Run container
docker run -d \
    --name mlsdm-api \
    -p 8000:8000 \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    mlsdm-cognitive:1.0.0

# Or use docker-compose
docker-compose up -d
```

**Pros:**
- Consistent environment
- Easy deployment
- Isolated dependencies
- Portable across clouds

**Cons:**
- Container overhead
- Requires Docker knowledge

---

### Pattern 4: Kubernetes Deployment

Scalable, production-grade deployment.

#### Deployment Manifest

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlsdm-api
  labels:
    app: mlsdm
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlsdm
  template:
    metadata:
      labels:
        app: mlsdm
        version: v1.0.0
    spec:
      containers:
      - name: mlsdm-api
        image: your-registry/mlsdm-cognitive:1.0.0
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: mlsdm-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: mlsdm-api-service
spec:
  selector:
    app: mlsdm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlsdm-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mlsdm-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Deploy:**
```bash
# Create secret
kubectl create secret generic mlsdm-secrets \
    --from-literal=openai-api-key=$OPENAI_API_KEY

# Deploy
kubectl apply -f k8s-deployment.yaml

# Check status
kubectl get pods -l app=mlsdm
kubectl get svc mlsdm-api-service
```

**Pros:**
- Auto-scaling
- Self-healing
- Load balancing
- Rolling updates

**Cons:**
- Kubernetes complexity
- Higher operational overhead

---

## Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# MLSDM Configuration
MLSDM_DIM=384
MLSDM_CAPACITY=20000
MLSDM_WAKE_DURATION=8
MLSDM_SLEEP_DURATION=3
MLSDM_INITIAL_THRESHOLD=0.50

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json  # json or text

# API Configuration
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=30

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
```

### Configuration File

```yaml
# config.yaml
mlsdm:
  dimension: 384
  capacity: 20000
  wake_duration: 8
  sleep_duration: 3
  initial_threshold: 0.50

llm:
  provider: openai
  model: gpt-3.5-turbo
  max_tokens: 2048
  temperature: 0.7

embeddings:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2
  dimension: 384

api:
  port: 8000
  workers: 4
  timeout: 30
  cors_origins:
    - "https://yourdomain.com"

logging:
  level: INFO
  format: json
  file: /var/log/mlsdm/app.log

monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
```

---

## Monitoring & Observability

### Key Metrics

**System Metrics:**
- `mlsdm_steps_total`: Total steps processed
- `mlsdm_accepted_total`: Accepted requests
- `mlsdm_rejected_total`: Rejected requests
- `mlsdm_memory_used`: Memory usage (vectors)
- `mlsdm_memory_bytes`: Memory usage (bytes)

**Performance Metrics:**
- `mlsdm_generation_duration_seconds`: Generation latency
- `mlsdm_retrieval_duration_seconds`: Context retrieval latency
- `mlsdm_moral_threshold`: Current moral threshold

**Phase Metrics:**
- `mlsdm_phase`: Current phase (0=wake, 1=sleep)
- `mlsdm_consolidations_total`: Total consolidations

### Prometheus Integration

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Metrics
requests_total = Counter('mlsdm_requests_total', 'Total requests')
accepted = Counter('mlsdm_accepted_total', 'Accepted requests')
rejected = Counter('mlsdm_rejected_total', 'Rejected requests')
memory_used = Gauge('mlsdm_memory_used', 'Memory vectors used')
moral_threshold = Gauge('mlsdm_moral_threshold', 'Moral threshold')
latency = Histogram('mlsdm_latency_seconds', 'Request latency')

# Start metrics server
start_http_server(9090)

# Update metrics
requests_total.inc()
memory_used.set(state['qilm_stats']['used'])
moral_threshold.set(state['moral_threshold'])
with latency.time():
    result = wrapper.generate(prompt, moral_value)
```

### Logging

```python
import logging
import json

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

logger = logging.getLogger(__name__)

def log_request(request, result):
    """Log request with structured data."""
    log_data = {
        "timestamp": time.time(),
        "prompt_length": len(request["prompt"]),
        "moral_value": request["moral_value"],
        "accepted": result["accepted"],
        "phase": result["phase"],
        "step": result["step"],
        "moral_threshold": result["moral_threshold"]
    }
    logger.info(json.dumps(log_data))
```

### Health Checks

```python
@app.get("/health")
async def health():
    """Comprehensive health check."""
    state = wrapper.get_state()
    
    # Check memory usage
    memory_pct = (state['qilm_stats']['used'] / 
                  state['qilm_stats']['capacity']) * 100
    
    # Check moral threshold bounds
    threshold_ok = 0.30 <= state['moral_threshold'] <= 0.90
    
    # Overall health
    healthy = memory_pct < 95 and threshold_ok
    
    return {
        "status": "healthy" if healthy else "degraded",
        "checks": {
            "memory": {
                "status": "ok" if memory_pct < 95 else "warning",
                "used": state['qilm_stats']['used'],
                "capacity": state['qilm_stats']['capacity'],
                "percentage": round(memory_pct, 2)
            },
            "moral_filter": {
                "status": "ok" if threshold_ok else "error",
                "threshold": state['moral_threshold']
            },
            "phase": {
                "current": state['phase'],
                "step": state['step']
            }
        }
    }
```

---

## Security Considerations

### API Security

1. **Authentication**
   ```python
   from fastapi import Header, HTTPException
   
   async def verify_api_key(x_api_key: str = Header()):
       if x_api_key != os.getenv("API_KEY"):
           raise HTTPException(status_code=401, detail="Invalid API key")
   
   @app.post("/v1/generate", dependencies=[Depends(verify_api_key)])
   async def generate(request: GenerateRequest):
       # ...
   ```

2. **Rate Limiting**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/v1/generate")
   @limiter.limit("100/minute")
   async def generate(request: Request, ...):
       # ...
   ```

3. **Input Validation**
   ```python
   class GenerateRequest(BaseModel):
       prompt: str = Field(..., max_length=10000)
       moral_value: float = Field(..., ge=0.0, le=1.0)
       
       @validator('prompt')
       def validate_prompt(cls, v):
           if not v.strip():
               raise ValueError("Prompt cannot be empty")
           return v
   ```

### Network Security

- Use HTTPS/TLS for all external communication
- Implement CORS properly
- Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- Network segmentation in Kubernetes

### Data Security

- Never log sensitive prompts or responses
- Sanitize outputs
- Implement audit logging
- Regular security updates

---

## Scaling & Performance

### Vertical Scaling

Single instance optimization:

```yaml
# Optimized configuration
mlsdm:
  capacity: 50000  # Increase for more memory
  
resources:
  requests:
    memory: "2Gi"
    cpu: "4000m"
```

### Horizontal Scaling

Multiple instances with load balancing:

- Each instance maintains its own memory (stateful)
- Use sticky sessions if context continuity needed
- Consider shared memory layer for advanced scenarios

### Performance Tuning

1. **Memory Capacity**
   - Default: 20,000 vectors (~30 MB)
   - High throughput: 50,000 vectors (~75 MB)
   - Low memory: 10,000 vectors (~15 MB)

2. **Phase Durations**
   - High throughput: wake=20, sleep=2
   - Balanced: wake=8, sleep=3 (default)
   - Frequent consolidation: wake=5, sleep=5

3. **Context Retrieval**
   - Fast: top_k=3
   - Balanced: top_k=5 (default)
   - Comprehensive: top_k=10

---

## Troubleshooting

### Common Issues

**Issue 1: High rejection rate**
```
Symptoms: Most requests rejected
Cause: Moral threshold too high or incorrect scoring
Solution: 
- Check moral_value scoring function
- Lower initial_threshold
- Monitor threshold adaptation
```

**Issue 2: Memory full**
```
Symptoms: Context retrieval returns old data
Cause: Memory at capacity, wrapping occurs
Solution:
- Increase capacity parameter
- Implement memory cleanup strategy
- Monitor usage patterns
```

**Issue 3: Slow response times**
```
Symptoms: High P95/P99 latency
Cause: Large context_top_k or slow LLM
Solution:
- Reduce context_top_k
- Optimize embedding function
- Use faster LLM model
- Add caching layer
```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose logging
result = wrapper.generate(prompt, moral_value)

# Inspect state
state = wrapper.get_state()
print(json.dumps(state, indent=2))
```

---

## Production Checklist

Before deploying to production:

### Infrastructure
- [ ] Resources provisioned (CPU, RAM, storage)
- [ ] Network connectivity verified
- [ ] Load balancer configured
- [ ] SSL/TLS certificates installed
- [ ] DNS configured

### Application
- [ ] All tests passing
- [ ] Configuration validated
- [ ] Secrets management configured
- [ ] Error handling comprehensive
- [ ] Logging configured

### Monitoring
- [ ] Metrics collection enabled
- [ ] Dashboards created
- [ ] Alerts configured
- [ ] Health checks working
- [ ] Log aggregation setup

### Security
- [ ] Authentication enabled
- [ ] Rate limiting configured
- [ ] Input validation implemented
- [ ] Security scan passed
- [ ] Secrets rotated

### Operations
- [ ] Deployment procedure documented
- [ ] Rollback procedure tested
- [ ] Backup strategy defined
- [ ] Incident response plan ready
- [ ] On-call schedule established

### Performance
- [ ] Load testing completed
- [ ] Performance benchmarks met
- [ ] Resource limits tuned
- [ ] Scaling strategy defined
- [ ] Capacity planning done

---

## Support

For deployment assistance:
- GitHub Issues: https://github.com/neuron7x/mlsdm/issues
- Documentation: See README.md and other guides
- Email: Contact maintainer for enterprise support

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Maintainer**: neuron7x
