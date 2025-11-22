# MLSDM Production Runbook

**Status**: Production Ready v1.0.0  
**Last Updated**: November 2025  
**Maintainer**: neuron7x

## Table of Contents

1. [Service Overview](#service-overview)
2. [Quick Reference](#quick-reference)
3. [Deployment](#deployment)
4. [Monitoring & Alerts](#monitoring--alerts)
5. [Common Issues](#common-issues)
6. [Troubleshooting](#troubleshooting)
7. [Incident Response](#incident-response)
8. [Maintenance](#maintenance)
9. [Disaster Recovery](#disaster-recovery)

---

## Service Overview

**Purpose**: Production-ready neurobiologically-grounded cognitive architecture with moral governance, phase-based memory, and cognitive rhythm.

**Architecture**:
- FastAPI-based REST API
- In-memory cognitive engine
- Prometheus metrics export
- Health check endpoints

**Key Features**:
- Hard memory limit (20k vectors, ≤1.4 GB RAM)
- Adaptive moral homeostasis (EMA + dynamic threshold)
- Circadian rhythm (8 wake + 3 sleep cycles)
- Thread-safe concurrent processing (1000+ RPS)

---

## Quick Reference

### Critical Endpoints

```
Health Checks:
- GET /health/liveness    - Process alive check (always 200)
- GET /health/readiness   - Ready for traffic (200/503)
- GET /health/detailed    - Comprehensive status
- GET /health/metrics     - Prometheus metrics

API Endpoints:
- POST /event             - Process cognitive event
- GET /state              - Get system state
```

### Key Metrics

```
# Process metrics
process_event_latency_seconds     - Event processing latency
total_events_processed            - Total events counter
accepted_events_count             - Accepted events counter
moral_filter_threshold            - Current moral threshold

# System metrics
memory_usage_bytes                - Memory consumption
cpu_usage_percent                 - CPU utilization
```

### Environment Variables

```bash
# Required
API_KEY=<secret>                  # API authentication key
CONFIG_PATH=/path/to/config.yaml  # Configuration file path

# Optional
MLSDM_ENV=production              # Environment (dev/staging/production)
DISABLE_RATE_LIMIT=0              # Disable rate limiting (testing only)
LOG_LEVEL=INFO                    # Logging level
```

---

## Deployment

### Docker Deployment

```bash
# Build image
docker build -f Dockerfile.neuro-engine-service -t mlsdm:latest .

# Run container
docker run -d \
  --name mlsdm-api \
  -p 8000:8000 \
  -e API_KEY=your-secret-key \
  -e MLSDM_ENV=production \
  -v $(pwd)/config:/etc/mlsdm:ro \
  --restart unless-stopped \
  mlsdm:latest

# Check health
curl http://localhost:8000/health/liveness

# View logs
docker logs -f mlsdm-api
```

### Kubernetes Deployment

```bash
# Deploy to production
kubectl apply -f deploy/k8s/production-deployment.yaml

# Check deployment status
kubectl get deployments -n mlsdm-production
kubectl get pods -n mlsdm-production

# Check pod health
kubectl describe pod -n mlsdm-production -l app=mlsdm-api

# View logs
kubectl logs -n mlsdm-production -l app=mlsdm-api --tail=100 -f

# Port forward for local testing
kubectl port-forward -n mlsdm-production svc/mlsdm-api 8000:80
```

### Rolling Update

```bash
# Update image
kubectl set image deployment/mlsdm-api \
  mlsdm-api=ghcr.io/neuron7x/mlsdm-neuro-engine:1.0.1 \
  -n mlsdm-production

# Watch rollout
kubectl rollout status deployment/mlsdm-api -n mlsdm-production

# Rollback if needed
kubectl rollout undo deployment/mlsdm-api -n mlsdm-production
```

---

## Monitoring & Alerts

### Health Checks

**Liveness Probe**:
- **Purpose**: Detect if process is alive
- **Endpoint**: `/health/liveness`
- **Expected**: 200 OK
- **Action**: Restart pod if failing

**Readiness Probe**:
- **Purpose**: Determine if ready for traffic
- **Endpoint**: `/health/readiness`
- **Expected**: 200 OK (ready), 503 (not ready)
- **Action**: Remove from load balancer if failing

**Checks Performed**:
- Memory manager initialized
- System memory < 95% used
- CPU usage < 98%

### Key Metrics to Monitor

```promql
# Request rate
rate(total_events_processed[5m])

# Error rate
rate(rejected_events_count[5m]) / rate(total_events_processed[5m])

# Latency (p95)
histogram_quantile(0.95, rate(process_event_latency_seconds_bucket[5m]))

# Memory usage
memory_usage_bytes / (1024^3)  # GB

# Moral threshold drift
delta(moral_filter_threshold[1h])
```

### Recommended Alerts

```yaml
# High error rate
- alert: HighRejectionRate
  expr: rate(rejected_events_count[5m]) / rate(total_events_processed[5m]) > 0.5
  for: 5m
  severity: warning
  
# High latency
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(process_event_latency_seconds_bucket[5m])) > 0.1
  for: 5m
  severity: warning

# Memory pressure
- alert: HighMemoryUsage
  expr: memory_usage_bytes > 1.8e9  # 1.8 GB
  for: 5m
  severity: critical

# Service down
- alert: ServiceDown
  expr: up{job="mlsdm-api"} == 0
  for: 1m
  severity: critical
```

---

## Common Issues

### Issue: High Rejection Rate

**Symptoms**:
- Many events rejected by moral filter
- `rejected_events_count` increasing rapidly

**Cause**:
- Moral threshold too high
- Input moral values consistently low

**Resolution**:
```bash
# Check current threshold
curl -H "Authorization: Bearer $API_KEY" \
  http://api/health/detailed | jq '.statistics.moral_filter_threshold'

# Monitor moral threshold
watch -n 5 'curl -s http://api/health/detailed | jq .statistics.moral_filter_threshold'

# Wait for adaptive adjustment (threshold will auto-adjust)
# If needed, restart service to reset threshold to initial value
```

### Issue: Memory Growth

**Symptoms**:
- Memory usage increasing beyond 1.4 GB
- OOM kills in Kubernetes

**Cause**:
- Memory leak (unlikely, system has hard limits)
- Configuration error (capacity too high)

**Resolution**:
```bash
# Check current memory state
curl http://api/health/detailed | jq '.memory_state'

# Verify configuration
kubectl get configmap mlsdm-config -n mlsdm-production -o yaml

# Check actual memory usage
kubectl top pods -n mlsdm-production

# If memory leak suspected, collect heap dump (if available)
# Otherwise, restart pod
kubectl delete pod -n mlsdm-production -l app=mlsdm-api --grace-period=30
```

### Issue: High Latency

**Symptoms**:
- P95 latency > 100ms
- Slow response times

**Cause**:
- High concurrent load
- Resource contention
- Sleep phase processing (intentionally slower)

**Resolution**:
```bash
# Check current phase
curl http://api/health/detailed | jq '.phase'

# If in sleep phase, latency is expected to be higher
# Otherwise, check CPU and memory
kubectl top pods -n mlsdm-production

# Check for resource throttling
kubectl describe pod -n mlsdm-production -l app=mlsdm-api

# Scale up if needed
kubectl scale deployment mlsdm-api --replicas=5 -n mlsdm-production
```

### Issue: Rate Limiting

**Symptoms**:
- Clients receiving 429 responses
- `rate_limit_exceeded` events in logs

**Cause**:
- Client exceeding 5 RPS limit
- Misconfigured rate limiter

**Resolution**:
```bash
# Check rate limiter configuration
curl http://api/health/detailed

# Review security logs for offending clients
kubectl logs -n mlsdm-production -l app=mlsdm-api | grep rate_limit_exceeded

# Temporarily disable for specific client (not recommended)
# Or increase rate limit in configuration

# Long-term: implement per-tier rate limits
```

---

## Troubleshooting

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/mlsdm-api LOG_LEVEL=DEBUG -n mlsdm-production

# View debug logs
kubectl logs -n mlsdm-production -l app=mlsdm-api --tail=100 -f | grep DEBUG

# Revert to INFO
kubectl set env deployment/mlsdm-api LOG_LEVEL=INFO -n mlsdm-production
```

### Pod Not Ready

```bash
# Check pod events
kubectl describe pod -n mlsdm-production <pod-name>

# Check readiness probe
kubectl logs -n mlsdm-production <pod-name> --previous

# Check health endpoint manually
kubectl port-forward -n mlsdm-production <pod-name> 8000:8000
curl http://localhost:8000/health/readiness
```

### Memory Manager Issues

```bash
# Check detailed health
curl http://api/health/detailed | jq .

# Look for error messages
kubectl logs -n mlsdm-production -l app=mlsdm-api | grep ERROR

# Check memory layer norms
curl http://api/health/detailed | jq '.memory_state'
```

### Configuration Issues

```bash
# Validate configuration
kubectl get configmap mlsdm-config -n mlsdm-production -o yaml

# Test configuration locally
python -c "
from mlsdm.utils.config_loader import ConfigLoader
config = ConfigLoader.load_config('config/production-ready.yaml')
print('Configuration valid')
"

# Apply updated configuration
kubectl apply -f config/production-ready.yaml
kubectl rollout restart deployment/mlsdm-api -n mlsdm-production
```

---

## Incident Response

### Severity Levels

**P0 - Critical**:
- Service completely down
- Data loss risk
- Security breach

**P1 - High**:
- Partial service degradation
- High error rate (>50%)
- Performance severely impacted

**P2 - Medium**:
- Minor service degradation
- Elevated error rate (>10%)
- Non-critical feature impaired

**P3 - Low**:
- Cosmetic issues
- Documentation updates needed

### Response Procedures

#### P0: Service Down

```bash
# 1. Check service status
kubectl get pods -n mlsdm-production

# 2. Check recent events
kubectl get events -n mlsdm-production --sort-by='.lastTimestamp'

# 3. Check logs
kubectl logs -n mlsdm-production -l app=mlsdm-api --tail=500

# 4. Quick restart if needed
kubectl rollout restart deployment/mlsdm-api -n mlsdm-production

# 5. If still down, rollback
kubectl rollout undo deployment/mlsdm-api -n mlsdm-production

# 6. Escalate if not resolved in 5 minutes
```

#### P1: High Error Rate

```bash
# 1. Identify error pattern
kubectl logs -n mlsdm-production -l app=mlsdm-api | grep ERROR

# 2. Check metrics
curl http://api/health/metrics | grep error_count

# 3. Check if specific to certain clients
kubectl logs -n mlsdm-production -l app=mlsdm-api | grep -A 5 "client_id"

# 4. Scale up if resource-related
kubectl scale deployment mlsdm-api --replicas=6 -n mlsdm-production

# 5. Monitor for improvement
watch kubectl get hpa -n mlsdm-production
```

---

## Maintenance

### Planned Maintenance

```bash
# 1. Announce maintenance window
# 2. Increase replicas for safety
kubectl scale deployment mlsdm-api --replicas=5 -n mlsdm-production

# 3. Perform updates
kubectl apply -f deploy/k8s/production-deployment.yaml

# 4. Monitor rollout
kubectl rollout status deployment/mlsdm-api -n mlsdm-production

# 5. Verify health
curl http://api/health/detailed

# 6. Return to normal replica count
kubectl scale deployment mlsdm-api --replicas=3 -n mlsdm-production
```

### Configuration Updates

```bash
# 1. Validate new configuration
python -c "
from mlsdm.utils.config_loader import ConfigLoader
config = ConfigLoader.load_config('config/production-ready.yaml')
print('Valid')
"

# 2. Update ConfigMap
kubectl create configmap mlsdm-config --from-file=config.yaml=config/production-ready.yaml \
  -n mlsdm-production --dry-run=client -o yaml | kubectl apply -f -

# 3. Rolling restart
kubectl rollout restart deployment/mlsdm-api -n mlsdm-production

# 4. Verify configuration loaded
kubectl logs -n mlsdm-production -l app=mlsdm-api | grep "Configuration loaded"
```

### Security Updates

```bash
# 1. Update base image
docker build -f Dockerfile.neuro-engine-service -t mlsdm:1.0.1 .

# 2. Scan for vulnerabilities
docker scan mlsdm:1.0.1

# 3. Push to registry
docker tag mlsdm:1.0.1 ghcr.io/neuron7x/mlsdm-neuro-engine:1.0.1
docker push ghcr.io/neuron7x/mlsdm-neuro-engine:1.0.1

# 4. Update deployment
kubectl set image deployment/mlsdm-api \
  mlsdm-api=ghcr.io/neuron7x/mlsdm-neuro-engine:1.0.1 \
  -n mlsdm-production

# 5. Monitor rollout
kubectl rollout status deployment/mlsdm-api -n mlsdm-production
```

---

## Disaster Recovery

### Backup Procedures

**Configuration**:
```bash
# Backup ConfigMaps and Secrets
kubectl get configmap mlsdm-config -n mlsdm-production -o yaml > backup/config-$(date +%Y%m%d).yaml
kubectl get secret mlsdm-secrets -n mlsdm-production -o yaml > backup/secrets-$(date +%Y%m%d).yaml
```

**Note**: This system uses in-memory storage. No persistent data to backup. State is ephemeral by design.

### Recovery Procedures

**Complete Cluster Failure**:
```bash
# 1. Restore configuration
kubectl apply -f backup/config-latest.yaml
kubectl apply -f backup/secrets-latest.yaml

# 2. Deploy service
kubectl apply -f deploy/k8s/production-deployment.yaml

# 3. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=mlsdm-api -n mlsdm-production --timeout=300s

# 4. Verify health
curl http://api/health/detailed

# 5. Resume traffic
# Update DNS/load balancer as needed
```

**Single Pod Failure**:
- Kubernetes will automatically restart failed pods
- No manual intervention required
- Monitor rollout: `kubectl get pods -n mlsdm-production -w`

---

## Contact Information

**On-Call Rotation**: [TBD]  
**Escalation Path**: [TBD]  
**Slack Channel**: #mlsdm-production  
**Documentation**: https://github.com/neuron7x/mlsdm

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-11 | Initial production runbook | neuron7x |

---

**Remember**: This is a cognitive architecture with adaptive behavior. Some variations in metrics are expected and healthy!
