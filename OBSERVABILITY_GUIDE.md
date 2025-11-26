# MLSDM Observability Guide

This guide explains how to set up and use observability features in MLSDM for monitoring, troubleshooting, and maintaining SLO compliance.

## Overview

MLSDM provides comprehensive observability through three pillars:

1. **Structured Logging** - JSON-formatted logs with correlation IDs and mandatory fields
2. **Prometheus Metrics** - Counters, gauges, and histograms for monitoring
3. **OpenTelemetry Tracing** - Distributed tracing for request path visibility

## Quick Start

### Enable Prometheus Metrics

Metrics are exposed at `/health/metrics` endpoint by default:

```bash
# Start the API server
uvicorn mlsdm.api.app:app --host 0.0.0.0 --port 8000

# Scrape metrics
curl http://localhost:8000/health/metrics
```

### Enable OpenTelemetry Tracing

Set environment variables to enable tracing:

```bash
# Enable tracing with console exporter (for debugging)
export MLSDM_OTEL_ENABLED=true
export OTEL_EXPORTER_TYPE=console

# Or use OTLP exporter for production
export MLSDM_OTEL_ENABLED=true
export OTEL_EXPORTER_TYPE=otlp
export MLSDM_OTEL_ENDPOINT=http://jaeger:4318
```

### Import Grafana Dashboard

1. Open Grafana and navigate to Dashboards → Import
2. Upload `deploy/grafana/mlsdm_observability_dashboard.json`
3. Select your Prometheus data source
4. Click Import

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLSDM_OTEL_ENABLED` | Enable/disable tracing | `false` |
| `MLSDM_OTEL_ENDPOINT` | OTLP endpoint URL | `http://localhost:4318` |
| `OTEL_SERVICE_NAME` | Service name for traces | `mlsdm` |
| `OTEL_EXPORTER_TYPE` | Exporter type: `console`, `otlp`, `jaeger`, `none` | `console` |
| `OTEL_TRACES_SAMPLER_ARG` | Sampling rate (0.0-1.0) | `1.0` |

### Prometheus Scrape Configuration

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'mlsdm'
    scrape_interval: 15s
    static_configs:
      - targets: ['mlsdm-api:8000']
    metrics_path: '/health/metrics'
```

## Key Metrics

### Request Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `mlsdm_requests_total` | Counter | Total requests | `endpoint`, `status` |
| `mlsdm_request_latency_seconds` | Histogram | Request latency | `endpoint`, `phase` |
| `mlsdm_events_processed_total` | Counter | Events processed | - |
| `mlsdm_events_rejected_total` | Counter | Events rejected | - |

### Moral Governance Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `mlsdm_moral_rejections_total` | Counter | Moral rejections | `reason` |
| `mlsdm_moral_threshold` | Gauge | Current threshold | - |

### Aphasia Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `mlsdm_aphasia_detected_total` | Counter | Aphasia detections | `severity_bucket` |
| `mlsdm_aphasia_repaired_total` | Counter | Successful repairs | - |
| `mlsdm_aphasia_events_total` | Counter | All aphasia events | `mode`, `is_aphasic`, `repair_applied` |

### Emergency Shutdown Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `mlsdm_emergency_shutdowns_total` | Counter | Shutdown events | `reason` |
| `mlsdm_emergency_shutdown_active` | Gauge | Shutdown active (1/0) | - |

### Cognitive State Metrics

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `mlsdm_phase` | Gauge | Current phase (wake=1, sleep=0) | - |
| `mlsdm_memory_usage_bytes` | Gauge | Memory usage | - |
| `mlsdm_memory_l1_norm` | Gauge | L1 memory layer norm | - |
| `mlsdm_memory_l2_norm` | Gauge | L2 memory layer norm | - |
| `mlsdm_memory_l3_norm` | Gauge | L3 memory layer norm | - |

## SLO Recommendations

Based on MLSDM architecture and production patterns, we recommend the following SLOs:

### Availability SLO: 99.5%

```promql
# Availability calculation
(1 - sum(rate(mlsdm_requests_total{status=~"5.."}[5m])) 
   / sum(rate(mlsdm_requests_total[5m]))) * 100
```

### Latency SLO: P95 < 500ms

```promql
# P95 latency
histogram_quantile(0.95, 
  sum(rate(mlsdm_request_latency_seconds_bucket[5m])) by (le)
)
```

### Error Budget: 0.5% of requests

```promql
# Error budget burn rate (per hour)
sum(increase(mlsdm_errors_total[1h])) 
/ (30 * 24 * 0.005 * sum(increase(mlsdm_requests_total[1h])))
```

### Moral Rejection Rate: < 5%

```promql
# Moral rejection rate
sum(rate(mlsdm_moral_rejections_total[5m])) 
/ sum(rate(mlsdm_requests_total[5m])) * 100
```

### Emergency Shutdown Frequency: < 1 per day

```promql
# Shutdowns per day
sum(increase(mlsdm_emergency_shutdowns_total[24h]))
```

## Tracing Structure

Each request generates a span tree with the following structure:

```
api.generate (SERVER)
├── engine.generate (INTERNAL)
│   ├── engine.moral_precheck (INTERNAL)
│   ├── engine.grammar_precheck (INTERNAL)
│   ├── engine.llm_generation (INTERNAL)
│   │   ├── mlsdm.llm_call (CLIENT)
│   │   └── mlsdm.speech_governance (INTERNAL)
│   │       ├── mlsdm.aphasia_detection (INTERNAL)
│   │       └── mlsdm.aphasia_repair (INTERNAL)
│   └── engine.post_moral_check (INTERNAL)
```

### Key Span Attributes

| Attribute | Description |
|-----------|-------------|
| `mlsdm.request_id` | Unique request identifier |
| `mlsdm.phase` | Cognitive phase (wake/sleep) |
| `mlsdm.moral_value` | Moral threshold used |
| `mlsdm.accepted` | Whether request was accepted |
| `mlsdm.rejected_at` | Stage where rejection occurred |
| `mlsdm.prompt_length` | Length of prompt (not content!) |
| `mlsdm.response_length` | Length of response |
| `mlsdm.latency_ms` | Processing latency |

## Logging Structure

All logs are JSON-formatted with mandatory fields:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "event_type": "request_completed",
  "correlation_id": "abc-123",
  "metrics": {
    "request_id": "abc-123",
    "phase": "wake",
    "step_counter": 42,
    "accepted": true,
    "reason": "normal",
    "moral_score_before": 0.75,
    "moral_score_after": 0.80,
    "latency_ms": 150.5
  }
}
```

### Privacy Invariant

**CRITICAL**: Raw user input and LLM responses are NEVER logged. Only metadata (lengths, scores, counts) is captured. The `payload_scrubber` function masks any text content before logging.

## Alerting Rules

Example Prometheus alerting rules:

```yaml
groups:
  - name: mlsdm-alerts
    rules:
      - alert: HighMoralRejectionRate
        expr: |
          sum(rate(mlsdm_moral_rejections_total[5m])) 
          / sum(rate(mlsdm_requests_total[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High moral rejection rate (> 10%)"
          
      - alert: EmergencyShutdownTriggered
        expr: mlsdm_emergency_shutdown_active == 1
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "MLSDM emergency shutdown is active"
          
      - alert: HighP95Latency
        expr: |
          histogram_quantile(0.95, 
            sum(rate(mlsdm_request_latency_seconds_bucket[5m])) by (le)
          ) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency exceeds 500ms"
          
      - alert: HighAphasiaCriticalRate
        expr: |
          sum(rate(mlsdm_aphasia_detected_total{severity_bucket="critical"}[5m])) 
          / sum(rate(mlsdm_aphasia_detected_total[5m])) > 0.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High rate of critical aphasia detections (> 20%)"
```

## Troubleshooting

### No Metrics on /health/metrics

1. Check the API is running: `curl http://localhost:8000/health`
2. Verify Prometheus client is installed: `pip show prometheus-client`
3. Check for import errors in logs

### Tracing Not Working

1. Verify MLSDM_OTEL_ENABLED is set to "true" (case-sensitive)
2. Check exporter type is valid: `console`, `otlp`, `jaeger`, or `none`
3. For OTLP, verify endpoint is reachable
4. Check for OpenTelemetry SDK initialization errors in logs

### High Latency

1. Check `mlsdm_request_latency_seconds` histogram for distribution
2. Look at individual span durations in traces
3. Common bottlenecks:
   - `mlsdm.llm_call`: LLM API latency
   - `mlsdm.memory_retrieval`: Memory search latency
   - `mlsdm.aphasia_repair`: Repair processing time

### High Moral Rejection Rate

1. Check `mlsdm_moral_threshold` gauge value
2. Review rejection reasons in `mlsdm_moral_rejections_total` labels
3. Check if threshold is adapting correctly
4. Review logs for `moral_precheck` and `post_moral_check` events

## Integration Examples

### Jaeger Setup

```yaml
# docker-compose.yml
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # UI
      - "4318:4318"    # OTLP HTTP
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  mlsdm-api:
    image: mlsdm-api:latest
    environment:
      - MLSDM_OTEL_ENABLED=true
      - OTEL_EXPORTER_TYPE=otlp
      - MLSDM_OTEL_ENDPOINT=http://jaeger:4318
```

### Prometheus + Grafana Stack

```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./deploy/grafana:/etc/grafana/provisioning/dashboards
```

## Best Practices

1. **Always use correlation IDs** - Pass `request_id` through the entire pipeline
2. **Never log PII** - Use `payload_scrubber` for any user content
3. **Set appropriate sampling** - Use 10% sampling in high-traffic production
4. **Monitor error budgets** - Set up alerts before budget exhaustion
5. **Review traces periodically** - Look for slow spans and optimization opportunities
