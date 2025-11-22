# Changelog

All notable changes to the MLSDM (Governed Cognitive Memory) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-22

### Added

#### Phase 6: API, Packaging, Deployment and Security Baseline

**Python SDK**
- Public Python SDK (`mlsdm.sdk.NeuroCognitiveClient`) for easy integration
- Support for multiple backends (`local_stub`, `openai`)
- Comprehensive type hints and docstrings
- 13 unit tests with full coverage

**HTTP API Service**
- FastAPI-based HTTP API service (`mlsdm.service`)
- `POST /v1/neuro/generate` endpoint for text generation
- `GET /healthz` for health checks
- `GET /metrics` for Prometheus-compatible metrics
- Full request/response validation with Pydantic models
- 15 integration tests including rate limiting tests

**Docker and Deployment**
- Multi-stage Dockerfile for optimized container images
- Docker Compose configuration for easy local deployment
- Kubernetes manifests (Deployment, Service) with:
  - Resource limits and requests
  - Health checks (liveness, readiness)
  - Security contexts (non-root, dropped capabilities)
  - Pod security policies

**Security Baseline**
- In-memory rate limiter (`mlsdm.security.RateLimiter`)
  - Configurable requests per window (default: 100/60s)
  - Per-client IP tracking
  - HTTP 429 responses on limit exceeded
- Payload scrubber for removing secrets from logs
  - Regex-based pattern matching for common secret formats
  - Support for API keys, tokens, passwords, AWS credentials, private keys
- Payload logging control via `LOG_PAYLOADS` environment variable
  - Defaults to `false` for privacy/compliance
  - Automatic secret scrubbing when enabled
- Comprehensive security documentation in `SECURITY_POLICY.md`

**Release Infrastructure**
- Semantic versioning (`__version__` in `mlsdm/__init__.py`)
- CHANGELOG.md for tracking releases
- GitHub Actions workflow for automated releases
- Docker image publishing to GitHub Container Registry
- Optional TestPyPI publishing support

### Documentation
- Updated README.md with SDK usage examples
- Added SECURITY_POLICY.md Phase 6 section
- API documentation via FastAPI's automatic Swagger UI
- Deployment guides for Docker and Kubernetes

### Infrastructure
- GitHub Actions CI/CD for testing
- Multi-stage Docker builds for smaller images
- Non-root container execution for security
- Environment-based configuration

## [Unreleased]

### Planned
- Distributed rate limiting with Redis
- API key authentication
- Request/response encryption
- Advanced monitoring and observability
- Additional language bindings (TypeScript, Go)

---

## Release Process

1. Update version in `src/mlsdm/__init__.py`
2. Update `CHANGELOG.md` with release notes
3. Create and push a git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will automatically:
   - Run tests
   - Build Docker image
   - Push to GitHub Container Registry
   - (Optional) Publish to PyPI

## Version History

- **0.1.0** (2025-11-22): Initial Phase 6 release with API, Docker, and security baseline
