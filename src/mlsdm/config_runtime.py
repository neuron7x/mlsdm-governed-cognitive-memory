"""
Runtime Configuration for MLSDM.

Provides centralized runtime configuration for different deployment modes:
- dev: Local development mode
- local-prod: Local production mode
- cloud-prod: Cloud production mode (Docker/k8s)
- agent-api: API/Agent mode for LLM platforms

Configuration priority (highest to lowest):
1. Environment variables (MLSDM_* prefix)
2. Mode-specific defaults
3. Base defaults

Usage:
    from mlsdm.config_runtime import get_runtime_config, RuntimeMode

    # Get configuration for a specific mode
    config = get_runtime_config(mode=RuntimeMode.DEV)

    # Get configuration from environment (auto-detect mode)
    config = get_runtime_config()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RuntimeMode(str, Enum):
    """Supported runtime modes for MLSDM."""

    DEV = "dev"                    # Local development
    LOCAL_PROD = "local-prod"      # Local production
    CLOUD_PROD = "cloud-prod"      # Cloud production (Docker/k8s)
    AGENT_API = "agent-api"        # API/Agent mode


@dataclass
class ServerConfig:
    """Server configuration for HTTP API."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"
    timeout_keep_alive: int = 30


@dataclass
class SecurityConfig:
    """Security configuration.
    
    Attributes:
        api_key: API key for basic authentication
        rate_limit_enabled: Enable rate limiting
        rate_limit_requests: Maximum requests per window
        rate_limit_window: Rate limit window in seconds
        secure_mode: Enable secure mode (training disabled)
        cors_origins: Allowed CORS origins
        
        # Advanced Security Features (v1.1)
        enable_oidc: Enable OIDC authentication (SEC-004)
        enable_mtls: Enable mutual TLS client certificate authentication (SEC-006)
        enable_rbac: Enable role-based access control
        enable_request_signing: Enable request signature verification
        enable_policy_engine: Enable policy-as-code request evaluation
        enable_guardrails: Enable LLM output guardrails
        enable_llm_safety: Enable LLM prompt/response safety analysis
        enable_pii_scrub_logs: Enable PII scrubbing in logs
        enable_multi_tenant_enforcement: Enable multi-tenant data isolation
    """

    api_key: str | None = None
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    secure_mode: bool = False
    cors_origins: list[str] = field(default_factory=list)
    
    # Advanced Security Features (v1.1)
    enable_oidc: bool = False
    enable_mtls: bool = False
    enable_rbac: bool = False
    enable_request_signing: bool = False
    enable_policy_engine: bool = False
    enable_guardrails: bool = False
    enable_llm_safety: bool = False
    enable_pii_scrub_logs: bool = False
    enable_multi_tenant_enforcement: bool = False


@dataclass
class ObservabilityConfig:
    """Observability configuration."""

    log_level: str = "INFO"
    json_logging: bool = False
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    otel_exporter_type: str = "none"
    otel_service_name: str = "mlsdm"


@dataclass
class EngineConfig:
    """NeuroCognitiveEngine configuration."""

    llm_backend: str = "local_stub"
    embedding_dim: int = 384
    enable_fslgs: bool = True
    enable_metrics: bool = True
    config_path: str = "config/default_config.yaml"


@dataclass
class RuntimeConfig:
    """Complete runtime configuration."""

    mode: RuntimeMode
    server: ServerConfig
    security: SecurityConfig
    observability: ObservabilityConfig
    engine: EngineConfig
    debug: bool = False

    def to_env_dict(self) -> dict[str, str]:
        """Convert configuration to environment variable dictionary.

        Returns:
            Dictionary of environment variable names and values.
        """
        env: dict[str, str] = {
            # Server
            "HOST": self.server.host,
            "PORT": str(self.server.port),
            # Security - Basic
            "DISABLE_RATE_LIMIT": "0" if self.security.rate_limit_enabled else "1",
            "RATE_LIMIT_REQUESTS": str(self.security.rate_limit_requests),
            "RATE_LIMIT_WINDOW": str(self.security.rate_limit_window),
            "MLSDM_SECURE_MODE": "1" if self.security.secure_mode else "0",
            # Security - Advanced (v1.1)
            "MLSDM_SECURITY_ENABLE_OIDC": "1" if self.security.enable_oidc else "0",
            "MLSDM_SECURITY_ENABLE_MTLS": "1" if self.security.enable_mtls else "0",
            "MLSDM_SECURITY_ENABLE_RBAC": "1" if self.security.enable_rbac else "0",
            "MLSDM_SECURITY_ENABLE_REQUEST_SIGNING": "1" if self.security.enable_request_signing else "0",
            "MLSDM_SECURITY_ENABLE_POLICY_ENGINE": "1" if self.security.enable_policy_engine else "0",
            "MLSDM_SECURITY_ENABLE_GUARDRAILS": "1" if self.security.enable_guardrails else "0",
            "MLSDM_SECURITY_ENABLE_LLM_SAFETY": "1" if self.security.enable_llm_safety else "0",
            "MLSDM_SECURITY_ENABLE_PII_SCRUB_LOGS": "1" if self.security.enable_pii_scrub_logs else "0",
            "MLSDM_SECURITY_ENABLE_MULTI_TENANT_ENFORCEMENT": "1" if self.security.enable_multi_tenant_enforcement else "0",
            # Observability
            "LOG_LEVEL": self.observability.log_level,
            "JSON_LOGGING": "true" if self.observability.json_logging else "false",
            "ENABLE_METRICS": "true" if self.observability.metrics_enabled else "false",
            "OTEL_SDK_DISABLED": "false" if self.observability.tracing_enabled else "true",
            "OTEL_EXPORTER_TYPE": self.observability.otel_exporter_type,
            "OTEL_SERVICE_NAME": self.observability.otel_service_name,
            # Engine
            "LLM_BACKEND": self.engine.llm_backend,
            "EMBEDDING_DIM": str(self.engine.embedding_dim),
            "ENABLE_FSLGS": "true" if self.engine.enable_fslgs else "false",
            "CONFIG_PATH": self.engine.config_path,
        }
        if self.security.api_key:
            env["API_KEY"] = self.security.api_key
        return env


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment."""
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def _get_mode_defaults(mode: RuntimeMode) -> dict[str, Any]:
    """Get default configuration values for a specific mode.

    Args:
        mode: Runtime mode.

    Returns:
        Dictionary of default values for the mode.
    """
    # Base defaults (shared across modes)
    base: dict[str, Any] = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "reload": False,
            "log_level": "info",
            "timeout_keep_alive": 30,
        },
        "security": {
            "api_key": None,
            "rate_limit_enabled": True,
            "rate_limit_requests": 100,
            "rate_limit_window": 60,
            "secure_mode": False,
            "cors_origins": [],
            # Advanced security features (default: disabled)
            "enable_oidc": False,
            "enable_mtls": False,
            "enable_rbac": False,
            "enable_request_signing": False,
            "enable_policy_engine": False,
            "enable_guardrails": False,
            "enable_llm_safety": False,
            "enable_pii_scrub_logs": False,
            "enable_multi_tenant_enforcement": False,
        },
        "observability": {
            "log_level": "INFO",
            "json_logging": False,
            "metrics_enabled": True,
            "tracing_enabled": False,
            "otel_exporter_type": "none",
            "otel_service_name": "mlsdm",
        },
        "engine": {
            "llm_backend": "local_stub",
            "embedding_dim": 384,
            "enable_fslgs": True,
            "enable_metrics": True,
            "config_path": "config/default_config.yaml",
        },
        "debug": False,
    }

    # Mode-specific overrides
    if mode == RuntimeMode.DEV:
        # Development: minimal security for ease of use
        base["server"]["reload"] = True
        base["server"]["workers"] = 1
        base["server"]["log_level"] = "debug"
        base["security"]["rate_limit_enabled"] = False
        base["security"]["secure_mode"] = False
        # Advanced security disabled in dev (can be overridden via env)
        base["security"]["enable_oidc"] = False
        base["security"]["enable_mtls"] = False
        base["security"]["enable_rbac"] = False
        base["security"]["enable_request_signing"] = False
        base["security"]["enable_policy_engine"] = False
        base["security"]["enable_guardrails"] = False
        base["security"]["enable_llm_safety"] = False
        base["security"]["enable_pii_scrub_logs"] = False
        base["security"]["enable_multi_tenant_enforcement"] = False
        base["observability"]["log_level"] = "DEBUG"
        base["observability"]["json_logging"] = False
        base["debug"] = True
        base["engine"]["config_path"] = "config/default_config.yaml"

    elif mode == RuntimeMode.LOCAL_PROD:
        # Local production: moderate security
        base["server"]["workers"] = 2
        base["server"]["log_level"] = "info"
        base["security"]["rate_limit_enabled"] = True
        base["security"]["secure_mode"] = True
        # Enable basic advanced security features
        base["security"]["enable_policy_engine"] = True
        base["security"]["enable_guardrails"] = True
        base["security"]["enable_llm_safety"] = True
        base["security"]["enable_pii_scrub_logs"] = True
        # Optional features disabled by default (can enable via env)
        base["security"]["enable_oidc"] = False
        base["security"]["enable_mtls"] = False
        base["security"]["enable_rbac"] = False
        base["security"]["enable_request_signing"] = False
        base["security"]["enable_multi_tenant_enforcement"] = False
        base["observability"]["log_level"] = "INFO"
        base["observability"]["json_logging"] = True
        base["observability"]["metrics_enabled"] = True
        base["engine"]["config_path"] = "config/production.yaml"

    elif mode == RuntimeMode.CLOUD_PROD:
        # Cloud production: full security hardening
        base["server"]["workers"] = 4
        base["server"]["log_level"] = "info"
        base["server"]["timeout_keep_alive"] = 60
        base["security"]["rate_limit_enabled"] = True
        base["security"]["secure_mode"] = True
        # Enable all advanced security features in production
        base["security"]["enable_oidc"] = True
        base["security"]["enable_mtls"] = True
        base["security"]["enable_rbac"] = True
        base["security"]["enable_request_signing"] = True
        base["security"]["enable_policy_engine"] = True
        base["security"]["enable_guardrails"] = True
        base["security"]["enable_llm_safety"] = True
        base["security"]["enable_pii_scrub_logs"] = True
        base["security"]["enable_multi_tenant_enforcement"] = True
        base["observability"]["log_level"] = "INFO"
        base["observability"]["json_logging"] = True
        base["observability"]["metrics_enabled"] = True
        base["observability"]["tracing_enabled"] = True
        base["observability"]["otel_exporter_type"] = "otlp"
        base["engine"]["config_path"] = "config/production.yaml"

    elif mode == RuntimeMode.AGENT_API:
        # Agent API: moderate to high security
        base["server"]["workers"] = 2
        base["server"]["log_level"] = "info"
        base["security"]["rate_limit_enabled"] = True
        base["security"]["secure_mode"] = True
        # Enable key security features for agent mode
        base["security"]["enable_policy_engine"] = True
        base["security"]["enable_guardrails"] = True
        base["security"]["enable_llm_safety"] = True
        base["security"]["enable_pii_scrub_logs"] = True
        # Optional features (can enable via env)
        base["security"]["enable_oidc"] = False
        base["security"]["enable_mtls"] = False
        base["security"]["enable_rbac"] = False
        base["security"]["enable_request_signing"] = False
        base["security"]["enable_multi_tenant_enforcement"] = False
        base["observability"]["log_level"] = "INFO"
        base["observability"]["json_logging"] = True
        base["observability"]["metrics_enabled"] = True
        base["engine"]["enable_fslgs"] = True
        base["engine"]["config_path"] = "config/production.yaml"

    return base


def get_runtime_mode() -> RuntimeMode:
    """Get the current runtime mode from environment.

    Uses MLSDM_RUNTIME_MODE environment variable.
    Defaults to DEV if not set or invalid.

    Returns:
        RuntimeMode enum value.
    """
    mode_str = os.environ.get("MLSDM_RUNTIME_MODE", "dev").lower()
    try:
        return RuntimeMode(mode_str)
    except ValueError:
        return RuntimeMode.DEV


def get_runtime_config(mode: RuntimeMode | None = None) -> RuntimeConfig:
    """Get runtime configuration for the specified mode.

    Args:
        mode: Runtime mode. If None, auto-detected from MLSDM_RUNTIME_MODE env var.

    Returns:
        RuntimeConfig instance with merged defaults and environment overrides.
    """
    if mode is None:
        mode = get_runtime_mode()

    defaults = _get_mode_defaults(mode)

    # Server config with env overrides
    server = ServerConfig(
        host=_get_env_str("HOST", defaults["server"]["host"]),
        port=_get_env_int("PORT", defaults["server"]["port"]),
        workers=_get_env_int("MLSDM_WORKERS", defaults["server"]["workers"]),
        reload=_get_env_bool("MLSDM_RELOAD", defaults["server"]["reload"]),
        log_level=_get_env_str("MLSDM_LOG_LEVEL", defaults["server"]["log_level"]),
        timeout_keep_alive=_get_env_int(
            "MLSDM_TIMEOUT_KEEP_ALIVE", defaults["server"]["timeout_keep_alive"]
        ),
    )

    # Security config with env overrides
    security = SecurityConfig(
        api_key=os.environ.get("API_KEY", defaults["security"]["api_key"]),
        rate_limit_enabled=_get_env_bool(
            "MLSDM_RATE_LIMIT_ENABLED", defaults["security"]["rate_limit_enabled"]
        )
        and os.environ.get("DISABLE_RATE_LIMIT") != "1",
        rate_limit_requests=_get_env_int(
            "RATE_LIMIT_REQUESTS", defaults["security"]["rate_limit_requests"]
        ),
        rate_limit_window=_get_env_int(
            "RATE_LIMIT_WINDOW", defaults["security"]["rate_limit_window"]
        ),
        secure_mode=_get_env_bool(
            "MLSDM_SECURE_MODE", defaults["security"]["secure_mode"]
        ),
        cors_origins=defaults["security"]["cors_origins"],
        # Advanced security features with env overrides
        enable_oidc=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_OIDC", defaults["security"]["enable_oidc"]
        ),
        enable_mtls=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_MTLS", defaults["security"]["enable_mtls"]
        ),
        enable_rbac=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_RBAC", defaults["security"]["enable_rbac"]
        ),
        enable_request_signing=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_REQUEST_SIGNING", defaults["security"]["enable_request_signing"]
        ),
        enable_policy_engine=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_POLICY_ENGINE", defaults["security"]["enable_policy_engine"]
        ),
        enable_guardrails=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_GUARDRAILS", defaults["security"]["enable_guardrails"]
        ),
        enable_llm_safety=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_LLM_SAFETY", defaults["security"]["enable_llm_safety"]
        ),
        enable_pii_scrub_logs=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_PII_SCRUB_LOGS", defaults["security"]["enable_pii_scrub_logs"]
        ),
        enable_multi_tenant_enforcement=_get_env_bool(
            "MLSDM_SECURITY_ENABLE_MULTI_TENANT_ENFORCEMENT", defaults["security"]["enable_multi_tenant_enforcement"]
        ),
    )

    # Observability config with env overrides
    observability = ObservabilityConfig(
        log_level=_get_env_str("LOG_LEVEL", defaults["observability"]["log_level"]),
        json_logging=_get_env_bool(
            "JSON_LOGGING", defaults["observability"]["json_logging"]
        ),
        metrics_enabled=_get_env_bool(
            "ENABLE_METRICS", defaults["observability"]["metrics_enabled"]
        ),
        tracing_enabled=_get_env_bool(
            "OTEL_TRACING_ENABLED", defaults["observability"]["tracing_enabled"]
        )
        and os.environ.get("OTEL_SDK_DISABLED", "true").lower() != "true",
        otel_exporter_type=_get_env_str(
            "OTEL_EXPORTER_TYPE", defaults["observability"]["otel_exporter_type"]
        ),
        otel_service_name=_get_env_str(
            "OTEL_SERVICE_NAME", defaults["observability"]["otel_service_name"]
        ),
    )

    # Engine config with env overrides
    engine = EngineConfig(
        llm_backend=_get_env_str("LLM_BACKEND", defaults["engine"]["llm_backend"]),
        embedding_dim=_get_env_int("EMBEDDING_DIM", defaults["engine"]["embedding_dim"]),
        enable_fslgs=_get_env_bool("ENABLE_FSLGS", defaults["engine"]["enable_fslgs"]),
        enable_metrics=_get_env_bool(
            "ENABLE_METRICS", defaults["engine"]["enable_metrics"]
        ),
        config_path=_get_env_str("CONFIG_PATH", defaults["engine"]["config_path"]),
    )

    return RuntimeConfig(
        mode=mode,
        server=server,
        security=security,
        observability=observability,
        engine=engine,
        debug=_get_env_bool("MLSDM_DEBUG", defaults["debug"]),
    )


def apply_runtime_config(config: RuntimeConfig) -> None:
    """Apply runtime configuration to environment.

    Sets environment variables based on the configuration.
    Useful for ensuring consistent configuration across all components.

    Args:
        config: RuntimeConfig instance to apply.
    """
    for key, value in config.to_env_dict().items():
        os.environ[key] = value


def print_runtime_config(config: RuntimeConfig) -> None:
    """Print runtime configuration in a human-readable format.

    Args:
        config: RuntimeConfig instance to print.
    """
    print("=" * 60)
    print(f"MLSDM Runtime Configuration ({config.mode.value})")
    print("=" * 60)
    print()
    print("Server:")
    print(f"  Host: {config.server.host}")
    print(f"  Port: {config.server.port}")
    print(f"  Workers: {config.server.workers}")
    print(f"  Reload: {config.server.reload}")
    print(f"  Log Level: {config.server.log_level}")
    print()
    print("Security:")
    print(f"  API Key: {'<set>' if config.security.api_key else '<not set>'}")
    print(f"  Rate Limit Enabled: {config.security.rate_limit_enabled}")
    print(f"  Secure Mode: {config.security.secure_mode}")
    print("  Advanced Security Features:")
    print(f"    OIDC: {config.security.enable_oidc}")
    print(f"    mTLS: {config.security.enable_mtls}")
    print(f"    RBAC: {config.security.enable_rbac}")
    print(f"    Request Signing: {config.security.enable_request_signing}")
    print(f"    Policy Engine: {config.security.enable_policy_engine}")
    print(f"    Guardrails: {config.security.enable_guardrails}")
    print(f"    LLM Safety: {config.security.enable_llm_safety}")
    print(f"    PII Scrubbing: {config.security.enable_pii_scrub_logs}")
    print(f"    Multi-Tenant: {config.security.enable_multi_tenant_enforcement}")
    print()
    print("Observability:")
    print(f"  Log Level: {config.observability.log_level}")
    print(f"  JSON Logging: {config.observability.json_logging}")
    print(f"  Metrics Enabled: {config.observability.metrics_enabled}")
    print(f"  Tracing Enabled: {config.observability.tracing_enabled}")
    print()
    print("Engine:")
    print(f"  LLM Backend: {config.engine.llm_backend}")
    print(f"  Embedding Dim: {config.engine.embedding_dim}")
    print(f"  FSLGS Enabled: {config.engine.enable_fslgs}")
    print(f"  Config Path: {config.engine.config_path}")
    print("=" * 60)
