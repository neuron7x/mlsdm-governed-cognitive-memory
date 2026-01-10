from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_POLICY_DIR = Path(__file__).resolve().parents[3] / "policy"

_UNIT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(ms|s|%|percent|ratio)\s*$", re.I)


class PolicyLoadError(RuntimeError):
    """Raised when policy loading or validation fails."""


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Owner(StrictBaseModel):
    name: str
    team: str | None = None
    email: str | None = None


class RequiredCheck(StrictBaseModel):
    name: str
    description: str | None = None
    github_action: str | None = None
    workflow_file: str | None = None
    command: str | None = None
    script: str | None = None
    severity_threshold: str | None = None
    confidence_threshold: str | None = None
    minimum_coverage: float | None = None
    fail_on_violation: bool = True


class Authentication(StrictBaseModel):
    api_key_required: bool
    api_key_env_var: str
    never_hardcode_secrets: bool
    secrets_in_env_only: bool


class InputValidation(StrictBaseModel):
    llm_safety_gateway_required: bool
    llm_safety_module: str
    payload_scrubber_module: str
    validate_all_external_input: bool


class DataProtection(StrictBaseModel):
    scrub_pii_from_logs: bool
    scrubber_implementation: str
    fields_always_scrubbed: list[str]


class DependencyControls(StrictBaseModel):
    sbom_generation_on_release: bool
    security_advisory_checks: bool
    outdated_dependency_monitoring: bool


class SecurityRequirements(StrictBaseModel):
    authentication: Authentication
    input_validation: InputValidation
    data_protection: DataProtection
    dependencies: DependencyControls


class AuditRequirements(StrictBaseModel):
    security_audit_frequency_days: int
    penetration_test_frequency_days: int
    security_training_required: bool
    incident_response_plan_required: bool


class BranchProtection(StrictBaseModel):
    require_pr_reviews: int
    require_status_checks: bool
    enforce_admins: bool
    required_status_checks: list[str]


class Integrations(StrictBaseModel):
    github_required_checks: list[str]
    branch_protection: BranchProtection


class CIWorkflowPolicy(StrictBaseModel):
    prohibited_permissions: list[str]
    first_party_action_owners: list[str]
    prohibited_mutable_refs: list[str]


class VulnerabilityThreshold(StrictBaseModel):
    max_allowed: int
    response_time_hours: int
    fix_timeline_days: int


class VulnerabilityThresholds(StrictBaseModel):
    critical: VulnerabilityThreshold
    high: VulnerabilityThreshold
    medium: VulnerabilityThreshold
    low: VulnerabilityThreshold


class SecurityControls(StrictBaseModel):
    required_checks: list[RequiredCheck]
    security_requirements: SecurityRequirements
    audit_requirements: AuditRequirements
    integrations: Integrations
    ci_workflow_policy: CIWorkflowPolicy


class SecurityThresholds(StrictBaseModel):
    vulnerability_thresholds: VulnerabilityThresholds
    coverage_gate_minimum_percent: float


class SecurityBaselinePolicy(StrictBaseModel):
    policy_contract_version: str
    version: str
    policy_name: str
    enforcement_level: str
    updated_at: str
    owner: Owner
    rationale: str
    controls: SecurityControls
    thresholds: SecurityThresholds


class Metric(StrictBaseModel):
    name: str
    type: str
    labels: list[str] | None = None
    buckets: list[float] | None = None


class Metrics(StrictBaseModel):
    required_exporters: list[str]
    collection_interval_seconds: int
    retention_days: int
    key_metrics: list[Metric]


class Monitoring(StrictBaseModel):
    metrics: Metrics


class LogLevels(StrictBaseModel):
    production: str
    development: str


class Logging(StrictBaseModel):
    structured_logging_required: bool
    log_levels: LogLevels
    required_fields: list[str]


class CriticalAlert(StrictBaseModel):
    name: str
    condition: str
    severity: str


class Alerting(StrictBaseModel):
    alert_manager_required: bool
    notification_channels: list[str]
    critical_alerts: list[CriticalAlert]


class TestSuite(StrictBaseModel):
    name: str
    marker: str
    max_duration_seconds: int


class SLOTests(StrictBaseModel):
    run_frequency: str
    timeout_seconds: int
    retry_on_flake: bool
    test_suites: list[TestSuite]


class MemoryTests(StrictBaseModel):
    deterministic_seeding: bool
    noise_tolerance_mb: float
    sample_count_min: int
    warmup_iterations: int


class Testing(StrictBaseModel):
    slo_tests: SLOTests
    memory_tests: MemoryTests


class Documentation(StrictBaseModel):
    slo_spec: str
    validation_protocol: str
    runbook: str
    observability_guide: str


class ObservabilityControls(StrictBaseModel):
    monitoring: Monitoring
    logging: Logging
    alerting: Alerting
    testing: Testing
    documentation: Documentation


class ApiEndpointTargets(StrictBaseModel):
    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    max_error_rate_percent: float | None = None
    availability_percent: float | None = None
    throughput_rps: float | None = None


class ApiEndpointSLO(StrictBaseModel):
    name: str
    endpoint: str
    targets: ApiEndpointTargets
    ci_thresholds: ApiEndpointTargets | None = None
    test_location: str | None = None


class SystemResourceTargets(StrictBaseModel):
    max_usage_mb: float | None = None
    growth_rate_mb_per_hour: float | None = None
    leak_detection_threshold: float | None = None
    p95_usage_percent: float | None = None
    max_usage_percent: float | None = None


class SystemResourceSLO(StrictBaseModel):
    name: str
    metric: str
    targets: SystemResourceTargets
    test_location: str | None = None


class CognitiveEngineTargets(StrictBaseModel):
    threshold_drift_max: float | None = None
    convergence_steps_max: int | None = None
    min_threshold: float | None = None
    max_threshold: float | None = None
    entangle_latency_ms: float | None = None
    retrieve_latency_ms: float | None = None
    corruption_rate_percent: float | None = None


class CognitiveEngineSLO(StrictBaseModel):
    name: str
    targets: CognitiveEngineTargets
    test_location: str | None = None


class ApiDefaults(StrictBaseModel):
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_error_rate_percent: float
    min_availability_percent: float


class SLODefinitions(StrictBaseModel):
    api_defaults: ApiDefaults
    api_endpoints: list[ApiEndpointSLO]
    system_resources: list[SystemResourceSLO]
    cognitive_engine: list[CognitiveEngineSLO]


class LatencyDefaults(StrictBaseModel):
    api_p50_ms: float
    api_p95_ms: float
    api_p99_ms: float
    engine_total_p50_ms: float
    engine_total_p95_ms: float
    engine_preflight_p95_ms: float
    generation_p95_ms: float


class ErrorRateDefaults(StrictBaseModel):
    max_error_rate_percent: float
    min_availability_percent: float
    expected_rejection_rate_percent_min: float
    expected_rejection_rate_percent_max: float


class ThroughputDefaults(StrictBaseModel):
    min_rps: float
    max_queue_depth: int
    min_concurrent_capacity: int


class LoadMultipliers(StrictBaseModel):
    moderate_load_slo: float
    moderate_load_error: float
    readiness_check: float
    liveness_check: float


class RuntimeDefaults(StrictBaseModel):
    latency: LatencyDefaults
    error_rate: ErrorRateDefaults
    throughput: ThroughputDefaults
    load_multipliers: LoadMultipliers


class ErrorBudgetMonthly(StrictBaseModel):
    availability_target: float
    error_budget_percent: float
    downtime_minutes_allowed: float


class ErrorBudgetBurnRates(StrictBaseModel):
    fast_burn: float
    slow_burn: float


class ErrorBudgets(StrictBaseModel):
    monthly: ErrorBudgetMonthly
    burn_rate_thresholds: ErrorBudgetBurnRates


class ObservabilityThresholds(StrictBaseModel):
    slos: SLODefinitions
    runtime_defaults: RuntimeDefaults
    error_budgets: ErrorBudgets


class ObservabilitySLOPolicy(StrictBaseModel):
    policy_contract_version: str
    version: str
    policy_name: str
    enforcement_level: str
    updated_at: str
    owner: Owner
    rationale: str
    controls: ObservabilityControls
    thresholds: ObservabilityThresholds


@dataclass(frozen=True)
class PolicyBundle:
    security_baseline: SecurityBaselinePolicy
    observability_slo: ObservabilitySLOPolicy
    policy_hash: str
    canonical_json: str
    canonical_data: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise PolicyLoadError(f"Policy file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise PolicyLoadError(f"YAML parsing error in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise PolicyLoadError(f"Policy file must contain a mapping: {path}")

    return data


def _parse_unit_string(value: str) -> float | None:
    match = _UNIT_RE.match(value)
    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2).lower()

    if unit == "s":
        return number * 1000.0
    return number


def _normalize_scalar(key: str | None, value: Any) -> Any:
    if isinstance(value, str):
        parsed = _parse_unit_string(value)
        if parsed is not None:
            return parsed
        return value

    if isinstance(value, (int, float)) and key:
        if key.endswith("_ms") or key.endswith("_percent") or key.endswith("_ratio"):
            return float(value)
    return value


def _sort_list(items: list[Any]) -> list[Any]:
    if not items:
        return items

    if all(isinstance(item, dict) and "name" in item for item in items):
        return sorted(items, key=lambda item: str(item.get("name")))

    if all(isinstance(item, dict) and "endpoint" in item for item in items):
        return sorted(items, key=lambda item: str(item.get("endpoint")))

    return items


def canonicalize_policy_data(data: Any) -> Any:
    if isinstance(data, dict):
        normalized = {key: canonicalize_policy_data(_normalize_scalar(key, value)) for key, value in data.items()}
        return {key: normalized[key] for key in sorted(normalized)}

    if isinstance(data, list):
        normalized_list = [canonicalize_policy_data(item) for item in data]
        normalized_list = _sort_list(normalized_list)
        return normalized_list

    return _normalize_scalar(None, data)


def _compute_hash(canonical_json: str) -> str:
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_policy_bundle(policy_dir: Path | None = None) -> PolicyBundle:
    policy_dir = policy_dir or DEFAULT_POLICY_DIR

    security_raw = _load_yaml(policy_dir / "security-baseline.yaml")
    observability_raw = _load_yaml(policy_dir / "observability-slo.yaml")

    try:
        security_policy = SecurityBaselinePolicy.model_validate(security_raw)
        observability_policy = ObservabilitySLOPolicy.model_validate(observability_raw)
    except ValidationError as exc:
        raise PolicyLoadError(f"Policy schema validation failed: {exc}") from exc

    canonical_data = canonicalize_policy_data(
        {
            "security_baseline": security_policy.model_dump(mode="python"),
            "observability_slo": observability_policy.model_dump(mode="python"),
        }
    )
    canonical_json = json.dumps(canonical_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    policy_hash = _compute_hash(canonical_json)

    return PolicyBundle(
        security_baseline=security_policy,
        observability_slo=observability_policy,
        policy_hash=policy_hash,
        canonical_json=canonical_json,
        canonical_data=canonical_data,
    )


def export_opa_policy_data(policy_dir: Path, output_path: Path) -> dict[str, Any]:
    bundle = load_policy_bundle(policy_dir)
    data = {
        "policy": bundle.canonical_data,
        "policy_hash": bundle.policy_hash,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def policy_required_checks(bundle: PolicyBundle) -> Iterable[RequiredCheck]:
    return bundle.security_baseline.controls.required_checks


def policy_test_locations(bundle: PolicyBundle) -> list[tuple[str, str]]:
    slos = bundle.observability_slo.thresholds.slos
    locations: list[tuple[str, str]] = []

    for endpoint in slos.api_endpoints:
        if endpoint.test_location:
            locations.append((endpoint.name, endpoint.test_location))

    for resource in slos.system_resources:
        if resource.test_location:
            locations.append((resource.name, resource.test_location))

    for component in slos.cognitive_engine:
        if component.test_location:
            locations.append((component.name, component.test_location))

    return locations


def policy_documentation_paths(bundle: PolicyBundle) -> dict[str, str]:
    docs = bundle.observability_slo.controls.documentation
    return {
        "slo_spec": docs.slo_spec,
        "validation_protocol": docs.validation_protocol,
        "runbook": docs.runbook,
        "observability_guide": docs.observability_guide,
    }
