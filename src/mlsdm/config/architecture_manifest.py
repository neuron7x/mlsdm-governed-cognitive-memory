"""Architecture manifest defining module boundaries and contracts.

This manifest provides a single source of truth for module responsibilities,
public interfaces, and allowed dependencies. It is intentionally lightweight
to keep validation fast and to avoid introducing new runtime dependencies.

The manifest is validated in tests to ensure:
* Every declared module directory exists
* Public interface files are present
* Allowed dependencies only reference known modules
* Layers use a constrained vocabulary for consistency
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
KNOWN_LAYERS = {
    "interface",
    "service",
    "engine",
    "cognitive-core",
    "memory",
    "cross-cutting",
    "integration",
    "foundation",
}


@dataclass(frozen=True)
class ArchitectureModule:
    """Declarative description of a module boundary."""

    name: str
    path: str
    layer: str
    responsibilities: Sequence[str]
    public_interfaces: Sequence[str]
    allowed_dependencies: Sequence[str]

    def absolute_path(self) -> Path:
        """Absolute filesystem path to the module directory."""
        return PACKAGE_ROOT / self.path

    def interface_paths(self) -> tuple[Path, ...]:
        """Absolute filesystem paths to declared public interface files."""
        base = self.absolute_path()
        return tuple(base / interface for interface in self.public_interfaces)


ARCHITECTURE_MANIFEST: tuple[ArchitectureModule, ...] = (
    ArchitectureModule(
        name="api",
        path="api",
        layer="interface",
        responsibilities=(
            "FastAPI surface area",
            "request validation and middleware",
            "lifecycle hooks for the cognitive engine",
        ),
        public_interfaces=("app.py", "health.py", "schemas.py"),
        allowed_dependencies=(
            "engine",
            "core",
            "router",
            "security",
            "observability",
            "utils",
        ),
    ),
    ArchitectureModule(
        name="sdk",
        path="sdk",
        layer="interface",
        responsibilities=(
            "client-facing SDK for embedding MLSDM",
            "configuration helpers for NeuroCognitiveEngine",
        ),
        public_interfaces=("neuro_engine_client.py",),
        allowed_dependencies=("engine", "utils"),
    ),
    ArchitectureModule(
        name="engine",
        path="engine",
        layer="engine",
        responsibilities=(
            "composition of cognitive subsystems",
            "engine configuration and factories",
            "routing to wrappers and adapters",
        ),
        public_interfaces=("neuro_cognitive_engine.py", "factory.py"),
        allowed_dependencies=("core", "memory", "router", "security", "observability", "utils"),
    ),
    ArchitectureModule(
        name="core",
        path="core",
        layer="cognitive-core",
        responsibilities=(
            "orchestration of cognitive pipeline",
            "LLM wrapper coordination",
            "memory manager lifecycle",
        ),
        public_interfaces=("cognitive_controller.py", "llm_pipeline.py", "memory_manager.py"),
        allowed_dependencies=("memory", "security", "observability", "utils"),
    ),
    ArchitectureModule(
        name="memory",
        path="memory",
        layer="memory",
        responsibilities=(
            "multi-level synaptic memory primitives",
            "phase-entangled lattice memory",
            "memory calibration and persistence",
        ),
        public_interfaces=("multi_level_memory.py", "phase_entangled_lattice_memory.py"),
        allowed_dependencies=("utils", "observability"),
    ),
    ArchitectureModule(
        name="router",
        path="router",
        layer="service",
        responsibilities=(
            "policy-aware routing between LLM providers",
            "adapter selection and failover",
        ),
        public_interfaces=("llm_router.py",),
        allowed_dependencies=("adapters", "security", "observability", "utils"),
    ),
    ArchitectureModule(
        name="adapters",
        path="adapters",
        layer="integration",
        responsibilities=(
            "provider-specific adapters",
            "provider factory and safety shims",
        ),
        public_interfaces=("provider_factory.py", "llm_provider.py"),
        allowed_dependencies=("security", "utils"),
    ),
    ArchitectureModule(
        name="security",
        path="security",
        layer="cross-cutting",
        responsibilities=(
            "policy engine and guardrails",
            "payload scrubbing and RBAC helpers",
            "rate limiting primitives",
        ),
        public_interfaces=("policy_engine.py", "guardrails.py", "payload_scrubber.py"),
        allowed_dependencies=("utils", "observability"),
    ),
    ArchitectureModule(
        name="observability",
        path="observability",
        layer="cross-cutting",
        responsibilities=("metrics, logging, and tracing infrastructure",),
        public_interfaces=("logger.py", "metrics.py", "tracing.py"),
        allowed_dependencies=("utils",),
    ),
    ArchitectureModule(
        name="utils",
        path="utils",
        layer="foundation",
        responsibilities=(
            "configuration loading and validation",
            "shared primitives (bulkheads, circuit breakers, caches)",
            "lightweight metrics helpers",
        ),
        public_interfaces=("config_loader.py", "config_validator.py", "metrics.py"),
        allowed_dependencies=(),
    ),
)


def validate_manifest(manifest: Iterable[ArchitectureModule]) -> list[str]:
    """Validate manifest consistency and return a list of issues."""
    modules = list(manifest)
    issues: list[str] = []

    names = [module.name for module in modules]
    if len(names) != len(set(names)):
        issues.append("Module names must be unique")

    for module in modules:
        if module.layer not in KNOWN_LAYERS:
            issues.append(f"Unknown layer '{module.layer}' for module '{module.name}'")

        module_path = module.absolute_path()
        if not module_path.exists():
            issues.append(f"Path does not exist for module '{module.name}': {module_path}")
        elif not module_path.is_dir():
            issues.append(f"Path for module '{module.name}' is not a directory: {module_path}")

        if not module.responsibilities:
            issues.append(f"No responsibilities defined for module '{module.name}'")
        if not module.public_interfaces:
            issues.append(f"No public interfaces defined for module '{module.name}'")

        for interface_path in module.interface_paths():
            if not interface_path.exists():
                issues.append(
                    f"Public interface '{interface_path.name}' missing for module '{module.name}'"
                )

        for dependency in module.allowed_dependencies:
            if dependency not in names:
                issues.append(
                    f"Module '{module.name}' declares unknown dependency '{dependency}'"
                )

    return issues
