"""Contract tests for the architecture manifest."""

from __future__ import annotations

from mlsdm.config.architecture_manifest import ARCHITECTURE_MANIFEST, validate_manifest


def test_architecture_manifest_is_consistent() -> None:
    """Manifest should have no validation issues."""
    issues = validate_manifest(ARCHITECTURE_MANIFEST)
    assert not issues, f"Architecture manifest violations: {issues}"


def test_manifest_covers_primary_modules() -> None:
    """Ensure the manifest declares the primary system boundaries."""
    names = {module.name for module in ARCHITECTURE_MANIFEST}
    expected = {
        "api",
        "sdk",
        "engine",
        "core",
        "memory",
        "router",
        "adapters",
        "security",
        "observability",
        "utils",
    }
    assert expected.issubset(names)
