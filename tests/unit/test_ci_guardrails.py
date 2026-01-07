from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_artifact_filtering_respects_allowlists() -> None:
    module = _load_module("no_generated_artifacts", "scripts/ci/no_generated_artifacts.py")
    sample_paths = [
        "coverage.xml",
        "artifacts/evidence/2025-12-26/2a6b52dd6fd4/coverage/coverage.xml",
        "tmp/local.db",
        "data/cache.sqlite",
    ]

    forbidden = module.find_forbidden(sample_paths)

    assert "coverage.xml" in forbidden
    assert "tmp/local.db" in forbidden
    # Evidence snapshots and allowlisted DB prefixes should pass
    assert (
        "artifacts/evidence/2025-12-26/2a6b52dd6fd4/coverage/coverage.xml" not in forbidden
    )
    assert "data/cache.sqlite" not in forbidden


def test_bidi_detector_flags_control_characters() -> None:
    module = _load_module("check_bidi", "scripts/check_bidi.py")

    issues = module.find_bidi_issues(f"safe{chr(0x202E)}text")
    assert any("U+202E" in item for item in issues)

    assert module.find_bidi_issues("plain ascii only") == []
