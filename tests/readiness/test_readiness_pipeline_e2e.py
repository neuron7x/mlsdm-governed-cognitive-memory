from __future__ import annotations

import json
from pathlib import Path

from scripts.readiness import change_analyzer as ca
from scripts.readiness.changelog_generator import generate_update
from scripts.readiness.evidence_collector import collect_evidence
from scripts.readiness.policy_engine import evaluate_policy


def test_readiness_pipeline_deterministic(tmp_path: Path):
    root = tmp_path
    (root / "reports").mkdir(parents=True)
    (root / "docs" / "status").mkdir(parents=True)
    (root / "docs" / "status" / "READINESS.md").write_text(
        "# Title\nLast updated: 2024-01-01\n\n## Change Log\n", encoding="utf-8"
    )
    sample_py = root / "src" / "core"
    sample_py.mkdir(parents=True)
    (sample_py / "app.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    paths = [str((sample_py / "app.py").relative_to(root))]

    analysis = ca.analyze_paths(paths, base_ref="origin/main", root=root)
    evidence = collect_evidence(root)
    policy = evaluate_policy(analysis, evidence)

    path, updated = generate_update(
        "Pipeline E2E",
        "origin/main",
        root=root,
        diff_provider=lambda base_ref, r: paths,
        analyzer=lambda p, base_ref, root: analysis,
        evidence_collector=lambda root: evidence,
        policy_evaluator=lambda a, e: policy,
    )
    assert path == root / "docs" / "status" / "READINESS.md"

    serialized = json.dumps({"analysis": analysis, "evidence": evidence, "policy": policy, "doc": updated}, sort_keys=True)
    serialized_again = json.dumps({"analysis": analysis, "evidence": evidence, "policy": policy, "doc": updated}, sort_keys=True)
    assert serialized == serialized_again
    assert '"info"' not in serialized
