from __future__ import annotations

import scripts.readiness.policy_engine as pe


def _base_change(paths, max_risk="high"):
    return {
        "max_risk": max_risk,
        "files": [{"path": p} for p in paths],
        "counts": {"categories": {}, "risks": {}},
    }


def _empty_evidence():
    return {
        "sources": {"junit": {"found": False}},
        "tests": {"totals": {"passed": 0, "failed": 0, "skipped": 0}},
        "security": {"tools": [], "measured": False},
    }


def test_core_changes_require_tests():
    change = _base_change(["src/mlsdm/core/module.py"], max_risk="critical")
    evidence = _empty_evidence()
    policy = pe.evaluate_policy(change, evidence)
    assert policy["verdict"] in ("reject", "manual_review")
    assert any(rule["rule_id"] == "CORE-001" for rule in policy["matched_rules"])
    assert policy["max_risk"] in ("high", "critical")


def test_infra_security_rejects_when_findings_present():
    change = _base_change([".github/workflows/readiness.yml"], max_risk="medium")
    evidence = {
        "sources": {"junit": {"found": True}},
        "tests": {"totals": {"passed": 1, "failed": 0, "skipped": 0}},
        "security": {
            "measured": True,
            "tools": [{"tool": "bandit", "high": 1, "medium": 0, "low": 0, "measured": True}],
        },
    }
    policy = pe.evaluate_policy(change, evidence)
    assert policy["verdict"] == "reject"
    assert any(rule["rule_id"] == "INFRA-001" for rule in policy["matched_rules"])
    assert policy["max_risk"] in ("high", "medium", "critical")


def test_security_rule_requires_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "ROOT", tmp_path)
    change = {
        "max_risk": "critical",
        "files": [{"path": "src/mlsdm/security/core.py", "category": "security_critical"}],
        "counts": {"categories": {"security_critical": 1}, "risks": {"critical": 1}},
    }
    evidence = {
        "sources": {"junit": {"found": False}},
        "tests": {"totals": {"passed": 0, "failed": 0, "skipped": 0}},
        "security": {"tools": [], "measured": False},
    }
    policy = pe.evaluate_policy(change, evidence)
    assert policy["verdict"] in ("approve_with_conditions", "reject", "manual_review")
    assert any(rule["rule_id"] == "SEC-001" for rule in policy["matched_rules"])
    assert list(policy) == ["verdict", "max_risk", "matched_rules", "blocking", "recommendations", "policy_hash"]


def test_info_input_is_normalized():
    change = _base_change(["src/core/app.py"], max_risk="info")
    evidence = {
        "sources": {"junit": {"found": False}},
        "tests": {"totals": {"passed": 0, "failed": 0, "skipped": 0}},
        "security": {"tools": [], "measured": False},
    }
    policy = pe.evaluate_policy(change, evidence)
    assert policy["max_risk"] != "info"


def test_infra_requires_pinned_actions(tmp_path, monkeypatch):
    workflows_dir = tmp_path / ".github" / "workflows"
    workflows_dir.mkdir(parents=True)
    wf = workflows_dir / "ci.yml"
    wf.write_text(
        """
name: CI
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@main
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(pe, "ROOT", tmp_path)
    change = _base_change([wf.relative_to(tmp_path).as_posix()], max_risk="medium")
    evidence = {
        "sources": {"junit": {"found": True}},
        "tests": {"totals": {"passed": 1, "failed": 0, "skipped": 0}},
        "security": {"tools": [], "measured": False},
    }
    policy = pe.evaluate_policy(change, evidence)
    rule = next(r for r in policy["matched_rules"] if r["rule_id"] == "INFRA-001")
    assert any("must be pinned" in m for m in rule["missing"])
