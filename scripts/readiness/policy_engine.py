"""Policy engine for readiness evidence and change analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from scripts.readiness import change_analyzer as ca

ROOT = Path(__file__).resolve().parents[2]

RISK_ORDER = ["informational", "low", "medium", "high", "critical"]
CATEGORY_ORDER = list(ca.CATEGORY_PRIORITY)


def _risk_rank(name: str) -> int:
    try:
        return RISK_ORDER.index(name)
    except ValueError:
        return 0


def _convert_risk(risk: str) -> str:
    if risk == "info":
        return "informational"
    return risk


def _highest_risk(risks: Iterable[str]) -> str:
    highest = "informational"
    for risk in risks:
        converted = _convert_risk(risk)
        if _risk_rank(converted) > _risk_rank(highest):
            highest = converted
    return highest


def _load_json(path: Path) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    return json.loads(content)


def _paths_from_analysis(change_analysis: dict[str, Any]) -> list[str]:
    files = change_analysis.get("files")
    if isinstance(files, list):
        paths = []
        for item in files:
            if isinstance(item, dict) and "path" in item:
                paths.append(str(item["path"]))
        return paths
    return []


def _is_doc(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith("docs/") or lowered.endswith((".md", ".rst", ".txt"))


def _is_test(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("tests/")


def _is_observability(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("src/mlsdm/observability/")


def _is_core(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith("src/mlsdm/") and not _is_observability(normalized)


def _is_infra(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized.startswith(".github/workflows/") or normalized == "scripts/readiness_check.py"


def _is_security(path: str) -> bool:
    return ca.classify_category(path) == "security_critical"


def _validate_workflow(path: Path) -> list[str]:
    missing: list[str] = []
    try:
        content = path.read_text(encoding="utf-8")
        yaml.safe_load(content)
    except Exception:
        missing.append("Workflow YAML invalid")
        return missing
    for line in content.splitlines():
        if "uses:" not in line:
            continue
        uses_part = line.split("uses:", 1)[1].strip()
        if "@main" in uses_part:
            missing.append(f"Action {uses_part} must be pinned (no @main)")
        if "@" not in uses_part:
            missing.append(f"Action {uses_part} must declare a pinned ref")
    return missing


def evaluate_policy(change_analysis: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate readiness policy given change analysis and collected evidence."""
    paths = sorted({p.strip() for p in _paths_from_analysis(change_analysis) if p.strip()})
    max_risk = _convert_risk(str(change_analysis.get("max_risk", "informational")))

    matched_rules: list[dict[str, Any]] = []
    blocking: list[str] = []
    recommendations: list[str] = []

    categories = []
    for file_entry in change_analysis.get("files", []):
        category = file_entry.get("category")
        if category:
            categories.append(str(category))
        else:
            categories.append(ca.classify_category(str(file_entry.get("path", ""))))
    categories = sorted({c for c in categories if c})
    is_critical_change = change_analysis.get("max_risk") == "critical"

    tests_totals = evidence.get("tests", {}).get("totals", {})
    tests_measured = bool(evidence.get("sources", {}).get("junit", {}).get("found")) and (
        tests_totals.get("passed", 0) + tests_totals.get("failed", 0) + tests_totals.get("skipped", 0) > 0
    )

    security_tools = evidence.get("security", {}).get("tools", []) or []
    security_measured = bool(evidence.get("security", {}).get("measured"))
    security_high = sum(int(t.get("high", 0)) for t in security_tools if isinstance(t, dict))
    security_medium = sum(int(t.get("medium", 0)) for t in security_tools if isinstance(t, dict))

    infra_changed = any(_is_infra(p) for p in paths) or "infrastructure" in categories
    core_changed = "functional_core" in categories
    security_changed = "security_critical" in categories or any(_is_security(p) for p in paths)
    observability_only = paths and all(_is_observability(p) or _is_doc(p) or _is_test(p) for p in paths)
    docs_only = paths and all(_is_doc(p) for p in paths)
    tests_only = paths and all(_is_test(p) for p in paths)

    if infra_changed:
        missing: list[str] = []
        workflow_paths = [p for p in paths if _is_infra(p) and p.endswith((".yml", ".yaml"))]
        for wf in workflow_paths:
            missing.extend(_validate_workflow(ROOT / wf))
        if security_measured and (security_high > 0 or security_medium > 0):
            missing.append("Resolve security findings (high/medium)")
            blocking.append("Security findings present in evidence")
        rule = {
            "rule_id": "INFRA-001",
            "title": "Infrastructure and readiness changes require policy review",
            "category": "infrastructure",
            "risk": "medium",
            "requirements": ["Infrastructure changes documented", "Security scans clean", "Pinned GitHub Actions"],
            "missing": sorted(dict.fromkeys(missing)),
        }
        matched_rules.append(rule)
        if missing:
            recommendations.append("Address infrastructure gaps (pinned actions, clean security scans)")
        max_risk = _highest_risk([max_risk, rule["risk"], "high" if missing else rule["risk"]])

    if core_changed:
        core_missing: list[str] = []
        if not tests_measured:
            core_missing.append("Tests evidence missing or empty")
        elif tests_totals.get("failed", 0) > 0:
            core_missing.append("Tests have failures")
        rule = {
            "rule_id": "CORE-001",
            "title": "Core cognitive changes require passing tests",
            "category": "functional_core",
            "risk": "high",
            "requirements": ["Tests executed", "No failing tests"],
            "missing": core_missing,
        }
        matched_rules.append(rule)
        max_risk = _highest_risk([max_risk, rule["risk"]])
        if core_missing and is_critical_change:
            blocking.append("Critical core change without passing tests")

    if security_changed:
        sec_missing: list[str] = []
        if not security_measured:
            sec_missing.append("Security evidence missing")
        if security_high > 0 or security_medium > 0:
            sec_missing.append("Security findings present")
            blocking.append("Security findings detected")
        rule = {
            "rule_id": "SEC-001",
            "title": "Security-sensitive changes require clean scans",
            "category": "security_critical",
            "risk": "critical",
            "requirements": ["Security scans executed", "No high/medium findings"],
            "missing": sec_missing,
        }
        matched_rules.append(rule)
        max_risk = _highest_risk([max_risk, rule["risk"]])
        if sec_missing and not blocking:
            recommendations.append("Provide security scan evidence for security-sensitive changes")

    if observability_only or tests_only:
        rule = {
            "rule_id": "OBS-001",
            "title": "Observability/docs/tests-only change",
            "category": "observability",
            "risk": "low",
            "requirements": ["Observability changes reviewed"],
            "missing": [],
        }
        matched_rules.append(rule)
        max_risk = _highest_risk([max_risk, rule["risk"]])

    if docs_only:
        rule = {
            "rule_id": "DOC-001",
            "title": "Documentation-only change",
            "category": "documentation",
            "risk": "informational",
            "requirements": ["Docs rendered and reviewed"],
            "missing": [],
        }
        matched_rules.append(rule)
        max_risk = _highest_risk([max_risk, rule["risk"]])

    matched_rules = sorted(matched_rules, key=lambda r: r["rule_id"])

    # verdict precedence: reject > manual_review > approve_with_conditions > approve
    verdict = "approve"
    if infra_changed or core_changed or security_changed:
        verdict = "approve_with_conditions"
    if blocking:
        verdict = "reject"
    elif is_critical_change and (core_changed or security_changed):
        verdict = "manual_review"
    if (infra_changed or security_changed) and security_measured and (security_high > 0 or security_medium > 0):
        verdict = "reject"

    if not matched_rules:
        verdict = "approve"

    policy = {
        "verdict": verdict,
        "max_risk": max_risk,
        "matched_rules": matched_rules,
        "blocking": blocking,
        "recommendations": recommendations,
    }

    canonical = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    policy_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    policy["policy_hash"] = f"sha256-{policy_hash}"
    return policy


def _write_output(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if not output:
        print(text)
        return
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(text + "\n", encoding="utf-8")
    tmp_path.replace(out_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate readiness policy")
    parser.add_argument("--change-analysis", required=True, help="Path to change analysis JSON")
    parser.add_argument("--evidence", required=True, help="Path to evidence JSON")
    parser.add_argument("--output", help="Output path (default: stdout)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        change_analysis = _load_json(Path(args.change_analysis))
        evidence = _load_json(Path(args.evidence))
        policy = evaluate_policy(change_analysis, evidence)
        _write_output(policy, args.output)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
