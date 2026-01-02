"""Deterministic CI failure intelligence summary generator.

This script is intentionally stdlib-only and resilient to missing inputs.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from xml.etree import ElementTree as ET


DEFAULT_MAX_LINES = 300
DEFAULT_MAX_BYTES = 200_000
TOP_FAILURE_LIMIT = 10


def _truncate_text(raw: str, max_lines: int, max_bytes: int) -> str:
    lines = (raw or "").splitlines()
    truncated_lines: List[str] = lines[:max_lines]
    joined = "\n".join(truncated_lines)
    if len(joined.encode("utf-8")) > max_bytes:
        encoded = joined.encode("utf-8")[:max_bytes]
        joined = encoded.decode("utf-8", errors="ignore")
    return joined.strip()


def _discover_file(explicit: Optional[str], patterns: Sequence[str]) -> Optional[str]:
    if explicit and os.path.isfile(explicit):
        return explicit
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            if os.path.isfile(path):
                return path
    return None


def parse_junit(path: Optional[str], max_lines: int, max_bytes: int) -> List[Dict[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []
    root = tree.getroot()
    candidates: List[Dict[str, str]] = []
    for case in root.iter("testcase"):
        node = case.find("failure")
        if node is None:
            node = case.find("error")
        if node is None:
            continue
        name = case.get("name") or ""
        classname = case.get("classname") or ""
        file_path = case.get("file") or ""
        test_id = f"{classname}::{name}" if classname else name
        message = (node.get("message") or "").strip()
        trace = _truncate_text((node.text or "").strip(), max_lines, max_bytes)
        candidates.append(
            {
                "id": test_id,
                "file": file_path,
                "message": message,
                "trace": trace,
            }
        )
    failures = sorted(candidates, key=lambda item: item.get("id") or "")
    return failures[:TOP_FAILURE_LIMIT]


def parse_coverage(path: Optional[str]) -> Optional[float]:
    if not path or not os.path.isfile(path):
        return None
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    rate = root.attrib.get("line-rate")
    if rate is None:
        return None
    try:
        return round(float(rate) * 100, 2)
    except ValueError:
        return None


def _read_changed_files(path: Optional[str]) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return [line.strip() for line in handle.readlines() if line.strip()]


def _collect_logs(glob_pattern: Optional[str], max_lines: int, max_bytes: int) -> Dict[str, str]:
    if not glob_pattern:
        return {}
    logs: Dict[str, str] = {}
    for log_path in sorted(glob.glob(glob_pattern, recursive=True)):
        if not os.path.isfile(log_path):
            continue
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        logs[log_path] = _truncate_text(content, max_lines, max_bytes)
    return logs


def classify_failure(
    failures: Sequence[Dict[str, str]],
    logs: Dict[str, str],
) -> Dict[str, str]:
    if not failures and not logs:
        return {"category": "pass", "reason": "No failures detected"}

    combined_text = " ".join(
        [
            *(f.get("message", "") for f in failures),
            *(f.get("trace", "") for f in failures),
            *logs.values(),
        ]
    ).lower()

    for keyword in ("connection refused", "network unreachable", "timeout connecting", "rate limit"):
        if keyword in combined_text:
            return {"category": "infra", "reason": f"Detected infra keyword '{keyword}'"}

    for keyword in ("ruff", "mypy", "flake8", "static analysis"):
        if keyword in combined_text:
            return {"category": "static analysis", "reason": f"Detected static analysis keyword '{keyword}'"}

    file_counts: Dict[str, int] = {}
    for failure in failures:
        file_path = failure.get("file") or ""
        if file_path:
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
    if any(count > 1 for count in file_counts.values()):
        return {"category": "deterministic test", "reason": "Repeated failures within the same test file"}

    for keyword in ("flaky", "intermittent", "timed out", "timeout", "race", "retry"):
        if keyword in combined_text:
            return {"category": "possible flake", "reason": f"Detected flake keyword '{keyword}'"}

    return {"category": "deterministic test", "reason": "Test failures without infra/static indicators"}


def _extract_module(path: str) -> str:
    parts = [p for p in Path(path).parts if p]
    if not parts:
        return path
    # Use up to three segments to capture paths like src/mlsdm/module
    return "/".join(parts[:3])


def impacted_modules(failures: Sequence[Dict[str, str]], changed_files: Sequence[str]) -> List[str]:
    fail_paths = {f.get("file") for f in failures if f.get("file")}
    modules: List[str] = []
    for path in fail_paths:
        module = _extract_module(path)
        if module not in modules:
            modules.append(module)
    intersecting: List[str] = []
    for changed in changed_files:
        changed_path = Path(changed)
        for fail_path in fail_paths:
            if not fail_path:
                continue
            fail_dir = Path(fail_path).parent
            if fail_dir == Path("."):
                continue
            if changed_path.is_relative_to(fail_dir):
                module = _extract_module(changed)
                if module not in intersecting:
                    intersecting.append(module)
    return intersecting or modules


def available_repro_commands() -> List[str]:
    makefile_path = "Makefile"
    commands: List[str] = []
    if os.path.isfile(makefile_path):
        with open(makefile_path, "r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
        if re.search(r"^test-fast:", content, flags=re.MULTILINE):
            commands.append("make test-fast")
        if re.search(r"^lint:", content, flags=re.MULTILINE):
            commands.append("make lint")
        if re.search(r"^type:", content, flags=re.MULTILINE):
            commands.append("make type")
    if not commands:
        commands.append("python -m pytest -q")
    return commands


def _redact(text: str) -> str:
    patterns = [
        (r"ghp_[A-Za-z0-9]{10,}", "ghp_[REDACTED]"),
        (r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]"),
        (r"AWS_SECRET_ACCESS_KEY[^\s]*", "AWS_SECRET_ACCESS_KEY[REDACTED]"),
        (r"BEGIN PRIVATE KEY[^-]*END PRIVATE KEY", "[REDACTED PRIVATE KEY]"),
    ]
    redacted = text
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
    return redacted


def _redact_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _redact_structure(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_structure(v) for v in value]
    if isinstance(value, str):
        return _redact(value)
    return value


def build_markdown(summary: Dict) -> str:
    lines = []
    lines.append("## Failure Intelligence")
    lines.append("")
    lines.append(f"**Signal:** {summary.get('signal')}")
    lines.append("")
    lines.append("### Top Failures")
    if summary["top_failures"]:
        for failure in summary["top_failures"]:
            lines.append(f"- {failure.get('id') or 'unknown'} ({failure.get('file') or 'n/a'})")
            if failure.get("message"):
                lines.append(f"  - Message: {failure['message']}")
            if failure.get("trace"):
                lines.append("  - Traceback:")
                trace_block = textwrap.indent(failure["trace"], "    ")
                lines.append(f"```\n{trace_block}\n```")
    else:
        lines.append("- No failing tests were detected.")
    lines.append("")
    classification = summary.get("classification", {})
    lines.append(
        f"### Classification\n- Category: {classification.get('category')}\n- Reason: {classification.get('reason')}"
    )
    lines.append("")
    lines.append("### Coverage")
    coverage = summary.get("coverage_percent")
    lines.append(f"- Line coverage: {coverage if coverage is not None else 'Unavailable'}")
    lines.append("")
    lines.append("### Impacted Modules")
    if summary.get("impacted_modules"):
        for module in summary["impacted_modules"]:
            lines.append(f"- {module}")
    else:
        lines.append("- Unable to determine from available data.")
    lines.append("")
    lines.append("### Reproduce Locally")
    for command in summary.get("repro_commands", []):
        lines.append(f"- `{command}`")
    lines.append("")
    lines.append("### Evidence")
    for pointer in summary.get("evidence", []):
        lines.append(f"- {pointer}")
    return "\n".join(lines)


def generate_summary(
    junit_path: Optional[str],
    coverage_path: Optional[str],
    changed_files_path: Optional[str],
    log_glob: Optional[str],
    max_lines: int,
    max_bytes: int,
) -> Dict:
    failures = parse_junit(junit_path, max_lines, max_bytes)
    coverage_percent = parse_coverage(coverage_path)
    changed_files = _read_changed_files(changed_files_path)
    logs = _collect_logs(log_glob, max_lines, max_bytes)
    classification = classify_failure(failures, logs)
    modules = impacted_modules(failures, changed_files)
    summary = {
        "signal": "Failures detected" if failures else "No failures detected",
        "top_failures": failures,
        "coverage_percent": coverage_percent,
        "classification": classification,
        "impacted_modules": modules,
        "repro_commands": available_repro_commands(),
        "evidence": [p for p in (junit_path, coverage_path, changed_files_path) if p],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic failure intelligence summary.")
    parser.add_argument("--junit", help="Path to junit XML file", default=None)
    parser.add_argument("--coverage", help="Path to coverage XML file", default=None)
    parser.add_argument("--changed-files", help="Path to file containing changed files list", default=None)
    parser.add_argument("--logs", help="Glob pattern to include log snippets", default=None)
    parser.add_argument("--out", required=True, help="Output markdown summary path")
    parser.add_argument("--json", required=True, dest="json_out", help="Output JSON summary path")
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args()

    junit_path = _discover_file(
        args.junit,
        patterns=[
            "junit.xml",
            "test-results.xml",
            "artifacts/junit*.xml",
            "reports/junit*.xml",
            "**/junit*.xml",
        ],
    )
    coverage_path = _discover_file(args.coverage, patterns=["coverage.xml", "reports/coverage.xml", "**/coverage.xml"])

    summary = generate_summary(
        junit_path=junit_path,
        coverage_path=coverage_path,
        changed_files_path=args.changed_files,
        log_glob=args.logs,
        max_lines=args.max_lines,
        max_bytes=args.max_bytes,
    )
    redacted_summary = _redact_structure(summary)

    markdown = build_markdown(redacted_summary)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(redacted_summary, handle, indent=2)


if __name__ == "__main__":
    main()
