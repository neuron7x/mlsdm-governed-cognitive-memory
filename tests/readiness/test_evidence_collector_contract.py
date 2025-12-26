from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path  # noqa: TC003

import scripts.readiness.evidence_collector as ec


def _write_junit(path: Path) -> None:
    content = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="unit" tests="2" failures="1" errors="0" skipped="0" time="0.12">
  <testcase classname="ExampleTest" name="test_ok" time="0.05"/>
  <testcase classname="ExampleTest" name="test_fail" time="0.07">
    <failure message="boom">boom</failure>
  </testcase>
</testsuite>
"""
    path.write_text(content, encoding="utf-8")


def _write_coverage(path: Path) -> None:
    path.write_text('<?xml version="1.0"?><coverage line-rate="0.8" branch-rate="0.5"></coverage>', encoding="utf-8")


def _write_bandit(path: Path) -> None:
    payload = {
        "results": [
            {"issue_severity": "LOW"},
            {"issue_severity": "MEDIUM"},
            {"issue_severity": "HIGH"},
        ]
    }
    path.write_text(ec.json.dumps(payload), encoding="utf-8")


def test_collect_evidence_contract_and_hash_stability(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    _write_junit(reports / "junit-unit.xml")
    _write_coverage(tmp_path / "coverage.xml")
    _write_bandit(reports / "bandit.json")

    fixed_now = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(ec, "_now", lambda: fixed_now)

    first = ec.collect_evidence(tmp_path)
    second = ec.collect_evidence(tmp_path)

    assert list(first) == [
        "timestamp_utc",
        "sources",
        "tests",
        "coverage",
        "security",
        "performance",
        "evidence_hash",
    ]
    assert first["timestamp_utc"] == fixed_now.isoformat()
    assert first["sources"]["junit"]["found"] is True
    assert "junit-unit.xml" in first["sources"]["junit"]["files"][0]
    assert first["coverage"]["measured"] is True
    assert first["coverage"]["line_rate"] == 0.8
    assert first["coverage"]["branch_rate"] == 0.5
    assert first["tests"]["totals"]["passed"] == 1
    assert first["tests"]["totals"]["failed"] == 1
    assert first["security"]["measured"] is True
    assert first["security"]["tools"][0]["high"] == 1
    assert first["security"]["tools"][0]["medium"] == 1
    assert first["security"]["tools"][0]["low"] == 1
    assert first["performance"]["measured"] is False

    assert first["evidence_hash"].startswith("sha256-")
    assert first["evidence_hash"] == second["evidence_hash"]


def test_collect_evidence_handles_invalid_files(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir(parents=True)
    (reports / "junit-unit.xml").write_text("not xml", encoding="utf-8")
    (tmp_path / "coverage.xml").write_text("broken", encoding="utf-8")
    (reports / "bandit.json").write_text("{", encoding="utf-8")

    fixed_now = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(ec, "_now", lambda: fixed_now)

    evidence = ec.collect_evidence(tmp_path)
    assert evidence["tests"]["suites"] == []
    assert evidence["coverage"]["measured"] is False
    assert evidence["security"]["tools"][0]["measured"] is False
    assert evidence["performance"]["measured"] is False
