import datetime
from pathlib import Path

import pytest

from scripts import readiness_check as rc


def _write_readiness(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "READINESS.md"
    path.write_text(content, encoding="utf-8")
    return path


def test_parse_last_updated_ok(tmp_path, monkeypatch):
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    readiness_path = _write_readiness(
        tmp_path,
        f"# System Readiness Status\nLast updated: {today}\n",
    )
    monkeypatch.setattr(rc, "READINESS_PATH", readiness_path)
    parsed = rc.parse_last_updated()
    assert parsed is not None
    assert parsed.strftime("%Y-%m-%d") == today


def test_parse_last_updated_missing_returns_none(tmp_path, monkeypatch):
    readiness_path = _write_readiness(tmp_path, "# Missing last updated\n")
    monkeypatch.setattr(rc, "READINESS_PATH", readiness_path)
    assert rc.parse_last_updated() is None


def test_stale_readiness_fails(tmp_path, monkeypatch):
    stale_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=rc.MAX_AGE_DAYS + 1)).strftime(
        "%Y-%m-%d"
    )
    readiness_path = _write_readiness(
        tmp_path,
        f"# System Readiness Status\nLast updated: {stale_date}\n",
    )
    monkeypatch.setattr(rc, "READINESS_PATH", readiness_path)
    monkeypatch.setattr(
        rc,
        "collect_changed_files",
        lambda: rc.DiffOutcome([], "test-base", True, None),
    )
    assert rc.main() == 1


def test_scope_changes_without_readiness_update_fail(tmp_path, monkeypatch):
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    readiness_path = _write_readiness(
        tmp_path,
        f"# System Readiness Status\nLast updated: {today}\n",
    )
    monkeypatch.setattr(rc, "READINESS_PATH", readiness_path)
    monkeypatch.setattr(
        rc,
        "collect_changed_files",
        lambda: rc.DiffOutcome(["src/new_module.py"], "test-base", True, None),
    )
    assert rc.main() == 1
