"""Tests for the safe Neuro service launcher example."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "run_neuro_service.py"
    spec = importlib.util.spec_from_file_location("run_neuro_service", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def run_neuro_service():
    return _load_module()


def test_dry_run_outputs_expected_command(
    run_neuro_service, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run should print the exact command without executing it."""
    monkeypatch.setenv("MLSDM_HOST", "127.0.0.1")
    monkeypatch.setenv("MLSDM_PORT", "9001")

    fake_subprocess = MagicMock()
    fake_subprocess.run.side_effect = AssertionError("subprocess.run should not be called during dry-run")
    monkeypatch.setattr(run_neuro_service, "subprocess", fake_subprocess)

    exit_code = run_neuro_service.main(["--dry-run"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "mlsdm.cli" in captured
    assert "--host 127.0.0.1" in captured
    assert "--port 9001" in captured
    fake_subprocess.run.assert_not_called()


def test_invalid_port_env_fails_cleanly(
    run_neuro_service, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invalid env port should produce a clear error and non-zero exit."""
    monkeypatch.setenv("PORT", "not-a-number")

    exit_code = run_neuro_service.main([])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Invalid port" in captured.err
    assert "1 and 65535" in captured.err
