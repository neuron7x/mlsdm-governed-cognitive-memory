"""CLI entrypoint tests for mlsdm.cli.main."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlsdm.cli as cli_pkg


def _load_cli_main_module():
    original_main = cli_pkg.main
    module = importlib.import_module("mlsdm.cli.main")
    cli_pkg.main = original_main  # Restore package attribute to function form for other tests
    return module


def _stub_app_module() -> dict[str, object]:
    return {"mlsdm.api.app": SimpleNamespace(app="fake-app")}


def test_main_api_invokes_uvicorn_run(monkeypatch) -> None:
    """Ensure --api uses uvicorn.run without requiring real uvicorn."""
    cli_main = _load_cli_main_module()
    mock_run = MagicMock()
    monkeypatch.setattr(cli_main, "uvicorn", MagicMock(run=mock_run))

    with patch.dict("sys.modules", _stub_app_module()):
        with patch("sys.argv", ["mlsdm", "--api"]):
            exit_code = cli_main.main()

    assert exit_code == 0
    assert mock_run.called
    args, kwargs = mock_run.call_args
    assert args[0] == "fake-app"
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000


def test_main_api_handles_missing_uvicorn(monkeypatch) -> None:
    """Gracefully handle missing uvicorn when --api is requested."""
    cli_main = _load_cli_main_module()
    monkeypatch.setattr(cli_main, "uvicorn", None)

    with patch.dict("sys.modules", _stub_app_module()):
        with patch("sys.argv", ["mlsdm", "--api"]):
            exit_code = cli_main.main()

    assert exit_code == 1
