import builtins
import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_app_import_has_no_file_io(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "mlsdm.api.app"
    repo_root = Path(__file__).resolve().parents[2]
    app_path = repo_root / "src" / "mlsdm" / "api" / "app.py"
    source = app_path.read_text(encoding="utf-8")

    for dependency in (
        "numpy",
        "psutil",
        "fastapi",
        "fastapi.responses",
        "fastapi.security",
        "pydantic",
        "mlsdm.api.health",
        "mlsdm.api.lifecycle",
        "mlsdm.api.middleware",
        "mlsdm.contracts",
        "mlsdm.core.memory_manager",
        "mlsdm.engine",
        "mlsdm.observability.tracing",
        "mlsdm.utils.config_loader",
        "mlsdm.utils.input_validator",
        "mlsdm.utils.rate_limiter",
        "mlsdm.utils.security_logger",
    ):
        importlib.import_module(dependency)

    sys.modules.pop(module_name, None)

    def _fail_open(*_: object, **__: object) -> object:
        raise AssertionError("file I/O during import")

    monkeypatch.setattr(builtins, "open", _fail_open)
    monkeypatch.setattr(Path, "open", _fail_open)
    monkeypatch.setattr(Path, "read_text", _fail_open)

    module = types.ModuleType(module_name)
    module.__file__ = str(app_path)
    module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
    sys.modules[module_name] = module

    exec(compile(source, str(app_path), "exec"), module.__dict__)
    imported = importlib.import_module(module_name)
    assert imported is module


def test_import_in_installed_mode_without_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("CONFIG_PATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; os.environ.pop('CONFIG_PATH', None); import mlsdm.api.app; print('ok')",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ok" in result.stdout


def test_lifespan_initializes_runtime_state() -> None:
    from mlsdm.api.app import create_app

    app = create_app()
    with TestClient(app) as client:
        assert getattr(app.state, "memory_manager", None) is not None
        assert getattr(app.state, "neuro_engine", None) is not None

        response = client.get("/health")
        assert response.status_code == 200
