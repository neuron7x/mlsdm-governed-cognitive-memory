import os
from importlib import resources
from pathlib import Path

import pytest

from mlsdm.utils.config_loader import ConfigLoader


def test_packaged_default_config_matches_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_config = repo_root / "config" / "default_config.yaml"
    if not repo_config.exists():
        pytest.skip("Repository default_config.yaml not present")

    packaged_text = (
        resources.files("mlsdm.config").joinpath("default_config.yaml").read_text(encoding="utf-8")
    )
    repo_text = repo_config.read_text(encoding="utf-8")
    assert packaged_text == repo_text


def test_config_loader_falls_back_to_packaged_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("MLSDM_"):
            monkeypatch.delenv(key, raising=False)

    config = ConfigLoader.load_config("config/default_config.yaml")
    assert isinstance(config, dict)
    assert "dimension" in config
    assert "multi_level_memory" in config
