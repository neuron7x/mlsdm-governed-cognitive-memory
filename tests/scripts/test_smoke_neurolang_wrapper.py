"""
Tests for scripts/smoke_neurolang_wrapper.py

Requires PyTorch (torch). Tests are skipped if torch is not installed.
"""

import sys
from pathlib import Path

import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch")

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import smoke_neurolang_wrapper


def test_smoke_neurolang_main_runs(capsys):
    """Test that smoke_neurolang_wrapper.main() runs successfully."""
    exit_code = smoke_neurolang_wrapper.main(["--prompt", "Test prompt."])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Response:" in out
    assert "Phase:" in out
    assert "Aphasia flags:" in out


def test_smoke_neurolang_default_prompt(capsys):
    """Test that smoke_neurolang_wrapper.main() works with default prompt."""
    exit_code = smoke_neurolang_wrapper.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Response:" in out
    assert "Accepted:" in out
