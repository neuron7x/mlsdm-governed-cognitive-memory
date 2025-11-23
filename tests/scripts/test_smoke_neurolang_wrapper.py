"""
Tests for scripts/smoke_neurolang_wrapper.py
"""

import sys
from pathlib import Path

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
