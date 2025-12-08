"""
Tests for scripts/smoke_neurolang_wrapper.py
"""

import sys
from pathlib import Path

import pytest

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import smoke_neurolang_wrapper

# Check if torch is available
try:
    import torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip entire module if torch is not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch not installed - install with 'pip install mlsdm[neurolang]'"
)


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
