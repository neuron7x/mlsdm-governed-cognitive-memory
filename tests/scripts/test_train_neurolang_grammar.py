"""
Tests for scripts/train_neurolang_grammar.py
"""

import sys
from pathlib import Path

import torch

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import train_neurolang_grammar


def test_train_neurolang_creates_checkpoint(tmp_path, monkeypatch):
    """Test that train_neurolang_grammar.main() creates a valid checkpoint."""
    output = tmp_path / "neurolang_grammar.pt"

    def dummy_train(self):
        """Mock training to avoid slow training in tests."""
        return None

    monkeypatch.setattr(
        train_neurolang_grammar.CriticalPeriodTrainer,
        "train",
        dummy_train,
    )

    exit_code = train_neurolang_grammar.main(
        ["--epochs", "1", "--output", str(output)],
    )
    assert exit_code == 0
    assert output.is_file()
    state = torch.load(output, map_location="cpu")
    assert isinstance(state, dict)
    assert "actor" in state
    assert "critic" in state


def test_train_neurolang_secure_mode(monkeypatch):
    """Test that train_neurolang_grammar.main() fails in secure mode."""
    monkeypatch.setenv("MLSDM_SECURE_MODE", "1")
    
    try:
        train_neurolang_grammar.main(["--epochs", "1"])
        assert False, "Expected SystemExit"
    except SystemExit as e:
        assert "Secure mode enabled" in str(e)
