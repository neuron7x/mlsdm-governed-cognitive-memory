"""
Tests for scripts/run_aphasia_eval.py
"""

import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

import run_aphasia_eval


def test_run_aphasia_eval_default_corpus(tmp_path, monkeypatch, capsys):
    """Test that run_aphasia_eval.main() runs successfully with a corpus."""
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        """
        {
          "telegraphic": ["This short. No connect. Bad."],
          "normal": ["This is a coherent answer with normal grammar."]
        }
        """,
        encoding="utf-8",
    )

    exit_code = run_aphasia_eval.main(["--corpus", str(corpus)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "AphasiaEvalSuite metrics:" in out
    assert "true_positive_rate:" in out
    assert "true_negative_rate:" in out


def test_run_aphasia_eval_missing_corpus(capsys):
    """Test that run_aphasia_eval.main() fails gracefully with missing corpus."""
    exit_code = run_aphasia_eval.main(["--corpus", "/nonexistent/corpus.json"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "Error:" in out or "not found" in out


def test_run_aphasia_eval_fail_on_low_metrics(tmp_path, monkeypatch, capsys):
    """Test that run_aphasia_eval.main() can fail on low metrics."""
    corpus = tmp_path / "corpus.json"
    # Create a corpus that will likely have low metrics
    corpus.write_text(
        """
        {
          "telegraphic": ["This is actually a normal sentence with proper grammar."],
          "normal": ["Short. Bad. Fragment."]
        }
        """,
        encoding="utf-8",
    )

    exit_code = run_aphasia_eval.main(["--corpus", str(corpus), "--fail-on-low-metrics"])
    # This should fail because the corpus has inverted samples
    assert exit_code == 1
