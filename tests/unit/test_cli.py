"""
Unit tests for CLI module.

Tests the CLI commands by directly calling the command functions
rather than through subprocess, for proper coverage tracking.
"""

import argparse
import os
from unittest.mock import patch

import pytest


class TestCmdCheck:
    """Test the cmd_check function."""

    def test_check_basic(self, capsys):
        """Test basic check command."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        result = cmd_check(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "MLSDM Environment Check" in captured.out
        assert "Python version" in captured.out

    def test_check_verbose(self, capsys):
        """Test check command with verbose flag."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=True)
        result = cmd_check(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Full status:" in captured.out

    def test_check_mlsdm_version(self, capsys):
        """Test that check shows mlsdm version."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "mlsdm v" in captured.out


class TestCmdDemo:
    """Test the cmd_demo function."""

    def test_demo_with_prompt(self, capsys):
        """Test demo with a single prompt."""
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt="Hello world",
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "MLSDM Demo" in captured.out
        assert "Prompt: Hello world" in captured.out

    def test_demo_with_verbose(self, capsys):
        """Test demo with verbose flag."""
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt="Test",
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=True,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Full result:" in captured.out

    def test_demo_without_prompt_runs_demo_prompts(self, capsys):
        """Test demo without prompt runs demo prompts."""
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt=None,
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Running demo prompts" in captured.out

    def test_demo_with_low_moral_value_rejected(self, capsys):
        """Test demo with low moral value is rejected."""
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt="Test",
            moral_value=0.1,  # Very low
            moral_threshold=0.9,  # High threshold
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Rejected" in captured.out

    def test_demo_interactive_exit_on_quit(self, capsys, monkeypatch):
        """Test interactive demo exits on 'quit'."""
        from mlsdm.cli import cmd_demo

        # Simulate user entering 'quit'
        inputs = iter(["quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        args = argparse.Namespace(
            interactive=True,
            prompt=None,
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_demo_interactive_exit_on_exit(self, capsys, monkeypatch):
        """Test interactive demo exits on 'exit'."""
        from mlsdm.cli import cmd_demo

        inputs = iter(["exit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        args = argparse.Namespace(
            interactive=True,
            prompt=None,
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0

    def test_demo_interactive_empty_input_skipped(self, capsys, monkeypatch):
        """Test interactive demo skips empty input."""
        from mlsdm.cli import cmd_demo

        inputs = iter(["", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        args = argparse.Namespace(
            interactive=True,
            prompt=None,
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0

    def test_demo_interactive_state_command(self, capsys, monkeypatch):
        """Test interactive demo 'state' command."""
        from mlsdm.cli import cmd_demo

        inputs = iter(["state", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        args = argparse.Namespace(
            interactive=True,
            prompt=None,
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "System State:" in captured.out

    def test_demo_interactive_handles_prompt(self, capsys, monkeypatch):
        """Test interactive demo handles prompts correctly."""
        from mlsdm.cli import cmd_demo

        inputs = iter(["Hello world", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        args = argparse.Namespace(
            interactive=True,
            prompt=None,
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "MLSDM" in captured.out

    def test_demo_interactive_eof_handling(self, capsys, monkeypatch):
        """Test interactive demo handles EOF correctly."""
        from mlsdm.cli import cmd_demo

        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)

        args = argparse.Namespace(
            interactive=True,
            prompt=None,
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        result = cmd_demo(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Exiting..." in captured.out


class TestMainFunction:
    """Test the main function."""

    def test_main_with_no_args_shows_help(self, capsys):
        """Test main with no arguments shows help."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm"]):
            result = main()

        assert result == 0

    def test_main_demo_command(self, capsys):
        """Test main with demo command."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm", "demo", "-p", "Hello"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "MLSDM Demo" in captured.out

    def test_main_check_command(self, capsys):
        """Test main with check command."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm", "check"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "MLSDM Environment Check" in captured.out


class TestCmdServe:
    """Test the cmd_serve function."""

    def test_serve_sets_env_vars(self, capsys):
        """Test that serve sets environment variables correctly."""
        from mlsdm.cli import cmd_serve

        with patch("uvicorn.run") as mock_uvicorn_run:
            args = argparse.Namespace(
                host="127.0.0.1",
                port=9000,
                config="config/default_config.yaml",  # Use valid config path
                backend="local_stub",
                disable_rate_limit=True,
                log_level="debug",
                reload=False,
            )

            # Save original env vars
            original_env = {
                key: os.environ.get(key)
                for key in ["CONFIG_PATH", "LLM_BACKEND", "DISABLE_RATE_LIMIT"]
            }

            try:
                cmd_serve(args)

                # Check environment variables were set
                assert os.environ.get("CONFIG_PATH") == "config/default_config.yaml"
                assert os.environ.get("LLM_BACKEND") == "local_stub"
                assert os.environ.get("DISABLE_RATE_LIMIT") == "1"

                # Check uvicorn.run was called
                mock_uvicorn_run.assert_called_once()
            finally:
                # Restore original env vars
                for key, val in original_env.items():
                    if val is None and key in os.environ:
                        del os.environ[key]
                    elif val is not None:
                        os.environ[key] = val

    def test_serve_without_optional_args(self, capsys):
        """Test serve without optional arguments."""
        from mlsdm.cli import cmd_serve

        with patch("uvicorn.run"):
            args = argparse.Namespace(
                host="0.0.0.0",
                port=8000,
                config=None,
                backend=None,
                disable_rate_limit=False,
                log_level="info",
                reload=False,
            )

            result = cmd_serve(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "MLSDM HTTP API Server" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
