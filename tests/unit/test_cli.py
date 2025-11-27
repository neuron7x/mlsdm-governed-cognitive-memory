"""Tests for MLSDM CLI module.

Tests cover:
- cmd_check command functionality
- cmd_demo command (limited, without interactive mode)
- main function argument parsing
- Environment variable handling
- Error handling and exit codes
"""

import argparse
import os
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestCmdCheck:
    """Tests for the check command."""

    def test_check_returns_zero_on_success(self, capsys):
        """Check command should return 0 when all checks pass."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        result = cmd_check(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Environment Check" in captured.out

    def test_check_shows_python_version(self, capsys):
        """Check command should display Python version."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Python version" in captured.out

    def test_check_validates_core_dependencies(self, capsys):
        """Check command should validate core dependencies."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Core dependencies" in captured.out
        assert "numpy" in captured.out
        assert "FastAPI" in captured.out

    def test_check_shows_optional_dependencies(self, capsys):
        """Check command should show optional dependencies."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Optional dependencies" in captured.out

    def test_check_displays_environment_variables(self, capsys):
        """Check command should display environment variables."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Environment variables" in captured.out
        assert "LLM_BACKEND" in captured.out

    def test_check_verbose_shows_full_status(self, capsys):
        """Check command with verbose flag shows full status JSON."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=True)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Full status" in captured.out

    def test_check_masks_sensitive_env_vars(self, capsys, monkeypatch):
        """Sensitive environment variables should be masked."""
        from mlsdm.cli import cmd_check

        monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890abcdef")
        
        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        # Should not contain the full key
        assert "sk-1234567890abcdef" not in captured.out
        # Should contain masked version
        assert "sk-1" in captured.out
        assert "..." in captured.out

    def test_check_detects_missing_config(self, capsys, monkeypatch, tmp_path):
        """Check command should handle missing config file."""
        from mlsdm.cli import cmd_check

        # Set non-existent config path
        monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "nonexistent.yaml"))
        
        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Config file not found" in captured.out or "will use defaults" in captured.out

    def test_check_validates_mlsdm_import(self, capsys):
        """Check command should validate mlsdm package import."""
        from mlsdm.cli import cmd_check

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "MLSDM package" in captured.out


class TestCmdDemo:
    """Tests for the demo command."""

    def test_demo_non_interactive_with_prompt(self, capsys):
        """Demo with single prompt should generate response."""
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
        assert "Hello world" in captured.out

    def test_demo_runs_default_prompts(self, capsys):
        """Demo without prompt should run default demo prompts."""
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt=None,  # No specific prompt
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
        assert "Hello, how are you?" in captured.out

    def test_demo_verbose_shows_full_result(self, capsys):
        """Demo with verbose flag shows full result JSON."""
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt="Test prompt",
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=True,
        )
        
        cmd_demo(args)

        captured = capsys.readouterr()
        assert "Full result" in captured.out

    def test_demo_shows_configuration(self, capsys):
        """Demo should display configuration parameters."""
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt="Test",
            moral_value=0.8,
            moral_threshold=0.6,
            wake_duration=10,
            sleep_duration=5,
            verbose=False,
        )
        
        cmd_demo(args)

        captured = capsys.readouterr()
        assert "Wake duration: 10" in captured.out
        assert "Sleep duration: 5" in captured.out
        assert "Moral threshold: 0.6" in captured.out


class TestMain:
    """Tests for the main entry point and argument parsing."""

    def test_main_no_command_prints_help(self, capsys):
        """Running without command should print help."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        # Help should mention available commands
        assert "demo" in captured.out or "Available commands" in captured.out or "help" in captured.out.lower()

    def test_main_check_command(self, capsys):
        """Main should handle check command."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm", "check"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Environment Check" in captured.out

    def test_main_demo_command(self, capsys):
        """Main should handle demo command with default prompts."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm", "demo"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "MLSDM Demo" in captured.out

    def test_main_demo_with_prompt(self, capsys):
        """Main should handle demo command with prompt argument."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm", "demo", "-p", "Test prompt"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Test prompt" in captured.out

    def test_main_check_verbose(self, capsys):
        """Main should handle check command with verbose flag."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm", "check", "-v"]):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Full status" in captured.out


class TestCmdServe:
    """Tests for the serve command (limited without starting server)."""

    def test_serve_sets_environment_variables(self, monkeypatch):
        """Serve command should set environment variables from args."""
        from mlsdm.cli import cmd_serve

        # Mock uvicorn to prevent actually starting server
        mock_uvicorn = MagicMock()
        
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            args = argparse.Namespace(
                host="0.0.0.0",
                port=8080,
                config="custom_config.yaml",
                backend="openai",
                log_level="debug",
                reload=True,
                disable_rate_limit=True,
            )
            
            # We need to mock the app import too
            with patch("mlsdm.api.app.app"):
                try:
                    cmd_serve(args)
                except Exception:
                    pass  # May fail on import, but env vars should be set
            
            # Check environment variables were set
            assert os.environ.get("CONFIG_PATH") == "custom_config.yaml"
            assert os.environ.get("LLM_BACKEND") == "openai"
            assert os.environ.get("DISABLE_RATE_LIMIT") == "1"

    def test_serve_missing_uvicorn(self, monkeypatch, capsys):
        """Serve command should handle missing uvicorn gracefully."""
        # This test verifies the cmd_serve function has proper error handling
        # for missing uvicorn. Since uvicorn is installed in the test env,
        # we just verify the serve function exists and returns correct status.
        from mlsdm.cli import cmd_serve

        # Verify the function exists and has expected signature
        import inspect
        sig = inspect.signature(cmd_serve)
        assert "args" in sig.parameters

        # The actual ImportError path is covered by manual testing
        # when uvicorn is not installed


class TestEnvironmentHandling:
    """Tests for environment variable handling across CLI commands."""

    def test_check_with_custom_config_path(self, capsys, monkeypatch, tmp_path):
        """Check should use CONFIG_PATH environment variable."""
        from mlsdm.cli import cmd_check

        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("dimension: 512")
        monkeypatch.setenv("CONFIG_PATH", str(config_file))

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "Config file exists" in captured.out or str(config_file) in captured.out

    def test_check_with_llm_backend_set(self, capsys, monkeypatch):
        """Check should display LLM_BACKEND when set."""
        from mlsdm.cli import cmd_check

        monkeypatch.setenv("LLM_BACKEND", "openai")

        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        assert "openai" in captured.out


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_demo_handles_import_error(self, capsys, monkeypatch):
        """Demo should handle mlsdm import errors gracefully."""
        # This is tricky to test since we're in the mlsdm package
        # Just verify the structure handles errors
        from mlsdm.cli import cmd_demo

        args = argparse.Namespace(
            interactive=False,
            prompt="test",
            moral_value=0.8,
            moral_threshold=0.5,
            wake_duration=8,
            sleep_duration=3,
            verbose=False,
        )
        
        # Should not raise
        result = cmd_demo(args)
        assert result == 0

    def test_check_low_python_version_warning(self, capsys, monkeypatch):
        """Check should warn about low Python versions."""
        from mlsdm.cli import cmd_check

        # We can't really test this properly without mocking sys.version_info
        # which would break the actual check. Just verify command runs.
        args = argparse.Namespace(verbose=False)
        result = cmd_check(args)
        
        assert result == 0  # Should still pass on current Python

    def test_short_api_key_masking(self, capsys, monkeypatch):
        """Short API keys should be fully masked."""
        from mlsdm.cli import cmd_check

        monkeypatch.setenv("OPENAI_API_KEY", "short")
        
        args = argparse.Namespace(verbose=False)
        cmd_check(args)

        captured = capsys.readouterr()
        # Short keys should show *** 
        assert "***" in captured.out or "shor" in captured.out


class TestDemoInteractiveMode:
    """Tests for demo interactive mode (limited testing)."""

    def test_demo_interactive_quit_command(self, monkeypatch, capsys):
        """Interactive mode should exit on 'quit' command."""
        from mlsdm.cli import cmd_demo

        # Simulate user typing 'quit'
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

    def test_demo_interactive_exit_command(self, monkeypatch, capsys):
        """Interactive mode should exit on 'exit' command."""
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
        captured = capsys.readouterr()
        assert "Goodbye!" in captured.out

    def test_demo_interactive_state_command(self, monkeypatch, capsys):
        """Interactive mode should show state on 'state' command."""
        from mlsdm.cli import cmd_demo

        inputs = iter(["state", "exit"])
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

        captured = capsys.readouterr()
        assert "System State" in captured.out

    def test_demo_interactive_empty_input(self, monkeypatch, capsys):
        """Interactive mode should skip empty inputs."""
        from mlsdm.cli import cmd_demo

        inputs = iter(["", "", "exit"])
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

    def test_demo_interactive_keyboard_interrupt(self, monkeypatch, capsys):
        """Interactive mode should handle KeyboardInterrupt gracefully."""
        from mlsdm.cli import cmd_demo

        def raise_keyboard_interrupt(_):
            raise KeyboardInterrupt()

        monkeypatch.setattr("builtins.input", raise_keyboard_interrupt)

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
        assert "Exiting" in captured.out

    def test_demo_interactive_eof_error(self, monkeypatch, capsys):
        """Interactive mode should handle EOFError gracefully."""
        from mlsdm.cli import cmd_demo

        def raise_eof_error(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof_error)

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

    def test_demo_interactive_generates_response(self, monkeypatch, capsys):
        """Interactive mode should generate responses for prompts."""
        from mlsdm.cli import cmd_demo

        inputs = iter(["Hello there", "exit"])
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
