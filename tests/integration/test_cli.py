"""
Integration tests for the MLSDM CLI.

Tests the CLI commands: demo, serve, check
"""

import subprocess
import sys
from unittest.mock import patch

import pytest


class TestCLICheck:
    """Test the 'mlsdm check' command."""

    def test_check_command_runs(self):
        """Test that check command runs without error."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "MLSDM Environment Check" in result.stdout

    def test_check_shows_version(self):
        """Test that check shows mlsdm version."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "mlsdm v" in result.stdout

    def test_check_validates_python_version(self):
        """Test that check validates Python version."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "Python version" in result.stdout

    def test_check_verbose_flag(self):
        """Test verbose flag outputs more info."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "check", "--verbose"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Verbose mode should show full status JSON
        assert "checks" in result.stdout


class TestCLIDemo:
    """Test the 'mlsdm demo' command."""

    def test_demo_with_prompt(self):
        """Test demo with single prompt."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "demo", "-p", "Hello world"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "MLSDM Demo" in result.stdout
        assert "Prompt: Hello world" in result.stdout

    def test_demo_without_prompt_runs_demo(self):
        """Test demo without prompt runs demo prompts."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "demo"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Running demo prompts" in result.stdout

    def test_demo_verbose_output(self):
        """Test demo verbose mode."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "demo", "-p", "Test", "--verbose"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Full result" in result.stdout

    def test_demo_custom_moral_value(self):
        """Test demo with custom moral value."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "demo", "-p", "Test", "-m", "0.9"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Moral Value: 0.9" in result.stdout

    def test_demo_low_moral_rejected(self):
        """Test demo with low moral value gets rejected."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlsdm.cli",
                "demo",
                "-p",
                "Test",
                "-m",
                "0.1",  # Very low moral value
                "--moral-threshold",
                "0.9",  # High threshold
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Rejected" in result.stdout


class TestCLIVersion:
    """Test version flag."""

    def test_version_flag(self):
        """Test --version flag."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "1.2.0" in result.stdout


class TestCLIHelp:
    """Test help output."""

    def test_help_flag(self):
        """Test --help flag."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "demo" in result.stdout
        assert "serve" in result.stdout
        assert "check" in result.stdout

    def test_demo_help(self):
        """Test demo --help."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "demo", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--prompt" in result.stdout
        assert "--interactive" in result.stdout

    def test_serve_help(self):
        """Test serve --help."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_check_help(self):
        """Test check --help."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "check", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--verbose" in result.stdout


class TestCLIModule:
    """Test CLI module can be imported and run directly."""

    def test_import_cli_module(self):
        """Test that CLI module can be imported."""
        from mlsdm import cli

        assert hasattr(cli, "main")
        assert callable(cli.main)

    def test_main_with_no_args(self):
        """Test main() with no arguments shows help."""
        from mlsdm.cli import main

        with patch("sys.argv", ["mlsdm"]):
            # Should print help and return 0
            result = main()
            assert result == 0


class TestCLIServe:
    """Test 'mlsdm serve' command (without actually starting server)."""

    def test_serve_help(self):
        """Test serve shows help with correct options."""
        result = subprocess.run(
            [sys.executable, "-m", "mlsdm.cli", "serve", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--backend" in result.stdout
        assert "--config" in result.stdout


class TestExampleWrapperSubprocessDelegation:
    """Test the example wrapper subprocess delegation security posture."""

    def test_example_wrapper_uses_constant_argv(self) -> None:
        """Verify the example wrapper uses constant argv without env-derived CLI args."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        # Load the example module without executing __main__
        spec = importlib.util.spec_from_file_location(
            "run_neuro_service",
            "examples/run_neuro_service.py"
        )
        assert spec is not None
        assert spec.loader is not None

        # Mock subprocess.run to capture the call
        mock_run = MagicMock()
        mock_run.return_value.returncode = 0

        with patch("subprocess.run", mock_run):
            example_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(example_module)
            example_module.main()

        # Verify subprocess.run was called with constant argv
        assert mock_run.called
        call_args = mock_run.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1].get("cmd")

        # The command should be constant: [sys.executable, "-m", "mlsdm.cli", "serve"]
        assert cmd[1:] == ["-m", "mlsdm.cli", "serve"], \
            f"Expected constant argv, got: {cmd}"

    def test_example_wrapper_passes_config_via_env(self) -> None:
        """Verify the example wrapper passes host/port/config via environment."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        spec = importlib.util.spec_from_file_location(
            "run_neuro_service",
            "examples/run_neuro_service.py"
        )
        assert spec is not None
        assert spec.loader is not None
        example_module = importlib.util.module_from_spec(spec)

        mock_run = MagicMock()
        mock_run.return_value.returncode = 0

        with patch("subprocess.run", mock_run):
            spec.loader.exec_module(example_module)
            example_module.main()

        # Verify env contains HOST, PORT, CONFIG_PATH
        call_kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
        env = call_kwargs.get("env", {})

        assert "HOST" in env, "HOST should be in subprocess env"
        assert "PORT" in env, "PORT should be in subprocess env"
        assert "CONFIG_PATH" in env, "CONFIG_PATH should be in subprocess env"

    def test_example_wrapper_uses_check_false(self) -> None:
        """Verify the example wrapper uses check=False explicitly."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        spec = importlib.util.spec_from_file_location(
            "run_neuro_service",
            "examples/run_neuro_service.py"
        )
        assert spec is not None
        assert spec.loader is not None
        example_module = importlib.util.module_from_spec(spec)

        mock_run = MagicMock()
        mock_run.return_value.returncode = 0

        with patch("subprocess.run", mock_run):
            spec.loader.exec_module(example_module)
            example_module.main()

        call_kwargs = mock_run.call_args[1] if mock_run.call_args[1] else {}
        assert call_kwargs.get("check") is False, "subprocess.run should use check=False"

    def test_example_wrapper_propagates_returncode(self) -> None:
        """Verify the example wrapper propagates subprocess return code."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        spec = importlib.util.spec_from_file_location(
            "run_neuro_service",
            "examples/run_neuro_service.py"
        )
        assert spec is not None
        assert spec.loader is not None
        example_module = importlib.util.module_from_spec(spec)

        mock_run = MagicMock()
        mock_run.return_value.returncode = 42  # Non-zero exit code

        with patch("subprocess.run", mock_run):
            spec.loader.exec_module(example_module)
            result = example_module.main()

        assert result == 42, "Should propagate subprocess return code"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
