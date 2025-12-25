"""Subprocess smoke test for the canonical runtime entrypoint."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from typing import Any

import pytest
import requests

from mlsdm.config.constants import DEFAULT_CONFIG_PATH


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1.0)
            if response.status_code == 200:
                return response.json()
        except Exception as exc:  # pragma: no cover - diagnostics only
            last_error = exc
        time.sleep(0.25)
    raise AssertionError(f"Health endpoint did not become ready: {last_error}")


@pytest.mark.e2e
def test_cloud_entrypoint_smoke(tmp_path) -> None:
    port = _get_free_port()
    env = os.environ.copy()
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "CONFIG_PATH": DEFAULT_CONFIG_PATH,
            "LLM_BACKEND": "local_stub",
            "MLSDM_RUNTIME_MODE": "cloud-prod",
            "DISABLE_RATE_LIMIT": "1",
            "OTEL_SDK_DISABLED": "true",
        }
    )
    assert os.path.exists(DEFAULT_CONFIG_PATH), f"Config not found: {DEFAULT_CONFIG_PATH}"

    stdout_path = tmp_path.joinpath("stdout.log")
    stderr_path = tmp_path.joinpath("stderr.log")

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.Popen(
            [sys.executable, "-m", "mlsdm.entrypoints.cloud"],
            env=env,
            stdout=stdout,
            stderr=stderr,
        )

        try:
            health = _wait_for_health(f"http://127.0.0.1:{port}/health/ready")
            assert health["status"] in {"ready", "healthy"}
        except AssertionError as err:
            stdout.flush()
            stderr.flush()
            stdout_tail = stdout_path.read_text(encoding="utf-8").splitlines()[-20:]
            stderr_tail = stderr_path.read_text(encoding="utf-8").splitlines()[-20:]
            raise AssertionError(
                f"{err}\nSTDOUT (tail):\n" + "\n".join(stdout_tail) + "\nSTDERR (tail):\n" + "\n".join(stderr_tail)
            ) from err
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
