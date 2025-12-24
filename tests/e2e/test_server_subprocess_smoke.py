import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            time.sleep(0.25)
            continue
        time.sleep(0.25)
    return False


def test_server_subprocess_health_smoke():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "default_config.yaml"
    port = _find_free_port()

    env = os.environ.copy()
    env.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "CONFIG_PATH": str(config_path),
            "LLM_BACKEND": "local_stub",
            "DISABLE_RATE_LIMIT": "1",
            "OTEL_SDK_DISABLED": "1",
        }
    )

    proc = subprocess.Popen(
        [sys.executable, "-m", "mlsdm.entrypoints.cloud"],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        healthy = _wait_for_health(f"http://127.0.0.1:{port}/health", timeout=35.0)
        assert healthy, "Server did not become healthy in time"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
