#!/usr/bin/env python3
"""Generate a minimal CI summary for evidence packs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def generate_summary(output_path: Path) -> None:
    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "git_ref": os.getenv("GITHUB_REF", "unknown"),
        "run_id": os.getenv("GITHUB_RUN_ID", "unknown"),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", "unknown"),
        "actor": os.getenv("GITHUB_ACTOR", "unknown"),
        "workflow": os.getenv("GITHUB_WORKFLOW", "local"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "artifacts" / "tmp" / "ci-summary.json"
    generate_summary(output_path)
    print(f"✓ CI summary written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
