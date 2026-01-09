from __future__ import annotations

import hashlib
import json
import mimetypes
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

import scripts.evidence.capture_evidence as capture_evidence


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parents[3] if len(current.parents) > 3 else current.parent


def _file_index_entry(evidence_dir: Path, rel_path: Path) -> dict[str, object]:
    data = (evidence_dir / rel_path).read_bytes()
    mime, _ = mimetypes.guess_type(str(rel_path))
    return {
        "path": str(rel_path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "mime_guess": mime or "application/octet-stream",
    }


def test_makefile_evidence_target_passes_mode_build() -> None:
    makefile = (_repo_root() / "Makefile").read_text(encoding="utf-8")
    assert "scripts/evidence/capture_evidence.py --mode build" in makefile


def test_capture_evidence_default_mode_build(monkeypatch: pytest.MonkeyPatch) -> None:
    script_path = str(_repo_root() / "scripts" / "evidence" / "capture_evidence.py")
    monkeypatch.setattr(sys, "argv", [script_path])
    args = capture_evidence.parse_args()
    assert args.mode == "build"


def test_verify_snapshot_smoke(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "artifacts" / "evidence" / "2026-01-01" / "deadbeef"
    coverage_dir = evidence_dir / "coverage"
    pytest_dir = evidence_dir / "pytest"
    audit_dir = evidence_dir / "audit"
    ci_dir = evidence_dir / "ci"
    env_dir = evidence_dir / "env"
    coverage_dir.mkdir(parents=True)
    pytest_dir.mkdir(parents=True)
    audit_dir.mkdir(parents=True)
    ci_dir.mkdir(parents=True)
    env_dir.mkdir(parents=True)

    coverage_xml = coverage_dir / "coverage.xml"
    coverage_xml.write_text('<coverage line-rate="0.80"></coverage>\n', encoding="utf-8")

    junit_xml = pytest_dir / "junit.xml"
    junit_xml.write_text(
        '<testsuite name="suite" tests="1" failures="0" errors="0" skipped="0">'
        "<testcase name=\"test_example\"/>"
        "</testsuite>\n",
        encoding="utf-8",
    )

    audit_json = audit_dir / "pip-audit.json"
    audit_json.write_text('{"dependencies": []}\n', encoding="utf-8")

    ci_summary = ci_dir / "summary.json"
    ci_summary.write_text('{"workflow": "test"}\n', encoding="utf-8")

    python_version = env_dir / "python_version.txt"
    python_version.write_text("3.11.0\n", encoding="utf-8")

    uv_lock_sha = env_dir / "uv_lock_sha256.txt"
    uv_lock_sha.write_text("deadbeef\n", encoding="utf-8")

    manifest = {
        "schema_version": "evidence-v1",
        "git_sha": "deadbeef00000000000000000000000000000000",
        "short_sha": "deadbeef",
        "created_utc": "2026-01-01T00:00:00Z",
        "source_ref": "refs/heads/test",
        "commands": [],
        "outputs": {
            "coverage_xml": "coverage/coverage.xml",
            "junit_xml": "pytest/junit.xml",
            "pip_audit_json": "audit/pip-audit.json",
            "ci_summary": "ci/summary.json",
            "python_version": "env/python_version.txt",
            "uv_lock_sha256": "env/uv_lock_sha256.txt",
        },
        "status": {"ok": True, "partial": False, "failures": []},
        "file_index": [
            _file_index_entry(evidence_dir, Path("coverage/coverage.xml")),
            _file_index_entry(evidence_dir, Path("pytest/junit.xml")),
            _file_index_entry(evidence_dir, Path("audit/pip-audit.json")),
            _file_index_entry(evidence_dir, Path("ci/summary.json")),
            _file_index_entry(evidence_dir, Path("env/python_version.txt")),
            _file_index_entry(evidence_dir, Path("env/uv_lock_sha256.txt")),
        ],
    }
    (evidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evidence/verify_evidence_snapshot.py",
            "--evidence-dir",
            str(evidence_dir),
        ],
        cwd=_repo_root(),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Verifier failed: {result.stderr}\nSTDOUT:\n{result.stdout}"
