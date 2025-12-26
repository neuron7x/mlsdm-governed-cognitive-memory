from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest  # noqa: TC002

import scripts.readiness.changelog_generator as cg


def test_generator_insertion_and_determinism(tmp_path: Path):
    readiness_dir = tmp_path / "docs" / "status"
    readiness_dir.mkdir(parents=True)
    readiness_path = readiness_dir / "READINESS.md"
    readiness_path.write_text("# Title\nLast updated: 2024-12-31\n\n## Change Log\n- old entry\n", encoding="utf-8")

    diff_paths = ["tests/test_app.py", "src/core/app.py"]
    analysis = {
        "counts": {
            "categories": {
                "security_critical": 0,
                "test_coverage": 1,
                "documentation": 0,
                "infrastructure": 0,
                "observability": 0,
                "functional_core": 1,
                "mixed": 0,
            },
            "risks": {"info": 1, "low": 0, "medium": 0, "high": 1, "critical": 0},
        },
        "primary_category": "mixed",
        "max_risk": "high",
    }

    fixed_now = lambda: datetime(2025, 1, 1, tzinfo=timezone.utc)
    path, updated = cg.generate_update(
        "Test Entry",
        "origin/main",
        root=tmp_path,
        diff_provider=lambda base_ref, root: diff_paths,
        analyzer=lambda paths, base_ref, root: analysis,
        now_provider=fixed_now,
    )

    expected_entry = (
        "- 2025-01-01 — **Test Entry** — Base: origin/main\n"
        "  - Changed files (2): `src/core/app.py`, `tests/test_app.py`\n"
        "  - Primary category: mixed; Max risk: high\n"
        "  - Category counts: {\"documentation\": 0, \"functional_core\": 1, \"infrastructure\": 0, \"mixed\": 0, \"observability\": 0, \"security_critical\": 0, \"test_coverage\": 1}\n"
        "  - Risk counts: {\"critical\": 0, \"high\": 1, \"info\": 1, \"low\": 0, \"medium\": 0}"
    )

    lines = updated.splitlines()
    assert path == readiness_path
    assert lines[1] == "Last updated: 2025-01-01"
    assert lines[4:9] == expected_entry.splitlines()
    assert updated.rstrip("\n").endswith("- old entry")

    # Running again with identical inputs is deterministic
    _, updated_again = cg.generate_update(
        "Test Entry",
        "origin/main",
        root=tmp_path,
        diff_provider=lambda base_ref, root: diff_paths,
        analyzer=lambda paths, base_ref, root: analysis,
        now_provider=fixed_now,
    )
    assert updated_again == updated


def test_generator_missing_readiness(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        cg.generate_update("Missing", "origin/main", root=tmp_path)


def test_generator_rejects_bidi_title(tmp_path: Path):
    readiness_dir = tmp_path / "docs" / "status"
    readiness_dir.mkdir(parents=True)
    (readiness_dir / "READINESS.md").write_text("# Title\nLast updated: 2024-12-31\n\n## Change Log\n", encoding="utf-8")

    with pytest.raises(ValueError):
        cg.generate_update("Bad\u202eTitle", "origin/main", root=tmp_path)
