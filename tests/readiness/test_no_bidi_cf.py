from __future__ import annotations

import unicodedata
from pathlib import Path

TARGETS = [
    Path("scripts/readiness/change_analyzer.py"),
    Path("scripts/readiness/evidence_collector.py"),
    Path("scripts/readiness/policy_engine.py"),
    Path("scripts/readiness/changelog_generator.py"),
    Path("docs/status/README.md"),
    Path("docs/status/READINESS.md"),
]


def test_files_contain_no_bidi_or_control_characters():
    bad = []
    for path in TARGETS:
        content = path.read_text(encoding="utf-8")
        for idx, ch in enumerate(content):
            if unicodedata.category(ch) in {"Cf", "Cs"}:
                bad.append((path, idx, hex(ord(ch))))
    assert not bad, f"Found hidden/bidi characters: {bad}"
