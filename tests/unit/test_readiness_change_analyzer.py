from __future__ import annotations

from pathlib import Path

import pytest

from scripts.readiness import change_analyzer as ca


def test_categorization_rules():
    assert ca.categorize_path("src/app/security/module.py") == "security_critical"
    assert ca.categorize_path("src/mlsdm/moral_filter.py") == "security_critical"
    assert ca.categorize_path("tests/unit/test_sample.py") == "test_coverage"
    assert ca.categorize_path("docs/readme.md") == "documentation"
    assert ca.categorize_path(".github/workflows/build.yml") == "infrastructure"
    assert ca.categorize_path("src/observability/logger.py") == "observability"
    assert ca.categorize_path("src/mlsdm/core/example.py") == "functional_core"


def test_risk_ordering_uses_numeric_level():
    files = {
        "a.py": {"category": "observability", "risk": "low", "semantic": {}},
        "b.py": {"category": "functional_core", "risk": "high", "semantic": {}},
        "c.py": {"category": "documentation", "risk": "informational", "semantic": {}},
    }
    primary, max_risk = ca.determine_overall(files)
    assert primary == "functional_core"
    assert max_risk == "high"


def test_extract_signatures_and_diff_respects_structure():
    base_content = """
def unchanged(x):
    return x

class Sample:
    def method(self, y):
        def ignored(): ...
        return y
"""
    current_content = """
def unchanged(x, z):
    return x

class Sample:
    @classmethod
    def method(cls, y):
        return y

def added(value: int) -> int:
    return value
"""
    module = "module.sample"
    base_signatures = ca.extract_signatures(base_content, module)
    current_signatures = ca.extract_signatures(current_content, module)
    diff = ca.diff_signatures(base_signatures, current_signatures, module)

    assert diff["functions_added"] == ["module.sample:added(value:int)->int"]
    assert diff["functions_removed"] == []
    assert "module.sample:unchanged(x)->None" not in diff["functions_added"]
    assert any("Sample.method" in item for item in diff["functions_modified"])
    assert not any("ignored" in item for values in diff.values() for item in values)


def test_stable_output_structure(tmp_path: Path):
    changed_file = "src/mlsdm/core/sample.py"
    changed_content = "def run(x):\n    return x\n"

    def base_loader(path: str, base_ref: str) -> str | None:  # noqa: ARG001
        return None

    def current_loader(path: str) -> str | None:
        if path == changed_file:
            return changed_content
        return None

    result = ca.analyze_files([changed_file], "origin/main", base_loader=base_loader, current_loader=current_loader)
    expected = {
        "primary_category": "functional_core",
        "max_risk": "high",
        "summary": {"files_analyzed": 1, "categories": {"functional_core": 1}},
        "files": {
            changed_file: {
                "category": "functional_core",
                "risk": "high",
                "semantic": {
                    "functions_added": ["src.mlsdm.core.sample:run(x)->None"],
                    "functions_removed": [],
                    "functions_modified": [],
                },
            }
        },
    }
    assert result == expected


def test_handles_empty_and_deleted_and_non_python(tmp_path: Path):
    deleted_path = "src/mlsdm/core/old.py"
    doc_path = "docs/status/READINESS.md"
    file_list = tmp_path / "files.txt"
    file_list.write_text(f"{deleted_path}\n{doc_path}\n", encoding="utf-8")

    def base_loader(path: str, base_ref: str) -> str | None:  # noqa: ARG001
        if path == deleted_path:
            return "def old():\n    return 1\n"
        return None

    def current_loader(path: str) -> str | None:
        return None

    analysis = ca.analyze_files(
        [], "origin/main", base_loader=base_loader, current_loader=current_loader
    )
    assert analysis["summary"]["files_analyzed"] == 0

    analysis_deleted = ca.analyze_files(
        [deleted_path, doc_path],
        "origin/main",
        base_loader=base_loader,
        current_loader=current_loader,
    )

    assert analysis_deleted["files"][deleted_path]["semantic"]["functions_removed"] == [
        "src.mlsdm.core.old:old()->None"
    ]
    assert analysis_deleted["files"][doc_path]["semantic"] == {
        "functions_added": [],
        "functions_removed": [],
        "functions_modified": [],
    }
    assert analysis_deleted["summary"]["files_analyzed"] == 2
