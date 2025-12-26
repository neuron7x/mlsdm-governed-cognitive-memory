from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest  # noqa: TC002

import scripts.readiness.change_analyzer as ca


def test_normalize_path_windows_separators():
    assert ca.normalize_path("src\\app\\mod.py") == "src/app/mod.py"


def test_classify_category_security_and_risk():
    path = "src/security/guardrails.py"
    category = ca.classify_category(path)
    assert category == "security_critical"
    assert ca.risk_for_category(category) == "critical"


def test_classify_category_observability_under_src():
    path = "src/app/logging/collector.py"
    assert ca.classify_category(path) == "observability"


def test_classify_category_infrastructure_and_docs_and_tests():
    assert ca.classify_category(".github/workflows/ci.yaml") == "infrastructure"
    assert ca.classify_category("docs/readme.md") == "documentation"
    assert ca.classify_category("tests/test_sample.py") == "test_coverage"


def test_module_name_src_and_tests():
    assert ca.module_name("src/pkg/mod.py") == "pkg.mod"
    assert ca.module_name("tests/test_mod.py") == "test_mod"


def test_semantic_diff_processes_all_top_level_items():
    source = (
        "def f1(a):\n"
        "    return a\n\n"
        "class C:\n"
        "    def m1(self, x):\n"
        "        def inner():\n"
        "            return x\n"
        "        return x\n\n"
        "def f2(b):\n"
        "    return b\n"
    )
    sigs, parse_error = ca.parse_python_signatures(source, "mod")
    assert parse_error is False
    assert "f1" in sigs and "f2" in sigs and "C.m1" in sigs
    assert all("inner" not in v for v in sigs.values())


def test_semantic_diff_added_removed_modified_functions():
    before = "def a(x):\n    return x\n\ndef c():\n    return 3\n"
    after = "def a(x, y):\n    return x\n\ndef b():\n    return 2\n"
    diff = ca.semantic_diff(before, after, module="mod")
    assert "mod:a(x)->None -> mod:a(x,y)->None" in diff["modified_functions"]
    assert "mod:b()->None" in diff["added_functions"]
    assert "mod:c()->None" in diff["removed_functions"]


def test_json_structure_and_counts(tmp_path: Path):
    (tmp_path / ".github/workflows").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "observability").mkdir()
    (tmp_path / "src/security").mkdir(parents=True)
    (tmp_path / "src/core").mkdir(parents=True)
    (tmp_path / "tests").mkdir()

    (tmp_path / ".github/workflows/ci.yaml").write_text("name: ci\n", encoding="utf-8")
    (tmp_path / "docs/readme.md").write_text("docs\n", encoding="utf-8")
    (tmp_path / "observability/logs.py").write_text("def log():\n    return True\n", encoding="utf-8")
    (tmp_path / "src/core/app.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src/security/auth.py").write_text("def bar(x):\n    return x\n", encoding="utf-8")
    (tmp_path / "tests/test_app.py").write_text("def test_sample():\n    assert True\n", encoding="utf-8")

    paths = [
        ".github/workflows/ci.yaml",
        "docs/readme.md",
        "observability/logs.py",
        "src/core/app.py",
        "src/security/auth.py",
        "tests/test_app.py",
    ]

    result = ca.analyze_paths(paths, base_ref="origin/main", root=tmp_path)

    expected_counts = {
        "categories": {
            "security_critical": 1,
            "test_coverage": 1,
            "documentation": 1,
            "infrastructure": 1,
            "observability": 1,
            "functional_core": 1,
            "mixed": 0,
        },
        "risks": {"info": 2, "low": 1, "medium": 1, "high": 1, "critical": 1},
    }

    expected_files = [
        {
            "path": ".github/workflows/ci.yaml",
            "category": "infrastructure",
            "risk": "medium",
            "metadata": {"missing": False, "new_file": True, "parse_error": False},
            "semantic_diff": None,
            "functions_added": [],
            "functions_removed": [],
            "yaml_diff": {"added_keys": ["name"], "changed_keys": [], "removed_keys": [], "parse_error": False},
        },
        {
            "path": "docs/readme.md",
            "category": "documentation",
            "risk": "info",
            "metadata": {"missing": False, "new_file": False, "parse_error": False},
            "semantic_diff": None,
            "functions_added": [],
            "functions_removed": [],
            "yaml_diff": None,
        },
        {
            "path": "observability/logs.py",
            "category": "observability",
            "risk": "low",
            "metadata": {"missing": False, "new_file": True, "parse_error": False},
            "semantic_diff": {
                "added_functions": ["observability.logs:log()->None"],
                "removed_functions": [],
                "modified_functions": [],
                "summary": {"added": 1, "removed": 0, "modified": 0},
                "parse_error": False,
            },
            "functions_added": ["observability.logs:log()->None"],
            "functions_removed": [],
            "yaml_diff": None,
        },
        {
            "path": "src/core/app.py",
            "category": "functional_core",
            "risk": "high",
            "metadata": {"missing": False, "new_file": True, "parse_error": False},
            "semantic_diff": {
                "added_functions": ["core.app:foo()->None"],
                "removed_functions": [],
                "modified_functions": [],
                "summary": {"added": 1, "removed": 0, "modified": 0},
                "parse_error": False,
            },
            "functions_added": ["core.app:foo()->None"],
            "functions_removed": [],
            "yaml_diff": None,
        },
        {
            "path": "src/security/auth.py",
            "category": "security_critical",
            "risk": "critical",
            "metadata": {"missing": False, "new_file": True, "parse_error": False},
            "semantic_diff": {
                "added_functions": ["security.auth:bar(x)->None"],
                "removed_functions": [],
                "modified_functions": [],
                "summary": {"added": 1, "removed": 0, "modified": 0},
                "parse_error": False,
            },
            "functions_added": ["security.auth:bar(x)->None"],
            "functions_removed": [],
            "yaml_diff": None,
        },
        {
            "path": "tests/test_app.py",
            "category": "test_coverage",
            "risk": "info",
            "metadata": {"missing": False, "new_file": True, "parse_error": False},
            "semantic_diff": {
                "added_functions": ["test_app:test_sample()->None"],
                "removed_functions": [],
                "modified_functions": [],
                "summary": {"added": 1, "removed": 0, "modified": 0},
                "parse_error": False,
            },
            "functions_added": ["test_app:test_sample()->None"],
            "functions_removed": [],
            "yaml_diff": None,
        },
    ]

    assert result == {
        "base_ref": "origin/main",
        "primary_category": "mixed",
        "max_risk": "critical",
        "counts": expected_counts,
        "files": expected_files,
    }


def test_analyze_paths_handles_missing_and_invalid_files(tmp_path: Path):
    missing = tmp_path / "missing.py"
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    text_file = tmp_path / "notes.txt"
    text_file.write_text("hello", encoding="utf-8")

    result = ca.analyze_paths(
        [str(missing.relative_to(tmp_path)), str(broken.relative_to(tmp_path)), str(text_file.relative_to(tmp_path))],
        base_ref="origin/main",
        root=tmp_path,
    )

    files = {entry["path"]: entry for entry in result["files"]}
    assert files[str(missing.relative_to(tmp_path))]["metadata"]["missing"] is True
    assert files[str(broken.relative_to(tmp_path))]["metadata"]["parse_error"] is True
    assert files[str(text_file.relative_to(tmp_path))]["semantic_diff"] is None


def test_analyze_paths_uses_base_ref_for_deleted_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    before_source = "def gone():\n    return 1\n"
    missing = "src/gone.py"

    def fake_show(path: str, ref: str, root: Path) -> str | None:  # type: ignore[override]
        return before_source if path == missing else None

    monkeypatch.setattr(ca, "get_file_at_ref", fake_show)
    result = ca.analyze_paths([missing], base_ref="origin/main", root=tmp_path)
    entry = result["files"][0]
    semantic = entry["semantic_diff"]
    assert semantic["added_functions"] == []
    assert semantic["modified_functions"] == []
    assert semantic["removed_functions"] == ["gone:gone()->None"]


def test_yaml_diff_reports_key_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    before = "a: 1\nb: 2\n"
    after_path = tmp_path / "config.yaml"
    after_path.write_text("a: 1\nb: 3\nc: 4\n", encoding="utf-8")

    def fake_show(path: str, ref: str, root: Path) -> str | None:  # type: ignore[override]
        return before if path == "config.yaml" else None

    monkeypatch.setattr(ca, "get_file_at_ref", fake_show)
    result = ca.analyze_paths(["config.yaml"], base_ref="origin/main", root=tmp_path)
    entry = result["files"][0]
    diff = entry["yaml_diff"]
    assert diff["added_keys"] == ["c"]
    assert diff["changed_keys"] == ["b"]
    assert diff["removed_keys"] == []


def test_no_control_characters_present():
    targets = [
        Path("scripts/readiness/change_analyzer.py"),
        Path("scripts/readiness/__init__.py"),
        Path("tests/unit/test_readiness_change_analyzer.py"),
    ]
    bad: list[str] = []
    for file in targets:
        text = file.read_text(encoding="utf-8")
        for ch in text:
            if unicodedata.category(ch) == "Cf":
                bad.append(f"{file}: U+{ord(ch):04X}")
    assert not bad, f"Control characters found: {bad}"


def test_read_paths_file_rejects_bidi(tmp_path: Path):
    files_file = tmp_path / "files.txt"
    files_file.write_text("normal.py\n", encoding="utf-8")
    assert ca._read_paths_file(str(files_file)) == ["normal.py"]

    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("good.py\n\u202e", encoding="utf-8")
    with pytest.raises(ValueError):
        ca._read_paths_file(str(bad_file))
