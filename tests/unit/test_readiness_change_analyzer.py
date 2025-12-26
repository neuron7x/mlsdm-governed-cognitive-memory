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
    files_file = tmp_path / "files.txt"
    files_file.write_text("src/a.py\ntests/test_b.py\ndocs/readme.rst\nobservability/logger.py\n", encoding="utf-8")
    paths = ca._read_paths_file(str(files_file))
    result = ca.analyze_paths(paths, base_ref="origin/main", root=tmp_path)

    assert set(result.keys()) >= {"primary_category", "max_risk", "counts", "files", "base_ref"}
    counts = result["counts"]
    assert counts["categories"]["functional_core"] == 1
    assert counts["categories"]["test_coverage"] == 1
    assert counts["categories"]["documentation"] == 1
    assert counts["categories"]["observability"] == 1
    assert counts["risks"]["high"] == 1
    assert counts["risks"]["info"] == 2
    assert counts["risks"]["low"] == 1

    files = result["files"]
    assert isinstance(files, list) and len(files) == 4
    assert [f["path"] for f in files] == sorted([f["path"] for f in files])
    for entry in files:
        assert set(entry.keys()) == {
            "path",
            "category",
            "risk",
            "metadata",
            "semantic_diff",
            "functions_added",
            "functions_removed",
            "yaml_diff",
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
