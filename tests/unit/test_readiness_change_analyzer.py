from pathlib import Path

import scripts.readiness.change_analyzer as ca


def test_normalize_path_windows_separators():
    assert ca.normalize_path(r"src\\module\\file.py") == "src/module/file.py"


def test_classify_category_security_critical_moral_filter():
    assert ca.classify_category("src/pkg/moral_filter/checks.py") == "security_critical"


def test_classify_category_infra_workflows():
    assert ca.classify_category(".github/workflows/readiness.yml") == "infrastructure"


def test_module_name_src_and_tests():
    assert ca.module_name("src/foo/bar.py") == "foo.bar"
    assert ca.module_name("tests/foo/test_bar.py") == "foo.test_bar"


def test_semantic_diff_added_removed_modified_functions():
    before = "def a(x):\n    return x\n\ndef c():\n    return 3\n"
    after = "def a(x, y):\n    return x\n\ndef b():\n    return 2\n"
    diff = ca.semantic_diff(before, after, module="mod")
    assert "mod:a(x)->None -> mod:a(x,y)->None" in diff["modified_functions"]
    assert "mod:b()->None" in diff["added_functions"]
    assert "mod:c()->None" in diff["removed_functions"]


def test_analyze_paths_handles_missing_files_gracefully(tmp_path: Path):
    files_list = ["missing.py", "docs/readme.md"]
    result = ca.analyze_paths(files_list, base_ref="HEAD", root=tmp_path)
    assert result["summary"]["files_analyzed"] == 2
    assert "missing.py" in result["files"]
    missing_entry = result["files"]["missing.py"]
    assert missing_entry["functions_added"] == []
    assert missing_entry["functions_removed"] == []
    assert missing_entry["functions_modified"] == []
