import json
from pathlib import Path

import scripts.readiness.change_analyzer as ca


def test_normalize_path_windows_separators():
    assert ca.normalize_path(r".\\src\\\\module\\file.py") == "src/module/file.py"


def test_classify_category_rules_and_risk_mapping():
    cases = [
        ("src/pkg/moral_filter/checks.py", "security_critical", "critical"),
        ("src/security/policy.py", "security_critical", "critical"),
        ("src/app/security/hook.py", "security_critical", "critical"),
        ("tests/test_example.py", "test_coverage", "informational"),
        ("docs/readme.md", "documentation", "informational"),
        ("deploy/k8s.yaml", "infrastructure", "medium"),
        ("config/service.yaml", "infrastructure", "medium"),
        (".github/workflows/ci.yml", "infrastructure", "medium"),
        ("metrics/collector.py", "observability", "low"),
        ("src/core/main.py", "functional_core", "high"),
        ("misc/other.txt", "documentation", "informational"),
    ]
    for path, expected_cat, expected_risk in cases:
        cat = ca.classify_category(path)
        assert cat == expected_cat
        assert ca.risk_for_category(cat) == expected_risk


def test_primary_category_priority_and_max_risk():
    paths = [
        "docs/readme.md",  # documentation
        "tests/test_a.py",  # test_coverage
        "src/security/check.py",  # security_critical
        "src/app/core.py",  # functional_core
    ]
    result = ca.analyze_paths(paths, base_ref="origin/main", root=Path.cwd())
    assert result["primary_category"] == "security_critical"
    assert result["max_risk"] == "critical"


def test_json_structure_and_counts(tmp_path: Path):
    files_file = tmp_path / "files.txt"
    files_file.write_text(
        "src/a.py\ntests/test_b.py\ndocs/readme.rst\nobservability/logger.py\n", encoding="utf-8"
    )

    paths = ca._read_paths_file(str(files_file))
    result = ca.analyze_paths(paths, base_ref="origin/main", root=tmp_path)

    assert set(result.keys()) >= {"primary_category", "max_risk", "counts", "files", "base_ref"}
    counts = result["counts"]
    assert "categories" in counts and "risks" in counts
    assert counts["categories"]["functional_core"] == 1
    assert counts["categories"]["test_coverage"] == 1
    assert counts["categories"]["documentation"] == 1
    assert counts["categories"]["observability"] == 1
    assert counts["risks"]["high"] == 1
    assert counts["risks"]["informational"] == 2
    assert counts["risks"]["low"] == 1

    files = result["files"]
    assert isinstance(files, list) and len(files) == 4
    for entry in files:
        assert set(entry.keys()) == {"path", "category", "risk", "details"}


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


def test_cli_reads_file_and_outputs_json(tmp_path: Path, monkeypatch):
    files_file = tmp_path / "paths.txt"
    files_file.write_text("src/a.py\ntests/test_me.py\n", encoding="utf-8")
    output_path = tmp_path / "out.json"
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")

    exit_code = ca.main(["--files", str(files_file), "--output", str(output_path)])
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["counts"]["categories"]["functional_core"] == 1
    assert payload["counts"]["categories"]["test_coverage"] == 1
    assert payload["max_risk"] == "high"


def test_analyze_paths_handles_missing_files_gracefully(tmp_path: Path):
    result = ca.analyze_paths(["missing.py", "docs/readme.md"], base_ref="HEAD", root=tmp_path)
    files = {entry["path"]: entry for entry in result["files"]}
    assert files["missing.py"]["details"]["semantic"]["summary"]["added"] == 0
    assert files["missing.py"]["risk"] == "high" or files["missing.py"]["risk"] == "informational"
