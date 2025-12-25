import json
from pathlib import Path

import scripts.readiness.change_analyzer as ca


def test_normalize_path_windows_separators():
    assert ca.normalize_path(r".\\src\\\\module\\file.py") == "src/module/file.py"


def test_classify_category_security_and_observability_priority():
    assert ca.classify_category("src/pkg/moral_filter/checks.py") == "security_critical"
    assert ca.classify_category("src/core/security/hook.py") == "security_critical"
    assert ca.classify_category("src/core/observability/logger.py") == "observability"


def test_security_keywords_map_to_critical_risk():
    for path in [
        "src/app/auth/login.py",
        "src/crypto/engine.py",
        "src/service/permission/check.py",
        "src/encryption/aes.py",
        "tests/security/test_auth.py",
    ]:
        cat = ca.classify_category(path)
        assert cat == "security_critical"
        assert ca.risk_for_category(cat) == "critical"


def test_category_order_priority():
    paths = [
        "docs/readme.md",
        "tests/test_example.py",
        "src/logging/collector.py",
        "src/app/module.py",
        "src/security/policy.py",
    ]
    result = ca.analyze_paths(paths, base_ref="origin/main", root=Path.cwd())
    assert result["primary_category"] == "security_critical"
    assert result["max_risk"] == "critical"


def test_json_structure_and_counts(tmp_path: Path):
    files_file = tmp_path / "files.txt"
    files_file.write_text("src/a.py\ntests/test_b.py\ndocs/readme.rst\nobservability/logger.py\n", encoding="utf-8")
    paths = ca._read_paths_file(str(files_file))
    result = ca.analyze_paths(paths, base_ref="origin/main", root=tmp_path)

    assert set(result.keys()) >= {"primary_category", "max_risk", "counts", "files", "base_ref"}
    counts = result["counts"]
    assert counts["categories"]["security_critical"] == 0
    assert counts["categories"]["functional_core"] == 1
    assert counts["categories"]["test_coverage"] == 1
    assert counts["categories"]["documentation"] == 1
    assert counts["categories"]["observability"] == 1
    assert counts["risks"]["critical"] == 0
    assert counts["risks"]["high"] == 1
    assert counts["risks"]["informational"] == 2
    assert counts["risks"]["low"] == 1

    files = result["files"]
    assert isinstance(files, list) and len(files) == 4
    assert [f["path"] for f in files] == sorted([f["path"] for f in files])
    for entry in files:
        assert set(entry.keys()) == {"path", "category", "risk", "details"}


def test_module_name_src_and_tests():
    assert ca.module_name("src/foo/bar.py") == "foo.bar"
    assert ca.module_name("tests/foo/test_bar.py") == "foo.test_bar"


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
    sigs = ca.parse_python_signatures(source, "mod")
    assert "f1" in sigs
    assert "f2" in sigs
    assert "C" in sigs
    assert "C.m1" in sigs
    assert all("inner" not in v for v in sigs.values())


def test_semantic_diff_added_removed_modified_functions():
    before = "def a(x):\n    return x\n\ndef c():\n    return 3\n"
    after = "def a(x, y):\n    return x\n\ndef b():\n    return 2\n"
    diff = ca.semantic_diff(before, after, module="mod")
    assert "mod:a(x)->None -> mod:a(x,y)->None" in diff["modified_functions"]
    assert "mod:b()->None" in diff["added_functions"]
    assert "mod:c()->None" in diff["removed_functions"]


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
    assert files[str(missing.relative_to(tmp_path))]["details"]["semantic"]["summary"]["added"] == 0
    assert files[str(broken.relative_to(tmp_path))]["details"]["semantic"]["summary"]["added"] == 0
    assert files[str(text_file.relative_to(tmp_path))]["details"]["semantic"]["summary"]["added"] == 0


def test_analyze_paths_uses_base_ref_for_deleted_file(tmp_path: Path, monkeypatch):
    before_source = "def gone():\n    return 1\n"
    missing = "src/gone.py"

    def fake_show(path: str, ref: str, root: Path) -> str | None:  # type: ignore[override]
        return before_source if path == missing else None

    monkeypatch.setattr(ca, "get_file_at_ref", fake_show)
    result = ca.analyze_paths([missing], base_ref="origin/main", root=tmp_path)
    entry = result["files"][0]
    semantic = entry["details"]["semantic"]
    assert semantic["added_functions"] == []
    assert semantic["modified_functions"] == []
    assert semantic["removed_functions"] == ["gone:gone()->None"]


def test_cli_outputs_json(tmp_path: Path):
    files_file = tmp_path / "paths.txt"
    files_file.write_text("src/a.py\ntests/test_me.py\n", encoding="utf-8")
    output_path = tmp_path / "out.json"

    exit_code = ca.main(["--files", str(files_file), "--output", str(output_path)])
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["counts"]["categories"]["functional_core"] == 1
    assert payload["counts"]["categories"]["test_coverage"] == 1
    assert payload["max_risk"] == "high"
