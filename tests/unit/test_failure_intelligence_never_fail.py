import json
import sys
from pathlib import Path

import scripts.ci.failure_intelligence as fi


def run_main(tmp_path: Path, args: list[str]) -> dict:
    out_md = tmp_path / "out.md"
    out_json = tmp_path / "out.json"
    argv = ["prog", "--out", str(out_md), "--json", str(out_json), *args]
    original = list(sys.argv)
    sys.argv = argv
    try:
        fi.main()
    finally:
        sys.argv = original
    assert out_md.exists()
    assert out_json.exists()
    return json.loads(out_json.read_text(encoding="utf-8"))


def test_missing_defusedxml_writes_outputs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(fi, "HAS_DEFUSEDXML", False)
    monkeypatch.setattr(fi, "DEFUSEDXML_ERR", "ImportError('defusedxml')")
    monkeypatch.setattr(fi, "parse", None, raising=False)
    summary = run_main(tmp_path, [])
    assert "defusedxml_missing" in summary.get("errors", [])
    assert summary["signal"].startswith("Failure intelligence")


def test_corrupt_xml_is_handled(tmp_path: Path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<testsuite><testcase></testsuite", encoding="utf-8")
    summary = run_main(tmp_path, ["--junit", str(bad)])
    assert summary["top_failures"] == []


def test_happy_path_outputs(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        """
        <testsuite>
          <testcase classname="pkg.test_sample" name="test_one" file="tests/unit/test_sample.py">
            <failure message="assert 1 == 0">Traceback line 1</failure>
          </testcase>
        </testsuite>
        """,
        encoding="utf-8",
    )
    coverage = tmp_path / "coverage.xml"
    coverage.write_text('<coverage line-rate="0.5" branch-rate="0.1" version="1.0"></coverage>', encoding="utf-8")
    summary = run_main(tmp_path, ["--junit", str(junit), "--coverage", str(coverage)])
    assert summary["coverage_percent"] == 50.0
    assert summary["top_failures"][0]["id"] == "pkg.test_sample::test_one"
