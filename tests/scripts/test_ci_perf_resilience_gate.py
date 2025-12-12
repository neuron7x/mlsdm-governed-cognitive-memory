"""
Unit tests for CI Performance & Resilience Gate script.
"""

import json
from unittest.mock import Mock, patch

import pytest

# Import the module to test
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from ci_perf_resilience_gate import (
    ChangeClass,
    CIInspector,
    CIPerfResilienceGate,
    FileChange,
    JobResult,
    JobStatus,
    MergeVerdictor,
    PRAnalyzer,
    RiskClassifier,
    RiskMode,
    parse_pr_url,
)


class TestPRAnalyzer:
    """Test PR analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = PRAnalyzer()

    def test_classify_documentation_file(self):
        """Test classification of documentation files."""
        change = self.analyzer.classify_file("README.md", "")
        assert change.change_class == ChangeClass.DOC_ONLY
        assert "Documentation" in change.reason

        change = self.analyzer.classify_file("docs/guide.md", "")
        assert change.change_class == ChangeClass.DOC_ONLY

    def test_classify_core_critical_path(self):
        """Test classification of core critical paths."""
        change = self.analyzer.classify_file("src/mlsdm/neuro_engine/core.py", "")
        assert change.change_class == ChangeClass.CORE_CRITICAL
        assert "critical path" in change.reason.lower()

        change = self.analyzer.classify_file("config/settings.yaml", "")
        assert change.change_class == ChangeClass.CORE_CRITICAL

    def test_classify_core_critical_content(self):
        """Test classification based on patch content."""
        patch = """
        async def process_request(timeout=5):
            await asyncio.sleep(1)
            return circuit_breaker.call()
        """
        change = self.analyzer.classify_file("src/utils.py", patch)
        assert change.change_class == ChangeClass.CORE_CRITICAL
        assert "critical patterns" in change.reason.lower()

    def test_classify_non_core_code(self):
        """Test classification of non-core code."""
        change = self.analyzer.classify_file("src/mlsdm/utils/helpers.py", "")
        assert change.change_class == ChangeClass.NON_CORE_CODE
        assert "non-critical" in change.reason.lower()

    def test_analyze_changes(self):
        """Test analyzing multiple file changes."""
        files = [
            {"filename": "README.md", "patch": ""},
            {"filename": "src/mlsdm/neuro_engine/main.py", "patch": ""},
            {"filename": "src/utils.py", "patch": ""},
        ]
        changes = self.analyzer.analyze_changes(files)
        assert len(changes) == 3
        assert changes[0].change_class == ChangeClass.DOC_ONLY
        assert changes[1].change_class == ChangeClass.CORE_CRITICAL


class TestCIInspector:
    """Test CI inspector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.inspector = CIInspector()

    def test_map_job_status_success(self):
        """Test mapping successful job status."""
        status = self.inspector.map_job_status("success", "completed")
        assert status == JobStatus.SUCCESS

    def test_map_job_status_failure(self):
        """Test mapping failed job status."""
        status = self.inspector.map_job_status("failure", "completed")
        assert status == JobStatus.FAILURE

    def test_map_job_status_skipped(self):
        """Test mapping skipped job status."""
        status = self.inspector.map_job_status("skipped", "completed")
        assert status == JobStatus.SKIPPED

        status = self.inspector.map_job_status(None, "skipped")
        assert status == JobStatus.SKIPPED

    def test_map_job_status_pending(self):
        """Test mapping pending job status."""
        status = self.inspector.map_job_status(None, "in_progress")
        assert status == JobStatus.PENDING

    def test_extract_key_facts_success(self):
        """Test extracting key facts from successful job."""
        job = {
            "name": "Test Job",
            "conclusion": "success",
            "status": "completed",
            "started_at": "2025-01-01T00:00:00Z",
            "completed_at": "2025-01-01T00:05:00Z",
        }
        facts = self.inspector._extract_key_facts(job, JobStatus.SUCCESS)
        assert "Passed" in facts
        assert "5m0s" in facts

    def test_extract_key_facts_failure(self):
        """Test extracting key facts from failed job."""
        job = {
            "name": "Test Job",
            "conclusion": "failure",
            "status": "completed",
            "steps": [
                {"name": "Setup", "conclusion": "success"},
                {"name": "Test", "conclusion": "failure"},
            ],
        }
        facts = self.inspector._extract_key_facts(job, JobStatus.FAILURE)
        assert "Failed" in facts
        assert "Test" in facts


class TestRiskClassifier:
    """Test risk classifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = RiskClassifier()

    def test_classify_green_light_doc_only(self):
        """Test green light classification for doc-only changes."""
        changes = [
            FileChange("README.md", ChangeClass.DOC_ONLY, "Doc file"),
            FileChange("docs/guide.md", ChangeClass.DOC_ONLY, "Doc file"),
        ]
        mode, reasons = self.classifier.classify(changes, [], [])
        assert mode == RiskMode.GREEN_LIGHT
        assert any("documentation-only" in r.lower() for r in reasons)

    def test_classify_green_light_non_core(self):
        """Test green light classification for non-core changes."""
        changes = [
            FileChange("src/utils.py", ChangeClass.NON_CORE_CODE, "Utility"),
            FileChange("tests/test_util.py", ChangeClass.NON_CORE_CODE, "Test"),
        ]
        mode, reasons = self.classifier.classify(changes, [], [])
        assert mode == RiskMode.GREEN_LIGHT
        assert any("no core critical" in r.lower() for r in reasons)

    def test_classify_yellow_moderate_critical(self):
        """Test yellow classification for moderate critical changes."""
        changes = [
            FileChange("src/mlsdm/neuro_engine/core.py", ChangeClass.CORE_CRITICAL, "Critical"),
            FileChange("src/utils.py", ChangeClass.NON_CORE_CODE, "Utility"),
        ]
        mode, reasons = self.classifier.classify(changes, [], [])
        assert mode == RiskMode.YELLOW_CRITICAL_PATH
        assert any("moderate" in r.lower() for r in reasons)

    def test_classify_red_many_critical(self):
        """Test red classification for many critical changes."""
        changes = [
            FileChange(f"src/core{i}.py", ChangeClass.CORE_CRITICAL, "Critical")
            for i in range(15)
        ]
        mode, reasons = self.classifier.classify(changes, [], [])
        assert mode == RiskMode.RED_HIGH_RISK_OR_RELEASE
        assert any("high number" in r.lower() for r in reasons)

    def test_classify_red_release_label(self):
        """Test red classification for release label."""
        changes = [
            FileChange("src/core.py", ChangeClass.CORE_CRITICAL, "Critical"),
        ]
        mode, reasons = self.classifier.classify(changes, [], ["release"])
        assert mode == RiskMode.RED_HIGH_RISK_OR_RELEASE
        assert any("release" in r.lower() for r in reasons)


class TestMergeVerdictor:
    """Test merge verdictor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.verdictor = MergeVerdictor()

    def test_verdict_green_light_safe(self):
        """Test safe verdict for green light mode."""
        job_results = [
            JobResult("CI / Lint", JobStatus.SUCCESS, "Passed"),
            JobResult("CI / Tests", JobStatus.SUCCESS, "Passed"),
        ]
        verdict, actions, reasons = self.verdictor.determine_verdict(
            RiskMode.GREEN_LIGHT, job_results
        )
        assert verdict == "SAFE_TO_MERGE_NOW"
        assert len(actions) == 0
        assert any("low-risk" in r.lower() for r in reasons)

    def test_verdict_green_light_base_failure(self):
        """Test do not merge for green light with base job failure."""
        job_results = [
            JobResult("Lint and Type Check", JobStatus.FAILURE, "Failed"),
            JobResult("CI / Tests", JobStatus.SUCCESS, "Passed"),
        ]
        verdict, actions, reasons = self.verdictor.determine_verdict(
            RiskMode.GREEN_LIGHT, job_results
        )
        assert verdict == "DO_NOT_MERGE_YET"
        assert len(actions) > 0
        assert any("base ci jobs failed" in r.lower() for r in reasons)

    def test_verdict_yellow_tests_passed(self):
        """Test safe verdict for yellow mode with tests passed."""
        job_results = [
            JobResult("Lint and Type Check", JobStatus.SUCCESS, "Passed"),
            JobResult(
                "Performance & Resilience Validation / Fast Resilience Tests",
                JobStatus.SUCCESS,
                "Passed",
            ),
            JobResult(
                "Performance & Resilience Validation / Performance & SLO Validation",
                JobStatus.SUCCESS,
                "Passed",
            ),
        ]
        verdict, actions, reasons = self.verdictor.determine_verdict(
            RiskMode.YELLOW_CRITICAL_PATH, job_results
        )
        assert verdict == "SAFE_TO_MERGE_NOW"
        assert len(actions) == 0
        assert any("passed" in r.lower() for r in reasons)

    def test_verdict_yellow_tests_skipped(self):
        """Test do not merge for yellow mode with tests skipped."""
        job_results = [
            JobResult("Lint and Type Check", JobStatus.SUCCESS, "Passed"),
            JobResult(
                "Performance & Resilience Validation / Fast Resilience Tests",
                JobStatus.SKIPPED,
                "Skipped",
            ),
        ]
        verdict, actions, reasons = self.verdictor.determine_verdict(
            RiskMode.YELLOW_CRITICAL_PATH, job_results
        )
        assert verdict == "DO_NOT_MERGE_YET"
        assert len(actions) > 0
        assert any("label" in action.lower() for action in actions)

    def test_verdict_red_all_passed(self):
        """Test safe verdict for red mode with all tests passed."""
        job_results = [
            JobResult("Lint and Type Check", JobStatus.SUCCESS, "Passed"),
            JobResult(
                "Performance & Resilience Validation / Fast Resilience Tests",
                JobStatus.SUCCESS,
                "Passed",
            ),
            JobResult(
                "Performance & Resilience Validation / Performance & SLO Validation",
                JobStatus.SUCCESS,
                "Passed",
            ),
            JobResult(
                "Performance & Resilience Validation / Comprehensive Resilience Tests",
                JobStatus.SUCCESS,
                "Passed",
            ),
        ]
        verdict, actions, reasons = self.verdictor.determine_verdict(
            RiskMode.RED_HIGH_RISK_OR_RELEASE, job_results
        )
        assert verdict == "SAFE_TO_MERGE_NOW"
        assert len(actions) == 0
        assert any("all" in r.lower() and "passed" in r.lower() for r in reasons)

    def test_verdict_red_missing_comprehensive(self):
        """Test do not merge for red mode missing comprehensive tests."""
        job_results = [
            JobResult("Lint and Type Check", JobStatus.SUCCESS, "Passed"),
            JobResult(
                "Performance & Resilience Validation / Fast Resilience Tests",
                JobStatus.SUCCESS,
                "Passed",
            ),
            JobResult(
                "Performance & Resilience Validation / Performance & SLO Validation",
                JobStatus.SUCCESS,
                "Passed",
            ),
        ]
        verdict, actions, reasons = self.verdictor.determine_verdict(
            RiskMode.RED_HIGH_RISK_OR_RELEASE, job_results
        )
        assert verdict == "DO_NOT_MERGE_YET"
        assert len(actions) > 0
        assert any("comprehensive" in r.lower() for r in reasons)


class TestCIPerfResilienceGate:
    """Test main gate functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gate = CIPerfResilienceGate()

    def test_generate_slo_improvements(self):
        """Test generating SLO improvement suggestions."""
        analysis = {
            "job_results": [
                JobResult(
                    "Performance & Resilience Validation / Fast Resilience Tests",
                    JobStatus.SKIPPED,
                    "Skipped",
                ),
            ],
            "mode": RiskMode.YELLOW_CRITICAL_PATH,
        }
        suggestions = self.gate.generate_slo_improvements(analysis)
        assert len(suggestions) <= 3
        assert len(suggestions) > 0

    def test_format_output_structure(self):
        """Test output formatting structure."""
        analysis = {
            "pr_number": 123,
            "pr_title": "Test PR",
            "pr_labels": ["test"],
            "changes": [
                FileChange("README.md", ChangeClass.DOC_ONLY, "Doc file"),
            ],
            "job_results": [
                JobResult("Test Job", JobStatus.SUCCESS, "Passed"),
            ],
            "mode": RiskMode.GREEN_LIGHT,
            "mode_reasons": ["Test reason"],
            "verdict": "SAFE_TO_MERGE_NOW",
            "required_actions": [],
            "verdict_reasons": ["All tests passed"],
        }
        output = self.gate.format_output(analysis)

        # Check for required sections
        assert "Section 1: MODE_CLASSIFICATION" in output
        assert "Section 2: CI_STATUS_TABLE" in output
        assert "Section 3: REQUIRED_ACTIONS_BEFORE_MERGE" in output
        assert "Section 4: MERGE_VERDICT" in output
        assert "Section 5: SLO/CI_IMPROVEMENT_IDEAS" in output
        assert "Appendix: Change Classification Details" in output


class TestParsePRUrl:
    """Test PR URL parsing."""

    def test_parse_valid_url(self):
        """Test parsing valid PR URL."""
        url = "https://github.com/neuron7x/mlsdm/pull/231"
        owner, repo, pr_number = parse_pr_url(url)
        assert owner == "neuron7x"
        assert repo == "mlsdm"
        assert pr_number == 231

    def test_parse_http_url(self):
        """Test parsing HTTP (non-HTTPS) URL."""
        url = "http://github.com/owner/repo/pull/42"
        owner, repo, pr_number = parse_pr_url(url)
        assert owner == "owner"
        assert repo == "repo"
        assert pr_number == 42

    def test_parse_invalid_url(self):
        """Test parsing invalid URL raises error."""
        with pytest.raises(ValueError):
            parse_pr_url("https://github.com/owner/repo")

        with pytest.raises(ValueError):
            parse_pr_url("not a url")


class TestIntegration:
    """Integration tests for the gate."""

    @patch("ci_perf_resilience_gate.requests.get")
    def test_analyze_pr_integration(self, mock_get):
        """Test full PR analysis integration."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        
        # Mock PR data, files, and workflow runs
        mock_response.json.side_effect = [
            # PR data (first call)
            {
                "title": "Test PR",
                "labels": [{"name": "test"}],
                "head": {"sha": "abc123"},
            },
            # Files data (second call)
            [
                {"filename": "README.md", "patch": ""},
                {"filename": "src/mlsdm/neuro_engine/core.py", "patch": ""},
            ],
        ]
        
        mock_get.return_value = mock_response

        gate = CIPerfResilienceGate()
        # Mock the CI inspector methods to avoid additional API calls
        with patch.object(
            gate.ci_inspector, "inspect_ci_jobs", return_value=[]
        ):
            analysis = gate.analyze_pr("owner", "repo", 123)

        assert analysis["pr_number"] == 123
        assert analysis["pr_title"] == "Test PR"
        assert len(analysis["changes"]) == 2
        assert isinstance(analysis["mode"], RiskMode)
        assert analysis["verdict"] in [
            "SAFE_TO_MERGE_NOW",
            "DO_NOT_MERGE_YET",
            "MERGE_ONLY_IF_YOU_CONSCIOUSLY_ACCEPT_RISK",
        ]
