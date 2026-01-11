"""
Smoke tests for Moral Filter Evaluation Suite.

Verifies that the eval runner:
- Loads scenarios correctly
- Executes without crashing
- Produces valid output structure
- Passes core invariant scenarios

Run with:
    pytest tests/evals/test_moral_filter_eval_smoke.py -v
"""

from __future__ import annotations

import pytest


class TestMoralFilterEvalSmoke:
    """Smoke tests for moral filter evaluation runner."""

    def test_runner_imports(self) -> None:
        """Verify runner module can be imported."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        assert MoralFilterEvalRunner is not None

    def test_scenarios_file_exists(self) -> None:
        """Verify scenarios YAML file exists."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        # Use the runner's default path which is relative to its own location
        runner = MoralFilterEvalRunner()
        assert runner.scenarios_path.exists(), f"Scenarios file not found: {runner.scenarios_path}"

    def test_load_scenarios(self) -> None:
        """Verify scenarios can be loaded from YAML."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        scenarios = runner.load_scenarios()

        assert scenarios is not None
        assert len(scenarios) >= 10, "Should have at least 10 scenarios"

        # Check scenario structure
        for scenario in scenarios:
            assert "id" in scenario, "Each scenario must have an id"
            assert "description" in scenario, "Each scenario must have a description"
            assert "input" in scenario, "Each scenario must have input"
            assert "expected" in scenario, "Each scenario must have expected"

    def test_run_single_scenario(self) -> None:
        """Verify single scenario can be executed."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        runner.load_scenarios()

        # Run first scenario
        result = runner.run_scenario(runner.scenarios[0])

        assert result is not None
        assert result.scenario_id is not None
        assert isinstance(result.passed, bool)

    def test_run_all_scenarios(self) -> None:
        """Verify all scenarios can be executed."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        assert results is not None
        assert results.total >= 10
        assert results.total == results.passed + results.failed
        assert results.pass_rate >= 0.0
        assert results.pass_rate <= 100.0

    def test_results_structure(self) -> None:
        """Verify results have correct structure for JSON export."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # Convert to dict for JSON compatibility
        results_dict = results.to_dict()

        assert "summary" in results_dict
        assert "total" in results_dict["summary"]
        assert "passed" in results_dict["summary"]
        assert "failed" in results_dict["summary"]
        assert "pass_rate" in results_dict["summary"]
        assert "by_property" in results_dict
        assert "by_label" in results_dict
        assert "per_scenario" in results_dict

    def test_threshold_bounds_scenarios_pass(self) -> None:
        """Verify core threshold bounds invariants are satisfied."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # Find threshold bounds scenarios
        bounds_scenarios = [s for s in results.scenarios if s.scenario_id.startswith("TH_BOUNDS_")]

        assert len(bounds_scenarios) >= 3, "Should have threshold bounds scenarios"

        # All threshold bounds scenarios should pass (INV-MF-1)
        failed = [s for s in bounds_scenarios if not s.passed]
        assert len(failed) == 0, (
            f"Threshold bounds scenarios failed: {[s.scenario_id for s in failed]}"
        )

    def test_evaluation_behavior_scenarios_pass(self) -> None:
        """Verify evaluation behavior scenarios work correctly."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # Find evaluation behavior scenarios
        eval_scenarios = [s for s in results.scenarios if s.scenario_id.startswith("EVAL_")]

        assert len(eval_scenarios) >= 5, "Should have evaluation behavior scenarios"

        # All should pass
        failed = [s for s in eval_scenarios if not s.passed]
        assert len(failed) == 0, f"Evaluation scenarios failed: {[s.scenario_id for s in failed]}"

    def test_drift_resistance_scenarios_pass(self) -> None:
        """Verify drift resistance scenarios maintain bounded threshold."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # Find drift resistance scenarios
        drift_scenarios = [s for s in results.scenarios if s.scenario_id.startswith("DRIFT_")]

        assert len(drift_scenarios) >= 3, "Should have drift resistance scenarios"

        # All should pass
        failed = [s for s in drift_scenarios if not s.passed]
        assert len(failed) == 0, (
            f"Drift resistance scenarios failed: {[s.scenario_id for s in failed]}"
        )

    def test_ema_stability_scenarios_pass(self) -> None:
        """Verify EMA stability scenarios maintain [0, 1] bounds."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # Find EMA stability scenarios
        ema_scenarios = [s for s in results.scenarios if s.scenario_id.startswith("EMA_")]

        assert len(ema_scenarios) >= 3, "Should have EMA stability scenarios"

        # All should pass
        failed = [s for s in ema_scenarios if not s.passed]
        assert len(failed) == 0, (
            f"EMA stability scenarios failed: {[s.scenario_id for s in failed]}"
        )

    @pytest.mark.slow
    def test_full_pass_rate(self) -> None:
        """Verify overall pass rate meets minimum threshold."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # We expect 100% pass rate for well-designed invariant scenarios
        assert results.pass_rate >= 90.0, f"Pass rate too low: {results.pass_rate}%"


class TestMoralFilterEvalProperties:
    """Tests for specific MoralFilterV2 properties via eval framework."""

    def test_inv_mf_1_threshold_always_bounded(self) -> None:
        """INV-MF-1: Threshold is always in [0.30, 0.90]."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # Check properties containing threshold bounds
        for scenario_result in results.scenarios:
            for prop, passed in scenario_result.properties_results.items():
                if "threshold >=" in prop or "threshold <=" in prop:
                    if "0.30" in prop or "0.90" in prop:
                        if not passed:
                            actual = scenario_result.actual_values.get("threshold", "N/A")
                            pytest.fail(
                                f"INV-MF-1 violated in {scenario_result.scenario_id}: "
                                f"threshold={actual}, property={prop}"
                            )

    def test_inv_mf_2_adaptation_bounded(self) -> None:
        """INV-MF-2: Single adaptation step changes threshold by at most 0.05."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        # Find adaptation bounded scenarios
        adapt_scenarios = [
            s for s in results.scenarios if "delta_threshold" in str(s.properties_results)
        ]

        for scenario in adapt_scenarios:
            for prop, passed in scenario.properties_results.items():
                if "delta_threshold" in prop and not passed:
                    delta = scenario.actual_values.get("delta_threshold", "N/A")
                    pytest.fail(
                        f"INV-MF-2 violated in {scenario.scenario_id}: delta_threshold={delta}"
                    )

    def test_inv_mf_3_ema_bounded(self) -> None:
        """INV-MF-3: EMA is always in [0.0, 1.0]."""
        from evals.moral_filter_runner import MoralFilterEvalRunner

        runner = MoralFilterEvalRunner()
        results = runner.run()

        for scenario_result in results.scenarios:
            ema = scenario_result.actual_values.get("ema")
            if ema is not None:
                assert 0.0 <= ema <= 1.0, (
                    f"INV-MF-3 violated in {scenario_result.scenario_id}: ema={ema}"
                )
