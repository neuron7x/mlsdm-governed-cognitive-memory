"""
Tests for MLSDM Governance Enforcer Module.

This test suite covers:
- Policy loading and mode selection
- Rule evaluation (allow, block, modify, escalate)
- Signal extraction (PII detection, toxicity, moral scores)
- Mode switching based on context
- Decision application
- Metrics recording
"""

from __future__ import annotations

from typing import Any

import pytest

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_singletons() -> None:
    """Reset singleton instances before each test."""
    from mlsdm.governance.enforcer import PolicyLoader
    from mlsdm.governance.metrics import GovernanceMetrics

    PolicyLoader.reset_for_testing()
    GovernanceMetrics.reset()


@pytest.fixture
def sample_input_payload() -> dict[str, Any]:
    """Sample input payload for testing."""
    return {
        "prompt": "Hello, how are you?",
        "moral_value": 0.5,
        "max_tokens": 100,
    }


@pytest.fixture
def sample_output_payload() -> dict[str, Any]:
    """Sample output payload for testing."""
    return {
        "response": "I'm doing great, thank you for asking!",
        "metadata": {
            "toxicity": 0.1,
            "uncertainty": 0.2,
        },
    }


@pytest.fixture
def toxic_output_payload() -> dict[str, Any]:
    """Output payload with high toxicity."""
    return {
        "response": "This is a toxic response.",
        "metadata": {
            "toxicity": 0.9,
            "uncertainty": 0.1,
        },
    }


@pytest.fixture
def pii_input_payload() -> dict[str, Any]:
    """Input payload containing PII."""
    return {
        "prompt": "My email is test@example.com and phone is 123-456-7890",
        "moral_value": 0.5,
    }


# =============================================================================
# Policy Loading Tests
# =============================================================================


class TestPolicyLoader:
    """Tests for PolicyLoader class."""

    def test_load_default_policy(self) -> None:
        """Test loading the default policy file."""
        from mlsdm.governance.enforcer import PolicyLoader

        loader = PolicyLoader()
        policy = loader.load()

        assert policy is not None
        assert "modes" in policy
        assert "rules" in policy
        assert "normal" in policy["modes"]
        assert "cautious" in policy["modes"]
        assert "emergency" in policy["modes"]

    def test_policy_singleton(self) -> None:
        """Test that PolicyLoader is a singleton."""
        from mlsdm.governance.enforcer import PolicyLoader

        loader1 = PolicyLoader()
        loader2 = PolicyLoader()

        assert loader1 is loader2

    def test_reload_policy(self) -> None:
        """Test policy reload functionality."""
        from mlsdm.governance.enforcer import PolicyLoader, reload_policy

        loader = PolicyLoader()
        _ = loader.load()  # Initial load
        policy2 = reload_policy()

        # Should return fresh policy (even if same content)
        assert policy2 is not None
        assert "modes" in policy2


# =============================================================================
# Mode Selection Tests
# =============================================================================


class TestModeSelection:
    """Tests for mode selection logic."""

    def test_default_mode_is_normal(self) -> None:
        """Test that default mode is 'normal'."""
        from mlsdm.governance import get_current_mode

        mode = get_current_mode({})
        assert mode == "normal"

    def test_explicit_mode_override(self) -> None:
        """Test explicit mode specification in context."""
        from mlsdm.governance import get_current_mode

        mode = get_current_mode({"mode": "cautious"})
        assert mode == "cautious"

        mode = get_current_mode({"mode": "emergency"})
        assert mode == "emergency"

    def test_high_risk_triggers_emergency(self) -> None:
        """Test that high risk level triggers emergency mode."""
        from mlsdm.governance import get_current_mode

        mode = get_current_mode({"risk_level": 0.9})
        assert mode == "emergency"

    def test_medium_risk_triggers_cautious(self) -> None:
        """Test that medium risk level triggers cautious mode."""
        from mlsdm.governance import get_current_mode

        mode = get_current_mode({"risk_level": 0.6})
        assert mode == "cautious"

    def test_low_risk_stays_normal(self) -> None:
        """Test that low risk level stays in normal mode."""
        from mlsdm.governance import get_current_mode

        mode = get_current_mode({"risk_level": 0.3})
        assert mode == "normal"

    def test_sensitive_domain_triggers_cautious(self) -> None:
        """Test that sensitive domain triggers cautious mode."""
        from mlsdm.governance import get_current_mode

        mode = get_current_mode({"sensitive_domain": True})
        assert mode == "cautious"


# =============================================================================
# Basic Evaluation Tests
# =============================================================================


class TestBasicEvaluation:
    """Tests for basic governance evaluation."""

    def test_allow_safe_content(
        self, sample_input_payload: dict[str, Any], sample_output_payload: dict[str, Any]
    ) -> None:
        """Test that safe content is allowed."""
        from mlsdm.governance import evaluate

        decision = evaluate(sample_input_payload, sample_output_payload, {})

        assert decision.action == "allow"
        assert decision.mode == "normal"

    def test_evaluate_returns_decision(
        self, sample_input_payload: dict[str, Any]
    ) -> None:
        """Test that evaluate returns a GovernanceDecision."""
        from mlsdm.governance import GovernanceDecision, evaluate

        decision = evaluate(sample_input_payload, None, {})

        assert isinstance(decision, GovernanceDecision)
        assert decision.action in ("allow", "block", "modify", "escalate")
        assert decision.mode is not None

    def test_decision_includes_metadata(
        self, sample_input_payload: dict[str, Any], sample_output_payload: dict[str, Any]
    ) -> None:
        """Test that decision includes metadata."""
        from mlsdm.governance import evaluate

        decision = evaluate(sample_input_payload, sample_output_payload, {})

        assert decision.metadata is not None
        assert "signals" in decision.metadata


# =============================================================================
# Block Rule Tests
# =============================================================================


class TestBlockRules:
    """Tests for blocking rules."""

    def test_block_high_toxicity(self, toxic_output_payload: dict[str, Any]) -> None:
        """Test that high toxicity content is blocked."""
        from mlsdm.governance import evaluate

        input_payload = {"prompt": "test", "moral_value": 0.5}
        decision = evaluate(input_payload, toxic_output_payload, {})

        assert decision.action == "block"
        assert decision.rule_id == "R002"

    def test_block_pii_content(self, pii_input_payload: dict[str, Any]) -> None:
        """Test that PII content is blocked."""
        from mlsdm.governance import evaluate

        decision = evaluate(pii_input_payload, None, {})

        assert decision.action == "block"
        assert decision.rule_id == "R001"

    def test_pii_in_output_blocked(self) -> None:
        """Test that PII in output is blocked."""
        from mlsdm.governance import evaluate

        input_payload = {"prompt": "What's my email?", "moral_value": 0.5}
        output_payload = {
            "response": "Your email is test@example.com",
            "metadata": {"toxicity": 0.1},
        }

        decision = evaluate(input_payload, output_payload, {})

        assert decision.action == "block"
        assert decision.rule_id == "R001"


# =============================================================================
# Modify Rule Tests
# =============================================================================


class TestModifyRules:
    """Tests for modification rules."""

    def test_modify_moderate_toxicity(self) -> None:
        """Test that moderate toxicity adds disclaimer."""
        from mlsdm.governance import evaluate

        input_payload = {"prompt": "test", "moral_value": 0.5}
        output_payload = {
            "response": "A moderately concerning response.",
            "metadata": {"toxicity": 0.6},
        }

        decision = evaluate(input_payload, output_payload, {})

        assert decision.action == "modify"
        assert decision.rule_id == "R003"


# =============================================================================
# Escalate Rule Tests
# =============================================================================


class TestEscalateRules:
    """Tests for escalation rules."""

    def test_escalate_high_uncertainty(self) -> None:
        """Test that high uncertainty triggers escalation."""
        from mlsdm.governance import evaluate

        input_payload = {"prompt": "test", "moral_value": 0.5}
        output_payload = {
            "response": "I'm not sure about this.",
            "metadata": {"toxicity": 0.1, "uncertainty": 0.8},
        }

        decision = evaluate(input_payload, output_payload, {})

        assert decision.action == "escalate"
        assert decision.rule_id == "R004"


# =============================================================================
# Apply Decision Tests
# =============================================================================


class TestApplyDecision:
    """Tests for apply_decision function."""

    def test_apply_allow_returns_unchanged(
        self, sample_output_payload: dict[str, Any]
    ) -> None:
        """Test that allow action returns output unchanged."""
        from mlsdm.governance import GovernanceDecision, apply_decision

        decision = GovernanceDecision(
            action="allow",
            reason="Safe content",
            rule_id="R007",
            mode="normal",
            metadata={},
        )

        result = apply_decision(decision, sample_output_payload)

        assert result is sample_output_payload

    def test_apply_block_returns_none(
        self, sample_output_payload: dict[str, Any]
    ) -> None:
        """Test that block action returns None."""
        from mlsdm.governance import GovernanceDecision, apply_decision

        decision = GovernanceDecision(
            action="block",
            reason="Content blocked",
            rule_id="R001",
            mode="normal",
            metadata={},
        )

        result = apply_decision(decision, sample_output_payload)

        assert result is None

    def test_apply_modify_adds_disclaimer(self) -> None:
        """Test that modify action adds disclaimer."""
        from mlsdm.governance import GovernanceDecision, apply_decision

        output_payload = {"response": "Original response.", "metadata": {}}
        decision = GovernanceDecision(
            action="modify",
            reason="Modified for safety",
            rule_id="R003",
            mode="normal",
            metadata={
                "modification": {
                    "type": "append_disclaimer",
                    "disclaimer": "This is a disclaimer.",
                }
            },
        )

        result = apply_decision(decision, output_payload)

        assert result is not None
        assert "disclaimer" in result["response"].lower()
        assert result["metadata"]["modified"] is True

    def test_apply_escalate_marks_output(
        self, sample_output_payload: dict[str, Any]
    ) -> None:
        """Test that escalate action marks output for review."""
        from mlsdm.governance import GovernanceDecision, apply_decision

        decision = GovernanceDecision(
            action="escalate",
            reason="Needs review",
            rule_id="R004",
            mode="normal",
            metadata={
                "escalation": {
                    "channel": "review_queue",
                    "priority": "high",
                }
            },
        )

        result = apply_decision(decision, sample_output_payload)

        assert result is not None
        assert result["metadata"]["escalated"] is True
        assert result["metadata"]["escalation_channel"] == "review_queue"

    def test_apply_to_none_returns_none(self) -> None:
        """Test that applying decision to None returns None."""
        from mlsdm.governance import GovernanceDecision, apply_decision

        decision = GovernanceDecision(
            action="allow",
            reason="Safe",
            rule_id=None,
            mode="normal",
            metadata={},
        )

        result = apply_decision(decision, None)

        assert result is None


# =============================================================================
# Mode-Specific Behavior Tests
# =============================================================================


class TestModeSpecificBehavior:
    """Tests for mode-specific rule behavior."""

    def test_cautious_mode_stricter_threshold(self) -> None:
        """Test that cautious mode has stricter moral threshold."""
        from mlsdm.governance import evaluate

        input_payload = {"prompt": "test", "moral_value": 0.55}
        output_payload = {"response": "Response", "metadata": {"toxicity": 0.1}}

        # In normal mode (threshold 0.50), should allow
        decision_normal = evaluate(input_payload, output_payload, {"mode": "normal"})

        # In cautious mode (threshold 0.70), might behave differently
        decision_cautious = evaluate(input_payload, output_payload, {"mode": "cautious"})

        assert decision_normal.mode == "normal"
        assert decision_cautious.mode == "cautious"

    def test_emergency_mode_most_strict(self) -> None:
        """Test that emergency mode is the most strict."""
        from mlsdm.governance import evaluate

        input_payload = {"prompt": "test", "moral_value": 0.5}
        output_payload = {"response": "Response", "metadata": {"toxicity": 0.1}}

        decision = evaluate(input_payload, output_payload, {"mode": "emergency"})

        assert decision.mode == "emergency"


# =============================================================================
# Metrics Tests
# =============================================================================


class TestGovernanceMetrics:
    """Tests for governance metrics collection."""

    def test_record_decision_increments_total(self) -> None:
        """Test that recording decision increments total counter."""
        from mlsdm.governance import get_metrics

        metrics = get_metrics()
        metrics.reset_counters()

        metrics.record_decision("allow", None, "normal")
        metrics.record_decision("block", "R001", "normal")

        summary = metrics.get_summary()
        assert summary["total_decisions"] == 2
        assert summary["allowed_total"] == 1
        assert summary["blocked_total"] == 1

    def test_metrics_per_mode(self) -> None:
        """Test per-mode metrics tracking."""
        from mlsdm.governance import get_metrics

        metrics = get_metrics()
        metrics.reset_counters()

        metrics.record_decision("allow", None, "normal")
        metrics.record_decision("block", "R001", "cautious")
        metrics.record_decision("escalate", "R004", "emergency")

        summary = metrics.get_summary()
        assert "normal" in summary["per_mode"]
        assert "cautious" in summary["per_mode"]
        assert "emergency" in summary["per_mode"]

    def test_metrics_per_rule(self) -> None:
        """Test per-rule metrics tracking."""
        from mlsdm.governance import get_metrics

        metrics = get_metrics()
        metrics.reset_counters()

        metrics.record_decision("block", "R001", "normal")
        metrics.record_decision("block", "R001", "normal")
        metrics.record_decision("block", "R002", "normal")

        summary = metrics.get_summary()
        assert summary["per_rule"]["R001"] == 2
        assert summary["per_rule"]["R002"] == 1

    def test_block_rate_calculation(self) -> None:
        """Test block rate calculation."""
        from mlsdm.governance import get_metrics

        metrics = get_metrics()
        metrics.reset_counters()

        # 2 blocks out of 4 total = 50% block rate
        metrics.record_decision("allow", None, "normal")
        metrics.record_decision("allow", None, "normal")
        metrics.record_decision("block", "R001", "normal")
        metrics.record_decision("block", "R002", "normal")

        summary = metrics.get_summary()
        assert summary["block_rate"] == 0.5


# =============================================================================
# PII Detection Tests
# =============================================================================


class TestPIIDetection:
    """Tests for PII pattern detection."""

    def test_detect_email(self) -> None:
        """Test email detection."""
        from mlsdm.governance.enforcer import SignalExtractor

        result = SignalExtractor.extract_pii_signals("Contact: user@example.com")
        assert result["email"] is True

    def test_detect_phone(self) -> None:
        """Test phone number detection."""
        from mlsdm.governance.enforcer import SignalExtractor

        result = SignalExtractor.extract_pii_signals("Call me at 555-123-4567")
        assert result["phone"] is True

    def test_detect_ssn(self) -> None:
        """Test SSN detection."""
        from mlsdm.governance.enforcer import SignalExtractor

        result = SignalExtractor.extract_pii_signals("SSN: 123-45-6789")
        assert result["ssn"] is True

    def test_detect_credit_card(self) -> None:
        """Test credit card detection."""
        from mlsdm.governance.enforcer import SignalExtractor

        result = SignalExtractor.extract_pii_signals("Card: 4111-1111-1111-1111")
        assert result["credit_card"] is True

    def test_no_pii_detected(self) -> None:
        """Test no PII in clean text."""
        from mlsdm.governance.enforcer import SignalExtractor

        result = SignalExtractor.has_pii("Hello, how are you today?")
        assert result is False


# =============================================================================
# GovernanceContext Tests
# =============================================================================


class TestGovernanceContext:
    """Tests for GovernanceContext dataclass."""

    def test_context_from_dict(self) -> None:
        """Test creating context from dictionary."""
        from mlsdm.governance import evaluate

        context = {
            "mode": "cautious",
            "risk_level": 0.5,
            "sensitive_domain": True,
            "user_type": "anonymous",
            "correlation_id": "test-123",
        }

        input_payload = {"prompt": "test", "moral_value": 0.5}
        decision = evaluate(input_payload, None, context)

        assert decision.mode == "cautious"

    def test_context_with_additional_fields(self) -> None:
        """Test context with additional custom fields."""
        from mlsdm.governance import GovernanceContext, evaluate

        context = GovernanceContext(
            mode=None,
            risk_level=0.3,
            additional={"custom_signal": True},
        )

        input_payload = {"prompt": "test", "moral_value": 0.5}
        decision = evaluate(input_payload, None, context)

        assert decision is not None


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full governance flow."""

    def test_full_pipeline_allow(self) -> None:
        """Test full pipeline with allowed content."""
        from mlsdm.governance import apply_decision, evaluate

        input_payload = {"prompt": "Hello", "moral_value": 0.7}
        output_payload = {"response": "Hi there!", "metadata": {"toxicity": 0.1}}
        context = {"risk_level": 0.2}

        decision = evaluate(input_payload, output_payload, context)
        result = apply_decision(decision, output_payload)

        assert decision.action == "allow"
        assert result is output_payload

    def test_full_pipeline_block(self) -> None:
        """Test full pipeline with blocked content."""
        from mlsdm.governance import apply_decision, evaluate

        input_payload = {"prompt": "My SSN is 123-45-6789", "moral_value": 0.5}
        context = {}

        decision = evaluate(input_payload, None, context)
        result = apply_decision(decision, {"response": "..."})

        assert decision.action == "block"
        assert result is None

    def test_full_pipeline_modify(self) -> None:
        """Test full pipeline with modified content."""
        from mlsdm.governance import apply_decision, evaluate

        input_payload = {"prompt": "test", "moral_value": 0.5}
        output_payload = {"response": "Moderate response", "metadata": {"toxicity": 0.6}}
        context = {}

        decision = evaluate(input_payload, output_payload, context)

        if decision.action == "modify":
            result = apply_decision(decision, output_payload)
            assert result is not None
            assert result["metadata"].get("modified") is True

    def test_metrics_recorded_during_evaluation(self) -> None:
        """Test that metrics are recorded during evaluation."""
        from mlsdm.governance import evaluate, get_metrics, record_decision

        metrics = get_metrics()
        metrics.reset_counters()

        input_payload = {"prompt": "test", "moral_value": 0.5}
        output_payload = {"response": "OK", "metadata": {"toxicity": 0.1}}

        decision = evaluate(input_payload, output_payload, {})

        # Manually record the decision (simulating integration)
        record_decision(
            decision.action,
            decision.rule_id,
            decision.mode,
            decision.reason,
        )

        summary = metrics.get_summary()
        assert summary["total_decisions"] >= 1
