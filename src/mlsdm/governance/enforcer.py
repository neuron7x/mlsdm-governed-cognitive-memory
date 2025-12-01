"""
MLSDM Governance Enforcer Module.

This module implements the core governance enforcement logic for MLSDM,
evaluating input/output payloads against configured policies and rules.

Architecture:
    1. Policy loader reads policy.yaml configuration
    2. Mode selector determines operational mode from context
    3. Rule evaluator matches signals against rule triggers
    4. Decision builder creates actionable governance decisions
    5. Action applier modifies/blocks/allows content based on decisions

Integration:
    The enforcer is called in the NeuroCognitiveEngine pipeline:
    - Before LLM: evaluate(input_payload, None, context)
    - After LLM: evaluate(input_payload, output_payload, context)
    - Apply: apply_decision(decision, output_payload)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

POLICY_FILE = Path(__file__).parent / "policy.yaml"

ActionType = Literal["allow", "block", "modify", "escalate"]
LogLevel = Literal["debug", "info", "warning", "error"]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GovernanceDecision:
    """Result of governance evaluation.

    Attributes:
        action: The action to take ("allow", "block", "modify", "escalate")
        reason: Human-readable explanation of the decision
        rule_id: ID of the rule that triggered this decision (if any)
        mode: The operational mode used for evaluation
        metadata: Additional data about the decision (scores, signals, etc.)
    """

    action: ActionType
    reason: str
    rule_id: str | None
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernanceContext:
    """Context for governance evaluation.

    Attributes:
        mode: Override mode (if None, mode is auto-selected)
        risk_level: Risk level signal (0.0 - 1.0)
        sensitive_domain: Whether the request is in a sensitive domain
        user_type: Type of user making the request
        correlation_id: Request correlation ID for logging
        additional: Any additional context signals
    """

    mode: str | None = None
    risk_level: float = 0.0
    sensitive_domain: bool = False
    user_type: str = "authenticated"
    correlation_id: str | None = None
    additional: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Policy Loader
# =============================================================================


class PolicyLoader:
    """Loads and caches governance policy configuration."""

    _instance: PolicyLoader | None = None
    _policy: dict[str, Any] | None = None

    def __new__(cls) -> PolicyLoader:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset singleton state for testing.

        This method is provided for test fixtures to reset the singleton
        between tests, ensuring test isolation.
        """
        cls._instance = None
        cls._policy = None

    def load(self, policy_path: Path | None = None) -> dict[str, Any]:
        """Load policy configuration from YAML file.

        Args:
            policy_path: Path to policy file (default: policy.yaml in module dir)

        Returns:
            Parsed policy configuration dictionary
        """
        if self._policy is not None:
            return self._policy

        path = policy_path or POLICY_FILE

        if not path.exists():
            logger.warning("Policy file not found at %s, using empty policy", path)
            self._policy = {"modes": {}, "rules": [], "defaults": {}}
            return self._policy

        with open(path, encoding="utf-8") as f:
            self._policy = yaml.safe_load(f)

        logger.info("Loaded governance policy from %s", path)
        return self._policy

    def reload(self, policy_path: Path | None = None) -> dict[str, Any]:
        """Force reload of policy configuration.

        Args:
            policy_path: Path to policy file

        Returns:
            Parsed policy configuration dictionary
        """
        self._policy = None
        return self.load(policy_path)


# =============================================================================
# Signal Extractor
# =============================================================================


class SignalExtractor:
    """Extracts governance signals from input/output payloads."""

    # PII pattern matchers
    _PII_PATTERNS = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    }

    @classmethod
    def extract_pii_signals(
        cls, text: str
    ) -> dict[str, bool]:
        """Check text for PII patterns.

        Args:
            text: Text to scan for PII

        Returns:
            Dict mapping pattern names to boolean detection status
        """
        results = {}
        for name, pattern in cls._PII_PATTERNS.items():
            results[name] = bool(pattern.search(text))
        return results

    @classmethod
    def has_pii(cls, text: str) -> bool:
        """Check if text contains any PII.

        Args:
            text: Text to scan

        Returns:
            True if any PII pattern is detected
        """
        return any(cls.extract_pii_signals(text).values())

    @classmethod
    def extract_signal(
        cls,
        signal_config: dict[str, Any],
        input_payload: dict[str, Any],
        output_payload: dict[str, Any] | None,
    ) -> Any:
        """Extract a signal value based on configuration.

        Args:
            signal_config: Signal configuration from policy
            input_payload: Input request payload
            output_payload: Output response payload (may be None)

        Returns:
            Extracted signal value or default
        """
        source = signal_config.get("source", "input")
        path = signal_config.get("path", "")
        default = signal_config.get("default")

        # Select source payload
        if source == "input":
            payload = input_payload
        elif source == "output":
            payload = output_payload if output_payload else {}
        elif source == "computed":
            # Handle computed signals
            extractor = signal_config.get("extractor")
            if extractor == "pii_pattern_matcher":
                # Check both input and output for PII
                input_text = input_payload.get("prompt", "")
                output_text = (
                    output_payload.get("response", "")
                    if output_payload
                    else ""
                )
                return cls.has_pii(input_text) or cls.has_pii(output_text)
            return default
        else:
            payload = {}

        # Navigate path
        value = payload
        for key in path.split("."):
            if not key:
                continue
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return default

        return value if value is not None else default


# =============================================================================
# Mode Selector
# =============================================================================


class ModeSelector:
    """Selects operational mode based on context signals."""

    def __init__(self, policy: dict[str, Any]):
        self._policy = policy

    def select_mode(self, context: GovernanceContext) -> str:
        """Select operational mode based on context.

        Args:
            context: Governance context with signals

        Returns:
            Selected mode name ("normal", "cautious", "emergency")
        """
        # If mode is explicitly specified in context, use it
        if context.mode is not None:
            modes = self._policy.get("modes", {})
            if context.mode in modes:
                return context.mode
            logger.warning("Invalid mode '%s' in context, using default", context.mode)

        # Evaluate mode selection rules
        mode_rules = self._policy.get("mode_selection", [])
        context_dict = {
            "risk_level": context.risk_level,
            "sensitive_domain": context.sensitive_domain,
            "user_type": context.user_type,
            **context.additional,
        }

        for rule in mode_rules:
            condition = rule.get("condition", {})
            if self._evaluate_condition(condition, context_dict):
                return rule.get("mode", "normal")

        # Fall back to default mode
        return self._policy.get("defaults", {}).get("mode", "normal")

    def _evaluate_condition(
        self, condition: dict[str, Any], context_dict: dict[str, Any]
    ) -> bool:
        """Evaluate a mode selection condition.

        Args:
            condition: Condition configuration
            context_dict: Context values

        Returns:
            True if condition is satisfied
        """
        context_key = condition.get("context_key")
        operator = condition.get("operator")
        expected = condition.get("value")

        if context_key is None or operator is None:
            return False

        actual = context_dict.get(context_key)
        if actual is None:
            return False

        return self._compare(actual, operator, expected)

    def _compare(self, actual: Any, operator: str, expected: Any) -> bool:
        """Compare values based on operator.

        Args:
            actual: Actual value
            operator: Comparison operator
            expected: Expected value

        Returns:
            Comparison result
        """
        ops: dict[str, Any] = {
            "eq": lambda a, e: a == e,
            "neq": lambda a, e: a != e,
            "gt": lambda a, e: a > e,
            "gte": lambda a, e: a >= e,
            "lt": lambda a, e: a < e,
            "lte": lambda a, e: a <= e,
            "in": lambda a, e: a in e,
            "not_in": lambda a, e: a not in e,
        }
        return ops.get(operator, lambda a, e: False)(actual, expected)


# =============================================================================
# Rule Evaluator
# =============================================================================


class RuleEvaluator:
    """Evaluates governance rules against signals."""

    def __init__(self, policy: dict[str, Any]):
        self._policy = policy

    def evaluate_rules(
        self,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any] | None,
        mode: str,
    ) -> tuple[str | None, str, dict[str, Any]]:
        """Evaluate all rules and return first matching rule's action.

        Args:
            input_payload: Input request payload
            output_payload: Output response payload (may be None)
            mode: Current operational mode

        Returns:
            Tuple of (rule_id, action, metadata) for matching rule
        """
        rules = self._policy.get("rules", [])
        mode_config = self._policy.get("modes", {}).get(mode, {})

        # Sort rules by priority (higher first)
        sorted_rules = sorted(
            [r for r in rules if r.get("enabled", True)],
            key=lambda r: r.get("priority", 0),
            reverse=True,
        )

        # Collect signals
        signals = self._collect_signals(input_payload, output_payload, mode_config)

        for rule in sorted_rules:
            trigger = rule.get("trigger", {})
            if self._evaluate_trigger(trigger, signals, mode, mode_config):
                action = rule.get("action", "allow")
                metadata = {
                    "log_level": rule.get("log", "info"),
                    "response_message": rule.get("response_message"),
                    "modification": rule.get("modification"),
                    "escalation": rule.get("escalation"),
                    "signals": signals,
                }
                return rule.get("id"), action, metadata

        # No rule matched - use default action
        default_action = self._policy.get("defaults", {}).get("fallback_action", "allow")
        return None, default_action, {"signals": signals}

    def _collect_signals(
        self,
        input_payload: dict[str, Any],
        output_payload: dict[str, Any] | None,
        mode_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Collect all signals for rule evaluation.

        Args:
            input_payload: Input request payload
            output_payload: Output response payload
            mode_config: Current mode configuration

        Returns:
            Dictionary of signal names to values
        """
        signals: dict[str, Any] = {}
        signal_configs = self._policy.get("signals", {})

        for signal_name, signal_config in signal_configs.items():
            signals[signal_name] = SignalExtractor.extract_signal(
                signal_config, input_payload, output_payload
            )

        # Add mode-specific thresholds
        signals["mode_moral_threshold"] = mode_config.get("moral_threshold", 0.5)
        signals["mode_uncertainty_threshold"] = mode_config.get(
            "uncertainty_escalation_threshold", 0.7
        )

        return signals

    def _evaluate_trigger(
        self,
        trigger: dict[str, Any],
        signals: dict[str, Any],
        mode: str,
        mode_config: dict[str, Any],
    ) -> bool:
        """Evaluate a rule trigger against collected signals.

        Args:
            trigger: Trigger configuration
            signals: Collected signal values
            mode: Current mode name
            mode_config: Current mode configuration

        Returns:
            True if trigger condition is satisfied
        """
        trigger_type = trigger.get("type", "default")

        if trigger_type == "default":
            return True

        elif trigger_type == "pattern":
            # Check if any of the specified signals are present
            trigger_signals = trigger.get("signals", [])
            return any(signals.get(sig) for sig in trigger_signals)

        elif trigger_type == "threshold":
            signal_name = trigger.get("signal")
            operator = trigger.get("operator", "gte")

            # Get threshold - either from trigger or from mode
            if trigger.get("mode_threshold"):
                threshold = mode_config.get("moral_threshold", 0.5)
            else:
                threshold = trigger.get("value", 0.0)

            actual = signals.get(signal_name, 0.0)
            return self._compare(actual, operator, threshold)

        elif trigger_type == "compound":
            # All conditions must be true
            conditions = trigger.get("conditions", [])
            for cond in conditions:
                signal_name = cond.get("signal")
                context_key = cond.get("context_key")
                operator = cond.get("operator", "eq")
                expected = cond.get("value")

                if signal_name:
                    actual = signals.get(signal_name)
                elif context_key:
                    actual = mode if context_key == "mode" else signals.get(context_key)
                else:
                    continue

                if not self._compare(actual, operator, expected):
                    return False
            return True

        return False

    def _compare(self, actual: Any, operator: str, expected: Any) -> bool:
        """Compare values based on operator.

        Args:
            actual: Actual value
            operator: Comparison operator
            expected: Expected value

        Returns:
            Comparison result
        """
        if actual is None:
            return False

        ops: dict[str, Any] = {
            "eq": lambda a, e: a == e,
            "neq": lambda a, e: a != e,
            "gt": lambda a, e: a > e,
            "gte": lambda a, e: a >= e,
            "lt": lambda a, e: a < e,
            "lte": lambda a, e: a <= e,
            "in": lambda a, e: a in e,
            "not_in": lambda a, e: a not in e,
        }
        return ops.get(operator, lambda a, e: False)(actual, expected)


# =============================================================================
# Public API
# =============================================================================


def evaluate(
    input_payload: dict[str, Any],
    output_payload: dict[str, Any] | None,
    context: dict[str, Any] | GovernanceContext,
    policy_path: Path | None = None,
) -> GovernanceDecision:
    """Evaluate governance rules against input/output payloads.

    This is the main entry point for governance evaluation. Call this:
    - Before LLM generation with output_payload=None for input validation
    - After LLM generation with full output for response validation

    Args:
        input_payload: Input request containing prompt, moral_value, etc.
        output_payload: Output response containing response text, metadata, etc.
        context: Governance context (dict or GovernanceContext object)
        policy_path: Optional path to policy file (for testing)

    Returns:
        GovernanceDecision with action, reason, and metadata

    Example:
        >>> decision = evaluate(
        ...     {"prompt": "Hello", "moral_value": 0.5},
        ...     {"response": "Hi there!", "metadata": {"toxicity": 0.1}},
        ...     {"risk_level": 0.3}
        ... )
        >>> if decision.action == "block":
        ...     return {"error": decision.reason}
    """
    # Convert context dict to GovernanceContext if needed
    if isinstance(context, dict):
        context = GovernanceContext(
            mode=context.get("mode"),
            risk_level=context.get("risk_level", 0.0),
            sensitive_domain=context.get("sensitive_domain", False),
            user_type=context.get("user_type", "authenticated"),
            correlation_id=context.get("correlation_id"),
            additional={
                k: v
                for k, v in context.items()
                if k
                not in ("mode", "risk_level", "sensitive_domain", "user_type", "correlation_id")
            },
        )

    # Load policy
    loader = PolicyLoader()
    policy = loader.load(policy_path)

    # Select mode
    mode_selector = ModeSelector(policy)
    mode = mode_selector.select_mode(context)

    # Evaluate rules
    rule_evaluator = RuleEvaluator(policy)
    rule_id, action, metadata = rule_evaluator.evaluate_rules(
        input_payload, output_payload, mode
    )

    # Build reason
    if rule_id:
        rule = next(
            (r for r in policy.get("rules", []) if r.get("id") == rule_id),
            None,
        )
        reason = (
            rule.get("description", f"Rule {rule_id} triggered")
            if rule
            else f"Rule {rule_id} triggered"
        )
        if metadata.get("response_message"):
            reason = metadata["response_message"]
    else:
        reason = "No rules matched - default action applied"

    # Validate action is a valid ActionType
    valid_actions = ("allow", "block", "modify", "escalate")
    if action not in valid_actions:
        logger.warning("Invalid action '%s', defaulting to 'block'", action)
        action = "block"

    # Log decision
    log_level = metadata.get("log_level", "info")
    log_func = getattr(logger, log_level, logger.info)
    log_func(
        "Governance decision: action=%s rule_id=%s mode=%s reason=%s",
        action,
        rule_id,
        mode,
        reason,
    )

    return GovernanceDecision(
        action=action,  # type: ignore[arg-type]  # Validated above
        reason=reason,
        rule_id=rule_id,
        mode=mode,
        metadata=metadata,
    )


def apply_decision(
    decision: GovernanceDecision,
    output_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Apply a governance decision to the output payload.

    This function modifies the output based on the decision action:
    - "allow": Return output unchanged
    - "block": Return None or error response
    - "modify": Append disclaimer or modify content
    - "escalate": Mark for review, may still return output

    Args:
        decision: Governance decision from evaluate()
        output_payload: Output payload to potentially modify

    Returns:
        Modified output payload, None for blocked content, or original for allowed

    Example:
        >>> decision = evaluate(input_payload, output_payload, context)
        >>> output_payload = apply_decision(decision, output_payload)
        >>> if output_payload is None:
        ...     return {"error": "Content blocked"}
    """
    if output_payload is None:
        return None

    action = decision.action

    if action == "allow":
        return output_payload

    elif action == "block":
        # Return None to signal blocked content
        logger.warning(
            "Content blocked: rule_id=%s reason=%s",
            decision.rule_id,
            decision.reason,
        )
        return None

    elif action == "modify":
        # Apply modification
        modification = decision.metadata.get("modification", {})
        mod_type = modification.get("type")

        if mod_type == "append_disclaimer":
            disclaimer = modification.get("disclaimer", "")
            response = output_payload.get("response", "")
            output_payload["response"] = f"{response}\n\n{disclaimer}"
            output_payload.setdefault("metadata", {})["modified"] = True
            output_payload["metadata"]["disclaimer_added"] = True

        return output_payload

    elif action == "escalate":
        # Mark for escalation but still return output
        escalation = decision.metadata.get("escalation", {})
        output_payload.setdefault("metadata", {})["escalated"] = True
        output_payload["metadata"]["escalation_channel"] = escalation.get("channel")
        output_payload["metadata"]["escalation_priority"] = escalation.get("priority")

        logger.warning(
            "Content escalated: rule_id=%s channel=%s priority=%s",
            decision.rule_id,
            escalation.get("channel"),
            escalation.get("priority"),
        )

        return output_payload

    return output_payload


def get_current_mode(context: dict[str, Any] | GovernanceContext) -> str:
    """Get the current operational mode based on context.

    Args:
        context: Governance context

    Returns:
        Current mode name
    """
    if isinstance(context, dict):
        context = GovernanceContext(**context)

    loader = PolicyLoader()
    policy = loader.load()
    mode_selector = ModeSelector(policy)
    return mode_selector.select_mode(context)


def reload_policy(policy_path: Path | None = None) -> dict[str, Any]:
    """Force reload of policy configuration.

    Useful for hot-reloading policy changes without restart.

    Args:
        policy_path: Optional path to policy file

    Returns:
        Reloaded policy configuration
    """
    loader = PolicyLoader()
    return loader.reload(policy_path)
