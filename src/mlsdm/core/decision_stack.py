"""Decision stack orchestration for risk gating, policy/action selection, and learning.

Order enforced:
    Risk Gate → Policy/Action Selection → Learning Update

This orchestrator ensures SafetyControlContour decisions are applied before any
downstream action selection (routers, iteration loops) or learning updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from mlsdm.risk import RiskAssessment, RiskDirective, RiskInputSignals, SafetyControlContour

ActionT = TypeVar("ActionT")
LearningT = TypeVar("LearningT")


@dataclass(frozen=True)
class DecisionStackResult(Generic[ActionT, LearningT]):
    """Outcome from executing the decision stack."""

    assessment: RiskAssessment
    directive: RiskDirective
    action: ActionT | None
    learning_update: LearningT | None
    blocked: bool


class DecisionStack(Generic[ActionT, LearningT]):
    """Orchestrates risk gating ahead of policy/action and learning steps."""

    def __init__(self, safety_contour: SafetyControlContour | None = None) -> None:
        self._safety_contour = safety_contour or SafetyControlContour()

    def assess_and_decide(self, signals: RiskInputSignals) -> tuple[RiskAssessment, RiskDirective]:
        """Run the risk contour and return assessment + directive."""
        assessment = self._safety_contour.assess(signals)
        directive = self._safety_contour.decide(assessment)
        return assessment, directive

    def apply(
        self,
        *,
        assessment: RiskAssessment,
        directive: RiskDirective,
        policy_action: Callable[[RiskDirective], ActionT],
        learning_update: Callable[[ActionT, RiskDirective], LearningT] | None = None,
    ) -> DecisionStackResult[ActionT, LearningT]:
        """Apply the decision stack using a precomputed assessment/directive."""
        if not directive.allow_execution:
            return DecisionStackResult(
                assessment=assessment,
                directive=directive,
                action=None,
                learning_update=None,
                blocked=True,
            )

        action_result = policy_action(directive)
        learning_result = learning_update(action_result, directive) if learning_update else None

        return DecisionStackResult(
            assessment=assessment,
            directive=directive,
            action=action_result,
            learning_update=learning_result,
            blocked=False,
        )

    def orchestrate(
        self,
        *,
        risk_signals: RiskInputSignals,
        policy_action: Callable[[RiskDirective], ActionT],
        learning_update: Callable[[ActionT, RiskDirective], LearningT] | None = None,
    ) -> DecisionStackResult[ActionT, LearningT]:
        """Run the full stack from risk gating through action and learning."""
        assessment, directive = self.assess_and_decide(risk_signals)
        return self.apply(
            assessment=assessment,
            directive=directive,
            policy_action=policy_action,
            learning_update=learning_update,
        )
