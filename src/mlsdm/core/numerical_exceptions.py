"""Custom exceptions for numerical stability and validation in MyceliumFractalNet.

This module provides specialized exceptions for numerical computation errors,
including stability violations and value range errors.

Exception Hierarchy:
    NumericalError (base)
    ├── StabilityError - NaN/Inf detection or numerical instability
    └── ValueOutOfRangeError - Values outside physically valid bounds

Example:
    >>> from mlsdm.core.numerical_exceptions import StabilityError, ValueOutOfRangeError
    >>> raise StabilityError("NaN detected in membrane potential at step 1000")
    >>> raise ValueOutOfRangeError("V", -120.0, -90.0, 40.0)

Reference:
    See docs/MATH_MODEL.md Section 5 for stability analysis.
"""

from __future__ import annotations

from typing import Any


class NumericalError(Exception):
    """Base exception for numerical computation errors.

    Attributes:
        message: Human-readable error description.
        step: Integration step where error occurred (optional).
        engine: Name of the engine that raised the error (optional).
        state_snapshot: Partial state information at time of error (optional).
    """

    def __init__(
        self,
        message: str,
        step: int | None = None,
        engine: str | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Initialize NumericalError.

        Args:
            message: Human-readable error description.
            step: Integration step where error occurred.
            engine: Name of the engine (e.g., "MembraneEngine").
            state_snapshot: Partial state for debugging.
        """
        self.message = message
        self.step = step
        self.engine = engine
        self.state_snapshot = state_snapshot or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with context."""
        parts = [self.message]
        if self.engine:
            parts.insert(0, f"[{self.engine}]")
        if self.step is not None:
            parts.append(f"(step={self.step})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for structured logging.

        Returns:
            Dictionary with error details.
        """
        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "step": self.step,
            "engine": self.engine,
            "state_snapshot": self.state_snapshot,
        }


class StabilityError(NumericalError):
    """Exception raised when numerical instability is detected.

    This includes NaN, Inf, or diverging values that indicate
    the numerical scheme has become unstable.

    Example:
        >>> raise StabilityError(
        ...     "NaN detected in state vector",
        ...     step=1000,
        ...     engine="MembraneEngine",
        ...     state_snapshot={"max_abs_value": float("inf")},
        ... )

    Reference:
        See docs/MATH_MODEL.md Section 5.1-5.2 for stability conditions.
    """

    def __init__(
        self,
        message: str = "Numerical instability detected",
        step: int | None = None,
        engine: str | None = None,
        state_snapshot: dict[str, Any] | None = None,
        has_nan: bool = False,
        has_inf: bool = False,
    ) -> None:
        """Initialize StabilityError.

        Args:
            message: Human-readable error description.
            step: Integration step where error occurred.
            engine: Name of the engine.
            state_snapshot: Partial state for debugging.
            has_nan: Whether NaN values were detected.
            has_inf: Whether Inf values were detected.
        """
        self.has_nan = has_nan
        self.has_inf = has_inf
        super().__init__(message, step, engine, state_snapshot)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with stability-specific details."""
        result = super().to_dict()
        result["has_nan"] = self.has_nan
        result["has_inf"] = self.has_inf
        return result


class ValueOutOfRangeError(NumericalError):
    """Exception raised when a value exceeds its valid physical/logical range.

    This is raised when computed values exceed bounds that would be
    physically meaningless (e.g., membrane potential > 100 mV).

    Example:
        >>> raise ValueOutOfRangeError(
        ...     variable_name="V",
        ...     value=-120.0,
        ...     min_value=-90.0,
        ...     max_value=40.0,
        ...     engine="MembraneEngine",
        ... )

    Reference:
        See docs/MATH_MODEL.md Section 2.4 and 3.6 for value bounds.
    """

    def __init__(
        self,
        variable_name: str,
        value: float,
        min_value: float,
        max_value: float,
        step: int | None = None,
        engine: str | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ValueOutOfRangeError.

        Args:
            variable_name: Name of the variable that exceeded bounds.
            value: The actual value that exceeded bounds.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            step: Integration step where error occurred.
            engine: Name of the engine.
            state_snapshot: Partial state for debugging.
        """
        self.variable_name = variable_name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value

        message = (
            f"{variable_name}={value:.6g} outside valid range "
            f"[{min_value:.6g}, {max_value:.6g}]"
        )
        super().__init__(message, step, engine, state_snapshot)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary with range-specific details."""
        result = super().to_dict()
        result["variable_name"] = self.variable_name
        result["value"] = self.value
        result["min_value"] = self.min_value
        result["max_value"] = self.max_value
        return result
