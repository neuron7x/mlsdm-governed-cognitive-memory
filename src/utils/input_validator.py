"""Input validation and sanitization utilities.

This module provides comprehensive input validation to prevent injection attacks,
data corruption, and ensure data integrity as per SECURITY_POLICY.md.
"""

import re
from typing import Any, List, Optional

import numpy as np


class InputValidator:
    """Utilities for validating and sanitizing user inputs."""

    # Constants for validation
    MAX_VECTOR_SIZE = 100_000  # Maximum vector dimension
    MAX_ARRAY_ELEMENTS = 1_000_000  # Maximum array size
    MIN_MORAL_VALUE = 0.0
    MAX_MORAL_VALUE = 1.0

    @staticmethod
    def validate_vector(
        vector: List[float],
        expected_dim: int,
        normalize: bool = False
    ) -> np.ndarray:
        """Validate and optionally normalize a vector input.

        Args:
            vector: Input vector as list of floats
            expected_dim: Expected dimension of the vector
            normalize: Whether to normalize the vector (default: False)

        Returns:
            Validated numpy array

        Raises:
            ValueError: If validation fails
        """
        # Check type
        if not isinstance(vector, (list, tuple, np.ndarray)):
            raise ValueError("Vector must be a list, tuple, or numpy array")

        # Check size limits
        if len(vector) > InputValidator.MAX_VECTOR_SIZE:
            raise ValueError(
                f"Vector size {len(vector)} exceeds maximum {InputValidator.MAX_VECTOR_SIZE}"
            )

        # Check dimension match
        if len(vector) != expected_dim:
            raise ValueError(
                f"Vector dimension {len(vector)} does not match expected {expected_dim}"
            )

        # Convert to numpy array
        try:
            arr = np.array(vector, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert vector to numpy array: {e}")

        # Check for invalid values (NaN, Inf)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Vector contains NaN or Inf values")

        # Check for zero vector if normalization is requested
        if normalize:
            norm = np.linalg.norm(arr)
            if norm < 1e-10:
                raise ValueError("Cannot normalize zero vector")
            arr = arr / norm

        return arr

    @staticmethod
    def validate_moral_value(value: float) -> float:
        """Validate moral value is in valid range [0.0, 1.0].

        Args:
            value: Moral value to validate

        Returns:
            Validated moral value

        Raises:
            ValueError: If value is out of range or invalid
        """
        # Check type
        if not isinstance(value, (int, float)):
            raise ValueError(f"Moral value must be numeric, got {type(value)}")

        # Convert to float
        try:
            val = float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert moral value to float: {e}")

        # Check for invalid values
        if not np.isfinite(val):
            raise ValueError("Moral value cannot be NaN or Inf")

        # Check range
        if val < InputValidator.MIN_MORAL_VALUE or val > InputValidator.MAX_MORAL_VALUE:
            raise ValueError(
                f"Moral value {val} must be between "
                f"{InputValidator.MIN_MORAL_VALUE} and {InputValidator.MAX_MORAL_VALUE}"
            )

        return val

    @staticmethod
    def sanitize_string(
        text: str,
        max_length: int = 10000,
        allow_newlines: bool = True
    ) -> str:
        """Sanitize string input to prevent injection attacks.

        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length (default: 10000)
            allow_newlines: Whether to allow newline characters (default: True)

        Returns:
            Sanitized string

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(text, str):
            raise ValueError(f"Input must be string, got {type(text)}")

        # Check length
        if len(text) > max_length:
            raise ValueError(
                f"String length {len(text)} exceeds maximum {max_length}"
            )

        # Remove null bytes (potential for injection)
        text = text.replace('\x00', '')

        # Remove or validate newlines
        if not allow_newlines:
            text = text.replace('\n', ' ').replace('\r', ' ')

        # Remove other control characters except tab and newline
        text = ''.join(
            char for char in text
            if char == '\t' or char == '\n' or not (0 <= ord(char) < 32)
        )

        return text.strip()

    @staticmethod
    def validate_client_id(client_id: str) -> str:
        """Validate and sanitize client ID (IP address or API key).

        Args:
            client_id: Client identifier

        Returns:
            Validated client ID

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(client_id, str):
            raise ValueError("Client ID must be a string")

        client_id = client_id.strip()

        if not client_id:
            raise ValueError("Client ID cannot be empty")

        if len(client_id) > 256:
            raise ValueError("Client ID too long")

        # Allow alphanumeric, dots, colons, and hyphens (for IPs and UUIDs)
        if not re.match(r'^[a-zA-Z0-9\.\:\-_]+$', client_id):
            raise ValueError("Client ID contains invalid characters")

        return client_id

    @staticmethod
    def validate_numeric_range(
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        name: str = "value"
    ) -> float:
        """Validate numeric value is within specified range.

        Args:
            value: Value to validate
            min_val: Minimum allowed value (optional)
            max_val: Maximum allowed value (optional)
            name: Name of the value for error messages

        Returns:
            Validated value

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric, got {type(value)}")

        val = float(value)

        if not np.isfinite(val):
            raise ValueError(f"{name} cannot be NaN or Inf")

        if min_val is not None and val < min_val:
            raise ValueError(f"{name} {val} is less than minimum {min_val}")

        if max_val is not None and val > max_val:
            raise ValueError(f"{name} {val} exceeds maximum {max_val}")

        return val

    @staticmethod
    def validate_array_size(
        array: Any,
        max_size: Optional[int] = None,
        name: str = "array"
    ) -> int:
        """Validate array size is within limits.

        Args:
            array: Array-like object to validate
            max_size: Maximum allowed size (default: MAX_ARRAY_ELEMENTS)
            name: Name of the array for error messages

        Returns:
            Size of the array

        Raises:
            ValueError: If validation fails
        """
        if max_size is None:
            max_size = InputValidator.MAX_ARRAY_ELEMENTS

        try:
            size = len(array)
        except TypeError:
            raise ValueError(f"{name} must have a length")

        if size > max_size:
            raise ValueError(
                f"{name} size {size} exceeds maximum {max_size}"
            )

        return size
