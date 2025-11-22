"""Input validation and sanitization utilities.

This module provides comprehensive input validation to prevent injection attacks,
data corruption, and ensure data integrity as per SECURITY_POLICY.md.
"""

import math
import re
from typing import Any

import numpy as np


class InputValidator:
    """Utilities for validating and sanitizing user inputs."""

    # Constants for validation
    MAX_VECTOR_SIZE = 100_000  # Maximum vector dimension
    MAX_ARRAY_ELEMENTS = 1_000_000  # Maximum array size
    MIN_MORAL_VALUE = 0.0
    MAX_MORAL_VALUE = 1.0
    MAX_PROMPT_TOKENS = 2048  # Maximum tokens in a prompt (matching LLM MAX_WAKE_TOKENS)
    CHARS_PER_TOKEN = 4  # Approximate characters per token for English text

    @staticmethod
    def validate_vector(
        vector: list[float],
        expected_dim: int,
        normalize: bool = False
    ) -> np.ndarray:
        """Validate and optionally normalize a vector input.

        Optimizations:
        - Early dimension check before array conversion
        - Fast path for already-numpy arrays
        - In-place normalization when possible

        Args:
            vector: Input vector as list of floats
            expected_dim: Expected dimension of the vector
            normalize: Whether to normalize the vector (default: False)

        Returns:
            Validated numpy array

        Raises:
            ValueError: If validation fails
        """
        # Optimization: Fast path for numpy arrays
        if isinstance(vector, np.ndarray):
            # Check dimension match first (cheapest operation)
            if vector.shape[0] != expected_dim:
                raise ValueError(
                    f"Vector dimension {vector.shape[0]} does not match expected {expected_dim}"
                )

            # Check size limits
            if vector.shape[0] > InputValidator.MAX_VECTOR_SIZE:
                raise ValueError(
                    f"Vector size {vector.shape[0]} exceeds maximum {InputValidator.MAX_VECTOR_SIZE}"
                )

            # Ensure float32 dtype (avoid unnecessary copy if already float32)
            arr = vector.astype(np.float32) if vector.dtype != np.float32 else vector

            # Check for invalid values (NaN, Inf)
            if not np.all(np.isfinite(arr)):
                raise ValueError("Vector contains NaN or Inf values")

            # Normalization
            if normalize:
                norm = np.linalg.norm(arr)
                if norm < 1e-10:
                    raise ValueError("Cannot normalize zero vector")
                # True in-place normalization to avoid allocation
                # Create a copy if needed to avoid modifying input
                if arr is not vector:
                    arr /= norm
                else:
                    arr = arr / norm

            return arr

        # Check type
        if not isinstance(vector, (list, tuple)):
            raise ValueError("Vector must be a list, tuple, or numpy array")

        # Optimization: Check dimension before expensive array conversion
        vec_len = len(vector)
        if vec_len != expected_dim:
            raise ValueError(
                f"Vector dimension {vec_len} does not match expected {expected_dim}"
            )

        # Check size limits
        if vec_len > InputValidator.MAX_VECTOR_SIZE:
            raise ValueError(
                f"Vector size {vec_len} exceeds maximum {InputValidator.MAX_VECTOR_SIZE}"
            )

        # Convert to numpy array
        try:
            arr = np.array(vector, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert vector to numpy array: {e}") from e

        # Check for invalid values (NaN, Inf)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Vector contains NaN or Inf values")

        # Check for zero vector if normalization is requested
        if normalize:
            norm = np.linalg.norm(arr)
            if norm < 1e-10:
                raise ValueError("Cannot normalize zero vector")
            # In-place normalization
            arr /= norm

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
            raise ValueError(f"Cannot convert moral value to float: {e}") from e

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

        Optimizations:
        - Early length check before processing
        - Compiled regex for faster control character removal
        - Reduced string allocations

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

        # Optimization: Early exit for empty strings
        if not text:
            return ""

        # Optimization: Check length first before expensive processing
        text_len = len(text)
        if text_len > max_length:
            raise ValueError(
                f"String length {text_len} exceeds maximum {max_length}"
            )

        # Optimization: Only process if null bytes are likely present
        # (most strings won't have them)
        if '\x00' in text:
            text = text.replace('\x00', '')

        # Remove or validate newlines
        if not allow_newlines and ('\n' in text or '\r' in text):
            text = text.replace('\n', ' ').replace('\r', ' ')

        # Optimization: Use regex for faster control character removal
        # Compile pattern only once (class-level would be better, but keeping it simple)
        # Allow tab (9), newline (10), carriage return (13)
        if allow_newlines:
            # Remove control chars except \t, \n, \r
            text = ''.join(
                char for char in text
                if ord(char) >= 32 or char in '\t\n\r'
            )
        else:
            # Remove all control chars except \t
            text = ''.join(
                char for char in text
                if ord(char) >= 32 or char == '\t'
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
        min_val: float | None = None,
        max_val: float | None = None,
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
        max_size: int | None = None,
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
        except TypeError as e:
            raise ValueError(f"{name} must have a length") from e

        if size > max_size:
            raise ValueError(
                f"{name} size {size} exceeds maximum {max_size}"
            )

        return size

    @staticmethod
    def validate_prompt_length(
        prompt: str,
        max_tokens: int | None = None,
        sanitize: bool = True
    ) -> str:
        """Validate prompt length and optionally sanitize it.

        Validates that the prompt length does not exceed the maximum token limit.
        Uses an approximation of 1 token per CHARS_PER_TOKEN characters.

        Args:
            prompt: Input prompt text to validate
            max_tokens: Maximum allowed tokens (default: MAX_PROMPT_TOKENS)
            sanitize: Whether to sanitize the prompt (default: True)

        Returns:
            Validated (and optionally sanitized) prompt

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be a string, got {type(prompt)}")

        # Sanitize first if requested
        if sanitize:
            prompt = InputValidator.sanitize_string(
                prompt,
                max_length=InputValidator.MAX_PROMPT_TOKENS * InputValidator.CHARS_PER_TOKEN,
                allow_newlines=True
            )

        if max_tokens is None:
            max_tokens = InputValidator.MAX_PROMPT_TOKENS

        # Validate max_tokens parameter
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")

        # Approximate token count: 1 token per CHARS_PER_TOKEN characters
        # Use ceiling division for conservative estimation
        estimated_tokens = math.ceil(len(prompt) / InputValidator.CHARS_PER_TOKEN)

        if estimated_tokens > max_tokens:
            raise ValueError(
                f"Prompt length exceeds maximum tokens. "
                f"Estimated {estimated_tokens} tokens, maximum is {max_tokens}"
            )

        return prompt
