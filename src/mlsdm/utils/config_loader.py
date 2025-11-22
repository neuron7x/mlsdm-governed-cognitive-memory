"""Configuration loading with validation for MLSDM Governed Cognitive Memory.

This module provides utilities for loading and validating configuration files
with comprehensive error messages and type safety.
"""

import configparser
import os
from pathlib import Path
from typing import Any

import yaml

from mlsdm.utils.config_schema import SystemConfig, validate_config_dict


class ConfigLoader:
    """Load and validate configuration files with schema validation."""

    @staticmethod
    def load_config(
        path: str,
        validate: bool = True,
        env_override: bool = True
    ) -> dict[str, Any]:
        """Load configuration from file with optional validation.

        Args:
            path: Path to configuration file (YAML or INI)
            validate: If True, validate against schema
            env_override: If True, allow environment variable overrides

        Returns:
            Configuration dictionary

        Raises:
            TypeError: If path is not a string
            ValueError: If file format is unsupported or validation fails
            FileNotFoundError: If configuration file does not exist
        """
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")

        # Check file exists
        if not Path(path).is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if not path.endswith((".yaml", ".yml", ".ini")):
            raise ValueError(
                "Unsupported configuration file format. "
                "Only YAML (.yaml, .yml) and INI (.ini) are supported."
            )

        config: dict[str, Any] = {}

        # Load from file
        if path.endswith((".yaml", ".yml")):
            config = ConfigLoader._load_yaml(path)
        else:
            config = ConfigLoader._load_ini(path)

        # Apply environment variable overrides if enabled
        if env_override:
            config = ConfigLoader._apply_env_overrides(config)

        # Validate against schema if enabled
        if validate:
            try:
                validated_config = validate_config_dict(config)
                # Convert back to dict for backward compatibility
                config = validated_config.model_dump()
            except ValueError as e:
                raise ValueError(
                    f"Configuration validation failed for '{path}':\n{str(e)}\n\n"
                    f"Please check your configuration file against the schema "
                    f"documentation in src/utils/config_schema.py"
                ) from e

        return config

    @staticmethod
    def _load_yaml(path: str) -> dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in '{path}': {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Error reading YAML file '{path}': {str(e)}") from e

    @staticmethod
    def _load_ini(path: str) -> dict[str, Any]:
        """Load INI configuration file."""
        try:
            config: dict[str, Any] = {}
            parser = configparser.ConfigParser()
            parser.read(path, encoding="utf-8")

            for section in parser.sections():
                for key, value in parser[section].items():
                    lower = value.lower()
                    if lower in ("true", "false"):
                        config[key] = lower == "true"
                    else:
                        try:
                            config[key] = int(value)
                        except ValueError:
                            try:
                                config[key] = float(value)
                            except ValueError:
                                config[key] = value

            return config
        except Exception as e:
            raise ValueError(f"Error reading INI file '{path}': {str(e)}") from e

    @staticmethod
    def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Environment variables should be prefixed with MLSDM_ and use
        double underscores for nested keys. For example:
        - MLSDM_DIMENSION=768
        - MLSDM_MORAL_FILTER__THRESHOLD=0.7
        - MLSDM_COGNITIVE_RHYTHM__WAKE_DURATION=10
        """
        prefix = "MLSDM_"

        for env_key, env_value in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            # Remove prefix and convert to lowercase
            config_key = env_key[len(prefix):].lower()

            # Handle nested keys (e.g., MORAL_FILTER__THRESHOLD)
            if "__" in config_key:
                parts = config_key.split("__")
                if len(parts) == 2:
                    section, key = parts
                    if section not in config:
                        config[section] = {}
                    config[section][key] = ConfigLoader._parse_env_value(env_value)
            else:
                # Top-level key
                config[config_key] = ConfigLoader._parse_env_value(env_value)

        return config

    @staticmethod
    def _parse_env_value(value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try boolean
        if value.lower() in ("true", "1", "yes", "on"):
            return True
        if value.lower() in ("false", "0", "no", "off"):
            return False

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string
        return value

    @staticmethod
    def load_validated_config(path: str) -> SystemConfig:
        """Load and return validated configuration as SystemConfig object.

        Args:
            path: Path to configuration file

        Returns:
            Validated SystemConfig instance

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If file does not exist
        """
        config_dict = ConfigLoader.load_config(path, validate=True)
        return validate_config_dict(config_dict)
