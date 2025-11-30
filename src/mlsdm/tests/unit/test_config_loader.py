"""Comprehensive unit tests for ConfigLoader."""

import os
import tempfile

import pytest

from mlsdm.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader."""

    def test_load_yaml_config(self):
        """Test loading a YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("dimension: 384\n")
            f.write("moral_filter:\n")
            f.write("  threshold: 0.5\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert config["dimension"] == 384
            assert config["moral_filter"]["threshold"] == 0.5
        finally:
            os.unlink(yaml_path)

    def test_load_yml_extension(self):
        """Test loading a .yml file (alternative YAML extension)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            # Use default dimension to avoid ontology mismatch
            f.write("strict_mode: false\n")
            yml_path = f.name

        try:
            config = ConfigLoader.load_config(yml_path)
            assert config["strict_mode"] is False
            assert config["dimension"] == 384  # default
        finally:
            os.unlink(yml_path)

    def test_load_ini_config(self):
        """Test loading an INI configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write("[section1]\n")
            f.write("dimension = 384\n")
            f.write("strict_mode = false\n")
            ini_path = f.name

        try:
            config = ConfigLoader.load_config(ini_path)
            assert config["dimension"] == 384
            assert config["strict_mode"] is False
        finally:
            os.unlink(ini_path)

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file - should use defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            # Empty config should be validated and get defaults
            assert "dimension" in config
            assert config["dimension"] == 384  # default
        finally:
            os.unlink(yaml_path)

    def test_load_nested_yaml(self):
        """Test loading a nested YAML configuration."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("cognitive_rhythm:\n")
            f.write("  wake_duration: 10\n")
            f.write("  sleep_duration: 5\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert "cognitive_rhythm" in config
            assert config["cognitive_rhythm"]["wake_duration"] == 10
            assert config["cognitive_rhythm"]["sleep_duration"] == 5
        finally:
            os.unlink(yaml_path)

    def test_invalid_file_format(self):
        """Test that unsupported file formats raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                ConfigLoader.load_config(txt_path)
        finally:
            os.unlink(txt_path)

    def test_invalid_path_type(self):
        """Test that non-string path raises TypeError."""
        with pytest.raises(TypeError, match="Path must be a string"):
            ConfigLoader.load_config(123)

    def test_ini_boolean_parsing(self):
        """Test that INI boolean values are parsed correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write("[section]\n")
            f.write("strict_mode = true\n")
            ini_path = f.name

        try:
            config = ConfigLoader.load_config(ini_path)
            assert config["strict_mode"] is True
        finally:
            os.unlink(ini_path)

    def test_ini_numeric_parsing(self):
        """Test that INI numeric values are parsed correctly - disable validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write("[section]\n")
            f.write("dimension = 768\n")
            ini_path = f.name

        try:
            # Disable validation to test numeric parsing without schema constraints
            config = ConfigLoader.load_config(ini_path, validate=False)
            assert config["dimension"] == 768
            assert isinstance(config["dimension"], int)
        finally:
            os.unlink(ini_path)

    def test_yaml_list_parsing(self):
        """Test that YAML lists are parsed correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("ontology_matcher:\n")
            f.write("  ontology_labels:\n")
            f.write("    - category1\n")
            f.write("    - category2\n")
            f.write("  ontology_vectors:\n")
            f.write("    - [1.0, 0.0]\n")
            f.write("    - [0.0, 1.0]\n")
            f.write("dimension: 2\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert "ontology_matcher" in config
            labels = config["ontology_matcher"]["ontology_labels"]
            assert len(labels) == 2
            assert labels[0] == "category1"
        finally:
            os.unlink(yaml_path)

    def test_multiple_ini_sections(self):
        """Test loading INI file with multiple sections - disable validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as f:
            f.write("[section1]\n")
            f.write("dimension = 512\n")
            f.write("[section2]\n")
            f.write("strict_mode = false\n")
            ini_path = f.name

        try:
            # Disable validation to test multiple sections without schema constraints
            config = ConfigLoader.load_config(ini_path, validate=False)
            assert config["dimension"] == 512
            assert config["strict_mode"] is False
        finally:
            os.unlink(ini_path)

    def test_yaml_with_special_types(self):
        """Test YAML with special types (null, lists, etc.) - disable validation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("string_key: hello\n")
            f.write("int_key: 123\n")
            f.write("float_key: 45.67\n")
            f.write("bool_key: true\n")
            f.write("null_key: null\n")
            yaml_path = f.name

        try:
            # Disable validation to test raw YAML parsing
            config = ConfigLoader.load_config(yaml_path, validate=False)
            assert config["string_key"] == "hello"
            assert config["int_key"] == 123
            assert config["float_key"] == 45.67
            assert config["bool_key"] is True
            assert config["null_key"] is None
        finally:
            os.unlink(yaml_path)

    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_config("/nonexistent/path/config.yaml")

    def test_yaml_encoding(self):
        """Test that YAML files with UTF-8 encoding are handled correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            # Disable validation to test encoding without schema constraints
            f.write("dimension: 384\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert config["dimension"] == 384
        finally:
            os.unlink(yaml_path)
