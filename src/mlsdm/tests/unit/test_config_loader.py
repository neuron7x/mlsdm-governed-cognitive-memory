"""Comprehensive unit tests for ConfigLoader."""
import os
import tempfile

import pytest

from mlsdm.utils.config_loader import ConfigLoader


class TestConfigLoader:
    """Test suite for ConfigLoader."""

    def test_load_yaml_config(self):
        """Test loading a YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("dimension: 384\n")
            f.write("threshold: 0.5\n")
            f.write("enabled: true\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert config["dimension"] == 384
            assert config["threshold"] == 0.5
            assert config["enabled"] is True
        finally:
            os.unlink(yaml_path)

    def test_load_yml_extension(self):
        """Test loading a .yml file (alternative YAML extension)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("test_key: test_value\n")
            yml_path = f.name

        try:
            config = ConfigLoader.load_config(yml_path)
            assert config["test_key"] == "test_value"
        finally:
            os.unlink(yml_path)

    def test_load_ini_config(self):
        """Test loading an INI configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[section1]\n")
            f.write("dimension = 384\n")
            f.write("threshold = 0.5\n")
            f.write("enabled = true\n")
            f.write("name = test\n")
            ini_path = f.name

        try:
            config = ConfigLoader.load_config(ini_path)
            assert config["dimension"] == 384
            assert config["threshold"] == 0.5
            assert config["enabled"] is True
            assert config["name"] == "test"
        finally:
            os.unlink(ini_path)

    def test_load_empty_yaml(self):
        """Test loading an empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert config == {}
        finally:
            os.unlink(yaml_path)

    def test_load_nested_yaml(self):
        """Test loading a nested YAML configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("database:\n")
            f.write("  host: localhost\n")
            f.write("  port: 5432\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert "database" in config
            assert config["database"]["host"] == "localhost"
            assert config["database"]["port"] == 5432
        finally:
            os.unlink(yaml_path)

    def test_invalid_file_format(self):
        """Test that unsupported file formats raise ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[section]\n")
            f.write("flag1 = true\n")
            f.write("flag2 = false\n")
            f.write("flag3 = True\n")
            f.write("flag4 = FALSE\n")
            ini_path = f.name

        try:
            config = ConfigLoader.load_config(ini_path)
            assert config["flag1"] is True
            assert config["flag2"] is False
            assert config["flag3"] is True
            assert config["flag4"] is False
        finally:
            os.unlink(ini_path)

    def test_ini_numeric_parsing(self):
        """Test that INI numeric values are parsed correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[section]\n")
            f.write("int_val = 42\n")
            f.write("float_val = 3.14\n")
            f.write("string_val = abc\n")
            ini_path = f.name

        try:
            config = ConfigLoader.load_config(ini_path)
            assert config["int_val"] == 42
            assert config["float_val"] == 3.14
            assert config["string_val"] == "abc"
        finally:
            os.unlink(ini_path)

    def test_yaml_list_parsing(self):
        """Test that YAML lists are parsed correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("items:\n")
            f.write("  - item1\n")
            f.write("  - item2\n")
            f.write("  - item3\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert "items" in config
            assert len(config["items"]) == 3
            assert config["items"][0] == "item1"
        finally:
            os.unlink(yaml_path)

    def test_multiple_ini_sections(self):
        """Test loading INI file with multiple sections."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("[section1]\n")
            f.write("key1 = value1\n")
            f.write("[section2]\n")
            f.write("key2 = value2\n")
            ini_path = f.name

        try:
            config = ConfigLoader.load_config(ini_path)
            assert "key1" in config
            assert "key2" in config
        finally:
            os.unlink(ini_path)

    def test_yaml_with_special_types(self):
        """Test YAML with special types (null, lists, etc.)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("string_key: hello\n")
            f.write("int_key: 123\n")
            f.write("float_key: 45.67\n")
            f.write("bool_key: true\n")
            f.write("null_key: null\n")
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            f.write("message: Привіт світ\n")  # Ukrainian: Hello world
            yaml_path = f.name

        try:
            config = ConfigLoader.load_config(yaml_path)
            assert config["message"] == "Привіт світ"
        finally:
            os.unlink(yaml_path)
