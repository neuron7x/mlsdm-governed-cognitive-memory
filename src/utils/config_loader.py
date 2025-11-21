"""Configuration loader for YAML and INI files."""
import configparser
from typing import Any, Dict

import yaml


class ConfigLoader:
    @staticmethod
    def load_config(path: str) -> Dict[str, Any]:
        if not isinstance(path, str):
            raise TypeError("Path must be a string.")

        if not path.endswith((".yaml", ".yml", ".ini")):
            raise ValueError("Unsupported configuration file format. Only YAML and INI are supported.")

        config: Dict[str, Any] = {}

        if path.endswith((".yaml", ".yml")):
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        else:
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
