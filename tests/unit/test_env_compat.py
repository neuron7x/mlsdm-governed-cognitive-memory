"""Tests for environment variable compatibility layer."""

import os
from unittest.mock import patch

import pytest

from mlsdm.config.env_compat import (
    apply_env_compat,
    get_env_compat_info,
    warn_if_legacy_vars_used,
)


class TestEnvCompat:
    """Test environment variable compatibility mapping."""

    def test_apply_env_compat_disable_rate_limit(self):
        """Test DISABLE_RATE_LIMIT=1 maps to MLSDM_RATE_LIMIT_ENABLED=0."""
        with patch.dict(os.environ, {"DISABLE_RATE_LIMIT": "1"}, clear=False):
            # Remove canonical if exists
            os.environ.pop("MLSDM_RATE_LIMIT_ENABLED", None)

            apply_env_compat()

            assert os.environ.get("MLSDM_RATE_LIMIT_ENABLED") == "0"

    def test_apply_env_compat_disable_rate_limit_false(self):
        """Test DISABLE_RATE_LIMIT=0 maps to MLSDM_RATE_LIMIT_ENABLED=1."""
        with patch.dict(os.environ, {"DISABLE_RATE_LIMIT": "0"}, clear=False):
            # Remove canonical if exists
            os.environ.pop("MLSDM_RATE_LIMIT_ENABLED", None)

            apply_env_compat()

            assert os.environ.get("MLSDM_RATE_LIMIT_ENABLED") == "1"

    def test_apply_env_compat_disable_rate_limit_true_string(self):
        """Test DISABLE_RATE_LIMIT=true maps to MLSDM_RATE_LIMIT_ENABLED=0."""
        with patch.dict(os.environ, {"DISABLE_RATE_LIMIT": "true"}, clear=False):
            # Remove canonical if exists
            os.environ.pop("MLSDM_RATE_LIMIT_ENABLED", None)

            apply_env_compat()

            assert os.environ.get("MLSDM_RATE_LIMIT_ENABLED") == "0"

    def test_apply_env_compat_does_not_overwrite_canonical(self):
        """Test that canonical variables are not overwritten."""
        with patch.dict(
            os.environ,
            {"DISABLE_RATE_LIMIT": "1", "MLSDM_RATE_LIMIT_ENABLED": "1"},
            clear=False,
        ):
            apply_env_compat()

            # Canonical should remain unchanged
            assert os.environ.get("MLSDM_RATE_LIMIT_ENABLED") == "1"

    def test_apply_env_compat_no_legacy_vars(self):
        """Test apply_env_compat works when no legacy vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise any errors
            apply_env_compat()

    def test_warn_if_legacy_vars_used_with_disable_rate_limit(self):
        """Test warning for DISABLE_RATE_LIMIT legacy var."""
        with patch.dict(os.environ, {"DISABLE_RATE_LIMIT": "1"}, clear=False):
            # Remove canonical if exists
            os.environ.pop("MLSDM_RATE_LIMIT_ENABLED", None)

            with pytest.warns(DeprecationWarning, match="DISABLE_RATE_LIMIT is deprecated"):
                legacy = warn_if_legacy_vars_used()

            assert "DISABLE_RATE_LIMIT" in legacy

    def test_warn_if_legacy_vars_used_no_warning_if_canonical_set(self):
        """Test no warning if canonical var is already set."""
        with patch.dict(
            os.environ,
            {"DISABLE_RATE_LIMIT": "1", "MLSDM_RATE_LIMIT_ENABLED": "0"},
            clear=False,
        ):
            # Should not warn if canonical is set
            legacy = warn_if_legacy_vars_used()

            assert "DISABLE_RATE_LIMIT" in legacy

    def test_warn_if_legacy_vars_used_no_legacy(self):
        """Test no warnings when no legacy vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            legacy = warn_if_legacy_vars_used()

            assert len(legacy) == 0

    def test_get_env_compat_info_structure(self):
        """Test get_env_compat_info returns expected structure."""
        info = get_env_compat_info()

        assert "DISABLE_RATE_LIMIT" in info
        assert "CONFIG_PATH" in info
        assert "LLM_BACKEND" in info

        # Check each entry has expected keys
        for var_info in info.values():
            assert "canonical" in var_info
            assert "current_value" in var_info
            assert "canonical_value" in var_info
            assert "note" in var_info

    def test_get_env_compat_info_with_values(self):
        """Test get_env_compat_info returns correct values."""
        with patch.dict(
            os.environ,
            {
                "DISABLE_RATE_LIMIT": "1",
                "CONFIG_PATH": "config/test.yaml",
                "LLM_BACKEND": "openai",
            },
            clear=False,
        ):
            info = get_env_compat_info()

            assert info["DISABLE_RATE_LIMIT"]["current_value"] == "1"
            assert info["CONFIG_PATH"]["current_value"] == "config/test.yaml"
            assert info["LLM_BACKEND"]["current_value"] == "openai"

    def test_apply_env_compat_is_idempotent(self):
        """Test that apply_env_compat can be called multiple times safely."""
        with patch.dict(os.environ, {"DISABLE_RATE_LIMIT": "1"}, clear=False):
            os.environ.pop("MLSDM_RATE_LIMIT_ENABLED", None)

            # First call
            apply_env_compat()
            value1 = os.environ.get("MLSDM_RATE_LIMIT_ENABLED")

            # Second call
            apply_env_compat()
            value2 = os.environ.get("MLSDM_RATE_LIMIT_ENABLED")

            assert value1 == value2 == "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
