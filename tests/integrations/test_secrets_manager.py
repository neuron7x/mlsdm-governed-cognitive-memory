"""
Tests for secrets manager integration.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from mlsdm.integrations import SecretProvider, SecretsManager


class TestSecretsManager:
    """Test secrets manager integration."""

    def test_initialization(self) -> None:
        """Test manager initialization."""
        manager = SecretsManager(
            provider=SecretProvider.VAULT,
            vault_addr="https://vault.example.com",
            vault_token="s.test",
        )

        assert manager.provider == SecretProvider.VAULT
        assert manager.vault_addr == "https://vault.example.com"
        assert manager.vault_token == "s.test"

    def test_get_secret_from_environment(self) -> None:
        """Test retrieving secret from environment."""
        manager = SecretsManager(provider=SecretProvider.ENVIRONMENT)

        with patch.dict(os.environ, {"TEST_SECRET": "test_value"}):
            secret = manager.get_secret("TEST_SECRET")
            assert secret == "test_value"

    def test_get_secret_with_default(self) -> None:
        """Test default value when secret not found."""
        manager = SecretsManager(provider=SecretProvider.ENVIRONMENT)

        secret = manager.get_secret("NONEXISTENT_KEY", default="default_value")
        assert secret == "default_value"

    def test_get_secret_from_vault_success(self) -> None:
        """Test retrieving secret from Vault."""
        manager = SecretsManager(
            provider=SecretProvider.VAULT,
            vault_addr="https://vault.example.com",
            vault_token="s.test",
        )

        mock_response = {"data": {"data": {"value": "secret_from_vault"}}}

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_response
            mock_get.return_value.raise_for_status = MagicMock()

            secret = manager.get_secret("myapp/api_key")
            assert secret == "secret_from_vault"

    def test_cache_functionality(self) -> None:
        """Test that secrets are cached."""
        manager = SecretsManager(provider=SecretProvider.ENVIRONMENT)

        with patch.dict(os.environ, {"CACHED_SECRET": "cached_value"}):
            # First call - should cache
            secret1 = manager.get_secret("CACHED_SECRET")
            
            # Clear environment
            os.environ.pop("CACHED_SECRET", None)
            
            # Second call - should return cached value
            secret2 = manager.get_secret("CACHED_SECRET")
            
            assert secret1 == "cached_value"
            assert secret2 == "cached_value"

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        manager = SecretsManager(provider=SecretProvider.ENVIRONMENT)

        with patch.dict(os.environ, {"TEST_KEY": "test"}):
            manager.get_secret("TEST_KEY")
            assert "TEST_KEY" in manager._cache

            manager.clear_cache()
            assert len(manager._cache) == 0
