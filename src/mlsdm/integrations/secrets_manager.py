"""
External Secret Management Integration

Integrates with HashiCorp Vault, AWS Secrets Manager, Azure Key Vault,
and other secret management systems.
"""

import json
import logging
import os
import re
import threading
from enum import Enum
from typing import Any, Dict, Optional

import requests


class SecretProvider(Enum):
    """Supported secret management providers."""

    ENVIRONMENT = "environment"
    VAULT = "vault"
    AWS_SECRETS = "aws_secrets"
    AZURE_KEYVAULT = "azure_keyvault"


class SecretsManager:
    """
    Universal secrets manager client.

    Provides unified interface for retrieving secrets from various
    secret management systems with caching and automatic refresh.

    Example:
        >>> manager = SecretsManager(
        ...     provider=SecretProvider.VAULT,
        ...     vault_addr="https://vault.example.com",
        ...     vault_token="s.xxxxx"
        ... )
        >>> api_key = manager.get_secret("mlsdm/openai_api_key")
        >>> print(api_key)
    """

    def __init__(
        self,
        provider: SecretProvider = SecretProvider.ENVIRONMENT,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        azure_vault_url: Optional[str] = None,
        cache_ttl: int = 300,
    ) -> None:
        """
        Initialize secrets manager.

        Args:
            provider: Secret provider to use
            vault_addr: HashiCorp Vault address (for VAULT provider)
            vault_token: HashiCorp Vault token (for VAULT provider)
            aws_region: AWS region (for AWS_SECRETS provider)
            azure_vault_url: Azure Key Vault URL (for AZURE_KEYVAULT provider)
            cache_ttl: Cache TTL in seconds for fetched secrets
        """
        self.provider = provider
        self.vault_addr = vault_addr
        self.vault_token = vault_token
        self.aws_region = aws_region
        self.azure_vault_url = azure_vault_url
        self.cache_ttl = cache_ttl

        self.logger = logging.getLogger(__name__)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_lock = threading.RLock()

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve secret value.

        Args:
            key: Secret key/path
            default: Default value if secret not found

        Returns:
            Secret value or default
            
        Raises:
            ValueError: If secret name contains invalid characters
        """
        # Validate secret name to prevent injection attacks
        # Allow alphanumeric, underscores, hyphens, forward slashes, and single dots (not ..)
        # Disallow path traversal (..), shell special chars, etc.
        if not re.match(r'^[\w\-/]+(?:\.[\w\-/]+)*$', key) or '..' in key:
            self.logger.error(f"Invalid secret name format: {key}")
            raise ValueError(
                f"Invalid secret name format: {key}. Only alphanumeric, "
                "underscore, hyphen, forward slash, and dot characters are "
                "allowed (no path traversal)."
            )
        
        # Check cache first and validate TTL (with thread-safety)
        with self._cache_lock:
            if key in self._cache:
                import time
                timestamp = self._cache_timestamps.get(key)
                if timestamp is not None:
                    cache_age = time.time() - timestamp
                    if cache_age < self.cache_ttl:
                        return self._cache[key]
                # Expired or invalid timestamp, remove from cache
                if key in self._cache:
                    del self._cache[key]
                if key in self._cache_timestamps:
                    del self._cache_timestamps[key]

        # Fetch from provider
        if self.provider == SecretProvider.ENVIRONMENT:
            value = self._get_from_env(key, default)
        elif self.provider == SecretProvider.VAULT:
            value = self._get_from_vault(key)
        elif self.provider == SecretProvider.AWS_SECRETS:
            value = self._get_from_aws(key)
        elif self.provider == SecretProvider.AZURE_KEYVAULT:
            value = self._get_from_azure(key)
        else:
            self.logger.error(f"Unsupported provider: {self.provider}")
            return default

        # Cache the result with timestamp (thread-safe)
        if value is not None:
            import time
            with self._cache_lock:
                self._cache[key] = value
                self._cache_timestamps[key] = time.time()

        return value or default

    def _get_from_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment variable."""
        return os.environ.get(key, default)

    def _get_from_vault(self, path: str) -> Optional[str]:
        """Get secret from HashiCorp Vault."""
        if not self.vault_addr or not self.vault_token:
            self.logger.error("Vault address or token not configured")
            return None

        try:
            url = f"{self.vault_addr}/v1/{path}"
            headers = {"X-Vault-Token": self.vault_token}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()
            # Vault returns data in nested structure
            if "data" in data and "data" in data["data"]:
                return data["data"]["data"].get("value")
            elif "data" in data:
                return data["data"].get("value")
            return None

        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch secret from Vault: {e}")
            return None

    def _get_from_aws(self, secret_name: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        try:
            # Lazy import boto3 to make it optional
            import boto3  # type: ignore[import]

            session = boto3.session.Session()
            client = session.client(
                service_name="secretsmanager", region_name=self.aws_region
            )

            response = client.get_secret_value(SecretId=secret_name)

            # Secrets can be string or binary
            if "SecretString" in response:
                secret = response["SecretString"]
                # Try to parse as JSON
                try:
                    secret_dict = json.loads(secret)
                    # If it's a dict, return the 'value' key or the whole dict as string
                    if isinstance(secret_dict, dict):
                        return secret_dict.get("value", json.dumps(secret_dict))
                except json.JSONDecodeError:
                    pass
                return secret
            else:
                # Binary secrets
                return response["SecretBinary"].decode("utf-8")

        except ImportError:
            self.logger.error("boto3 not installed. Install with: pip install boto3")
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch secret from AWS: {e}")
            return None

    def _get_from_azure(self, secret_name: str) -> Optional[str]:
        """Get secret from Azure Key Vault."""
        try:
            # Lazy import Azure SDK to make it optional
            from azure.identity import DefaultAzureCredential  # type: ignore[import]
            from azure.keyvault.secrets import SecretClient  # type: ignore[import]

            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=self.azure_vault_url or "", credential=credential)

            secret = client.get_secret(secret_name)
            return secret.value

        except ImportError:
            self.logger.error(
                "Azure SDK not installed. Install with: "
                "pip install azure-keyvault-secrets azure-identity"
            )
            return None
        except Exception as e:
            self.logger.error(f"Failed to fetch secret from Azure: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear the secrets cache (thread-safe)."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_timestamps.clear()
        self.logger.info("Secrets cache cleared")
