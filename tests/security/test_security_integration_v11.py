"""
Integration tests for Security Hardening v1.1 features.

Tests the integration of:
- Policy engine
- Guardrails
- LLM safety
- Multi-tenancy
- Security profiles
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request


class TestSecurityProfiles:
    """Test security profile configuration."""

    def test_dev_profile_disables_security_features(self, monkeypatch):
        """Dev profile should disable advanced security features."""
        monkeypatch.setenv("MLSDM_RUNTIME_MODE", "dev")
        from mlsdm.config_runtime import get_runtime_config
        
        config = get_runtime_config()
        
        assert config.mode.value == "dev"
        assert config.security.enable_oidc is False
        assert config.security.enable_mtls is False
        assert config.security.enable_rbac is False
        assert config.security.enable_policy_engine is False
        assert config.security.enable_guardrails is False
        
    def test_prod_profile_enables_security_features(self, monkeypatch):
        """Cloud prod profile should enable all security features."""
        monkeypatch.setenv("MLSDM_RUNTIME_MODE", "cloud-prod")
        from mlsdm.config_runtime import get_runtime_config
        
        config = get_runtime_config()
        
        assert config.mode.value == "cloud-prod"
        assert config.security.enable_oidc is True
        assert config.security.enable_mtls is True
        assert config.security.enable_rbac is True
        assert config.security.enable_policy_engine is True
        assert config.security.enable_guardrails is True
        assert config.security.enable_llm_safety is True
        assert config.security.enable_pii_scrub_logs is True
        assert config.security.enable_multi_tenant_enforcement is True


class TestSecurityIntegration:
    """Test security integration helper functions."""
    
    def test_create_policy_context_from_request(self):
        """Test creating PolicyContext from FastAPI request."""
        from mlsdm.api.security_integration import create_policy_context_from_request
        
        # Create mock request
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/generate"
        request.headers = {"User-Agent": "test-client"}
        request.state.user_id = "user123"
        request.state.user_roles = ["user", "write"]
        request.state.tenant_id = "tenant456"
        request.state.client_id = "client789"
        request.state.has_valid_token = True
        request.state.has_valid_signature = False
        request.state.has_mtls_cert = False
        request.state.request_id = "req-001"
        
        context = create_policy_context_from_request(
            request=request,
            route="/generate",
            prompt="Hello, world!",
        )
        
        assert context.user_id == "user123"
        assert context.user_roles == ["user", "write"]
        assert context.client_id == "client789"
        assert context.route == "/generate"
        assert context.method == "POST"
        assert context.prompt == "Hello, world!"
        assert context.has_valid_token is True
        assert context.metadata["tenant_id"] == "tenant456"
        assert context.metadata["request_id"] == "req-001"
        
    def test_evaluate_request_policy_disabled(self):
        """Test policy evaluation when disabled."""
        from mlsdm.api.security_integration import evaluate_request_policy_if_enabled
        from mlsdm.security.policy_engine import PolicyContext
        
        # Mock config to disable policy engine
        with patch("mlsdm.api.security_integration.get_runtime_config") as mock_config:
            mock_config.return_value.security.enable_policy_engine = False
            
            context = PolicyContext(
                client_id="test",
                route="/generate",
                method="POST",
            )
            
            decision = evaluate_request_policy_if_enabled(context)
            
            # Should return None when disabled
            assert decision is None
            
    def test_evaluate_request_policy_enabled(self):
        """Test policy evaluation when enabled."""
        from mlsdm.api.security_integration import evaluate_request_policy_if_enabled
        from mlsdm.security.policy_engine import PolicyContext
        
        # Mock config to enable policy engine
        with patch("mlsdm.api.security_integration.get_runtime_config") as mock_config:
            mock_config.return_value.security.enable_policy_engine = True
            
            context = PolicyContext(
                client_id="test",
                route="/generate",
                method="POST",
                has_valid_token=True,
                user_roles=["user"],
            )
            
            decision = evaluate_request_policy_if_enabled(context)
            
            # Should return a decision when enabled
            assert decision is not None
            assert hasattr(decision, "allow")
            
    def test_apply_guardrails_disabled(self):
        """Test guardrails when disabled."""
        from mlsdm.api.security_integration import apply_guardrails_if_enabled
        
        # Mock config to disable guardrails
        with patch("mlsdm.api.security_integration.get_runtime_config") as mock_config:
            mock_config.return_value.security.enable_guardrails = False
            
            result = apply_guardrails_if_enabled(
                prompt="Hello, world!",
            )
            
            # Should return None when disabled
            assert result is None
            
    def test_analyze_llm_safety_disabled(self):
        """Test LLM safety analysis when disabled."""
        from mlsdm.api.security_integration import analyze_llm_safety_if_enabled
        
        # Mock config to disable LLM safety
        with patch("mlsdm.api.security_integration.get_runtime_config") as mock_config:
            mock_config.return_value.security.enable_llm_safety = False
            
            result = analyze_llm_safety_if_enabled(
                text="Hello, world!",
                is_prompt=True,
            )
            
            # Should return None when disabled
            assert result is None


class TestMultiTenancy:
    """Test multi-tenant isolation features."""
    
    def test_tenant_id_extraction_from_oidc(self):
        """Test tenant_id extraction from OIDC JWT claims."""
        from mlsdm.security.oidc import UserInfo
        
        # Create UserInfo with tenant in custom claims
        user_info = UserInfo(
            subject="user123",
            email="user@example.com",
            roles=["user"],
            custom_claims={"tenant_id": "tenant456"},
        )
        
        assert user_info.custom_claims["tenant_id"] == "tenant456"
        
    def test_check_multi_tenant_enforcement_disabled(self):
        """Test multi-tenant enforcement when disabled."""
        from mlsdm.api.security_integration import check_multi_tenant_enforcement
        
        # Mock config and request
        with patch("mlsdm.api.security_integration.get_runtime_config") as mock_config:
            mock_config.return_value.security.enable_multi_tenant_enforcement = False
            
            request = MagicMock(spec=Request)
            request.state.tenant_id = "tenant1"
            
            # Should allow when disabled
            allowed = check_multi_tenant_enforcement(
                request=request,
                resource_tenant_id="tenant2",  # Different tenant
            )
            
            assert allowed is True
            
    def test_check_multi_tenant_enforcement_same_tenant(self):
        """Test multi-tenant enforcement with matching tenant."""
        from mlsdm.api.security_integration import check_multi_tenant_enforcement
        
        # Mock config and request
        with patch("mlsdm.api.security_integration.get_runtime_config") as mock_config:
            mock_config.return_value.security.enable_multi_tenant_enforcement = True
            
            request = MagicMock(spec=Request)
            request.state.tenant_id = "tenant1"
            
            # Should allow same tenant
            allowed = check_multi_tenant_enforcement(
                request=request,
                resource_tenant_id="tenant1",
            )
            
            assert allowed is True
            
    def test_check_multi_tenant_enforcement_different_tenant(self):
        """Test multi-tenant enforcement with different tenant."""
        from mlsdm.api.security_integration import check_multi_tenant_enforcement
        
        # Mock config and request
        with patch("mlsdm.api.security_integration.get_runtime_config") as mock_config:
            mock_config.return_value.security.enable_multi_tenant_enforcement = True
            
            request = MagicMock(spec=Request)
            request.state.tenant_id = "tenant1"
            
            # Should block different tenant
            allowed = check_multi_tenant_enforcement(
                request=request,
                resource_tenant_id="tenant2",  # Different tenant
            )
            
            assert allowed is False


class TestPIIScrubbing:
    """Test PII scrubbing in logs."""
    
    def test_payload_scrubber_scrubs_secrets(self):
        """Test that payload scrubber redacts secrets."""
        from mlsdm.security.payload_scrubber import scrub_text
        
        # Test various secret patterns
        text_with_secrets = """
        api_key=sk-1234567890abcdefghij
        password=MySecretP@ssw0rd
        token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9
        """
        
        scrubbed = scrub_text(text_with_secrets)
        
        # Secrets should be redacted
        assert "sk-1234567890abcdefghij" not in scrubbed
        assert "MySecretP@ssw0rd" not in scrubbed
        assert "***REDACTED***" in scrubbed
        
    def test_payload_scrubber_scrubs_dict(self):
        """Test that payload scrubber redacts dict values."""
        from mlsdm.security.payload_scrubber import scrub_dict
        
        payload = {
            "prompt": "Hello, world!",
            "api_key": "sk-1234567890abcdefghij",
            "password": "MySecretP@ssw0rd",
            "user": "john.doe@example.com",
        }
        
        scrubbed = scrub_dict(payload)
        
        # Secrets should be redacted
        assert scrubbed["api_key"] == "***REDACTED***"
        assert scrubbed["password"] == "***REDACTED***"
        # Non-secrets should remain
        assert scrubbed["prompt"] == "Hello, world!"


class TestSecurityLogging:
    """Test security event logging."""
    
    def test_security_logger_logs_policy_violation(self):
        """Test that security logger logs policy violations."""
        from mlsdm.utils.security_logger import get_security_logger
        
        logger = get_security_logger()
        
        # Should not raise exception
        logger.log_policy_violation(
            client_id="test-client",
            policy="test-policy",
            reason="Test violation",
        )
        
    def test_security_logger_logs_auth_failure(self):
        """Test that security logger logs auth failures."""
        from mlsdm.utils.security_logger import get_security_logger
        
        logger = get_security_logger()
        
        # Should not raise exception
        logger.log_auth_failure(
            client_id="test-client",
            reason="Invalid token",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
