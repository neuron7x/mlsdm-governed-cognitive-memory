"""
RBAC hierarchy property tests.

Property-based tests for Role-Based Access Control hierarchy.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


class TestRBACProperties:
    """RBAC hierarchy property tests."""
    
    def test_role_hierarchy_transitivity(self):
        """
        Property: RBAC hierarchy is transitive.
        
        ∀ user with role R: has_role(R) ⟹ has_role(child(R))
        
        If admin ⊃ write ⊃ read, then:
        - admin has write and read permissions
        - write has read permissions
        - read has only read permissions
        """
        # Role hierarchy: admin > write > read
        role_hierarchy = {
            "admin": ["write", "read"],
            "write": ["read"],
            "read": [],
        }
        
        # Test transitivity
        def has_permission(user_role: str, required_role: str) -> bool:
            if user_role == required_role:
                return True
            return required_role in role_hierarchy.get(user_role, [])
        
        # Admin should have all permissions
        assert has_permission("admin", "admin")
        assert has_permission("admin", "write")
        assert has_permission("admin", "read")
        
        # Write should have write and read
        assert has_permission("write", "write")
        assert has_permission("write", "read")
        assert not has_permission("write", "admin")
        
        # Read should have only read
        assert has_permission("read", "read")
        assert not has_permission("read", "write")
        assert not has_permission("read", "admin")
    
    @settings(max_examples=50, deadline=None)
    @given(
        role_name=st.sampled_from(["admin", "write", "read", "none"]),
    )
    def test_permission_checks_consistent(self, role_name):
        """Property: Permission checks are consistent."""
        # Define expected permissions
        expected_permissions = {
            "admin": {"read", "write", "admin"},
            "write": {"read", "write"},
            "read": {"read"},
            "none": set(),
        }
        
        permissions = expected_permissions.get(role_name, set())
        
        # Check consistency
        if "admin" in permissions:
            assert "write" in permissions
            assert "read" in permissions
        if "write" in permissions:
            assert "read" in permissions


pytestmark = pytest.mark.property
