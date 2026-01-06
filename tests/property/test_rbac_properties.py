"""
RBAC hierarchy property tests.

Property-based tests for Role-Based Access Control hierarchy.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsdm.security.rbac import Role, UserContext, ROLE_HIERARCHY


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
        # Test using actual ROLE_HIERARCHY
        # admin should include write and read
        assert Role.ADMIN in ROLE_HIERARCHY
        assert Role.WRITE in ROLE_HIERARCHY[Role.ADMIN]
        assert Role.READ in ROLE_HIERARCHY[Role.ADMIN]
        
        # write should include read
        assert Role.WRITE in ROLE_HIERARCHY
        assert Role.READ in ROLE_HIERARCHY[Role.WRITE]
        
        # read should only include itself
        assert Role.READ in ROLE_HIERARCHY
        assert ROLE_HIERARCHY[Role.READ] == {Role.READ}
        
        # Test using UserContext
        admin_user = UserContext(user_id="admin", roles={Role.ADMIN})
        assert admin_user.has_role(Role.ADMIN)
        assert admin_user.has_role(Role.WRITE)
        assert admin_user.has_role(Role.READ)
        
        write_user = UserContext(user_id="writer", roles={Role.WRITE})
        assert write_user.has_role(Role.WRITE)
        assert write_user.has_role(Role.READ)
        assert not write_user.has_role(Role.ADMIN)
        
        read_user = UserContext(user_id="reader", roles={Role.READ})
        assert read_user.has_role(Role.READ)
        assert not read_user.has_role(Role.WRITE)
        assert not read_user.has_role(Role.ADMIN)
    
    @settings(max_examples=50, deadline=None)
    @given(
        role_name=st.sampled_from(["READ", "WRITE", "ADMIN"]),
    )
    def test_permission_checks_consistent(self, role_name):
        """Property: Permission checks are consistent."""
        role = Role[role_name]
        
        # Get permissions for this role
        permissions = ROLE_HIERARCHY.get(role, set())
        
        # Check consistency
        if Role.ADMIN in permissions:
            assert Role.WRITE in permissions
            assert Role.READ in permissions
        if Role.WRITE in permissions:
            assert Role.READ in permissions
        
        # Test with UserContext
        user = UserContext(user_id="test", roles={role})
        
        # User should have all permissions in hierarchy
        for perm in permissions:
            assert user.has_role(perm)


pytestmark = pytest.mark.property
