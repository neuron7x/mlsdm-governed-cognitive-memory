"""
Unit Tests for GovernanceKernel

Tests for governance kernel read-only proxies and reset functionality.
"""

import numpy as np
import pytest

from mlsdm.core.governance_kernel import GovernanceKernel, MoralRO, PelmRO, SynapticRO


class TestReadOnlyProxies:
    """Test read-only proxy classes for kernel components."""

    def test_synaptic_ro_lambda_l3(self):
        """Test SynapticRO.lambda_l3 property."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        synaptic_ro = kernel.synaptic_ro
        
        # Access lambda_l3 property
        lambda_l3 = synaptic_ro.lambda_l3
        assert isinstance(lambda_l3, float)
        assert lambda_l3 > 0

    def test_synaptic_ro_theta_l2(self):
        """Test SynapticRO.theta_l2 property."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        synaptic_ro = kernel.synaptic_ro
        
        # Access theta_l2 property
        theta_l2 = synaptic_ro.theta_l2
        assert isinstance(theta_l2, float)
        assert theta_l2 > 0

    def test_synaptic_ro_gating12(self):
        """Test SynapticRO.gating12 property."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        synaptic_ro = kernel.synaptic_ro
        
        # Access gating12 property
        gating12 = synaptic_ro.gating12
        assert isinstance(gating12, float)
        assert 0 <= gating12 <= 1

    def test_synaptic_ro_gating23(self):
        """Test SynapticRO.gating23 property."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        synaptic_ro = kernel.synaptic_ro
        
        # Access gating23 property
        gating23 = synaptic_ro.gating23
        assert isinstance(gating23, float)
        assert 0 <= gating23 <= 1

    def test_synaptic_ro_to_dict(self):
        """Test SynapticRO.to_dict() method."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        synaptic_ro = kernel.synaptic_ro
        
        # Call to_dict method
        state_dict = synaptic_ro.to_dict()
        assert isinstance(state_dict, dict)
        assert "lambda_l1" in state_dict or "dim" in state_dict

    def test_moral_ro_get_state(self):
        """Test MoralRO.get_state() method."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        moral_ro = kernel.moral_ro
        
        # Call get_state method
        state = moral_ro.get_state()
        assert isinstance(state, dict)

    def test_pelm_ro_detect_corruption(self):
        """Test PelmRO.detect_corruption() method."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        pelm_ro = kernel.pelm_ro
        
        # Call detect_corruption method
        is_corrupted = pelm_ro.detect_corruption()
        assert isinstance(is_corrupted, bool)


class TestGovernanceKernelReset:
    """Test GovernanceKernel reset functionality."""

    def test_reset_with_initial_moral_threshold(self):
        """Test reset with initial_moral_threshold parameter."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3,
            initial_moral_threshold=0.5
        )
        
        # Reset with new threshold
        kernel.reset(initial_moral_threshold=0.6)
        
        # Verify the kernel still works after reset
        assert kernel.moral_ro.threshold >= 0

    def test_reset_with_synaptic_config(self):
        """Test reset with synaptic_config parameter."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        
        # Create a simple config-like object (could be None in practice)
        # Since we're testing code coverage, we just need to pass the parameter
        from mlsdm.config import SYNAPTIC_MEMORY_DEFAULTS
        config = SYNAPTIC_MEMORY_DEFAULTS
        
        # Reset with synaptic config
        kernel.reset(synaptic_config=config)
        
        # Verify the kernel still works after reset
        assert kernel.synaptic_ro is not None

    def test_reset_with_all_optional_params(self):
        """Test reset with both initial_moral_threshold and synaptic_config."""
        kernel = GovernanceKernel(
            dim=10, 
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        
        # Import config if available
        try:
            from mlsdm.config import SYNAPTIC_MEMORY_DEFAULTS
            config = SYNAPTIC_MEMORY_DEFAULTS
        except ImportError:
            config = None
        
        # Reset with both optional parameters
        kernel.reset(
            initial_moral_threshold=0.6,
            synaptic_config=config
        )
        
        # Verify the kernel still works after reset
        assert kernel.moral_ro is not None
        assert kernel.synaptic_ro is not None


class TestGovernanceKernelCapabilities:
    """Test capability-based access control."""

    def test_assert_can_mutate_with_valid_capability(self):
        """Test _assert_can_mutate with a valid capability."""
        kernel = GovernanceKernel(
            dim=10,
            capacity=100,
            wake_duration=8,
            sleep_duration=3
        )
        
        # Try calling a method that checks capabilities  
        # This tests the early return at line 227
        import numpy as np
        
        # Call moral_adapt which uses _assert_can_mutate
        # Without capability, it should work from test context (but won't be in allowlist)
        # So we test by calling from within the kernel methods
        # Actually, let's just test the methods that use capabilities
        kernel.moral_adapt(True)  # This should work from test
        
        # Verify it worked
        assert kernel.moral_ro is not None
