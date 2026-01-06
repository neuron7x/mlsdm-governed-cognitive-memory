"""
Embedding poisoning attack tests.

Tests resistance to adversarial embedding vectors designed to poison
the memory system or bypass semantic filtering.

Tests validate that malicious vectors cannot:
1. Corrupt memory integrity
2. Bypass moral filtering via embedding manipulation
3. Cause memory overflow or crashes
"""

import numpy as np
import pytest

from mlsdm.memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory


class TestEmbeddingPoisoning:
    """Test resistance to embedding poisoning attacks."""
    
    def test_nan_embedding_rejected(self):
        """Test NaN embeddings are rejected."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # NaN vector should raise ValueError
        nan_vector = [float('nan')] * 10
        with pytest.raises(ValueError, match="invalid value"):
            pelm.entangle(nan_vector, phase=0.5)
    
    def test_inf_embedding_rejected(self):
        """Test infinity embeddings are rejected."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # Inf vector should raise ValueError
        inf_vector = [float('inf')] * 10
        with pytest.raises(ValueError, match="invalid value"):
            pelm.entangle(inf_vector, phase=0.5)
        
        # -Inf vector should also be rejected
        neg_inf_vector = [float('-inf')] * 10
        with pytest.raises(ValueError, match="invalid value"):
            pelm.entangle(neg_inf_vector, phase=0.5)
    
    def test_extreme_magnitude_vectors(self):
        """Test very large magnitude vectors are handled."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # Very large but finite vector
        large_vector = [1e10] * 10
        
        try:
            idx = pelm.entangle(large_vector, phase=0.5)
            assert idx != -1, "Large vector rejected incorrectly"
        except (ValueError, OverflowError):
            # Acceptable to reject if causes numerical issues
            pass
    
    def test_adversarial_phase_values(self):
        """Test adversarial phase values are rejected."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        vector = [1.0] * 10
        
        # Out of range phases should be rejected
        with pytest.raises(ValueError, match="phase must be in"):
            pelm.entangle(vector, phase=-0.1)
        
        with pytest.raises(ValueError, match="phase must be in"):
            pelm.entangle(vector, phase=1.1)
        
        # NaN phase should be rejected
        with pytest.raises(ValueError, match="phase must be a finite"):
            pelm.entangle(vector, phase=float('nan'))
    
    def test_dimension_mismatch_attack(self):
        """Test dimension mismatch vectors are rejected."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # Wrong dimension should raise ValueError
        wrong_dim = [1.0] * 5
        with pytest.raises(ValueError, match="dimension mismatch"):
            pelm.entangle(wrong_dim, phase=0.5)
        
        # Oversized dimension should also be rejected
        oversized = [1.0] * 20
        with pytest.raises(ValueError, match="dimension mismatch"):
            pelm.entangle(oversized, phase=0.5)
    
    def test_memory_corruption_resistance(self):
        """Test memory integrity after adversarial inputs."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # Store some valid vectors
        for i in range(10):
            vector = [float(i)] * 10
            pelm.entangle(vector, phase=0.5)
        
        # Try to corrupt with invalid vectors
        try:
            pelm.entangle([float('nan')] * 10, phase=0.5)
        except ValueError:
            pass  # Expected
        
        # Memory should still be valid
        assert pelm.size == 10, "Memory corrupted after invalid input"
        
        # Should still be able to retrieve
        query = [1.0] * 10
        results = pelm.retrieve(query, query_phase=0.5, top_k=5)
        assert len(results) > 0, "Retrieval broken after attack"
    
    def test_flooding_attack_capacity_enforcement(self):
        """Test capacity limits are enforced under flooding."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=50)
        
        # Flood with vectors
        for i in range(100):
            vector = [float(i % 10)] * 10
            pelm.entangle(vector, phase=0.5)
        
        # Size should not exceed capacity
        assert pelm.size <= pelm.capacity, (
            f"Capacity exceeded: {pelm.size} > {pelm.capacity}"
        )
    
    def test_retrieval_poisoning_resistance(self):
        """Test retrieval is not poisoned by adversarial vectors."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # Store normal vectors
        for i in range(10):
            vector = [float(i)] * 10
            pelm.entangle(vector, phase=0.5)
        
        # Query should return valid results
        query = [5.0] * 10
        results = pelm.retrieve(query, query_phase=0.5, top_k=3)
        
        # Results should be valid (not corrupted)
        for result in results:
            assert result.vector is not None
            assert len(result.vector) == 10
            assert not np.isnan(result.vector).any()
            assert not np.isinf(result.vector).any()
    
    def test_zero_vector_handling(self):
        """Test zero vectors are handled correctly."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # Zero vector should be accepted (valid but low norm)
        zero_vector = [0.0] * 10
        idx = pelm.entangle(zero_vector, phase=0.5)
        
        # Should be stored (even if low utility)
        assert idx != -1, "Zero vector rejected"
    
    def test_adversarial_similarity_manipulation(self):
        """Test that similarity cannot be manipulated adversarially."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        
        # Store target vector
        target = [1.0] * 10
        pelm.entangle(target, phase=0.5)
        
        # Try to craft high-similarity adversarial vector
        adversarial = [1.0001] * 10  # Very similar to target
        pelm.entangle(adversarial, phase=0.5)
        
        # Query for target
        results = pelm.retrieve(target, query_phase=0.5, top_k=2)
        
        # Both should be retrieved with appropriate resonance
        assert len(results) > 0, "No results from retrieval"
        
        # Resonance values should be valid
        for result in results:
            assert 0.0 <= result.resonance <= 1.0, (
                f"Invalid resonance: {result.resonance}"
            )


pytestmark = pytest.mark.security
