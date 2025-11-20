"""
Fault Injection Tests

Test system behavior under various failure scenarios.
Principal-level chaos engineering validation.
"""

import numpy as np
import pytest
import sys
import threading
import time
from unittest.mock import patch, MagicMock

sys.path.insert(0, '.')
from src.core.cognitive_controller import CognitiveController


class TestFaultInjection:
    """Test fault injection scenarios"""
    
    def test_high_concurrency_race_conditions(self):
        """Test system under extreme concurrent load"""
        controller = CognitiveController(dim=384)
        errors = []
        results = []
        
        def worker(worker_id):
            try:
                for _ in range(100):
                    vec = np.random.randn(384).astype(np.float32)
                    vec = vec / np.linalg.norm(vec)
                    state = controller.process_event(vec, moral_value=0.8)
                    results.append(state)
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Launch 50 concurrent workers
        threads = []
        for i in range(50):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Encountered {len(errors)} errors: {errors[:5]}"
        assert len(results) == 5000, f"Expected 5000 results, got {len(results)}"
        print(f"✅ High concurrency test passed: {len(results)} events processed")
    
    def test_invalid_vector_inputs(self):
        """Test graceful handling of invalid inputs"""
        controller = CognitiveController(dim=384)
        
        # Test 1: Zero vector
        zero_vec = np.zeros(384, dtype=np.float32)
        try:
            state = controller.process_event(zero_vec, moral_value=0.8)
            # Should either handle gracefully or reject
            assert isinstance(state, dict)
        except (ValueError, ZeroDivisionError):
            # Expected - zero norm vectors are invalid
            pass
        
        # Test 2: NaN vector - system may handle gracefully
        nan_vec = np.full(384, np.nan, dtype=np.float32)
        try:
            state = controller.process_event(nan_vec, moral_value=0.8)
            # System handled it gracefully
            assert isinstance(state, dict)
        except (ValueError, RuntimeError):
            # Or raised an exception - both are acceptable
            pass
        
        # Test 3: Inf vector - system may handle gracefully
        inf_vec = np.full(384, np.inf, dtype=np.float32)
        try:
            state = controller.process_event(inf_vec, moral_value=0.8)
            # System handled it gracefully
            assert isinstance(state, dict)
        except (ValueError, RuntimeError):
            # Or raised an exception - both are acceptable
            pass
        
        print("✅ Invalid input handling test passed")
    
    def test_extreme_moral_values(self):
        """Test behavior with extreme moral values"""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Test boundary values
        test_cases = [0.0, 0.001, 0.999, 1.0, -0.1, 1.1]
        
        for moral_val in test_cases:
            try:
                state = controller.process_event(vec, moral_value=moral_val)
                assert isinstance(state, dict)
                # System should clamp or handle extremes
                assert 0.3 <= state["moral_threshold"] <= 0.9
            except ValueError:
                # Some extreme values may be rejected
                pass
        
        print("✅ Extreme moral values test passed")
    
    def test_rapid_phase_transitions(self):
        """Test rapid wake/sleep transitions"""
        controller = CognitiveController(dim=384)
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        
        # Force rapid phase transitions
        for _ in range(100):
            controller.rhythm.step()
            state = controller.process_event(vec, moral_value=0.8)
            assert isinstance(state, dict)
            assert "phase" in state
        
        print("✅ Rapid phase transitions test passed")
    
    def test_memory_under_toxic_bombardment(self):
        """Test system stability under sustained toxic input"""
        controller = CognitiveController(dim=384)
        
        # Send 1000 toxic events
        toxic_accepted = 0
        toxic_rejected = 0
        
        for _ in range(1000):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            
            # Toxic moral value
            moral_val = np.random.uniform(0.1, 0.3)
            state = controller.process_event(vec, moral_value=moral_val)
            
            if state["rejected"]:
                toxic_rejected += 1
            else:
                toxic_accepted += 1
        
        rejection_rate = toxic_rejected / 1000
        print(f"Toxic bombardment: {rejection_rate*100:.1f}% rejection rate")
        
        # System should reject most toxic content
        assert rejection_rate > 0.5, f"Expected >50% rejection, got {rejection_rate*100:.1f}%"
        print("✅ Toxic bombardment test passed")
    
    def test_concurrent_phase_transitions(self):
        """Test concurrent operations during phase transitions"""
        controller = CognitiveController(dim=384)
        errors = []
        
        def phase_stepper():
            """Thread that steps rhythm"""
            for _ in range(50):
                controller.rhythm.step()
                time.sleep(0.001)
        
        def event_processor():
            """Thread that processes events"""
            try:
                for _ in range(100):
                    vec = np.random.randn(384).astype(np.float32)
                    vec = vec / np.linalg.norm(vec)
                    controller.process_event(vec, moral_value=0.8)
            except Exception as e:
                errors.append(str(e))
        
        # Start threads
        stepper = threading.Thread(target=phase_stepper)
        processors = [threading.Thread(target=event_processor) for _ in range(10)]
        
        stepper.start()
        for p in processors:
            p.start()
        
        stepper.join()
        for p in processors:
            p.join()
        
        assert len(errors) == 0, f"Encountered errors: {errors}"
        print("✅ Concurrent phase transitions test passed")
    
    def test_sustained_load_memory_stability(self):
        """Test memory stability under sustained load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        controller = CognitiveController(dim=384)
        
        # Process 5000 events
        for i in range(5000):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            moral_val = np.random.uniform(0.3, 0.95)
            controller.process_event(vec, moral_value=moral_val)
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"Memory: {initial_memory:.2f} MB → {final_memory:.2f} MB (+{memory_increase:.2f} MB)")
        
        # Memory should not grow significantly
        assert memory_increase < 100, f"Memory leak detected: {memory_increase:.2f} MB increase"
        print("✅ Memory stability test passed")


def test_chaos_resilience():
    """Run all chaos tests"""
    test_suite = TestFaultInjection()
    
    print("\n" + "="*60)
    print("CHAOS ENGINEERING TESTS")
    print("="*60 + "\n")
    
    test_suite.test_high_concurrency_race_conditions()
    test_suite.test_invalid_vector_inputs()
    test_suite.test_extreme_moral_values()
    test_suite.test_rapid_phase_transitions()
    test_suite.test_memory_under_toxic_bombardment()
    test_suite.test_concurrent_phase_transitions()
    test_suite.test_sustained_load_memory_stability()
    
    print("\n" + "="*60)
    print("✅ ALL CHAOS TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    test_chaos_resilience()
