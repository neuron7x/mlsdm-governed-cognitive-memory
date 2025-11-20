"""
Performance Benchmark Tests

Validate latency and throughput SLOs.
Principal-level performance validation.
"""

import numpy as np
import time
import sys
from dataclasses import dataclass
from typing import List

sys.path.insert(0, '.')
from src.core.cognitive_controller import CognitiveController


@dataclass
class LatencyMetrics:
    """Latency measurement results"""
    samples: int
    mean: float
    p50: float
    p95: float
    p99: float
    p999: float
    min: float
    max: float


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def measure_latencies(self, num_samples: int = 1000) -> LatencyMetrics:
        """Measure operation latencies"""
        controller = CognitiveController(dim=384)
        latencies = []
        
        # Warmup
        for _ in range(100):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            controller.process_event(vec, moral_value=0.8)
        
        # Actual measurement
        for _ in range(num_samples):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            moral_val = np.random.uniform(0.3, 0.95)
            
            start = time.perf_counter()
            state = controller.process_event(vec, moral_value=moral_val)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        latencies.sort()
        return LatencyMetrics(
            samples=num_samples,
            mean=float(np.mean(latencies)),
            p50=float(np.percentile(latencies, 50)),
            p95=float(np.percentile(latencies, 95)),
            p99=float(np.percentile(latencies, 99)),
            p999=float(np.percentile(latencies, 99.9)),
            min=float(min(latencies)),
            max=float(max(latencies))
        )
    
    def test_p95_latency_slo(self):
        """Test P95 latency meets SLO (<120ms)"""
        metrics = self.measure_latencies(num_samples=2000)
        
        print(f"\nLatency Profile ({metrics.samples} samples):")
        print(f"  Mean:  {metrics.mean:.2f} ms")
        print(f"  P50:   {metrics.p50:.2f} ms")
        print(f"  P95:   {metrics.p95:.2f} ms")
        print(f"  P99:   {metrics.p99:.2f} ms")
        print(f"  P99.9: {metrics.p999:.2f} ms")
        
        # Validate SLO
        assert metrics.p95 < 120, \
            f"P95 latency {metrics.p95:.2f}ms exceeds SLO (120ms)"
        
        print(f"✅ P95 latency SLO met: {metrics.p95:.2f}ms < 120ms")
    
    def test_p99_latency_slo(self):
        """Test P99 latency meets SLO (<200ms)"""
        metrics = self.measure_latencies(num_samples=2000)
        
        # Validate SLO
        assert metrics.p99 < 200, \
            f"P99 latency {metrics.p99:.2f}ms exceeds SLO (200ms)"
        
        print(f"✅ P99 latency SLO met: {metrics.p99:.2f}ms < 200ms")
    
    def test_throughput_baseline(self):
        """Test throughput baseline (target: >1000 ops/sec)"""
        controller = CognitiveController(dim=384)
        
        num_ops = 5000
        start_time = time.time()
        
        for _ in range(num_ops):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            moral_val = np.random.uniform(0.3, 0.95)
            controller.process_event(vec, moral_value=moral_val)
        
        elapsed = time.time() - start_time
        throughput = num_ops / elapsed
        
        print(f"\nThroughput: {throughput:.2f} ops/sec")
        print(f"  Total ops: {num_ops}")
        print(f"  Duration: {elapsed:.2f}s")
        
        # Baseline validation
        assert throughput > 1000, \
            f"Throughput {throughput:.2f} ops/sec below baseline (1000 ops/sec)"
        
        print(f"✅ Throughput baseline met: {throughput:.2f} ops/sec")
    
    def test_concurrent_throughput(self):
        """Test concurrent throughput with multiple threads"""
        import threading
        
        controller = CognitiveController(dim=384)
        num_threads = 10
        ops_per_thread = 500
        
        def worker():
            for _ in range(ops_per_thread):
                vec = np.random.randn(384).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                moral_val = np.random.uniform(0.3, 0.95)
                controller.process_event(vec, moral_value=moral_val)
        
        start_time = time.time()
        
        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        elapsed = time.time() - start_time
        total_ops = num_threads * ops_per_thread
        throughput = total_ops / elapsed
        
        print(f"\nConcurrent Throughput:")
        print(f"  Threads: {num_threads}")
        print(f"  Total ops: {total_ops}")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} ops/sec")
        
        # Should scale reasonably with concurrency
        assert throughput > 1000, \
            f"Concurrent throughput {throughput:.2f} ops/sec too low"
        
        print(f"✅ Concurrent throughput: {throughput:.2f} ops/sec")
    
    def test_memory_footprint(self):
        """Test memory footprint stays within bounds"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_rss = process.memory_info().rss / 1024 / 1024  # MB
        
        controller = CognitiveController(dim=384)
        after_init_rss = process.memory_info().rss / 1024 / 1024
        
        # Process events
        for _ in range(10000):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            moral_val = np.random.uniform(0.3, 0.95)
            controller.process_event(vec, moral_value=moral_val)
        
        final_rss = process.memory_info().rss / 1024 / 1024
        
        print(f"\nMemory Footprint:")
        print(f"  Initial:     {initial_rss:.2f} MB")
        print(f"  After init:  {after_init_rss:.2f} MB")
        print(f"  Final:       {final_rss:.2f} MB")
        print(f"  Increase:    {final_rss - initial_rss:.2f} MB")
        
        # Validate memory bounds (≤1.4 GB as per spec)
        assert final_rss < 1400, \
            f"Memory footprint {final_rss:.2f}MB exceeds limit (1400MB)"
        
        # Check for memory leaks
        event_increase = final_rss - after_init_rss
        assert event_increase < 100, \
            f"Memory grew by {event_increase:.2f}MB (potential leak)"
        
        print(f"✅ Memory footprint within bounds: {final_rss:.2f} MB")
    
    def test_latency_stability_over_time(self):
        """Test that latency remains stable over extended operation"""
        controller = CognitiveController(dim=384)
        
        # Measure latency in windows
        window_size = 500
        num_windows = 10
        window_p95s = []
        
        for window in range(num_windows):
            latencies = []
            
            for _ in range(window_size):
                vec = np.random.randn(384).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                moral_val = np.random.uniform(0.3, 0.95)
                
                start = time.perf_counter()
                controller.process_event(vec, moral_value=moral_val)
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)
            
            p95 = np.percentile(latencies, 95)
            window_p95s.append(p95)
        
        # Check stability
        p95_variance = np.var(window_p95s)
        p95_mean = np.mean(window_p95s)
        
        print(f"\nLatency Stability:")
        print(f"  Windows: {num_windows}")
        print(f"  Mean P95: {p95_mean:.2f} ms")
        print(f"  Variance: {p95_variance:.2f}")
        print(f"  Min P95:  {min(window_p95s):.2f} ms")
        print(f"  Max P95:  {max(window_p95s):.2f} ms")
        
        # P95 should be relatively stable
        assert p95_variance < 100, \
            f"P95 variance {p95_variance:.2f} too high (unstable latency)"
        
        print(f"✅ Latency stable over time")


def test_performance_suite():
    """Run all performance tests"""
    test_suite = TestPerformanceBenchmarks()
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK TESTS")
    print("="*60)
    
    test_suite.test_p95_latency_slo()
    test_suite.test_p99_latency_slo()
    test_suite.test_throughput_baseline()
    test_suite.test_concurrent_throughput()
    test_suite.test_memory_footprint()
    test_suite.test_latency_stability_over_time()
    
    print("\n" + "="*60)
    print("✅ ALL PERFORMANCE TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    test_performance_suite()
