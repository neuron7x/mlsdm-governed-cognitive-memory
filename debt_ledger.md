DL-001
- Priority: P3
- Gate: test
- Symptom: RuntimeWarning about overflow encountered in dot during TestMemoryContentSafety::test_extreme_magnitude_vectors.
- Evidence: artifacts/baseline/test.log (numpy/linalg/_linalg.py:2792 RuntimeWarning: overflow encountered in dot, triggered by tests/safety/test_memory_leakage.py::TestMemoryContentSafety::test_extreme_magnitude_vectors).
- Likely root cause: Test inputs use extremely large vectors causing numpy.linalg dot product to overflow.
- Smallest fix: Reduce magnitude of the test vectors or guard the computation (e.g., normalization/clipping) to avoid overflow or mark the warning as expected.
- Proof command: source .venv/bin/activate && make test
- Risk: Changing test data or numerical handling could hide real overflow issues or alter safety checks on extreme inputs.
