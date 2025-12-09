# Documentation Improvements Backlog

**Generated**: 2025-12-09  
**Source**: REALITY_CHECK_REPORT.md  
**Priority**: P2 (Medium Risk) and P3 (Nice to Have)

---

## Priority 2: Should Fix Before v1.0

### P2-1: Test Count Clarification (5 min)

**Issue**: "577 tests" appears in CORE_IMPLEMENTATION_VALIDATION.md but scope unclear. Full suite is 1,587 tests.

**Current State**:
- CORE_IMPLEMENTATION_VALIDATION.md: "577 tests collected"
- COVERAGE_REPORT_2025.md: "1,587 tests passed"
- Scope difference not explained

**Fix**:
```markdown
# In CORE_IMPLEMENTATION_VALIDATION.md, line 29
Replace: "577 tests collected for core components"
With: "577 tests collected for core cognitive modules (memory, cognition, rhythm, speech)"

# Add clarification
"Note: Full test suite contains 1,587 tests (see COVERAGE_REPORT_2025.md). 
This validation focuses only on core cognitive components."
```

**Files to Update**:
- `CORE_IMPLEMENTATION_VALIDATION.md` (add scope clarification)

---

### P2-2: Memory Footprint Reproduction (5 min)

**Issue**: Benchmark script exists but dependency requirements not documented in user-facing docs.

**Fix**:
Add to GETTING_STARTED.md or TESTING_GUIDE.md:

```markdown
## Verifying Key Metrics

### Memory Footprint (29.37 MB claim)

```bash
# Install dependencies
pip install numpy

# Run benchmark
python benchmarks/measure_memory_footprint.py

# Expected output: ~29.37 MB (within 10% margin)
```

### Effectiveness Metrics

```bash
# Install full dependencies
pip install -e .

# Moral filter effectiveness (93.3% toxic rejection)
pytest tests/validation/test_moral_filter_effectiveness.py -v -s

# Wake/sleep effectiveness (89.5% resource savings)
pytest tests/validation/test_wake_sleep_effectiveness.py -v -s

# Aphasia detection (100% TPR, 80% TNR)
pytest tests/eval/test_aphasia_eval_suite.py -v
```
```

**Files to Update**:
- `GETTING_STARTED.md` or `TESTING_GUIDE.md` (add "Verifying Metrics" section)

---

### P2-3: Aphasia Corpus Size Caveat (2 min)

**Issue**: "100% TPR" claimed but corpus is only 50 telegraphic + 50 normal samples.

**Fix**:
```markdown
# In README.md, line 135 and 572
Replace: "100% TPR, 80% TNR"
With: "100% TPR, 80% TNR (on 50+50 sample corpus)"

# In CLAIMS_TRACEABILITY.md, add note
"Corpus Size: 100 samples (50 telegraphic, 50 normal) - adequate for validation 
but limited for production claims. See tests/eval/aphasia_corpus.json."
```

**Files to Update**:
- `README.md` (add caveat to aphasia metrics)
- `CLAIMS_TRACEABILITY.md` (add corpus limitation note)

---

### P2-4: Zero Growth Runtime Validation (1 hour)

**Issue**: "Zero allocation after init" claimed but only validated via static analysis, not runtime profiling.

**Fix**: Create property-based test for memory growth:

```python
# tests/property/test_zero_allocation.py
import pytest
import psutil
import numpy as np
from mlsdm.memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory

def test_pelm_zero_growth_after_init():
    """Verify PELM memory does not grow after reaching capacity."""
    pelm = PhaseEntangledLatticeMemory(dimension=384, capacity=1000)
    
    # Fill to capacity
    for i in range(1000):
        vec = np.random.randn(384).astype(np.float32).tolist()
        pelm.entangle(vec, phase=0.5)
    
    # Measure baseline
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    
    # Insert 1000 more items (should trigger eviction, not growth)
    for i in range(1000):
        vec = np.random.randn(384).astype(np.float32).tolist()
        pelm.entangle(vec, phase=0.5)
    
    # Measure after
    after_mb = process.memory_info().rss / (1024 * 1024)
    
    # Allow 5% growth for GC overhead
    assert after_mb <= baseline_mb * 1.05, \
        f"Memory grew from {baseline_mb:.2f} MB to {after_mb:.2f} MB"
```

**Files to Create**:
- `tests/property/test_zero_allocation.py`

**Dependencies**: `psutil` (add to requirements.txt or test requirements)

---

### P2-5: "97.8% Comprehensive Safety" Definition (5 min)

**Issue**: Unclear what "comprehensive" means - is it aggregated metrics? Different datasets?

**Fix**:
```markdown
# In CLAIMS_TRACEABILITY.md, line 38
Add clarification:
"Comprehensive Toxic Rejection: 97.8% (aggregated across toxic, unsafe, PII datasets)"

# Or if definition is different, document actual calculation:
"Comprehensive Safety Metric = (toxic_rejection_rate × 0.7) + 
                                 (pii_detection_rate × 0.2) + 
                                 (safety_check_pass_rate × 0.1)"
```

**Files to Update**:
- `CLAIMS_TRACEABILITY.md` (add metric definition)
- `tests/validation/test_moral_filter_effectiveness.py` (add comment explaining calculation)

---

## Priority 3: Nice to Have

### P3-1: STRIDE Security Independent Verification (15 min)

**Issue**: Tests exist but not verified in reality check audit.

**Action**: Run and document:
```bash
pytest tests/security/test_guardrails_stride.py -v
```

**Expected**: All 21 tests pass (9 STRIDE + 12 integration)

**Files to Update**:
- `REALITY_CHECK_REPORT.md` (update C15 status from B to A if tests pass)

---

### P3-2: Test Organization Documentation (20 min)

**Issue**: Different test scopes not clearly mapped in documentation.

**Fix**: Add to TESTING_GUIDE.md:

```markdown
## Test Organization

| Directory | Scope | Test Count | Coverage Focus |
|-----------|-------|------------|----------------|
| `tests/unit/` | Unit tests for all modules | ~1,200 | All source modules |
| `tests/state/` | State persistence unit tests | ~31 | State management |
| `tests/integration/` | Component integration | ~50 | Module interactions |
| `tests/validation/` | Effectiveness validation | ~4 | Key metrics |
| `tests/eval/` | Evaluation suites | ~9 | Aphasia, Sapolsky |
| `tests/property/` | Property-based tests | ~50 | Invariants |
| `tests/security/` | Security tests | ~38 | STRIDE controls |
| `tests/load/` | Load tests (Locust) | ~3 | Throughput |
| **Total** | **Full Suite** | **~1,587** | **70.85% coverage** |

### Test Scopes

**Core Cognitive Modules Only** (used in CORE_IMPLEMENTATION_VALIDATION.md):
```bash
pytest tests/unit/test_cognitive_controller.py \
       tests/unit/test_llm_wrapper.py \
       tests/unit/test_*_memory.py \
       tests/unit/test_moral_filter*.py \
       tests/unit/test_cognitive_rhythm.py \
       --co -q
# Result: ~577 tests
```

**Full Coverage Suite** (used in coverage_gate.sh):
```bash
pytest tests/unit/ tests/state/ --cov=src/mlsdm
# Result: 1,587 tests, 70.85% coverage
```
```

**Files to Update**:
- `TESTING_GUIDE.md` (add "Test Organization" section)

---

### P3-3: Neurobiological Language Disclaimer (0 min - Already Done)

**Status**: ✅ Already handled well

**Location**: CLAIMS_TRACEABILITY.md lines 121-124 clarify metaphorical nature.

**No Action Required**: Documentation is already clear that "neurobiologically-inspired" means computational principles inspired by neuroscience, not neural simulation.

---

## Summary

### Effort Estimates

| Priority | Tasks | Total Effort |
|----------|-------|--------------|
| P2 | 5 tasks | ~1.5 hours |
| P3 | 3 tasks | ~35 minutes |
| **Total** | **8 tasks** | **~2 hours** |

### Impact

- **P2 fixes** make documentation **professional-grade**
- **P3 fixes** are optional polish

### Recommendation

Complete **P2 tasks before v1.0 release**. P3 tasks can be done incrementally.

---

## References

- [REALITY_CHECK_REPORT.md](REALITY_CHECK_REPORT.md) - Full analysis
- [CLAIMS_TRACEABILITY.md](CLAIMS_TRACEABILITY.md) - Current claims matrix
- [COVERAGE_REPORT_2025.md](COVERAGE_REPORT_2025.md) - Coverage details
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Test execution guide
