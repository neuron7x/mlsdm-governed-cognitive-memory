# API Reference

**Document Version:** 1.0.0  
**Project Version:** 1.0.0  
**Last Updated:** November 2025  
**Status:** Production

Complete API reference for MLSDM Governed Cognitive Memory v1.0.0.

## Table of Contents

- [LLMWrapper](#llmwrapper)
- [CognitiveController](#cognitivecontroller)
- [Memory Components](#memory-components)
  - [QILM_v2](#qilm_v2)
  - [MultiLevelSynapticMemory](#multilevelsyn apticmemory)
- [Filtering Components](#filtering-components)
  - [MoralFilterV2](#moralfilterv2)
  - [OntologyMatcher](#ontologymatcher)
- [Rhythm Components](#rhythm-components)
  - [CognitiveRhythm](#cognitiverhythm)
- [Utilities](#utilities)
  - [MetricsCollector](#metricscollector)

---

## LLMWrapper

Universal wrapper for any LLM with cognitive governance and biological constraints.

### Constructor

```python
LLMWrapper(
    llm_generate_fn: Callable[[str, int], str],
    embedding_fn: Callable[[str], np.ndarray],
    dim: int = 384,
    capacity: int = 20000,
    wake_duration: int = 8,
    sleep_duration: int = 3,
    initial_moral_threshold: float = 0.50
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `llm_generate_fn` | Callable | Required | Function that takes (prompt: str, max_tokens: int) and returns generated text |
| `embedding_fn` | Callable | Required | Function that takes text: str and returns embedding as np.ndarray |
| `dim` | int | 384 | Dimension of embeddings (must match embedding_fn output) |
| `capacity` | int | 20000 | Maximum number of vectors in memory |
| `wake_duration` | int | 8 | Number of steps in wake phase |
| `sleep_duration` | int | 3 | Number of steps in sleep phase |
| `initial_moral_threshold` | float | 0.50 | Initial moral filtering threshold (0.30-0.90) |

**Raises:**
- `ValueError`: If parameters are invalid

**Example:**
```python
from src.core.llm_wrapper import LLMWrapper
import numpy as np

def my_llm(prompt: str, max_tokens: int) -> str:
    return "Generated response"

def my_embed(text: str) -> np.ndarray:
    return np.random.randn(384).astype(np.float32)

wrapper = LLMWrapper(
    llm_generate_fn=my_llm,
    embedding_fn=my_embed,
    dim=384
)
```

### Methods

#### generate

Generate text with cognitive governance.

```python
def generate(
    prompt: str,
    moral_value: float,
    max_tokens: Optional[int] = None,
    context_top_k: int = 5
) -> dict
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | str | Required | Input prompt text |
| `moral_value` | float | Required | Moral acceptability score (0.0-1.0) |
| `max_tokens` | int | None | Override default max tokens |
| `context_top_k` | int | 5 | Number of context items to retrieve |

**Returns:** Dictionary with keys:
- `response` (str): Generated text (empty if rejected)
- `accepted` (bool): Whether request was accepted
- `phase` (str): Current phase ("wake" or "sleep")
- `step` (int): Current step counter
- `note` (str): Processing note
- `moral_threshold` (float): Current moral threshold
- `context_items` (int): Number of context items retrieved
- `max_tokens_used` (int): Max tokens used for generation

**Raises:**
- `ValueError`: If moral_value not in [0.0, 1.0]

**Example:**
```python
result = wrapper.generate(
    prompt="Explain quantum computing",
    moral_value=0.9,
    context_top_k=10
)

if result["accepted"]:
    print(result["response"])
else:
    print(f"Rejected: {result['note']}")
```

#### get_state

Get current system state.

```python
def get_state() -> dict
```

**Returns:** Dictionary with keys:
- `step` (int): Current step counter
- `phase` (str): Current phase
- `phase_counter` (int): Counter within current phase
- `moral_threshold` (float): Current moral threshold
- `moral_ema` (float): Exponential moving average of acceptance rate
- `accepted_count` (int): Total accepted requests
- `rejected_count` (int): Total rejected requests
- `qilm_stats` (dict): QILM memory statistics
  - `capacity` (int): Maximum capacity
  - `used` (int): Current usage
  - `memory_mb` (float): Memory usage in MB
- `synaptic_norms` (dict): Synaptic memory L2 norms
  - `L1` (float): L1 layer norm
  - `L2` (float): L2 layer norm
  - `L3` (float): L3 layer norm

**Example:**
```python
state = wrapper.get_state()
print(f"Phase: {state['phase']}, Step: {state['step']}")
print(f"Memory: {state['qilm_stats']['used']}/{state['qilm_stats']['capacity']}")
print(f"Moral threshold: {state['moral_threshold']:.2f}")
```

#### reset

Reset system state (primarily for testing).

```python
def reset() -> None
```

**Example:**
```python
wrapper.reset()
```

---

## NeuroLangWrapper

Extended LLM wrapper with NeuroLang + Aphasia-Broca detection and repair capabilities.

### Constructor

```python
NeuroLangWrapper(
    llm_generate_fn: Callable[[str, int], str],
    embedding_fn: Callable[[str], np.ndarray],
    dim: int = 384,
    capacity: int = 20000,
    wake_duration: int = 8,
    sleep_duration: int = 3,
    initial_moral_threshold: float = 0.50,
    aphasia_detect_enabled: bool = True,
    aphasia_repair_enabled: bool = True,
    aphasia_severity_threshold: float = 0.3
)
```

**Parameters:**

All parameters from `LLMWrapper`, plus:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `aphasia_detect_enabled` | bool | True | Enable/disable Aphasia-Broca detection |
| `aphasia_repair_enabled` | bool | True | Enable/disable automatic repair of detected aphasia |
| `aphasia_severity_threshold` | float | 0.3 | Minimum severity (0.0-1.0) to trigger repair |

**Raises:**
- `ValueError`: If parameters are invalid

**Example:**
```python
from mlsdm.extensions import NeuroLangWrapper
import numpy as np

def my_llm(prompt: str, max_tokens: int) -> str:
    return "Generated response"

def my_embed(text: str) -> np.ndarray:
    return np.random.randn(384).astype(np.float32)

# Full detection + repair (default)
wrapper = NeuroLangWrapper(
    llm_generate_fn=my_llm,
    embedding_fn=my_embed,
    dim=384,
    aphasia_detect_enabled=True,
    aphasia_repair_enabled=True,
    aphasia_severity_threshold=0.3
)

# Monitoring only (detect but don't repair)
wrapper_monitor = NeuroLangWrapper(
    llm_generate_fn=my_llm,
    embedding_fn=my_embed,
    dim=384,
    aphasia_detect_enabled=True,
    aphasia_repair_enabled=False
)

# Detection disabled
wrapper_disabled = NeuroLangWrapper(
    llm_generate_fn=my_llm,
    embedding_fn=my_embed,
    dim=384,
    aphasia_detect_enabled=False
)
```

### Methods

#### generate

Generate text with cognitive governance and Aphasia-Broca detection/repair.

```python
def generate(
    prompt: str,
    moral_value: float = 0.5,
    max_tokens: int = 50
) -> dict
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `prompt` | str | Required | Input prompt text |
| `moral_value` | float | 0.5 | Moral acceptability score (0.0-1.0) |
| `max_tokens` | int | 50 | Maximum tokens for generation |

**Returns:** Dictionary with keys:
- `response` (str): Generated (and possibly repaired) text
- `accepted` (bool): Whether request was accepted by moral filter
- `phase` (str): Current phase ("wake" or "sleep")
- `neuro_enhancement` (str): NeuroLang processing result
- `aphasia_flags` (dict or None): Aphasia detection report (None if detection disabled)
  - `is_aphasic` (bool): Whether text shows aphasia symptoms
  - `severity` (float): Severity score (0.0-1.0)
  - `avg_sentence_len` (float): Average sentence length
  - `function_word_ratio` (float): Ratio of function words
  - `fragment_ratio` (float): Ratio of sentence fragments
  - `flags` (list): List of specific issues detected

**Behavior based on configuration:**

1. **Detection disabled** (`aphasia_detect_enabled=False`):
   - Returns base LLM response without analysis
   - `aphasia_flags` will be `None`

2. **Detection enabled, repair disabled** (`aphasia_detect_enabled=True`, `aphasia_repair_enabled=False`):
   - Analyzes response and includes `aphasia_flags`
   - Does not modify response text (monitoring mode)

3. **Both enabled** (default):
   - Analyzes response
   - Repairs if `is_aphasic=True` and `severity >= aphasia_severity_threshold`
   - Always includes final `aphasia_flags` (reflects original analysis)

**Example:**
```python
# Full detection + repair
result = wrapper.generate(
    prompt="Explain the system",
    moral_value=0.8,
    max_tokens=100
)

if result["accepted"]:
    print(result["response"])
    if result["aphasia_flags"]:
        print(f"Aphasia detected: {result['aphasia_flags']['is_aphasic']}")
        print(f"Severity: {result['aphasia_flags']['severity']:.2f}")
```

---

## CognitiveController

Low-level cognitive controller for direct memory operations.

### Constructor

```python
CognitiveController(dim: int = 384, capacity: int = 20000)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dim` | int | 384 | Vector dimension |
| `capacity` | int | 20000 | Memory capacity |

### Methods

#### process_event

Process a single event vector.

```python
def process_event(
    event_vector: np.ndarray,
    moral_value: float
) -> dict
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `event_vector` | np.ndarray | Event vector of shape (dim,), normalized |
| `moral_value` | float | Moral value (0.0-1.0) |

**Returns:** Dictionary with processing state

**Raises:**
- `ValueError`: If event_vector has wrong shape or moral_value invalid

**Example:**
```python
from src.core.cognitive_controller import CognitiveController
import numpy as np

controller = CognitiveController(dim=384)
vector = np.random.randn(384).astype(np.float32)
vector = vector / np.linalg.norm(vector)

state = controller.process_event(vector, moral_value=0.8)
print(state)
```

#### retrieve_context

Retrieve relevant context vectors from memory.

```python
def retrieve_context(
    query_vector: np.ndarray,
    top_k: int = 5
) -> List[np.ndarray]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query_vector` | np.ndarray | Required | Query vector for retrieval |
| `top_k` | int | 5 | Number of vectors to retrieve |

**Returns:** List of retrieved vectors (up to top_k)

---

## Memory Components

### QILM_v2

Quantum-Inspired Latent Memory with phase entanglement.

#### Constructor

```python
QILM_v2(dim: int, capacity: int = 20000)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dim` | int | Required | Vector dimension |
| `capacity` | int | 20000 | Maximum capacity |

#### Methods

##### entangle_phase

Store vector with phase entanglement.

```python
def entangle_phase(
    event_vector: np.ndarray,
    phase: float
) -> None
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `event_vector` | np.ndarray | Vector to store (dim,) |
| `phase` | float | Phase value for entanglement |

##### retrieve

Retrieve vectors by phase similarity.

```python
def retrieve(
    phase: float,
    tolerance: float = 0.2,
    top_k: int = 5
) -> List[np.ndarray]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `phase` | float | Required | Target phase |
| `tolerance` | float | 0.2 | Phase tolerance |
| `top_k` | int | 5 | Max results |

**Returns:** List of retrieved vectors

---

### MultiLevelSynapticMemory

Three-level synaptic memory with differential decay.

#### Constructor

```python
MultiLevelSynapticMemory(
    dim: int,
    lambda_l1: float = 0.50,
    lambda_l2: float = 0.10,
    lambda_l3: float = 0.01
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `dim` | int | Required | Vector dimension |
| `lambda_l1` | float | 0.50 | L1 decay rate (fast) |
| `lambda_l2` | float | 0.10 | L2 decay rate (medium) |
| `lambda_l3` | float | 0.01 | L3 decay rate (slow) |

#### Methods

##### update

Update memory with new event.

```python
def update(event_vector: np.ndarray) -> None
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `event_vector` | np.ndarray | Event vector to integrate |

##### get_state

Get current memory state.

```python
def get_state() -> dict
```

**Returns:** Dictionary with L1, L2, L3 vectors and norms

---

## Filtering Components

### MoralFilterV2

Adaptive moral filter with homeostatic threshold.

#### Constructor

```python
MoralFilterV2(
    initial_threshold: float = 0.50,
    adapt_rate: float = 0.05,
    ema_alpha: float = 0.1
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `initial_threshold` | float | 0.50 | Starting threshold (0.30-0.90) |
| `adapt_rate` | float | 0.05 | Adaptation step size |
| `ema_alpha` | float | 0.1 | EMA smoothing factor |

#### Methods

##### evaluate

Evaluate if moral value passes threshold.

```python
def evaluate(moral_value: float) -> bool
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `moral_value` | float | Moral score (0.0-1.0) |

**Returns:** True if accepted, False if rejected

##### adapt

Adapt threshold based on acceptance.

```python
def adapt(accepted: bool) -> None
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `accepted` | bool | Whether last evaluation was accepted |

**Example:**
```python
from src.cognition.moral_filter_v2 import MoralFilterV2

filter = MoralFilterV2(0.5)
accepted = filter.evaluate(0.8)
filter.adapt(accepted)
```

---

### OntologyMatcher

Semantic ontology matching with multiple metrics.

#### Constructor

```python
OntologyMatcher(
    ontology_vectors: np.ndarray,
    labels: List[str]
)
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `ontology_vectors` | np.ndarray | Ontology vectors (n_concepts, dim) |
| `labels` | List[str] | Concept labels |

#### Methods

##### match

Find best matching ontology concept.

```python
def match(
    event_vector: np.ndarray,
    metric: str = "cosine"
) -> Tuple[str, float]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `event_vector` | np.ndarray | Required | Query vector |
| `metric` | str | "cosine" | Distance metric ("cosine" or "euclidean") |

**Returns:** Tuple of (label, score)

---

## Rhythm Components

### CognitiveRhythm

Wake/sleep cycle management.

#### Constructor

```python
CognitiveRhythm(
    wake_duration: int = 8,
    sleep_duration: int = 3
)
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `wake_duration` | int | 8 | Wake phase steps |
| `sleep_duration` | int | 3 | Sleep phase steps |

#### Methods

##### step

Advance one step and get current phase.

```python
def step() -> str
```

**Returns:** Current phase ("wake" or "sleep")

##### get_phase_value

Get numerical phase value for memory entanglement.

```python
def get_phase_value() -> float
```

**Returns:** 0.1 (wake) or 0.9 (sleep)

---

## Utilities

### MetricsCollector

Performance and behavior metrics collection.

#### Methods

##### record_event_processing

Record event processing time.

```python
def record_event_processing(duration_ms: float) -> None
```

##### get_statistics

Get collected statistics.

```python
def get_statistics() -> dict
```

**Returns:** Dictionary with metrics

---

## Type Definitions

### Common Types

```python
from typing import Callable, Optional, List, Tuple, Dict
import numpy as np

# LLM generation function type
LLMGenerateFn = Callable[[str, int], str]

# Embedding function type
EmbeddingFn = Callable[[str], np.ndarray]

# State dictionary type
State = Dict[str, Any]
```

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid input parameters
- `RuntimeError`: System state errors
- `TypeError`: Type mismatches

### Error Examples

```python
# Invalid moral value
try:
    result = wrapper.generate("Hello", moral_value=1.5)
except ValueError as e:
    print(f"Invalid moral value: {e}")

# Invalid vector dimension
try:
    vector = np.random.randn(512)  # Wrong dimension
    controller.process_event(vector, 0.8)
except ValueError as e:
    print(f"Dimension mismatch: {e}")
```

---

## Performance Characteristics

### Latency

- **process_event**: ~2ms (P50), ~10ms (P95)
- **retrieve_context**: ~5-15ms depending on top_k
- **generate**: Depends on LLM + overhead (~2-20ms)

### Memory

- **QILM_v2**: Fixed allocation (capacity × dim × 4 bytes)
- **Example**: 20,000 × 384 × 4 = 29.37 MB
- **Total system**: ~30 MB (well under 1.4 GB limit)

### Concurrency

- Thread-safe with internal locking
- Tested at 1000+ concurrent requests
- No lost updates or race conditions

---

## Examples

### Complete Integration Example

```python
from src.core.llm_wrapper import LLMWrapper
import numpy as np
import openai

# Setup OpenAI
openai.api_key = "your-key"

def openai_gen(prompt: str, max_tokens: int) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def simple_embed(text: str) -> np.ndarray:
    # Replace with actual embeddings
    return np.random.randn(384).astype(np.float32)

# Create wrapper
wrapper = LLMWrapper(
    llm_generate_fn=openai_gen,
    embedding_fn=simple_embed,
    dim=384,
    capacity=20000
)

# Generate with governance
result = wrapper.generate(
    prompt="What is artificial intelligence?",
    moral_value=0.9
)

if result["accepted"]:
    print(f"Response: {result['response']}")
    print(f"Phase: {result['phase']}")
    print(f"Context items: {result['context_items']}")
else:
    print(f"Rejected: {result['note']}")

# Check system state
state = wrapper.get_state()
print(f"\nSystem State:")
print(f"  Steps: {state['step']}")
print(f"  Phase: {state['phase']}")
print(f"  Memory: {state['qilm_stats']['used']}/{state['qilm_stats']['capacity']}")
print(f"  Moral threshold: {state['moral_threshold']:.2f}")
```

---

## See Also

- [README.md](README.md) - Project overview
- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Usage examples
- [ARCHITECTURE_SPEC.md](ARCHITECTURE_SPEC.md) - Architecture details
- [examples/](examples/) - Code examples

---

**Version**: 1.0.0  
**Last Updated**: November 2025  
**Maintainer**: neuron7x
