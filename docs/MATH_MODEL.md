# Mathematical Model Formalization for MyceliumFractalNet

**Document Version:** 1.0.0  
**Project Version:** 1.2.0  
**Last Updated:** November 2025  
**Status:** Phase 2 - Numerical Implementation

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Membrane Potential Dynamics](#2-membrane-potential-dynamics)
- [3. Reaction-Diffusion Field](#3-reaction-diffusion-field)
- [4. Fractal Growth Model](#4-fractal-growth-model)
- [5. Numerical Stability Analysis](#5-numerical-stability-analysis)
- [6. Parameter Reference](#6-parameter-reference)
- [7. References](#7-references)

---

## 1. Overview

MyceliumFractalNet is a bio-inspired neural network architecture that models:

1. **Membrane Potentials**: Electrical dynamics of neural membranes using Hodgkin-Huxley-inspired ODEs
2. **Reaction-Diffusion Fields**: Morphogen concentration fields for adaptive signaling
3. **Fractal Growth**: Self-similar network topology generation via DLA-based growth

This document formalizes the mathematical equations implemented in the numerical core.

---

## 2. Membrane Potential Dynamics

### 2.1 Governing Equations

The membrane potential `V` of a neural unit evolves according to a simplified Hodgkin-Huxley model:

```
dV/dt = (1/τ) * (-g_L * (V - E_L) + I_ext + I_syn)
```

Where:
- `V` ∈ [-90, 40] mV: Membrane potential
- `τ` = 10 ms: Membrane time constant (default)
- `g_L` = 0.1 mS/cm²: Leak conductance (default)
- `E_L` = -65 mV: Leak reversal potential (default)
- `I_ext`: External input current (bounded ∈ [-100, 100] µA/cm²)
- `I_syn`: Synaptic input current from connected units

### 2.2 Synaptic Current

Synaptic current is computed from pre-synaptic activity:

```
I_syn = Σ_j (w_ij * S_j * (E_syn - V))
```

Where:
- `w_ij` ∈ [0, 1]: Synaptic weight from unit j to unit i
- `S_j` ∈ [0, 1]: Synaptic gating variable of pre-synaptic unit
- `E_syn`: Synaptic reversal potential (0 mV excitatory, -80 mV inhibitory)

### 2.3 Numerical Integration

**Scheme**: 4th-order Runge-Kutta (RK4) or explicit Euler

**Stability Conditions**:
- Time step `dt` ≤ τ/10 for Euler stability
- For RK4: `dt` ≤ τ/2 provides accurate integration

**Default Parameters**:
- `dt` = 0.1 ms (satisfies CFL condition for τ = 10 ms)
- `max_steps` = 10000 per simulation epoch

### 2.4 Value Bounds

| Parameter | Min | Max | Unit |
|-----------|-----|-----|------|
| V | -90 | 40 | mV |
| g_L | 0.01 | 1.0 | mS/cm² |
| τ | 1.0 | 100.0 | ms |
| I_ext | -100 | 100 | µA/cm² |

---

## 3. Reaction-Diffusion Field

### 3.1 Governing Equations

The morphogen concentration field `u(x, y, t)` evolves according to a reaction-diffusion PDE:

```
∂u/∂t = D * ∇²u + R(u)
```

Where:
- `u` ∈ [0, 1]: Normalized morphogen concentration
- `D` ∈ [0.01, 1.0]: Diffusion coefficient (default: 0.1)
- `R(u)`: Reaction term (logistic growth)
- `∇²`: Laplacian operator (2D discrete approximation)

### 3.2 Reaction Term

Logistic growth with saturation:

```
R(u) = α * u * (1 - u) - β * u
```

Where:
- `α` = 0.1: Growth rate (default)
- `β` = 0.01: Decay rate (default)

### 3.3 Spatial Discretization

**5-point stencil Laplacian** on uniform grid:

```
∇²u[i,j] ≈ (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / h²
```

Where:
- `h`: Grid spacing (default: 1.0)
- Grid size: N × N (default: 64 × 64)

### 3.4 Boundary Conditions

Supported boundary conditions:
1. **Periodic**: `u[0, j] = u[N-1, j]`, `u[i, 0] = u[i, N-1]`
2. **Neumann** (zero-flux): `∂u/∂n = 0` at boundaries
3. **Dirichlet**: `u = u_boundary` at boundaries

Default: **Periodic** (for toroidal topology)

### 3.5 Numerical Integration

**Scheme**: Forward Time Centered Space (FTCS) explicit scheme

**CFL Stability Condition**:

```
dt ≤ h² / (4 * D)
```

For default parameters (D = 0.1, h = 1.0):
```
dt ≤ 2.5
```

**Recommended**: `dt` = 0.1 (10× safety margin)

### 3.6 Value Bounds

| Parameter | Min | Max | Default | Unit |
|-----------|-----|-----|---------|------|
| u | 0.0 | 1.0 | 0.5 | - |
| D | 0.001 | 10.0 | 0.1 | - |
| α | 0.0 | 1.0 | 0.1 | 1/time |
| β | 0.0 | 1.0 | 0.01 | 1/time |
| h | 0.1 | 10.0 | 1.0 | - |
| N | 8 | 512 | 64 | cells |

---

## 4. Fractal Growth Model

### 4.1 Diffusion-Limited Aggregation (DLA)

The fractal structure grows via DLA algorithm:

1. **Initialize** seed particle at center of grid
2. **Launch** random walker from boundary
3. **Random walk** until:
   - Walker touches aggregate → stick (with probability `p_stick`)
   - Walker exits boundary → discard
4. **Repeat** for `n_particles` iterations

### 4.2 Growth Probability

Sticking probability with distance modulation:

```
p_stick(r) = p_base * exp(-λ * (r - r_min))
```

Where:
- `p_base` = 1.0: Base sticking probability
- `λ` = 0.0: Distance decay rate (0 = uniform)
- `r`: Distance from center
- `r_min`: Minimum growth radius

### 4.3 L-System Alternative

Optional L-system based growth:

**Axiom**: `F`

**Rules**:
- `F → F[+F]F[-F]F` (branching)
- Angle: θ = 25.7° (golden angle variant)
- Iterations: 4-6 (default: 5)

### 4.4 Stochastic Control

All random operations use controlled seeding:

```python
rng = np.random.default_rng(seed=random_seed)
```

**Default seed**: 42 (for reproducibility in tests)

### 4.5 Parameters

| Parameter | Min | Max | Default | Description |
|-----------|-----|-----|---------|-------------|
| grid_size | 32 | 1024 | 128 | DLA grid dimension |
| n_particles | 100 | 100000 | 5000 | Number of particles to add |
| p_stick | 0.1 | 1.0 | 1.0 | Sticking probability |
| max_walk_steps | 1000 | 1000000 | 100000 | Max steps per walker |
| random_seed | 0 | 2³² | 42 | RNG seed |

---

## 5. Numerical Stability Analysis

### 5.1 Membrane Engine Stability

**Euler Method Stability**:
For linear ODE `dV/dt = -V/τ`, explicit Euler is stable if:

```
|1 - dt/τ| ≤ 1  →  dt ≤ 2τ
```

Conservative choice: `dt ≤ τ/10`

**RK4 Stability Region**:
RK4 has a larger stability region and is unconditionally stable for
linear problems with appropriate step size.

### 5.2 Reaction-Diffusion Stability

**FTCS CFL Condition**:

For 2D diffusion:
```
dt ≤ h² / (4D)
```

For combined reaction-diffusion with reaction rate `k`:
```
dt ≤ min(h² / (4D), 1/k)
```

### 5.3 NaN/Inf Detection

All engines implement post-step validation. Example from MembraneEngine:

```python
def _validate_state(self) -> None:
    has_nan = bool(np.any(np.isnan(self._V)))
    has_inf = bool(np.any(np.isinf(self._V)))

    if has_nan or has_inf:
        raise StabilityError(
            "NaN or Inf detected in membrane potential",
            step=self._step_count,
            engine=self.ENGINE_NAME,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    v_min, v_max = float(np.min(self._V)), float(np.max(self._V))
    if v_min < self.config.V_min or v_max > self.config.V_max:
        if self.config.clamp_values:
            np.clip(self._V, self.config.V_min, self.config.V_max, out=self._V)
        else:
            raise ValueOutOfRangeError(
                variable_name="V",
                value=v_min if v_min < self.config.V_min else v_max,
                min_value=self.config.V_min,
                max_value=self.config.V_max,
            )
```

### 5.4 Metrics Collection

Each engine collects per-step statistics via dataclass metrics:

```python
@dataclass
class MembraneMetrics:
    max_V: float = float("-inf")
    min_V: float = float("inf")
    mean_V: float = 0.0
    std_V: float = 0.0
    steps_completed: int = 0
    stability_violations: int = 0
```

---

## 6. Parameter Reference

### 6.1 Complete Parameter Table

| Module | Parameter | Type | Default | Range | Description |
|--------|-----------|------|---------|-------|-------------|
| Membrane | τ | float | 10.0 | [1, 100] | Time constant (ms) |
| Membrane | g_L | float | 0.1 | [0.01, 1] | Leak conductance |
| Membrane | E_L | float | -65.0 | [-90, -40] | Leak potential (mV) |
| Membrane | V_init | float | -65.0 | [-90, 40] | Initial potential |
| Membrane | dt | float | 0.1 | [0.001, 1] | Time step (ms) |
| RD | D | float | 0.1 | [0.001, 10] | Diffusion coefficient |
| RD | α | float | 0.1 | [0, 1] | Growth rate |
| RD | β | float | 0.01 | [0, 1] | Decay rate |
| RD | h | float | 1.0 | [0.1, 10] | Grid spacing |
| RD | N | int | 64 | [8, 512] | Grid size |
| RD | boundary | str | "periodic" | - | Boundary condition |
| DLA | grid_size | int | 128 | [32, 1024] | Grid dimension |
| DLA | n_particles | int | 5000 | [100, 100000] | Particle count |
| DLA | p_stick | float | 1.0 | [0.1, 1] | Sticking probability |
| DLA | random_seed | int | 42 | [0, 2³²] | RNG seed |

### 6.2 Configuration Files

Parameters can be loaded from YAML configuration:

```yaml
mycelium_fractal_net:
  membrane:
    tau: 10.0
    g_L: 0.1
    E_L: -65.0
    dt: 0.1
  reaction_diffusion:
    D: 0.1
    alpha: 0.1
    beta: 0.01
    grid_size: 64
    boundary: "periodic"
  fractal_growth:
    method: "dla"
    grid_size: 128
    n_particles: 5000
    random_seed: 42
```

---

## 7. References

### Neuroscience

- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. *The Journal of Physiology*, 117(4), 500-544.

### Reaction-Diffusion

- Turing, A. M. (1952). The Chemical Basis of Morphogenesis. *Philosophical Transactions of the Royal Society of London*, 237(641), 37-72.

### Fractal Growth

- Witten, T. A., & Sander, L. M. (1981). Diffusion-Limited Aggregation, a Kinetic Critical Phenomenon. *Physical Review Letters*, 47(19), 1400-1403.

- Lindenmayer, A. (1968). Mathematical models for cellular interactions in development. *Journal of Theoretical Biology*, 18(3), 280-299.

### Numerical Methods

- Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.

---

## Related Documentation

- [NEURO_FOUNDATIONS.md](NEURO_FOUNDATIONS.md) - Neuroscience foundations
- [SCIENTIFIC_RATIONALE.md](SCIENTIFIC_RATIONALE.md) - Scientific rationale
- [FORMAL_INVARIANTS.md](FORMAL_INVARIANTS.md) - Formal invariants and properties

---

**Status**: ✅ Documented  
**Coverage**: All numerical modules  
**Last Updated**: 2025-11-28  
**Maintainer**: neuron7x
