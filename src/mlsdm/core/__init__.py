"""Core numerical engines for MyceliumFractalNet.

This package contains the numerical simulation engines:
- MembraneEngine: ODE integration for membrane potential dynamics
- ReactionDiffusionEngine: PDE solver for reaction-diffusion fields
- FractalGrowthEngine: DLA-based fractal structure generation

See docs/MATH_MODEL.md for mathematical foundations.
"""

from mlsdm.core.fractal_growth_engine import (
    FractalGrowthConfig,
    FractalGrowthEngine,
    FractalGrowthMetrics,
    GrowthMethod,
)
from mlsdm.core.membrane_engine import (
    IntegrationScheme,
    MembraneConfig,
    MembraneEngine,
    MembraneMetrics,
)
from mlsdm.core.numerical_exceptions import (
    NumericalError,
    StabilityError,
    ValueOutOfRangeError,
)
from mlsdm.core.reaction_diffusion_engine import (
    BoundaryCondition,
    ReactionDiffusionConfig,
    ReactionDiffusionEngine,
    ReactionDiffusionMetrics,
)

__all__ = [
    # Exceptions
    "NumericalError",
    "StabilityError",
    "ValueOutOfRangeError",
    # Membrane Engine
    "IntegrationScheme",
    "MembraneConfig",
    "MembraneEngine",
    "MembraneMetrics",
    # Reaction-Diffusion Engine
    "BoundaryCondition",
    "ReactionDiffusionConfig",
    "ReactionDiffusionEngine",
    "ReactionDiffusionMetrics",
    # Fractal Growth Engine
    "FractalGrowthConfig",
    "FractalGrowthEngine",
    "FractalGrowthMetrics",
    "GrowthMethod",
]
