"""Domain-Prior Mask-Constrained Neural Networks (priorneuralnet).

Interpretable, structurally-sparsed PyTorch heads for p >> n tabular data:
features are hard-wired into named domain concepts via a binary connectivity
mask, the resulting concept layer is benchmarked against parameter-matched
unconstrained MLPs, and concept-level Integrated Gradients attributions are
exported as compact machine-readable + interactive HTML artifacts.
"""

from priorneuralnet.engine.config import (
    BenchmarkManifest,
    ConceptAttribution,
    PriorGraphSpec,
    TrainConfig,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkManifest",
    "ConceptAttribution",
    "PriorGraphSpec",
    "TrainConfig",
    "__version__",
]
