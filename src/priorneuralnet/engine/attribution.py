"""Concept-level and input-level Integrated Gradients (pure PyTorch).

No external attribution libraries: the straight-line path integral is short,
exact for the linear concept layer, and free of dependency drift.

For `PriorNeuralNet`:
  * `concept_attributions` integrates d(prediction)/d(latent) along the path
    0 -> encode(x) and returns the first `n_concepts` entries: the named
    concept -> prediction edge weights.
  * `input_attributions` integrates along the path x=0 -> x for standardized
    inputs: the feature -> prediction edge weights.
  * `feature_concept_flows` gives (feature -> concept) edge weights. For the
    linear concept layer the Integrated Gradients edge weight is exactly
    w_eff[concept, feature] * x_feature (the path integral of a linear map),
    averaged in magnitude over the dataset — no approximation involved.
"""

from __future__ import annotations

import torch

from priorneuralnet.engine.config import ConceptAttribution, FeatureContribution
from priorneuralnet.engine.layers import PriorNeuralNet

DEFAULT_STEPS = 32


def input_attributions(
    model: PriorNeuralNet,
    x: torch.Tensor,
    target_index: int = 1,
    steps: int = DEFAULT_STEPS,
) -> torch.Tensor:
    """IG w.r.t. input features along the straight-line path 0 -> x. [n, F]"""
    model.eval()
    x = x.detach()
    grad_sum = torch.zeros_like(x)
    with torch.enable_grad():
        for alpha in torch.linspace(0.0, 1.0, steps + 1)[1:]:
            xa = (alpha * x).requires_grad_(True)
            out = model(xa)
            out[:, target_index].sum().backward()
            grad_sum += xa.grad
    return x * (grad_sum / steps)


def concept_attributions(
    model: PriorNeuralNet,
    x: torch.Tensor,
    target_index: int = 1,
    steps: int = DEFAULT_STEPS,
) -> torch.Tensor:
    """IG w.r.t. the latent vector returned by `encode` along 0 -> encode(x).

    Rows correspond to [concept_0..C-1, residual_0..R-1]; callers wanting only
    the named concepts slice out the leading `n_concepts` columns. [n, latent]
    """
    model.eval()
    with torch.no_grad():
        latent = model.encode(x)
    grad_sum = torch.zeros_like(latent)
    with torch.enable_grad():
        for alpha in torch.linspace(0.0, 1.0, steps + 1)[1:]:
            la = (alpha * latent).requires_grad_(True)
            out = model.head(la)
            out[:, target_index].sum().backward()
            grad_sum += la.grad
    return latent * (grad_sum / steps)


def feature_concept_flows(model: PriorNeuralNet, x: torch.Tensor) -> torch.Tensor:
    """Exact feature -> concept edge weights. [C, F]

    entry[c, f] = w_eff[c, f] * mean_n(|x_n,f|): magnitude of the IG edge for
    a linear layer on typical data. Zero for mask-inactive edges.
    """
    with torch.no_grad():
        w_eff = model.concept_layer.effective_weight()
        mean_abs = x.detach().abs().mean(dim=0)
    return w_eff * mean_abs.unsqueeze(0)


def rank_concepts(
    model: PriorNeuralNet,
    x: torch.Tensor,
    concept_names: list[str],
    feature_names: list[str],
    target_index: int = 1,
    steps: int = DEFAULT_STEPS,
    top_features_per_concept: int = 10,
) -> list[ConceptAttribution]:
    """Rank named concepts by |IG| and list their top feature drivers."""
    latent_ig = concept_attributions(model, x, target_index=target_index, steps=steps)
    concept_ig = latent_ig[:, : model.n_concepts].abs().mean(dim=0)
    flows = feature_concept_flows(model, x)

    order = torch.argsort(concept_ig, descending=True).tolist()
    out: list[ConceptAttribution] = []
    for rank, c in enumerate(order):
        edges = flows[c]
        nonzero = torch.nonzero(edges).flatten()
        if nonzero.numel():
            top = nonzero[torch.argsort(edges[nonzero].abs(), descending=True)]
            top = top[:top_features_per_concept].tolist()
        else:
            top = []
        out.append(
            ConceptAttribution(
                concept=concept_names[c],
                attribution=float(concept_ig[c]),
                rank=rank + 1,
                top_features=[
                    FeatureContribution(
                        feature=feature_names[f],
                        concept=concept_names[c],
                        contribution=float(edges[f]),
                    )
                    for f in top
                ],
            )
        )
    return out
