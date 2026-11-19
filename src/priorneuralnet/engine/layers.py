"""Hard structural sparsity: masked layers, the prior network, and baselines.

`MaskedPriorLinear` enforces that every hidden neuron of the concept layer
represents exactly one named domain concept: its incoming connectivity is
restricted to the features belonging to that concept by a binary mask buffer.
Enforcement is threefold:

1. Initialization is zeroed outside the mask and scaled by *per-row fan-in*
   (a sparse row inited at dense fan-in scale is under-scaled by ~sqrt(n)).
2. `forward()` multiplies weights by the mask, so any stray weight value is
   invisible to the function.
3. A gradient hook zeroes `dW` outside the mask, so Adam-family momentum and
   variance stay exactly zero for inactive connections; `apply_mask_constraints`
   re-projects weights after every optimizer step because decoupled weight
   decay (AdamW) still shrinks parameters whose gradient is zero.

The concept layer is kept *linear* by design: for a linear first layer,
Integrated Gradients against concept activations is exact along the
straight-line path, which is what makes the reported concept/edge attributions
mathematically faithful. Non-linearity is applied downstream of the named
concepts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu, "tanh": torch.tanh}
# Module factories for use inside nn.Sequential (functions are not Modules).
_ACTIVATION_MODULES = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class MaskedPriorLinear(nn.Linear):
    """Linear layer whose connectivity obeys a hard binary mask.

    Parameters mirror `nn.Linear`: mask is a `[out_features, in_features]`
    0/1 tensor where `mask[o, i] = 1` allows connection i -> o.
    """

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor):
        super().__init__(in_features, out_features)
        mask = mask.detach().to(dtype=torch.float32)
        if mask.shape != (out_features, in_features):
            raise ValueError(f"mask shape {tuple(mask.shape)} != ({out_features}, {in_features})")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary")
        self.register_buffer("mask", mask)
        # nn.Linear's default init ignores the mask; redo it mask-aware.
        self._init_masked_weights()
        # Zero gradients outside the mask: Adam momentum/variance for inactive
        # connections then remain exactly zero across all steps.
        self.weight.register_hook(lambda grad: grad * self.mask)

    def _init_masked_weights(self) -> None:
        fan_in = self.mask.sum(dim=1)
        bound = torch.where(
            fan_in > 0,
            torch.sqrt(6.0 / fan_in.clamp(min=1.0)),
            torch.zeros_like(fan_in),
        )
        with torch.no_grad():
            self.weight.uniform_(-1.0, 1.0)
            self.weight.mul_(bound.unsqueeze(1))
            self.weight.mul_(self.mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)

    def effective_weight(self) -> torch.Tensor:
        return (self.weight * self.mask).detach()

    @torch.no_grad()
    def apply_mask_constraint(self) -> None:
        self.weight.mul_(self.mask)


def apply_mask_constraints(root: nn.Module) -> int:
    """Re-project every `MaskedPriorLinear` weight inside `root` onto its mask.

    Call after each optimizer step; returns the number of layers touched.
    """
    count = 0
    for module in root.modules():
        if isinstance(module, MaskedPriorLinear):
            module.apply_mask_constraint()
            count += 1
    return count


def count_active_parameters(root: nn.Module) -> int:
    """Trainable parameters that are structurally free to carry signal.

    Counts every `requires_grad` entry, minus mask-inactive weight entries
    of `MaskedPriorLinear` layers (those are permanently pinned to zero).
    """
    total = sum(p.numel() for p in root.parameters() if p.requires_grad)
    for module in root.modules():
        if isinstance(module, MaskedPriorLinear):
            inactive = module.weight.numel() - int(module.mask.sum().item())
            total -= inactive
    return int(total)


class PriorNeuralNet(nn.Module):
    """Concept-constrained network: features -> named concepts -> head.

    * `concept_layer` (`MaskedPriorLinear`): one neuron per domain concept,
      connected only to that concept's features. Kept linear.
    * optional `residual` channel: a small regularized linear block over
      *orphan* (unmapped) features only, concatenated to the concept layer;
      gives the model a supervised place to put background signal instead of
      leaking it into concept neurons.
    * head: `Linear -> activation -> Linear`, the only nonlinear mixing.
    """

    def __init__(
        self,
        n_features: int,
        n_concepts: int,
        mask: torch.Tensor,
        n_outputs: int,
        hidden_units: int = 16,
        residual_feature_indices: tuple[int, ...] | list[int] = (),
        residual_units: int = 0,
        activation: str = "gelu",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        self.n_features, self.n_concepts, self.n_outputs = n_features, n_concepts, n_outputs
        self.concept_layer = MaskedPriorLinear(n_features, n_concepts, mask)
        self.residual_indices = list(residual_feature_indices)
        if self.residual_indices and residual_units > 0:
            self.residual: nn.Linear | None = nn.Linear(len(self.residual_indices), residual_units)
            self.latent_dim = n_concepts + residual_units
        else:
            self.residual = None
            self.residual_indices = []
            self.latent_dim = n_concepts
        self.head_hidden = nn.Linear(self.latent_dim, hidden_units)
        self.head_out = nn.Linear(hidden_units, n_outputs)
        self.act = _ACTIVATIONS[activation]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Latent vector = [concept activations (linear), residual channel]."""
        concept = self.concept_layer(x)
        if self.residual is not None:
            residual = self.residual(x[:, self.residual_indices])
            return torch.cat([concept, residual], dim=1)
        return concept

    def head(self, latent: torch.Tensor) -> torch.Tensor:
        return self.head_out(self.act(self.head_hidden(latent)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def concept_activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.concept_layer(x)


class UnconstrainedMLP(nn.Module):
    """Plain fully-connected baseline with arbitrary `hidden_widths`."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        hidden_widths: tuple[int, ...] = (64,),
        activation: str = "gelu",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        widths = list(hidden_widths)
        if not widths or any(w < 1 for w in widths):
            raise ValueError("hidden_widths must be positive widths")
        layers: list[nn.Module] = []
        dim = n_features
        for width in widths:
            layers.append(nn.Linear(dim, width))
            layers.append(_ACTIVATION_MODULES[activation]())
            dim = width
        layers.append(nn.Linear(dim, n_outputs))
        self.net = nn.Sequential(*layers)
        self.hidden_widths = tuple(widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_matched_mlp(
    n_features: int,
    n_outputs: int,
    target_active_params: int,
    activation: str = "gelu",
) -> tuple[UnconstrainedMLP, int]:
    """Build a single-hidden-layer MLP with (near-)equal trainable parameters.

    For widths `(h,)`: params = h*(F+1) + O*(h+1) = h*(F+O+1) + O, solved in
    closed form. Returns the model and its achieved parameter count.
    """
    width = max(1, round((target_active_params - n_outputs) / (n_features + n_outputs + 1)))
    mlp = UnconstrainedMLP(n_features, n_outputs, (width,), activation=activation)
    return mlp, count_active_parameters(mlp)


def standard_mlp(
    n_features: int, n_outputs: int, widths: tuple[int, ...] = (64,), activation: str = "gelu"
) -> UnconstrainedMLP:
    """Standard full-width black-box MLP baseline."""
    return UnconstrainedMLP(n_features, n_outputs, widths, activation=activation)
