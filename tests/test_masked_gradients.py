"""Hard correctness of `MaskedPriorLinear`: forbidden connections stay exactly
zero in weights AND gradients across full AdamW steps (incl. weight decay)."""

from __future__ import annotations

import math

import pytest
import torch

from priorneuralnet.engine.layers import (
    MaskedPriorLinear,
    PriorNeuralNet,
    apply_mask_constraints,
    count_active_parameters,
    parameter_matched_mlp,
)


def _mask(rows: int = 6, cols: int = 20, density: float = 0.25, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(rows, cols, generator=g) < density).float()
    m[:, :rows] = 1.0  # every row has >= 1 active connection
    return m


def _layer(seed: int = 0) -> tuple[MaskedPriorLinear, torch.Tensor]:
    torch.manual_seed(seed)
    m = _mask(seed=seed)
    return MaskedPriorLinear(m.shape[1], m.shape[0], m), m


def test_initial_weights_zero_outside_mask():
    layer, m = _layer()
    assert torch.all(layer.weight.data[m == 0] == 0.0)


def test_initialization_scales_by_per_row_fan_in_not_dense_fan_in():
    layer, m = _layer()
    fan_in = m.sum(dim=1)
    bound = torch.sqrt(6.0 / fan_in)
    dense_bound = math.sqrt(6.0 / m.shape[1])
    exceeded_dense_bound = False
    for row in range(m.shape[0]):
        active = layer.weight.data[row][m[row] == 1]
        assert active.numel() == int(fan_in[row])
        assert torch.all(active.abs() <= bound[row] + 1e-7)
        # A sparse row's per-row bound is LARGER than the dense bound; if init
        # used the dense fan-in it could never exceed it. Sampling evidence that
        # scaling is per-row (deterministic: fixed seed).
        if bound[row].item() > dense_bound and active.abs().max().item() > dense_bound:
            exceeded_dense_bound = True
    assert exceeded_dense_bound, "init does not scale per-row fan-in"


def test_forward_uses_effective_weight_only():
    layer, m = _layer()
    x = torch.randn(7, layer.in_features)
    expected = x @ (layer.weight.data * m).T + layer.bias.data
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_adamw_steps_never_leak_weight_or_gradient_outside_mask():
    """The mathematical core: after each backward, dW is exactly zero on
    forbidden entries (gradient hook); after each AdamW step with nonzero
    weight decay, re-projection keeps weights exactly zero there."""
    layer, m = _layer()
    opt = torch.optim.AdamW(layer.parameters(), lr=0.05, weight_decay=0.2)
    x = torch.randn(16, layer.in_features)
    forbidden = m == 0
    initial = layer.weight.data.clone()
    changed = False
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        loss = layer(x).square().mean()
        loss.backward()
        assert torch.all(layer.weight.grad[forbidden] == 0.0), "gradient leaked"
        opt.step()
        apply_mask_constraints(layer)
        assert torch.all(layer.weight.data[forbidden] == 0.0), (
            "AdamW weight decay leaked value into a forbidden connection"
        )
        if not torch.allclose(layer.weight.data, initial, atol=1e-12):
            changed = True
    assert changed, "training must actually move the active weights"


def test_zero_fan_in_row_stays_exactly_zero():
    m = _mask()
    m[0, :] = 0.0  # degenerate concept: no features
    torch.manual_seed(1)
    layer = MaskedPriorLinear(m.shape[1], m.shape[0], m)
    assert torch.all(layer.weight.data[0] == 0.0)
    out = layer(torch.randn(4, m.shape[1]))
    assert torch.all(out[:, 0] == layer.bias.data[0])


def test_apply_mask_constraints_reports_layers_touched():
    net = _tiny_prior_net()
    assert apply_mask_constraints(net) == 1  # exactly one masked layer


def test_count_active_parameters_matches_manual_arithmetic():
    net = _tiny_prior_net()
    assert net.residual is not None  # helper always constructs one
    m = net.concept_layer.mask
    manual = (
        int(m.sum().item())  # active concept-layer weights
        + m.shape[0]  # concept biases
        + net.residual.weight.numel()
        + net.residual.bias.numel()
        + net.head_hidden.weight.numel()
        + net.head_hidden.bias.numel()
        + net.head_out.weight.numel()
        + net.head_out.bias.numel()
    )
    assert count_active_parameters(net) == manual
    dense = net.concept_layer.weight.numel() - int(m.sum().item())
    assert dense > 0  # a dense counterfactual would count strictly more


def test_parameter_matched_mlp_hits_target_within_small_tolerance():
    net = _tiny_prior_net()
    target = count_active_parameters(net)
    mlp, achieved = parameter_matched_mlp(net.n_features, net.n_outputs, target)
    assert abs(achieved - target) <= max(
        net.n_features + net.n_outputs + 1, 0
    )  # one hidden-row worth of slack, by construction of the closed form


def _tiny_prior_net() -> PriorNeuralNet:
    torch.manual_seed(2)
    m = _mask(rows=5, cols=12, density=0.3, seed=3)
    return PriorNeuralNet(
        n_features=12,
        n_concepts=5,
        mask=m,
        n_outputs=2,
        hidden_units=4,
        residual_feature_indices=[11],
        residual_units=2,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
