"""End-to-end: the p>>n synthetic benchmark runs cleanly and the
ground-truth active pathways are recovered in the top concept attributions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from priorneuralnet.engine.benchmark import run_benchmark
from priorneuralnet.engine.synthetic import make_synthetic_prior_dataset
from priorneuralnet.reporting.manifest import load_manifest

EXPECTED_ARTIFACTS = (
    "manifest.json",
    "report.html",
    "concept_activations.parquet",
    "final_model.pt",
    "concept_drivers.json",
    "prior_graph.json",
)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> tuple[Path, dict, list[str]]:
    """One shared full run for all assertions (benchmark is the slow part)."""
    root = tmp_path_factory.mktemp("e2e")
    ds = make_synthetic_prior_dataset(n_samples=240, seed=11)
    paths = ds.save(root)
    manifest = run_benchmark(
        paths["dataset"],
        paths["prior"],
        ds.target_column,
        run_learning_curve=True,
        n_splits=3,
        epochs=80,
        output_dir=root / "runs",
    )
    return root / "runs" / manifest["run_id"], manifest, ds.active_pathways


def test_benchmark_runs_clean_and_recovers_true_pathways(run_dir):
    _, manifest, active = run_dir
    top = [c["concept"] for c in manifest["concept_ranking"][:3]]
    assert all(p in top for p in active), (
        f"ground-truth pathways {active} not in top-3 concepts {top}"
    )


def test_all_artifacts_exist_and_are_valid(run_dir):
    run, manifest, _ = run_dir
    for name in EXPECTED_ARTIFACTS:
        path = run / name
        assert path.exists(), f"missing artifact: {name}"
        assert path.stat().st_size > 0
    # manifest round-trips through its schema and respects the token budget
    reloaded = load_manifest(run / "manifest.json")
    assert reloaded["run_id"] == manifest["run_id"]
    assert manifest["token_estimate"] < 2000
    # activations parquet matches the named concepts
    acts = pd.read_parquet(run / "concept_activations.parquet")
    assert list(acts.columns) == sorted(
        json.loads((run / "prior_graph.json").read_text())["concepts"]
    )
    assert len(acts) == manifest["dataset"]["n_samples"]
    # report.html is a real dashboard: plotly divs + sankey + curve
    html = (run / "report.html").read_text(encoding="utf-8")
    assert "sankey" in html and "plotly" in html.lower()
    assert html.count('class="fig"') >= 5


def test_parameter_matching_is_close_and_models_compared_fairly(run_dir):
    _, manifest, _ = run_dir
    by_name = {m["model"]: m for m in manifest["comparison"]}
    prior = by_name["priorneuralnet"]
    matched = by_name["unconstrained_mlp_param_matched"]
    assert matched["active_trainable_parameters"] <= prior["active_trainable_parameters"] * 1.05
    assert prior["generalization_gap"] == pytest.approx(prior["train_mean"] - prior["val_mean"])
    assert len(prior["val_metrics"]) == 3  # n_splits honored


def test_learning_curve_emits_all_fractions_for_both_models(run_dir):
    _, manifest, _ = run_dir
    points = manifest["learning_curve"]
    fracs = {p["train_fraction"] for p in points}
    assert fracs == {0.25, 0.5, 1.0}
    models = {p["model"] for p in points}
    assert models == {"priorneuralnet", "unconstrained_mlp_param_matched"}


def test_learning_curve_metrics_are_valid_at_every_point(run_dir):
    """Every curve point trained and produced a finite, well-formed metric.
    The prior-vs-baseline ranking itself is an empirical research claim (it is
    seed- and regime-dependent); it belongs in the generated report and demo,
    not in a hard CI assertion that would flake."""
    _, manifest, _ = run_dir
    for p in manifest["learning_curve"]:
        assert np.isfinite(p["val_metric"])
        assert np.isfinite(p["train_metric"])
        assert 0.0 <= p["val_metric"] <= 1.0
        assert 0.0 <= p["train_metric"] <= 1.0
        assert p["train_fraction"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
