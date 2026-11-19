"""Head-to-head CV benchmark: priorneuralnet vs parameter-matched baselines.

Protocol:
  * Stratified K-Fold for classification (K-Fold for regression), identical
    folds and seeds for every model.
  * Three competitors: the mask-constrained `PriorNeuralNet`, an
    `UnconstrainedMLP` automatically resized to match its *active trainable
    parameter* count, and a standard full-width MLP.
  * Optional sample-efficiency sweep: validation metric vs training fraction
    (25%, 50%, 100%) for prior vs parameter-matched MLP.
  * A final model trained on the full dataset exports concept activations
    (.parquet), weights (.pt) and concept-level attribution tables used by
    the manifest, the HTML report and `inspect_concept_drivers`.

Mask enforcement is re-projected after every optimizer step
(`apply_mask_constraints`) so decoupled weight decay cannot leak signal into
structurally forbidden connections.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from priorneuralnet.engine.attribution import rank_concepts
from priorneuralnet.engine.config import (
    LearningCurvePoint,
    ModelMetrics,
    TrainConfig,
)
from priorneuralnet.engine.layers import (
    PriorNeuralNet,
    apply_mask_constraints,
    count_active_parameters,
    parameter_matched_mlp,
    standard_mlp,
)
from priorneuralnet.engine.prior_graph import (
    PriorGraph,
    build_prior_graph,
    load_dataframe,
)

TRAIN_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 1.0)
PRIOR = "priorneuralnet"
PARAM_MATCHED = "unconstrained_mlp_param_matched"
FULL_MLP = "unconstrained_mlp_full"


class BenchmarkError(RuntimeError):
    pass


def detect_task(y: np.ndarray) -> tuple[str, int | None, str]:
    """Return (task, n_classes, primary_metric) from the target column."""
    as_float = pd.to_numeric(pd.Series(y), errors="coerce")
    unique = np.unique(as_float.dropna().to_numpy())
    is_int = np.all(np.isclose(unique, np.round(unique)))
    if is_int and len(unique) <= 10:
        return "classification", int(len(unique)), "accuracy"
    return "regression", None, "rmse"


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_models(
    graph: PriorGraph, n_outputs: int, cfg: TrainConfig
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Instantiate the three competitors sharing identical hyperparameters."""
    mask = torch.from_numpy(graph.mask)
    orphan_idx = [graph.feature_index[f] for f in graph.orphan_features]
    prior = PriorNeuralNet(
        n_features=graph.mask.shape[1],
        n_concepts=graph.mask.shape[0],
        mask=mask,
        n_outputs=n_outputs,
        hidden_units=cfg.hidden_units,
        residual_feature_indices=orphan_idx,
        residual_units=cfg.residual_units if orphan_idx else 0,
        activation=cfg.activation,
    )
    active = count_active_parameters(prior)
    matched, matched_params = parameter_matched_mlp(
        graph.mask.shape[1], n_outputs, active, activation=cfg.activation
    )
    full = standard_mlp(graph.mask.shape[1], n_outputs, (64,), activation=cfg.activation)
    return {
        PRIOR: (
            prior,
            {
                "type": "prior_masked_concepts",
                "n_concepts": graph.mask.shape[0],
                "hidden_units": cfg.hidden_units,
                "residual_units": prior.latent_dim - graph.mask.shape[0],
                "active_trainable_parameters": active,
            },
        ),
        PARAM_MATCHED: (
            matched,
            {
                "type": "unconstrained_mlp",
                "hidden_widths": list(matched.hidden_widths),
                "parameter_target": active,
                "achieved_parameters": matched_params,
            },
        ),
        FULL_MLP: (
            full,
            {
                "type": "unconstrained_mlp",
                "hidden_widths": list(full.hidden_widths),
                "achieved_parameters": count_active_parameters(full),
            },
        ),
    }


def _evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor, task: str) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(X)
    if task == "classification":
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax(dim=1)
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average="macro", zero_division=0))
        try:
            auc = float(
                roc_auc_score(y, prob[:, 1] if prob.shape[1] == 2 else prob, multi_class="ovr")
            )
        except ValueError:
            auc = float("nan")
        return {"accuracy": acc, "f1_macro": f1, "roc_auc": auc}
    err = out.squeeze(-1) - y
    rmse = float(torch.sqrt((err * err).mean()))
    var = float(y.var())
    return {"rmse": rmse, "r2": 1.0 - rmse * rmse / var if var > 0 else float("nan")}


def train_model(
    model: torch.nn.Module,
    Xtr: torch.Tensor,
    ytr: torch.Tensor,
    Xva: torch.Tensor,
    yva: torch.Tensor,
    task: str,
    cfg: TrainConfig,
) -> dict:
    """Train one model; returns per-split metric dicts. Constrained models are
    re-projected onto their masks after every optimizer step."""
    constrained = isinstance(model, PriorNeuralNet)
    if constrained:
        apply_mask_constraints(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    n = Xtr.shape[0]
    if task == "classification":
        criterion = torch.nn.CrossEntropyLoss()
        ytr_l, yva_l = ytr.long(), yva.long()
    else:
        criterion = torch.nn.MSELoss()
        ytr_l, yva_l = ytr.float(), yva.float()

    rng = np.random.default_rng(cfg.seed)
    bs = max(1, min(cfg.batch_size, n))

    def val_loss() -> float:
        model.eval()
        with torch.no_grad():
            return float(criterion(model(Xva), yva_l))

    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = torch.from_numpy(order[start : start + bs]).to(torch.long)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(Xtr[idx]), ytr_l[idx])
            loss.backward()
            opt.step()
            if constrained:
                apply_mask_constraints(model)
        if cfg.early_stopping:
            vl = val_loss()
            if vl < best_loss - 1e-4:
                best_loss, wait = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train": _evaluate(model, Xtr, ytr_l, task),
        "val": _evaluate(model, Xva, yva_l, task),
    }


@dataclass
class _Tensors:
    X: torch.Tensor
    y: torch.Tensor
    scaler: StandardScaler


def _prepare(df: pd.DataFrame, graph: PriorGraph, target_column: str) -> _Tensors:
    Xn = df[graph.feature_names].to_numpy(dtype=np.float32)
    y = pd.to_numeric(df[target_column], errors="raise").to_numpy()
    scaler = StandardScaler()
    return _Tensors(
        X=torch.tensor(scaler.fit_transform(Xn), dtype=torch.float32),
        y=torch.from_numpy(np.array(y, copy=True)),
        scaler=scaler,
    )


def run_benchmark(
    data: pd.DataFrame | str | Path,
    prior: dict[str, str] | str | Path,
    target_column: str,
    run_learning_curve: bool = True,
    n_splits: int = 5,
    epochs: int = 60,
    output_dir: str | Path = "runs",
    train_config: TrainConfig | None = None,
) -> dict:
    """Run the full benchmark and write all artifacts; returns the manifest."""
    df = data if isinstance(data, pd.DataFrame) else load_dataframe(data)
    graph = build_prior_graph(df, target_column, prior)
    cfg = (train_config or TrainConfig()).model_copy(update={"epochs": epochs})
    if n_splits < 2:
        raise BenchmarkError("n_splits must be >= 2")

    task, n_classes, metric = detect_task(df[target_column])
    tensors = _prepare(df, graph, target_column)
    X, y = tensors.X, tensors.y
    y_np = y.numpy()
    if task == "classification":
        codes = {v: i for i, v in enumerate(sorted(set(y_np.tolist())))}
        n_outputs = len(codes)
        y = torch.tensor([codes[v] for v in y_np], dtype=torch.long)
    else:
        n_outputs = 1

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{secrets.token_hex(2)}"
    out = Path(output_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- cross-validated comparison ----------------
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        if task == "classification"
        else KFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    )
    folds = list(splitter.split(X.numpy(), y.numpy() if task == "classification" else None))
    results: dict[str, dict] = {}
    for model_name in (PRIOR, PARAM_MATCHED, FULL_MLP):
        fold_metrics = []
        for fold_i, (tr, va) in enumerate(folds):
            set_global_seed(cfg.seed + fold_i)
            model, _arch = build_models(graph, n_outputs, cfg)[model_name]
            res = train_model(
                model,
                X[torch.tensor(tr)],
                y[torch.tensor(tr)],
                X[torch.tensor(va)],
                y[torch.tensor(va)],
                task,
                cfg,
            )
            fold_metrics.append(res)
        _, arch = build_models(graph, n_outputs, cfg)[model_name]
        results[model_name] = {
            "arch": arch,
            "train": [m["train"][metric] for m in fold_metrics],
            "val": [m["val"][metric] for m in fold_metrics],
            "aux": {
                k: [m["val"][k] for m in fold_metrics]
                for k in fold_metrics[0]["val"]
                if k != metric
            },
        }

    # ---------------- sample-efficiency sweep ----------------
    curve: list[dict] = []
    if run_learning_curve:
        # fixed inner split for the curve
        inner = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed + 99)
            if task == "classification"
            else KFold(n_splits=5, shuffle=True, random_state=cfg.seed + 99)
        )
        inner_tr, inner_va = next(
            iter(inner.split(X.numpy(), y.numpy() if task == "classification" else None))
        )
        for frac in TRAIN_FRACTIONS:
            size = max(8, int(len(inner_tr) * frac))
            subset = inner_tr[:size]
            for model_name in (PRIOR, PARAM_MATCHED):
                set_global_seed(cfg.seed + 1000 + int(frac * 100))
                model, _ = build_models(graph, n_outputs, cfg)[model_name]
                res = train_model(
                    model,
                    X[torch.tensor(subset)],
                    y[torch.tensor(subset)],
                    X[torch.tensor(inner_va)],
                    y[torch.tensor(inner_va)],
                    task,
                    cfg,
                )
                curve.append(
                    LearningCurvePoint(
                        train_fraction=frac,
                        model=model_name,
                        val_metric=res["val"][metric],
                        train_metric=res["train"][metric],
                    ).model_dump()
                )

    # ---------------- final model: attributions + artifacts ----------------
    set_global_seed(cfg.seed)
    final_model, final_arch = build_models(graph, n_outputs, cfg)[PRIOR]
    train_model(final_model, X, y, X, y, task, cfg)
    if n_classes == 2 or task == "regression":
        target_index = 0 if task == "regression" else 1
    else:
        target_index = 1  # positive class for the multiclass case too

    # one attribution pass, generous feature lists; the manifest slices its own budget
    concept_ranking = rank_concepts(
        final_model,
        X,
        graph.concept_names,
        graph.feature_names,
        target_index=target_index,
        top_features_per_concept=50,
    )
    with torch.no_grad():
        activations = pd.DataFrame(
            final_model.concept_activations(X).numpy(),
            columns=graph.concept_names,
        )
    activations.to_parquet(out / "concept_activations.parquet")
    torch.save(
        {"state_dict": final_model.state_dict(), "architecture": final_arch}, out / "final_model.pt"
    )
    # full driver lists (manifest keeps only the top slice)
    (out / "concept_drivers.json").write_text(
        json.dumps(
            {d.concept: [f.model_dump() for f in d.top_features] for d in concept_ranking}, indent=1
        ),
        encoding="utf-8",
    )

    warnings = []
    if graph.orphan_features:
        warnings.append(
            f"{len(graph.orphan_features)} orphan features have no concept mapping; "
            f"a {cfg.residual_units}-unit regularized residual channel absorbed them."
        )
    prior_val = float(np.mean(results[PRIOR]["val"]))
    matched_val = float(np.mean(results[PARAM_MATCHED]["val"]))
    improvement = prior_val - matched_val if metric == "rmse" else matched_val - prior_val
    verdict = (
        f"{PRIOR} vs parameter-matched MLP on {metric}: {prior_val:.4f} vs "
        f"{matched_val:.4f} -> {'prior wins by' if improvement > 0 else 'parameter-matched MLP wins by'} "
        f"{abs(improvement):.4f}."
    )

    comparison = [
        ModelMetrics(
            model=name,
            architecture=body["arch"],
            active_trainable_parameters=body["arch"].get(
                "active_trainable_parameters", body["arch"].get("achieved_parameters", 0)
            ),
            metric_name=metric,
            train_metrics=body["train"],
            val_metrics=body["val"],
            train_mean=float(np.mean(body["train"])),
            val_mean=float(np.mean(body["val"])),
            generalization_gap=float(np.mean(body["train"]) - np.mean(body["val"])),
            auxiliary=body["aux"],
        )
        for name, body in results.items()
    ]

    from priorneuralnet.reporting.manifest import build_manifest, save_manifest

    manifest = build_manifest(
        run_id=run_id,
        df=df,
        graph=graph,
        target_column=target_column,
        task=task,
        n_classes=n_classes,
        metric=metric,
        cfg=cfg,
        comparison=comparison,
        curve=curve,
        concept_ranking=concept_ranking,
        verdict=verdict,
        warnings=warnings,
    )
    save_manifest(manifest, out / "manifest.json")
    (out / "prior_graph.json").write_text(
        json.dumps(
            {
                "mapping": graph.mapping,
                "concepts": graph.concept_names,
                "orphans": graph.orphan_features,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    manifest["artifacts"] = {
        "manifest": "manifest.json",
        "report": "report.html",
        "concept_activations": "concept_activations.parquet",
        "final_model": "final_model.pt",
        "prior_graph": "prior_graph.json",
        "concept_drivers": "concept_drivers.json",
    }
    save_manifest(manifest, out / "manifest.json")

    from priorneuralnet.reporting.html_report import render_report

    render_report(manifest, activations, out)

    return manifest


def inspect_concept_drivers(run_dir: str | Path, concept_name: str, top_k: int = 10) -> dict:
    """Top feature drivers of a concept from a completed run."""
    run = Path(run_dir)
    drivers_path = run / "concept_drivers.json"
    if not drivers_path.exists():
        raise BenchmarkError(f"no concept_drivers.json under {run}")
    drivers = json.loads(drivers_path.read_text(encoding="utf-8"))
    if concept_name not in drivers:
        raise BenchmarkError(f"unknown concept {concept_name!r}; available: {sorted(drivers)}")
    ranked = sorted(drivers[concept_name], key=lambda d: abs(d["contribution"]), reverse=True)
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    concept_meta = next(
        (c for c in manifest.get("concept_ranking", []) if c["concept"] == concept_name), None
    )
    return {
        "run_id": manifest.get("run_id", run.name),
        "concept": concept_name,
        "attribution": concept_meta["attribution"] if concept_meta else None,
        "rank": concept_meta["rank"] if concept_meta else None,
        "top_features": ranked[:top_k],
    }
