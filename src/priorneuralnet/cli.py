"""Typer CLI: validate-graph, benchmark, generate-demo, inspect-concepts."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from priorneuralnet.engine.benchmark import inspect_concept_drivers, run_benchmark
from priorneuralnet.engine.prior_graph import (
    PriorGraphError,
    build_prior_graph,
    load_dataframe,
    load_mapping,
)
from priorneuralnet.engine.synthetic import make_synthetic_prior_dataset

app = typer.Typer(
    help="Domain-Prior Mask-Constrained Neural Networks: validate priors, "
    "benchmark them against parameter-matched MLPs, export interpretable artifacts.",
    no_args_is_help=True,
)


@app.command("validate-graph")
def validate_graph(
    data_path: str = typer.Argument(..., help="dataset .csv/.tsv/.parquet"),
    prior_path: str = typer.Argument(..., help="prior mapping .json or 2-column .csv"),
    target_column: str = typer.Argument(..., help="name of the label column"),
) -> None:
    """Validate a feature->concept prior against a dataset; print graph stats."""
    try:
        df = load_dataframe(data_path)
        mapping = load_mapping(prior_path)
        graph = build_prior_graph(df, target_column, mapping)
    except PriorGraphError as exc:
        typer.secho(f"prior graph invalid: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    stats = graph.stats()
    typer.echo(json.dumps(stats, indent=1))
    if graph.orphan_features:
        typer.secho(
            f"WARNING: {len(graph.orphan_features)} orphan features are unmapped; "
            "they will be routed to the regularized residual channel.",
            fg=typer.colors.YELLOW,
        )
    typer.secho("prior graph is valid ✓", fg=typer.colors.GREEN)


@app.command("benchmark")
def benchmark(
    data_path: str = typer.Argument(...),
    prior_path: str = typer.Argument(...),
    target_column: str = typer.Argument(...),
    learning_curve: bool = typer.Option(True, "--learning-curve/--no-learning-curve"),
    n_splits: int = typer.Option(5, min=2),
    epochs: int = typer.Option(60, min=1),
    output_dir: str = typer.Option("runs", "--output-dir", "-o"),
) -> None:
    """Run the full CV benchmark and write manifest/report/activations."""
    manifest = run_benchmark(
        data_path,
        prior_path,
        target_column,
        run_learning_curve=learning_curve,
        n_splits=n_splits,
        epochs=epochs,
        output_dir=output_dir,
    )
    run_dir = Path(output_dir) / manifest["run_id"]
    typer.echo(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "output_dir": str(run_dir),
                "verdict": manifest["verdict"],
                "top_concepts": [c["concept"] for c in manifest["concept_ranking"][:5]],
            },
            indent=1,
        )
    )


@app.command("generate-demo")
def generate_demo(
    output_dir: str = typer.Option("examples/demo_run", "--output-dir", "-o"),
    n_samples: int = typer.Option(240, min=40),
    seed: int = typer.Option(7),
    epochs: int = typer.Option(120, min=10),
) -> None:
    """Generate the synthetic p>>n demo dataset, prior, then benchmark it."""
    out = Path(output_dir)
    typer.echo(f"generating synthetic dataset in {out} ...")
    ds = make_synthetic_prior_dataset(n_samples=n_samples, seed=seed)
    paths = ds.save(out)
    typer.echo(f"ground-truth active pathways: {ds.active_pathways}")
    manifest = run_benchmark(
        paths["dataset"],
        paths["prior"],
        ds.target_column,
        run_learning_curve=True,
        n_splits=5,
        epochs=epochs,
        output_dir=str(out),
    )
    typer.echo(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "output_dir": str(out / manifest["run_id"]),
                "verdict": manifest["verdict"],
                "top_concepts": [c["concept"] for c in manifest["concept_ranking"][:5]],
                "recovered_ground_truth": sorted(ds.active_pathways)
                == sorted(
                    c["concept"] for c in manifest["concept_ranking"][: len(ds.active_pathways)]
                ),
            },
            indent=1,
        )
    )


@app.command("inspect-concepts")
def inspect_concepts(
    concept: str = typer.Argument(..., help="concept name to inspect"),
    run_id: str = typer.Argument(..., help="run id (directory under --output-dir)"),
    top_k: int = typer.Option(10, min=1),
    output_dir: str = typer.Option("runs", "--output-dir", "-o"),
) -> None:
    """Show a concept's attribution and top-k feature drivers from a run."""
    result = inspect_concept_drivers(Path(output_dir) / run_id, concept, top_k=top_k)
    typer.echo(json.dumps(result, indent=1))


if __name__ == "__main__":
    app()
