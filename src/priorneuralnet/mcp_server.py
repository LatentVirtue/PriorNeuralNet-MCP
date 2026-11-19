"""MCPServer (mcp 2.x) exposing the priorneuralnet engine as agent tools.

Transport: stdio (`priorneuralnet-mcp`). Register with any MCP client using:

    {"mcpServers": {"priorneuralnet": {"command": "priorneuralnet-mcp"}}}

Design: the tools are thin, JSON-safe wrappers over the engine; all real
logic lives in `priorneuralnet.engine`, which never imports `mcp`.
Run artifacts are stored under `<output_dir>/<run_id>/`; the default
`output_dir` for the manifest resource is `runs` (override per tool call).
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from priorneuralnet.engine.benchmark import (
    BenchmarkError,
    run_benchmark,
)
from priorneuralnet.engine.benchmark import (
    inspect_concept_drivers as engine_inspect_concept_drivers,
)
from priorneuralnet.engine.prior_graph import (
    PriorGraphError,
    build_prior_graph,
    load_dataframe,
    load_mapping,
)

mcp = MCPServer("priorneuralnet")


def validate_domain_prior(data_path: str, prior_path: str, target_column: str) -> dict:
    """Check a dataset + feature->concept prior and report graph statistics.

    Returns {valid: bool, stats: {...}} or {valid: false, error: ...}.
    Stats include sparsity density, per-concept fan-in and orphan features.
    """
    try:
        df = load_dataframe(data_path)
        mapping = load_mapping(prior_path)
        graph = build_prior_graph(df, target_column, mapping)
    except (PriorGraphError, ValueError) as exc:
        return {"valid": False, "error": str(exc)}
    stats = graph.stats()
    warnings = []
    if graph.orphan_features:
        warnings.append(
            f"{len(graph.orphan_features)} features are unmapped (orphans); "
            "benchmark will route them to a regularized residual channel."
        )
    return {"valid": True, "stats": stats, "warnings": warnings}


def run_priorneuralnet_benchmark(
    data_path: str,
    prior_path: str,
    target_column: str,
    run_learning_curve: bool = True,
    n_splits: int = 5,
    epochs: int = 60,
    output_dir: str = "runs",
) -> dict:
    """Benchmark the prior-constrained network against parameter-matched MLPs.

    Writes artifacts under `<output_dir>/<run_id>/` (manifest.json,
    report.html, concept_activations.parquet, final_model.pt,
    concept_drivers.json) and returns the run id, verdict, per-model val
    scores and a pointer to the full manifest.
    """
    try:
        manifest = run_benchmark(
            data_path,
            prior_path,
            target_column,
            run_learning_curve=run_learning_curve,
            n_splits=n_splits,
            epochs=epochs,
            output_dir=output_dir,
        )
    except (PriorGraphError, BenchmarkError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "run_id": manifest["run_id"],
        "output_dir": str(Path(output_dir) / manifest["run_id"]),
        "verdict": manifest["verdict"],
        "model_val_scores": {
            m["model"]: {
                "mean": m["val_mean"],
                "gap": m["generalization_gap"],
                "metric": m["metric_name"],
            }
            for m in manifest["comparison"]
        },
        "top_concepts": [c["concept"] for c in manifest["concept_ranking"][:5]],
        "manifest": manifest,
    }


def inspect_concept_drivers(
    run_id: str, concept_name: str, top_k: int = 10, output_dir: str = "runs"
) -> dict:
    """Return a concept's attribution rank plus its top-k feature drivers."""
    try:
        return engine_inspect_concept_drivers(Path(output_dir) / run_id, concept_name, top_k=top_k)
    except (BenchmarkError, FileNotFoundError) as exc:
        return {"ok": False, "error": str(exc)}


mcp.tool()(validate_domain_prior)
mcp.tool()(run_priorneuralnet_benchmark)
mcp.tool()(inspect_concept_drivers)


@mcp.resource("priorneuralnet://runs/{run_id}/manifest")
def run_manifest_resource(run_id: str) -> str:
    """The compact LLM-facing benchmark manifest for a completed run."""
    path = Path("runs") / run_id / "manifest.json"
    if not path.exists():
        return json.dumps({"error": f"no manifest for run_id {run_id!r} under ./runs"})
    return path.read_text(encoding="utf-8")


def main() -> None:
    """Entry point for the `priorneuralnet-mcp` console script (stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()
