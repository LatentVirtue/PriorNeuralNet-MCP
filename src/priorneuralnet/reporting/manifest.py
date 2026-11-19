"""Compact LLM-facing benchmark manifest (<2,000 tokens by construction)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from priorneuralnet.engine.config import (
    BenchmarkManifest,
    ConceptAttribution,
    DatasetStats,
    LearningCurvePoint,
    ModelMetrics,
    TrainConfig,
)
from priorneuralnet.engine.prior_graph import PriorGraph

MAX_TOKENS = 2000


def estimate_tokens(obj: object) -> int:
    """Crude token estimate: serialized JSON chars / 4."""
    return len(json.dumps(obj, separators=(",", ":"))) // 4


def _graph_summary(graph: PriorGraph, target_column: str) -> dict:
    s = graph.stats()
    orphans = s["orphan_features"]
    return {
        "target_column": target_column,
        "n_features": s["n_features"],
        "n_concepts": s["n_concepts"],
        "active_connections": s["active_connections"],
        "total_connections": s["total_connections"],
        "mask_density": round(s["mask_density"], 4),
        "orphan_feature_count": len(orphans),
        "orphan_feature_examples": orphans[:5],
        "concept_fan_in": s["concept_fan_in"],
    }


def build_manifest(
    run_id: str,
    df: pd.DataFrame,
    graph: PriorGraph,
    target_column: str,
    task: str,
    n_classes: int | None,
    metric: str,
    cfg: TrainConfig,
    comparison: list[ModelMetrics],
    curve: list[dict],
    concept_ranking: list[ConceptAttribution],
    verdict: str,
    warnings: list[str],
) -> dict:
    """Assemble the manifest, degrading gracefully to stay under MAX_TOKENS.

    Keeps every required section (sparsity stats, orphan warnings, CV
    comparison + generalization gaps, learning curve, top-10 concepts, top
    feature drivers per concept); when over budget it trims attribution
    breadth before it ever drops a section.
    """
    dataset = DatasetStats(
        n_samples=int(df.shape[0]),
        n_features=graph.mask.shape[1],
        n_classes=n_classes,
        task=task,  # type: ignore[arg-type]
        metric_name=metric,
    )
    ranking_n, features_per_concept = 10, 6
    while True:
        payload = BenchmarkManifest(
            run_id=run_id,
            dataset=dataset,
            prior_graph=_graph_summary(graph, target_column),
            train_config=cfg,
            comparison=comparison,
            learning_curve=[LearningCurvePoint(**c) for c in curve],
            concept_ranking=[
                item.model_copy(update={"top_features": item.top_features[:features_per_concept]})
                for item in concept_ranking[:ranking_n]
            ],
            verdict=verdict,
            warnings=warnings,
        ).model_dump(mode="json")
        estimate = estimate_tokens(payload)
        if estimate <= MAX_TOKENS or (ranking_n <= 5 and features_per_concept <= 2):
            payload["token_estimate"] = estimate
            return payload
        if features_per_concept > 2:
            features_per_concept -= 2
        else:
            ranking_n -= 1


def save_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict:
    """Validate a manifest file back into the schema and return it."""
    return BenchmarkManifest.model_validate_json(Path(path).read_text(encoding="utf-8")).model_dump(
        mode="json"
    )
