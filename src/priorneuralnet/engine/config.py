"""Pydantic v2 schemas shared by the engine, CLI, reporting and MCP layers.

This module must stay import-cheap: no torch, no mcp, no plotting imports.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Prior graph
# --------------------------------------------------------------------------


class PriorGraphSpec(BaseModel):
    """Validated feature -> concept mapping and its structural statistics."""

    target_column: str
    features: list[str]
    concepts: list[str]
    mapping: dict[str, str] = Field(description="feature name -> concept name")
    orphan_features: list[str] = Field(
        default_factory=list,
        description="dataset columns with no concept mapping (background/unknown)",
    )
    n_features: int
    n_concepts: int
    active_connections: int = Field(ge=0)
    total_connections: int = Field(ge=0)
    mask_density: float = Field(ge=0.0, le=1.0, description="active / total edges")


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class TrainConfig(BaseModel):
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    hidden_units: int = 4
    residual_units: int = 8
    seed: int = 13
    activation: Literal["gelu", "relu", "tanh"] = "gelu"
    device: str = "cpu"
    # identical early-stopping protocol for every model (monitors fold val loss);
    # off by default: it protects accuracy but freezes attributions early
    patience: int = 15
    early_stopping: bool = False


# --------------------------------------------------------------------------
# Attributions
# --------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    feature: str
    concept: str
    contribution: float = Field(
        description="signed mean |w_eff * x| flow feature -> concept "
        "(exact Integrated-Gradients edge weight for the linear concept layer)"
    )


class ConceptAttribution(BaseModel):
    concept: str
    attribution: float = Field(description="mean |Integrated Gradients| w.r.t. the target")
    rank: int
    top_features: list[FeatureContribution]


# --------------------------------------------------------------------------
# Benchmark + manifest
# --------------------------------------------------------------------------


class ModelMetrics(BaseModel):
    model: str
    architecture: dict = Field(default_factory=dict)
    active_trainable_parameters: int
    metric_name: str
    train_metrics: list[float] = Field(default_factory=list)
    val_metrics: list[float] = Field(default_factory=list)
    train_mean: float
    val_mean: float
    generalization_gap: float = Field(description="train_mean - val_mean")
    auxiliary: dict[str, list[float]] = Field(default_factory=dict)


class LearningCurvePoint(BaseModel):
    train_fraction: float
    model: str
    val_metric: float
    train_metric: float | None = None


class DatasetStats(BaseModel):
    n_samples: int
    n_features: int
    n_classes: int | None = None
    task: Literal["classification", "regression"]
    metric_name: str


class BenchmarkManifest(BaseModel):
    """Compact artifact designed to be consumed by LLM agents (<2k tokens)."""

    schema_version: int = 1
    run_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    dataset: DatasetStats
    prior_graph: dict = Field(
        description="compact graph summary (stats, concepts, fan-in; full mapping "
        "lives in the separate prior artifact, not the LLM-facing manifest)"
    )
    train_config: TrainConfig
    comparison: list[ModelMetrics]
    learning_curve: list[LearningCurvePoint] = Field(default_factory=list)
    concept_ranking: list[ConceptAttribution]
    verdict: str = Field(description="one-line machine summary of the head-to-head outcome")
    warnings: list[str] = Field(default_factory=list)
    token_estimate: int | None = Field(
        default=None, description="crude len(json)//4 token count of this manifest"
    )
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="run-dir-relative artifact filenames"
    )
