"""Loading and validating feature -> concept prior mappings.

A *prior mapping* assigns dataset features (e.g. genes, sensors) to named
domain concepts (e.g. pathways, physical subsystems). This module turns such
a mapping plus a concrete dataset into a validated `PriorGraph` carrying the
binary connectivity mask consumed by `layers.MaskedPriorLinear`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from priorneuralnet.engine.config import PriorGraphSpec


class PriorGraphError(ValueError):
    """Raised when a prior mapping or dataset fails structural validation."""


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Read a tabular dataset from .csv/.tsv/.parquet into a DataFrame."""
    p = Path(path)
    if not p.exists():
        raise PriorGraphError(f"dataset file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(p, sep="\t")
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise PriorGraphError(f"unsupported dataset format: {suffix} (use .csv/.tsv/.parquet)")


def load_mapping(path: str | Path) -> dict[str, str]:
    """Read a feature -> concept mapping from JSON or two-column CSV.

    Accepted shapes:
      * JSON object `{"feature": "concept", ...}`
      * JSON list   `[{"feature": ..., "concept": ...}, ...]`
      * CSV/TSV with two columns (feature, concept); header optional.
    """
    p = Path(path)
    if not p.exists():
        raise PriorGraphError(f"prior mapping file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            mapping = {str(k): str(v) for k, v in raw.items()}
        elif isinstance(raw, list):
            try:
                mapping = {str(row["feature"]): str(row["concept"]) for row in raw}
            except (KeyError, TypeError) as exc:
                raise PriorGraphError(
                    "JSON list entries must have 'feature' and 'concept' keys"
                ) from exc
        else:
            raise PriorGraphError("prior mapping JSON must be an object or a list")
    elif suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix != ".csv" else ","
        raw = pd.read_csv(p, sep=sep, header=None, dtype=str)
        if raw.shape[1] != 2:
            raise PriorGraphError("CSV prior mapping must have exactly 2 columns")
        first = [str(v).strip().lower() for v in raw.iloc[0].tolist()]
        rows = raw
        if first[0] in {"feature", "gene", "sensor"} and first[1] in {
            "concept",
            "pathway",
            "group",
        }:
            rows = raw.iloc[1:]  # strip header row
        mapping = {str(a).strip(): str(b).strip() for a, b in rows.itertuples(index=False)}
    else:
        raise PriorGraphError(f"unsupported prior mapping format: {suffix} (use .json/.csv/.tsv)")
    if not mapping:
        raise PriorGraphError("prior mapping is empty")
    return mapping


@dataclass
class PriorGraph:
    """Validated dataset-aware prior: names, mapping and binary mask."""

    feature_names: list[str]
    concept_names: list[str]
    mapping: dict[str, str]
    orphan_features: list[str] = field(default_factory=list)
    mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))

    @property
    def feature_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.feature_names)}

    @property
    def concept_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.concept_names)}

    def stats(self) -> dict:
        active = int(self.mask.sum())
        total = int(self.mask.size)
        per_concept = self.mask.sum(axis=1).astype(int)
        return {
            "n_features": self.mask.shape[1],
            "n_concepts": self.mask.shape[0],
            "active_connections": active,
            "total_connections": total,
            "mask_density": active / total if total else 0.0,
            "orphan_features": list(self.orphan_features),
            "concept_fan_in": {
                c: int(f) for c, f in zip(self.concept_names, per_concept, strict=False)
            },
            "concept_fan_in_min": int(per_concept.min()) if len(per_concept) else 0,
            "concept_fan_in_max": int(per_concept.max()) if len(per_concept) else 0,
        }

    def to_spec(self, target_column: str) -> PriorGraphSpec:
        s = self.stats()
        return PriorGraphSpec(
            target_column=target_column,
            features=list(self.feature_names),
            concepts=list(self.concept_names),
            mapping=dict(self.mapping),
            orphan_features=list(self.orphan_features),
            n_features=s["n_features"],
            n_concepts=s["n_concepts"],
            active_connections=s["active_connections"],
            total_connections=s["total_connections"],
            mask_density=s["mask_density"],
        )


def build_prior_graph(
    df: pd.DataFrame,
    target_column: str,
    mapping: dict[str, str] | str | Path,
) -> PriorGraph:
    """Validate a mapping against a dataset and build the binary mask.

    Raises `PriorGraphError` on: missing target, non-numeric feature columns,
    mapped features that do not exist in the dataset, or degenerate mappings.
    Dataset columns absent from the mapping become *orphan features*.
    """
    if isinstance(mapping, (str, Path)):
        mapping = load_mapping(mapping)
    if target_column not in df.columns:
        raise PriorGraphError(
            f"target column {target_column!r} not in dataset "
            f"(available: {list(df.columns)[:20]}...)"
        )
    dataset_cols = [c for c in df.columns if c != target_column]

    missing = [f for f in mapping if f not in dataset_cols]
    if missing:
        raise PriorGraphError(
            f"{len(missing)} mapped features are absent from the dataset "
            f"(first examples: {missing[:5]})"
        )
    for col in dataset_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            raise PriorGraphError(
                f"feature column {col!r} is not fully numeric; "
                "encode/clean it before building a prior graph"
            )

    feature_names = dataset_cols  # mask covers every non-target column
    concept_names = sorted({concept for concept in mapping.values()})
    if not concept_names:
        raise PriorGraphError("mapping defines no concepts")

    orphans = [c for c in dataset_cols if c not in mapping]

    fidx = {name: i for i, name in enumerate(feature_names)}
    cidx = {name: i for i, name in enumerate(concept_names)}
    mask = np.zeros((len(concept_names), len(feature_names)), dtype=np.float32)
    for feature, concept in mapping.items():
        mask[cidx[concept], fidx[feature]] = 1.0

    if mask.sum() == 0:
        raise PriorGraphError("mask has no active connections")

    return PriorGraph(
        feature_names=feature_names,
        concept_names=concept_names,
        mapping=dict(mapping),
        orphan_features=orphans,
        mask=mask,
    )
