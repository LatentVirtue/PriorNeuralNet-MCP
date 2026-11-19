"""Prior-graph loading, validation, mask construction, orphan reporting."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from priorneuralnet.engine.prior_graph import (
    PriorGraphError,
    build_prior_graph,
    load_dataframe,
    load_mapping,
)


@pytest.fixture()
def dataset_csv(tmp_path):
    df = pd.DataFrame(
        {
            "gene_A1": [1.0, 2.0, 3.0, 4.0],
            "gene_A2": [0.5, 0.5, 0.5, 0.5],
            "gene_B1": [2.0, 1.0, 0.0, 3.0],
            "stray_C": [1.0, 1.0, 1.0, 1.0],
            "label": [0, 1, 0, 1],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def prior_json(tmp_path):
    mapping = {"gene_A1": "pathway_A", "gene_A2": "pathway_A", "gene_B1": "pathway_B"}
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    return path


def test_load_mapping_json_object_and_list(tmp_path):
    obj = tmp_path / "a.json"
    obj.write_text(json.dumps({"f1": "c1"}))
    assert load_mapping(obj) == {"f1": "c1"}
    lst = tmp_path / "b.json"
    lst.write_text(json.dumps([{"feature": "f1", "concept": "c1"}]))
    assert load_mapping(lst) == {"f1": "c1"}


def test_load_mapping_csv_two_column_with_optional_header(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("feature,concept\ng1,pathA\ng2,pathA\ng3,pathB\n")
    assert load_mapping(p) == {"g1": "pathA", "g2": "pathA", "g3": "pathB"}
    q = tmp_path / "q.csv"
    q.write_text("g1,pathA\ng2,pathB\n")
    assert load_mapping(q) == {"g1": "pathA", "g2": "pathB"}


def test_mask_shape_density_and_orphans(dataset_csv, prior_json):
    df = load_dataframe(dataset_csv)
    graph = build_prior_graph(df, "label", prior_json)
    assert graph.concept_names == ["pathway_A", "pathway_B"]
    assert graph.feature_names == ["gene_A1", "gene_A2", "gene_B1", "stray_C"]
    assert graph.orphan_features == ["stray_C"]
    # 3 active edges out of 2 concepts x 4 features
    assert graph.mask.sum() == 3.0
    assert graph.stats()["mask_density"] == pytest.approx(3 / 8)
    assert graph.stats()["concept_fan_in"] == {"pathway_A": 2, "pathway_B": 1}
    spec = graph.to_spec("label")
    assert spec.n_features == 4 and spec.n_concepts == 2 and spec.active_connections == 3


def test_missing_target_rejected(dataset_csv, prior_json):
    df = load_dataframe(dataset_csv)
    with pytest.raises(PriorGraphError, match="target column"):
        build_prior_graph(df, "nope", prior_json)


def test_mapped_feature_absent_from_dataset_rejected(tmp_path, dataset_csv):
    df = load_dataframe(dataset_csv)
    mapping = {"gene_A1": "pA", "ghost_gene": "pA"}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(mapping))
    with pytest.raises(PriorGraphError, match="absent from the dataset"):
        build_prior_graph(df, "label", p)


def test_non_numeric_feature_column_rejected(tmp_path):
    df = pd.DataFrame({"a": ["x", "y"], "b": [1.0, 2.0], "t": [0, 1]})
    with pytest.raises(PriorGraphError, match="not fully numeric"):
        build_prior_graph(df, "t", {"a": "c1", "b": "c1"})


def test_empty_mapping_and_bad_format_rejected(tmp_path, dataset_csv):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({}))
    with pytest.raises(PriorGraphError, match="empty"):
        load_mapping(p)
    q = tmp_path / "weird.txt"
    q.write_text("nonsense")
    with pytest.raises(PriorGraphError):
        load_mapping(q)
    df = load_dataframe(dataset_csv)
    with pytest.raises(PriorGraphError):
        load_mapping(tmp_path / "does-not-exist.json")
        build_prior_graph(df, "label", {"missing_col": "c"})
