"""MCP surface: tool registration, JSON-safe tool behavior, resource, and the
engine-layer isolation invariant (engine never imports mcp)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from priorneuralnet import mcp_server
from priorneuralnet.engine.synthetic import make_synthetic_prior_dataset

SRC_ENGINE = Path(__file__).resolve().parents[1] / "src" / "priorneuralnet" / "engine"


def test_engine_layer_never_imports_mcp():
    """Definition-of-done invariant: engine/ has zero imports from `mcp`."""
    offenders = []
    for py in SRC_ENGINE.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import mcp", "from mcp")):
                offenders.append(f"{py.name}: {stripped}")
    assert not offenders, f"engine must not depend on mcp: {offenders}"


def test_tools_and_resource_are_registered():
    tools = asyncio.run(mcp_server.mcp.list_tools())
    names = {t.name for t in tools}
    assert {
        "validate_domain_prior",
        "run_priorneuralnet_benchmark",
        "inspect_concept_drivers",
    } <= names
    resources = asyncio.run(mcp_server.mcp.list_resource_templates())
    assert any("manifest" in str(r.uri_template) for r in resources)


@pytest.fixture()
def small_dataset(tmp_path):
    ds = make_synthetic_prior_dataset(
        n_samples=160, n_pathways=6, pathway_size=10, n_orphans=15, seed=5
    )
    paths = ds.save(tmp_path)
    return tmp_path, ds, paths


def test_validate_domain_prior_tool(small_dataset):
    tmp, ds, paths = small_dataset
    result = mcp_server.validate_domain_prior(paths["dataset"], paths["prior"], ds.target_column)
    assert result["valid"] is True
    assert result["stats"]["n_concepts"] == 6
    assert len(result["stats"]["orphan_features"]) == 15
    assert result["warnings"]  # orphan warning present

    bad = mcp_server.validate_domain_prior(paths["dataset"], paths["prior"], "ghost_target")
    assert bad["valid"] is False and "error" in bad


def test_run_and_inspect_roundtrip(small_dataset):
    tmp, ds, paths = small_dataset
    out = str(tmp / "runs")
    result = mcp_server.run_priorneuralnet_benchmark(
        paths["dataset"],
        paths["prior"],
        ds.target_column,
        run_learning_curve=False,
        n_splits=2,
        epochs=40,
        output_dir=out,
    )
    assert result["ok"] is True, result.get("error")
    assert result["verdict"]
    top = result["top_concepts"]
    assert top, "benchmark produced no concept ranking"
    inspected = mcp_server.inspect_concept_drivers(
        result["run_id"], top[0], top_k=5, output_dir=out
    )
    assert inspected["concept"] == top[0]
    assert 0 < len(inspected["top_features"]) <= 5
    missing = mcp_server.inspect_concept_drivers(result["run_id"], "not_a_concept", output_dir=out)
    assert missing["ok"] is False


def test_manifest_resource_reads_default_runs_dir(tmp_path, monkeypatch):
    """Resource path handling without a full run: stub a run directory."""
    monkeypatch.chdir(tmp_path)
    fake_run = tmp_path / "runs" / "stub-run"
    fake_run.mkdir(parents=True)
    payload = {"run_id": "stub-run", "hello": "world"}
    (fake_run / "manifest.json").write_text(json.dumps(payload))
    contents = asyncio.run(mcp_server.mcp.read_resource("priorneuralnet://runs/stub-run/manifest"))
    body = _first_text(contents)
    assert json.loads(body)["run_id"] == "stub-run"
    missing = asyncio.run(mcp_server.mcp.read_resource("priorneuralnet://runs/ghost/manifest"))
    body = _first_text(missing)
    assert "error" in json.loads(body)


def _first_text(contents) -> str:
    """mcp 2.x yields ReadResourceContents (`.content`); tolerate plain str/tuple."""
    for c in contents:
        text = getattr(c, "content", None) or (c if isinstance(c, str) else None)
        if text:
            return str(text)
    raise AssertionError("resource returned no text content")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
