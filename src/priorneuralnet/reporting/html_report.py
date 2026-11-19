"""Standalone interactive Plotly HTML report for a benchmark run.

Layout (per the project spec):
  a) 3-column Sankey: top input features -> named domain concepts -> target,
     edges colored by contribution sign, widths proportional to magnitude.
  b) Sample-efficiency curve (validation metric vs training fraction) for
     priorneuralnet vs the parameter-matched MLP, plus generalization-gap
     bars (train - val) for every model.
  c) Named-concept importance bar chart.
  d) Sample x concept latent-activation heatmap.

Figures embed via CDN plotly.js; the page shell is a Jinja2 template so the
HTML is self-contained and opens directly in a browser from disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template

_PAGE = Template(
    """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>priorneuralnet run {{ run_id }}</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 1100px;
        color: #14161a; background: #fbfbfd; }
 h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2.2rem; }
 .meta { background: #eef1f6; border-radius: 8px; padding: .8rem 1rem;
         font-size: .85rem; display: grid; gap: .2rem; }
 .verdict { background: #e7f4ea; border-left: 4px solid #2e7d32; padding: .7rem 1rem;
            margin: 1rem 0; }
 .warn { color: #8a6d1a; }
 .fig { margin-bottom: 1.5rem; }
</style>
</head>
<body>
<h1>Domain-Prior Network benchmark — run {{ run_id }}</h1>
<div class="meta">
  <span><b>dataset:</b> {{ n_samples }} samples × {{ n_features }} features
        (p&gt;&gt;n regime) · task: {{ task }} · metric: {{ metric }}</span>
  <span><b>prior graph:</b> {{ n_concepts }} concepts, {{ active_connections }} active
        edges, mask density {{ mask_density }}</span>
  <span><b>generated:</b> {{ created_at }}</span>
</div>
<div class="verdict">{{ verdict }}</div>
{% for w in warnings %}<p class="warn">⚠ {{ w }}</p>{% endfor %}
{% for title, div in figures %}<div class="fig"><h2>{{ title }}</h2>{{ div }}</div>{% endfor %}
</body>
</html>
""",
    autoescape=True,
)


def _short(name: str, width: int = 26) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def _sankey(manifest: dict) -> go.Figure:
    nodes: list[str] = ["target (positive class)"]
    src, tgt, val, color = [], [], [], []
    palette_pos, palette_neg = "rgba(46,125,50,0.55)", "rgba(178,34,34,0.5)"
    ranking = [c for c in manifest.get("concept_ranking", []) if c["attribution"] > 1e-9][:6]
    for concept in ranking:
        nodes.append(concept["concept"])
        c = len(nodes) - 1
        src.append(c)
        tgt.append(0)
        val.append(max(abs(concept["attribution"]), 1e-4))
        color.append(palette_pos)
        for feat in concept["top_features"][:4]:
            nodes.append(_short(feat["feature"]))
            f = len(nodes) - 1
            src.append(f)
            tgt.append(c)
            val.append(max(abs(feat["contribution"]), 1e-4))
            color.append(palette_pos if feat["contribution"] > 0 else palette_neg)
    fig = go.Figure(
        go.Sankey(
            node=dict(label=nodes, pad=14, thickness=14),
            link=dict(source=src, target=tgt, value=val, color=color),
        )
    )
    fig.update_layout(title="Signal flow: top features → named concepts → target", height=620)
    return fig


def _learning_curve(points: list[dict], metric: str) -> go.Figure:
    fig = go.Figure()
    for p in points:
        fig.add_trace(
            go.Scatter(
                x=[p["train_fraction"]],
                y=[p["val_metric"]],
                mode="lines+markers",
                name=f"{p['model']} (val)",
            )
        )
    fig.update_layout(
        title=f"Sample-efficiency curve — validation {metric} vs training fraction",
        xaxis_title="training fraction",
        yaxis_title=f"validation {metric}",
        height=450,
    )
    return fig


def _gap_bars(manifest: dict) -> go.Figure:
    models = [m["model"] for m in manifest["comparison"]]
    gaps = [m["generalization_gap"] for m in manifest["comparison"]]
    fig = go.Figure(go.Bar(x=models, y=gaps, marker_color="#3b6ea5"))
    fig.update_layout(
        title=f"Generalization gap (train − val, {manifest['dataset']['metric_name']})",
        yaxis_title="gap",
        height=400,
    )
    return fig


def _concept_bars(manifest: dict) -> go.Figure:
    ranking = manifest["concept_ranking"][:15][::-1]
    fig = go.Figure(
        go.Bar(
            x=[c["attribution"] for c in ranking],
            y=[c["concept"] for c in ranking],
            orientation="h",
            marker_color="#2e7d32",
        )
    )
    fig.update_layout(title="Named concept importance (mean |Integrated Gradients|)", height=560)
    return fig


def _activation_heatmap(activations: pd.DataFrame) -> go.Figure:
    max_samples = 120
    sub = activations.iloc[:max_samples]
    fig = go.Figure(
        go.Heatmap(
            z=sub.T.values,
            x=[str(i) for i in sub.index],
            y=list(activations.columns),
            colorscale="RdBu_r",
            zmid=0.0,
            colorbar=dict(title="activation"),
        )
    )
    fig.update_layout(
        title=f"Concept activations (samples × concepts, model trained on full data"
        f"{' — first 120 samples' if len(activations) > max_samples else ''})",
        height=520,
        xaxis_title="sample",
        yaxis_title="concept",
    )
    return fig


def render_report(
    manifest: dict, activations: pd.DataFrame, run_dir: str | Path, out_name: str = "report.html"
) -> Path:
    """Render the interactive dashboard into `run_dir/out_name`."""
    figures: list[tuple[str, str]] = [
        ("Feature → Concept → Target flow", _sankey(manifest)),
        (
            "Sample efficiency",
            _learning_curve(manifest.get("learning_curve", []), manifest["dataset"]["metric_name"]),
        ),
        ("Generalization gap", _gap_bars(manifest)),
        ("Concept importance", _concept_bars(manifest)),
        ("Concept activations", _activation_heatmap(activations)),
    ]
    divs: list[tuple[str, str]] = []
    for i, (title, fig) in enumerate(figures):
        divs.append(
            (
                title,
                pio.to_html(
                    fig,
                    full_html=False,
                    include_plotlyjs="cdn" if i == 0 else False,
                ),
            )
        )
    ds, pg = manifest["dataset"], manifest["prior_graph"]
    html = _PAGE.render(
        run_id=manifest["run_id"],
        created_at=manifest["created_at"],
        n_samples=ds["n_samples"],
        n_features=ds["n_features"],
        task=ds["task"],
        metric=ds["metric_name"],
        n_concepts=pg["n_concepts"],
        active_connections=pg["active_connections"],
        mask_density=pg["mask_density"],
        verdict=manifest["verdict"],
        warnings=manifest.get("warnings", []),
        figures=divs,
    )
    out = Path(run_dir) / out_name
    out.write_text(html, encoding="utf-8")
    return out
