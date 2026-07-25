"""
Graph generation utilities for BenchLab.

Creates Plotly and Matplotlib visualizations for:
- TPS timelines
- GPU telemetry trends
- Configuration heatmaps
- Scatter plots (TPS vs GPU util)
- Histograms (latency distribution)
- Radar charts (category scores)
- Box plots (performance distribution)
- Correlation matrices
- Time series decomposition
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchlab.logging import get_logger

logger = get_logger("graphs")

PLOTLY_AVAILABLE = True
MATPLOTLIB_AVAILABLE = True


def generate_tps_timeline(
    data: list[dict[str, Any]],
    output_path: str | None = None,
) -> str | None:
    """Generate TPS timeline chart."""
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d.get("config_hash", "") for d in data],
        y=[d.get("avg_tps", 0) for d in data],
        mode="lines+markers",
        name="Avg TPS",
        line=dict(color="#58a6ff"),
    ))
    fig.add_trace(go.Scatter(
        x=[d.get("config_hash", "") for d in data],
        y=[d.get("peak_tps", 0) for d in data],
        mode="lines+markers",
        name="Peak TPS",
        line=dict(color="#3fb950"),
    ))

    fig.update_layout(
        title="Throughput Timeline",
        xaxis_title="Configuration",
        yaxis_title="Tokens/sec",
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        return output_path
    return None


def generate_gpu_heatmap(
    data: list[dict[str, Any]],
    output_path: str | None = None,
) -> str | None:
    """Generate GPU utilization heatmap."""
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    configs = [d.get("config_hash", "") for d in data]
    eu_utils = [d.get("gpu_util", 0) for d in data]
    powers = [d.get("gpu_power", 0) for d in data]
    temps = [d.get("gpu_temp", 0) for d in data]

    fig = go.Figure(data=go.Heatmap(
        z=[eu_utils, powers, temps],
        x=configs,
        y=["EU Util %", "Power W", "Temp C"],
        colorscale="RdYlGn",
    ))

    fig.update_layout(
        title="GPU Telemetry Heatmap",
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        return output_path
    return None


def generate_radar_chart(
    data: list[dict[str, Any]],
    top_n: int = 5,
    output_path: str | None = None,
) -> str | None:
    """Generate radar chart for top configurations."""
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    sorted_data = sorted(data, key=lambda d: d.get("score", 0), reverse=True)[:top_n]

    categories = ["Performance", "Stability", "Efficiency"]
    fig = go.Figure()

    for config in sorted_data:
        fig.add_trace(go.Scatterpolar(
            r=[
                config.get("performance", 0) * 100,
                config.get("stability", 0) * 100,
                config.get("efficiency", 0) * 100,
            ],
            theta=categories,
            name=config.get("config_hash", "")[:8],
            fill="toself",
        ))

    fig.update_layout(
        title="Top Configs - Category Scores",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        return output_path
    return None


def generate_latency_histogram(
    data: list[dict[str, Any]],
    output_path: str | None = None,
) -> str | None:
    """Generate latency distribution histogram."""
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    ttfts = [d.get("ttft", 0) for d in data if d.get("ttft")]

    fig = go.Figure(data=go.Histogram(
        x=ttfts,
        nbinsx=30,
        marker_color="#58a6ff",
    ))

    fig.update_layout(
        title="TTFT Distribution",
        xaxis_title="Time to First Token (s)",
        yaxis_title="Count",
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        return output_path
    return None


def generate_scatter_tps_vs_gpu(
    data: list[dict[str, Any]],
    output_path: str | None = None,
) -> str | None:
    """Generate scatter plot of TPS vs GPU utilization."""
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d.get("gpu_util", 0) for d in data],
        y=[d.get("avg_tps", 0) for d in data],
        mode="markers",
        marker=dict(
            size=[d.get("score", 0) * 10 for d in data],
            color=[d.get("score", 0) for d in data],
            colorscale="Viridis",
            showscale=True,
        ),
        text=[d.get("config_hash", "") for d in data],
    ))

    fig.update_layout(
        title="TPS vs GPU Utilization",
        xaxis_title="GPU Utilization %",
        yaxis_title="Avg TPS",
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        return output_path
    return None


def generate_box_plot(
    data: list[dict[str, Any]],
    metric: str = "avg_tps",
    output_path: str | None = None,
) -> str | None:
    """Generate box plot for performance distribution."""
    if not PLOTLY_AVAILABLE:
        return None

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=[d.get(metric, 0) for d in data],
        name=metric,
        marker_color="#58a6ff",
    ))

    fig.update_layout(
        title=f"{metric} Distribution",
        yaxis_title=metric,
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        return output_path
    return None


def generate_correlation_matrix(
    data: list[dict[str, Any]],
    output_path: str | None = None,
) -> str | None:
    """Generate correlation matrix heatmap."""
    if not PLOTLY_AVAILABLE:
        return None

    import numpy as np

    metrics = ["avg_tps", "peak_tps", "ttft", "gpu_util", "gpu_power", "score"]
    values = [[d.get(m, 0) for m in metrics] for d in data]

    corr = np.corrcoef(values, rowvar=False)

    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=metrics,
        y=metrics,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
    ))

    fig.update_layout(
        title="Metric Correlation Matrix",
        template="plotly_dark",
    )

    if output_path:
        fig.write_html(output_path)
        return output_path
    return None


def generate_all_graphs(
    data: list[dict[str, Any]],
    output_dir: str = "reports",
) -> dict[str, str]:
    """Generate all graphs."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    graphs = {}
    generators = [
        ("tps_timeline", generate_tps_timeline),
        ("gpu_heatmap", generate_gpu_heatmap),
        ("radar_chart", generate_radar_chart),
        ("latency_histogram", generate_latency_histogram),
        ("scatter_tps_vs_gpu", generate_scatter_tps_vs_gpu),
        ("box_plot", generate_box_plot),
        ("correlation_matrix", generate_correlation_matrix),
    ]

    for name, gen_fn in generators:
        try:
            path = gen_fn(data, str(output_dir_path / f"{name}.html"))
            if path:
                graphs[name] = path
        except Exception as e:
            logger.debug(f"Failed to generate {name}: {e}")

    logger.info(f"Generated {len(graphs)} graphs")
    return graphs
