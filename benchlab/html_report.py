"""
Interactive HTML report generation for BenchLab.

Generates a full interactive dashboard with:
- Plotly charts (timeline, scatter, heatmap, radar, histogram)
- Configuration comparison tables
- Corruption viewer
- GPU telemetry graphs
- vLLM metrics explorer
- Dark mode
- Search and filter
- Export capabilities
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from benchlab.config import BenchLabConfig
from benchlab.logging import get_logger

logger = get_logger("html_report")


class HTMLReportGenerator:
    """Generate interactive HTML reports."""

    def __init__(self, config: BenchLabConfig) -> None:
        self.config = config
        self.output_dir = Path(config.reports_dir) / "html"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate(
        self,
        results: list[Any],
        scores: list[Any],
    ) -> str:
        """Generate the interactive HTML dashboard."""
        summary = self._build_summary(results)
        configs_data = self._build_configs_data(results, scores)
        timeline_data = self._build_timeline_data(results)
        gpu_data = self._build_gpu_data(results)
        vllm_data = self._build_vllm_data(results)
        corruption_data = self._build_corruption_data(results)
        comparison_data = self._build_comparison_data(results)

        html = self._render_template(
            summary=summary,
            configs_data=configs_data,
            timeline_data=timeline_data,
            gpu_data=gpu_data,
            vllm_data=vllm_data,
            corruption_data=corruption_data,
            comparison_data=comparison_data,
        )

        output_path = self.output_dir / "dashboard.html"
        with open(output_path, "w") as f:
            f.write(html)

        logger.info("HTML dashboard generated: %s", output_path)
        return str(output_path)

    def _build_summary(self, results: list[Any]) -> dict[str, Any]:
        total = len(results)
        success = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        corrupted = sum(1 for r in results if r.status == "corrupted")
        crashed = sum(1 for r in results if r.status == "crashed")
        successful = [r for r in results if r.status == "success"]
        avg_tps = [r.metrics.get("avg_tps", 0) for r in successful]
        peak_tps = [r.metrics.get("peak_tps", 0) for r in successful]
        ttft = [r.metrics.get("avg_ttft", 0) for r in successful]

        def avg(data: list[float]) -> float:
            return sum(data) / len(data) if data else 0

        return {
            "total_runs": total,
            "successful": success,
            "failed": failed,
            "corrupted": corrupted,
            "crashed": crashed,
            "avg_tps": round(avg(avg_tps), 2),
            "peak_tps": round(max(peak_tps), 2) if peak_tps else 0,
            "avg_ttft": round(avg(ttft), 4) if ttft else 0,
            "best_tps": round(max(avg_tps), 2) if avg_tps else 0,
        }

    def _build_configs_data(self, results: list[Any], scores: list[Any]) -> list[dict]:
        data = []
        for score in scores:
            d = score.to_dict() if hasattr(score, "to_dict") else score
            data.append({
                "rank": d.get("rank", 0),
                "config_hash": d.get("config_hash", ""),
                "overall_score": round(d.get("overall_score", 0), 4),
                "performance": round(d.get("categories", {}).get("performance", 0), 4),
                "stability": round(d.get("categories", {}).get("stability", 0), 4),
                "efficiency": round(d.get("categories", {}).get("efficiency", 0), 4),
            })
        return data

    def _build_timeline_data(self, results: list[Any]) -> list[dict]:
        data = []
        for r in results:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            data.append({
                "config_hash": d.get("config_hash", ""),
                "avg_tps": d.get("metrics", {}).get("avg_tps", 0),
                "peak_tps": d.get("metrics", {}).get("peak_tps", 0),
                "ttft": d.get("metrics", {}).get("avg_ttft", 0),
                "gpu_util": d.get("gpu_metrics", {}).get("eu_utilization", {}).get("avg", 0),
                "score": d.get("score", 0),
            })
        return data

    def _build_gpu_data(self, results: list[Any]) -> list[dict]:
        data = []
        for r in results:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            gpu = d.get("gpu_metrics", {})
            data.append({
                "config_hash": d.get("config_hash", ""),
                "eu_util_avg": gpu.get("eu_utilization", {}).get("avg", 0),
                "eu_util_max": gpu.get("eu_utilization", {}).get("max", 0),
                "power_avg": gpu.get("power", {}).get("avg", 0),
                "power_max": gpu.get("power", {}).get("max", 0),
                "temp_avg": gpu.get("temperature", {}).get("avg", 0),
                "temp_max": gpu.get("temperature", {}).get("max", 0),
            })
        return data

    def _build_vllm_data(self, results: list[Any]) -> list[dict]:
        data = []
        for r in results:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            vllm = d.get("vllm_metrics", {})
            data.append({
                "config_hash": d.get("config_hash", ""),
                "running_avg": vllm.get("running_requests", {}).get("avg", 0),
                "waiting_avg": vllm.get("waiting_requests", {}).get("avg", 0),
                "cache_avg": vllm.get("gpu_cache", {}).get("avg", 0),
                "prompt_tp": vllm.get("prompt_throughput", {}).get("avg", 0),
                "decode_tp": vllm.get("decode_throughput", {}).get("avg", 0),
            })
        return data

    def _build_corruption_data(self, results: list[Any]) -> list[dict]:
        events = []
        for r in results:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            for event in d.get("corruption_events", []):
                sev = event.get("severity", "")
                badge = "danger" if sev == "critical" else ("warning" if sev == "high" else "success")
                events.append({
                    "config_hash": d.get("config_hash", ""),
                    "type": event.get("corruption_type", ""),
                    "severity": sev,
                    "badge": badge,
                    "evidence": event.get("evidence", ""),
                })
        return events

    def _build_comparison_data(self, results: list[Any]) -> list[dict]:
        data = []
        for r in results:
            d = r.to_dict() if hasattr(r, "to_dict") else r
            metrics = d.get("metrics", {})
            gpu = d.get("gpu_metrics", {})
            data.append({
                "config_hash": d.get("config_hash", ""),
                "config_name": d.get("config_name", ""),
                "status": d.get("status", ""),
                "avg_tps": metrics.get("avg_tps", 0),
                "peak_tps": metrics.get("peak_tps", 0),
                "min_tps": metrics.get("min_tps", 0),
                "ttft": metrics.get("avg_ttft", 0),
                "decode_time": metrics.get("avg_decode_time", 0),
                "gpu_util": gpu.get("eu_utilization", {}).get("avg", 0),
                "power": gpu.get("power", {}).get("avg", 0),
                "score": d.get("score", 0),
            })
        return data

    def _render_template(
        self,
        summary: dict,
        configs_data: list[dict],
        timeline_data: list[dict],
        gpu_data: list[dict],
        vllm_data: list[dict],
        corruption_data: list[dict],
        comparison_data: list[dict],
    ) -> str:
        """Render the full HTML template."""
        generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        configs_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                c["rank"], c["config_hash"], c["overall_score"],
                c["performance"], c["stability"], c["efficiency"]
            ) for c in configs_data[:20]
        )
        corruption_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td><span class='badge badge-{}'>{}</span></td><td>{}</td></tr>".format(
                e["config_hash"], e["type"], e["badge"],
                e["severity"], e["evidence"][:100]
            ) for e in corruption_data[:50]
        )
        comparison_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                c["config_hash"], c["config_name"], c["status"],
                c["avg_tps"], c["peak_tps"], c["ttft"],
                c["gpu_util"], c["power"], c["score"]
            ) for c in comparison_data[:20]
        )

        configs_json = json.dumps(configs_data)
        timeline_json = json.dumps(timeline_data)
        gpu_json = json.dumps(gpu_data)
        vllm_json = json.dumps(vllm_data)
        corruption_json = json.dumps(corruption_data)
        comparison_json = json.dumps(comparison_data)

        parts: list[str] = []
        parts.append("""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>B70 Benchmark Lab Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --success: #3fb950;
            --danger: #f85149;
            --warning: #d29922;
        }
        body { background: var(--bg-primary); color: var(--text-primary); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
        .card { background: var(--bg-secondary); border: 1px solid var(--bg-tertiary); border-radius: 8px; }
        .card-header { background: var(--bg-tertiary); border-bottom: 1px solid var(--bg-tertiary); font-weight: 600; }
        .stat-card { text-align: center; padding: 20px; }
        .stat-value { font-size: 2.5rem; font-weight: 700; color: var(--accent); }
        .stat-label { color: var(--text-secondary); font-size: 0.9rem; }
        .table { color: var(--text-primary); }
        .table-dark { background: var(--bg-secondary); }
        .badge-success { background: var(--success); }
        .badge-danger { background: var(--danger); }
        .badge-warning { background: var(--warning); }
        .chart-container { margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container-fluid py-4">
        <h1 class="mb-4">B70 Benchmark Lab Dashboard</h1>
        <p class="text-secondary">Generated: {}</p>

        <div class="row g-3 mb-4">
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Total Runs</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value" style="color: var(--success)">{}</div>
                    <div class="stat-label">Successful</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value">{}</div>
                    <div class="stat-label">Best TPS</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card stat-card">
                    <div class="stat-value">{}s</div>
                    <div class="stat-label">Avg TTFT</div>
                </div>
            </div>
        </div>

        <ul class="nav nav-tabs mb-3" id="dashTabs" role="tablist">
            <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#overview">Overview</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#configs">Configs</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#gpu">GPU</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#vllm">vLLM</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#corruption">Corruption</button></li>
            <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#compare">Compare</button></li>
        </ul>

        <div class="tab-content">
            <div class="tab-pane fade show active" id="overview">
                <div class="row g-3">
                    <div class="col-md-6 chart-container">
                        <div class="card"><div class="card-header">TPS by Configuration</div><div id="chart-tps"></div></div>
                    </div>
                    <div class="col-md-6 chart-container">
                        <div class="card"><div class="card-header">Score Radar</div><div id="chart-radar"></div></div>
                    </div>
                </div>
            </div>
            <div class="tab-pane fade" id="configs">
                <div class="card">
                    <div class="card-header">Configuration Rankings</div>
                    <div class="card-body">
                        <table class="table table-dark table-striped">
                            <thead><tr><th>Rank</th><th>Config</th><th>Score</th><th>Performance</th><th>Stability</th><th>Efficiency</th></tr></thead>
                            <tbody>
{}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="tab-pane fade" id="gpu">
                <div class="row g-3">
                    <div class="col-md-6 chart-container">
                        <div class="card"><div class="card-header">GPU EU Utilization</div><div id="chart-gpu-util"></div></div>
                    </div>
                    <div class="col-md-6 chart-container">
                        <div class="card"><div class="card-header">GPU Power</div><div id="chart-gpu-power"></div></div>
                    </div>
                </div>
            </div>
            <div class="tab-pane fade" id="vllm">
                <div class="row g-3">
                    <div class="col-md-6 chart-container">
                        <div class="card"><div class="card-header">vLLM Prompt Throughput</div><div id="chart-vllm-prompt"></div></div>
                    </div>
                    <div class="col-md-6 chart-container">
                        <div class="card"><div class="card-header">vLLM Decode Throughput</div><div id="chart-vllm-decode"></div></div>
                    </div>
                </div>
            </div>
            <div class="tab-pane fade" id="corruption">
                <div class="card">
                    <div class="card-header">Corruption Events</div>
                    <div class="card-body">
                        <table class="table table-dark table-striped">
                            <thead><tr><th>Config</th><th>Type</th><th>Severity</th><th>Evidence</th></tr></thead>
                            <tbody>
{}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="tab-pane fade" id="compare">
                <div class="card">
                    <div class="card-header">Configuration Comparison</div>
                    <div class="card-body">
                        <table class="table table-dark table-striped">
                            <thead><tr><th>Config</th><th>Name</th><th>Status</th><th>Avg TPS</th><th>Peak TPS</th><th>TTFT</th><th>GPU %</th><th>Power</th><th>Score</th></tr></thead>
                            <tbody>
{}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const configsData = {};
        const timelineData = {};
        const gpuData = {};
        const vllmData = {};
        const corruptionData = {};
        const comparisonData = {};

        const plotlyConfig = {{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }};
        const plotlyLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#c9d1d9' }},
            margin: {{ t: 30, b: 50, l: 60, r: 20 }}
        }};

        if (timelineData.length) {{
            Plotly.newPlot('chart-tps', [{{
                x: timelineData.map(d => d.config_hash),
                y: timelineData.map(d => d.avg_tps),
                type: 'bar',
                marker: {{ color: timelineData.map(d => d.score > 0.7 ? '#3fb950' : d.score > 0.4 ? '#d29922' : '#f85149') }}
            }}], {{ ...plotlyLayout, title: 'Average TPS by Configuration', xaxis: {{ title: 'Config', tickangle: -45 }} }});
        }}

        if (gpuData.length) {{
            Plotly.newPlot('chart-gpu-util', [{{
                x: gpuData.map(d => d.config_hash),
                y: gpuData.map(d => d.eu_util_avg),
                type: 'bar',
                name: 'Avg EU Util'
            }}, {{
                x: gpuData.map(d => d.config_hash),
                y: gpuData.map(d => d.eu_util_max),
                type: 'bar',
                name: 'Max EU Util'
            }}], {{ ...plotlyLayout, barmode: 'group', title: 'GPU EU Utilization', yaxis: {{ title: '%' }} }});
        }}

        if (gpuData.length) {{
            Plotly.newPlot('chart-gpu-power', [{{
                x: gpuData.map(d => d.config_hash),
                y: gpuData.map(d => d.power_avg),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Avg Power'
            }}], {{ ...plotlyLayout, title: 'GPU Power Consumption', yaxis: {{ title: 'Watts' }} }});
        }}

        if (vllmData.length) {{
            Plotly.newPlot('chart-vllm-prompt', [{{
                x: vllmData.map(d => d.config_hash),
                y: vllmData.map(d => d.prompt_tp),
                type: 'bar',
                name: 'Prompt TPS'
            }}], {{ ...plotlyLayout, title: 'vLLM Prompt Throughput', yaxis: {{ title: 'Tokens/sec' }} }});
        }}

        if (configsData.length) {{
            const top5 = configsData.slice(0, 5);
            Plotly.newPlot('chart-radar', [{{
                r: top5.map(d => d.performance * 100),
                theta: top5.map(d => d.config_hash.substring(0, 8)),
                type: 'scatterpolar',
                name: 'Performance',
                fill: 'toself'
            }}, {{
                r: top5.map(d => d.stability * 100),
                theta: top5.map(d => d.config_hash.substring(0, 8)),
                type: 'scatterpolar',
                name: 'Stability',
                fill: 'toself'
            }}, {{
                r: top5.map(d => d.efficiency * 100),
                theta: top5.map(d => d.config_hash.substring(0, 8)),
                type: 'scatterpolar',
                name: 'Efficiency',
                fill: 'toself'
            }}], {{ ...plotlyLayout, title: 'Top 5 Configs - Category Scores', polar: {{ radialaxis: {{ visible: true, range: [0, 100] }} }} });
        }}
    </script>
</body>
</html>""".format(
            generated_at,
            summary["total_runs"],
            summary["successful"],
            summary["best_tps"],
            summary["avg_ttft"],
            configs_rows,
            corruption_rows,
            comparison_rows,
            configs_json,
            timeline_json,
            gpu_json,
            vllm_json,
            corruption_json,
            comparison_json,
        ))

        return "\n".join(parts)
