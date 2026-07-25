"""
Report generation for BenchLab.

Generates CSV, JSON, Markdown, and structured reports
from benchmark results and scoring data.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from benchlab.config import BenchLabConfig
from benchlab.logging import get_logger

logger = get_logger("report")


class ReportGenerator:
    """Generate benchmark reports in various formats."""

    def __init__(self, config: BenchLabConfig) -> None:
        self.config = config
        self.results_dir = Path(config.results_dir)
        self.reports_dir = Path(config.reports_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def generate_all(
        self,
        results: list[Any],
        scores: list[Any],
    ) -> dict[str, str]:
        """Generate all report formats."""
        paths: dict[str, str] = {}

        paths["json"] = await self.generate_json(results, scores)
        paths["csv"] = self.generate_csv(results)
        paths["markdown"] = self.generate_markdown(results, scores)

        logger.info(f"Generated reports: {list(paths.keys())}")
        return paths

    async def generate_json(
        self,
        results: list[Any],
        scores: list[Any],
    ) -> str:
        """Generate JSON report."""
        report: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runs": len(results),
            "successful_runs": sum(1 for r in results if r.status == "success"),
            "failed_runs": sum(1 for r in results if r.status == "failed"),
            "corrupted_runs": sum(1 for r in results if r.status == "corrupted"),
            "crashed_runs": sum(1 for r in results if r.status == "crashed"),
            "results": [r.to_dict() if hasattr(r, "to_dict") else r for r in results],
            "scores": [s.to_dict() if hasattr(s, "to_dict") else s for s in scores],
        }

        # Find best configs
        successful = [r for r in results if r.status == "success"]
        if successful:
            report["best_throughput"] = max(successful, key=lambda r: r.metrics.get("avg_tps", 0)).to_dict()
            report["best_latency"] = min(successful, key=lambda r: r.metrics.get("avg_total_time", float("inf"))).to_dict()
            report["most_stable"] = min(successful, key=lambda r: r.metrics.get("stddev_tps", float("inf"))).to_dict()

        path = self.reports_dir / "report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return str(path)

    def generate_csv(self, results: list[Any]) -> str:
        """Generate CSV report."""
        path = self.reports_dir / "results.csv"

        if not results:
            with open(path, "w") as f:
                f.write("config_hash,status,duration,avg_tps,peak_tps,ttft\n")
            return str(path)

        # Collect all field names
        fieldnames = [
            "config_hash", "config_name", "status", "duration",
            "avg_tps", "peak_tps", "min_tps", "stddev_tps",
            "avg_ttft", "avg_decode_time", "avg_total_time",
            "total_tokens", "total_requests", "corruption_count",
            "gpu_util_avg", "gpu_util_max", "gpu_power_avg",
            "gpu_temp_avg", "memory_usage", "score",
        ]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for result in results:
                if hasattr(result, "to_dict"):
                    d = result.to_dict()
                else:
                    d = result

                row = {
                    "config_hash": d.get("config_hash", ""),
                    "config_name": d.get("config_name", ""),
                    "status": d.get("status", ""),
                    "duration": d.get("duration", 0),
                    "avg_tps": d.get("metrics", {}).get("avg_tps", 0),
                    "peak_tps": d.get("metrics", {}).get("peak_tps", 0),
                    "min_tps": d.get("metrics", {}).get("min_tps", 0),
                    "stddev_tps": d.get("metrics", {}).get("stddev_tps", 0),
                    "avg_ttft": d.get("metrics", {}).get("avg_ttft", 0),
                    "avg_decode_time": d.get("metrics", {}).get("avg_decode_time", 0),
                    "avg_total_time": d.get("metrics", {}).get("avg_total_time", 0),
                    "total_tokens": d.get("metrics", {}).get("total_tokens", 0),
                    "total_requests": d.get("metrics", {}).get("total_requests", 0),
                    "corruption_count": len(d.get("corruption_events", [])),
                    "gpu_util_avg": d.get("gpu_metrics", {}).get("eu_utilization", {}).get("avg", 0),
                    "gpu_util_max": d.get("gpu_metrics", {}).get("eu_utilization", {}).get("max", 0),
                    "gpu_power_avg": d.get("gpu_metrics", {}).get("power", {}).get("avg", 0),
                    "gpu_temp_avg": d.get("gpu_metrics", {}).get("temperature", {}).get("avg", 0),
                    "memory_usage": d.get("metrics", {}).get("memory_usage", 0),
                    "score": d.get("score", 0),
                }
                writer.writerow(row)

        return str(path)

    def generate_markdown(
        self,
        results: list[Any],
        scores: list[Any],
    ) -> str:
        """Generate Markdown report."""
        lines: list[str] = []
        lines.append("# B70 Benchmark Lab Report\n")
        lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary
        total = len(results)
        success = sum(1 for r in results if r.status == "success")
        failed = sum(1 for r in results if r.status == "failed")
        corrupted = sum(1 for r in results if r.status == "corrupted")
        crashed = sum(1 for r in results if r.status == "crashed")

        lines.append("## Summary\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Runs | {total} |")
        lines.append(f"| Successful | {success} |")
        lines.append(f"| Failed | {failed} |")
        lines.append(f"| Corrupted | {corrupted} |")
        lines.append(f"| Crashed | {crashed} |")
        lines.append("")

        # Top configurations
        if scores:
            lines.append("## Top Configurations\n")
            lines.append("| Rank | Config | Score | Avg TPS | Peak TPS | Stability |")
            lines.append("|------|--------|-------|---------|----------|-----------|")
            for score in scores[:10]:
                if hasattr(score, "to_dict"):
                    d = score.to_dict()
                else:
                    d = score
                lines.append(
                    f"| {d.get('rank', 0)} | "
                    f"{d.get('config_hash', '')} | "
                    f"{d.get('overall_score', 0):.4f} | "
                    f"{d.get('categories', {}).get('performance', 0):.4f} | "
                    f"{d.get('categories', {}).get('performance', 0):.4f} | "
                    f"{d.get('categories', {}).get('stability', 0):.4f} |"
                )
            lines.append("")

        # Detailed results
        lines.append("## Detailed Results\n")
        lines.append("| Config | Status | Duration | Avg TPS | TTFT | GPU Util | Score |")
        lines.append("|--------|--------|----------|---------|------|----------|-------|")

        for result in results:
            if hasattr(result, "to_dict"):
                d = result.to_dict()
            else:
                d = result
            metrics = d.get("metrics", {})
            gpu = d.get("gpu_metrics", {})
            lines.append(
                f"| {d.get('config_hash', '')} | "
                f"{d.get('status', '')} | "
                f"{d.get('duration', 0):.1f}s | "
                f"{metrics.get('avg_tps', 0):.1f} | "
                f"{metrics.get('avg_ttft', 0):.3f}s | "
                f"{gpu.get('eu_utilization', {}).get('avg', 0):.0f}% | "
                f"{d.get('score', 0):.4f} |"
            )
        lines.append("")

        # Corruption events
        corruption_events = [
            e for r in results
            for e in r.get("corruption_events", [])
        ]
        if corruption_events:
            lines.append("## Corruption Events\n")
            for event in corruption_events[:20]:
                lines.append(f"- **{event.get('corruption_type', '')}** [{event.get('severity', '')}]: {event.get('evidence', '')}")
            lines.append("")

        path = self.reports_dir / "report.md"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        return str(path)

    def generate_summary(self, results: list[Any]) -> dict[str, Any]:
        """Generate a summary dictionary."""
        successful = [r for r in results if r.status == "success"]

        avg_tps = [r.metrics.get("avg_tps", 0) for r in successful]
        peak_tps = [r.metrics.get("peak_tps", 0) for r in successful]
        ttft = [r.metrics.get("avg_ttft", 0) for r in successful]
        gpu_util = [r.gpu_metrics.get("eu_utilization", {}).get("avg", 0) for r in successful]

        def avg(data: list[float]) -> float:
            return sum(data) / len(data) if data else 0

        return {
            "total_runs": len(results),
            "successful_runs": len(successful),
            "avg_tps": avg(avg_tps),
            "peak_tps": max(peak_tps) if peak_tps else 0,
            "avg_ttft": avg(ttft),
            "avg_gpu_util": avg(gpu_util),
            "best_tps": max(avg_tps) if avg_tps else 0,
            "worst_tps": min(avg_tps) if avg_tps else 0,
        }
