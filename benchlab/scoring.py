"""
Scoring engine for BenchLab.

Computes weighted scores for each configuration based on
performance, stability, efficiency, and corruption metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchlab.config import ScoringConfig
from benchlab.logging import get_logger

logger = get_logger("scoring")


@dataclass
class ScoreBreakdown:
    """Individual score components."""
    peak_tps_score: float = 0.0
    average_tps_score: float = 0.0
    p95_tps_score: float = 0.0
    min_tps_score: float = 0.0
    ttft_score: float = 0.0
    latency_score: float = 0.0
    gpu_util_score: float = 0.0
    power_efficiency_score: float = 0.0
    vram_efficiency_score: float = 0.0
    stability_score: float = 0.0
    long_context_stability_score: float = 0.0
    scheduler_efficiency_score: float = 0.0
    graph_stability_score: float = 0.0

    corruption_penalty: float = 0.0
    oom_penalty: float = 0.0
    crash_penalty: float = 0.0
    memory_growth_penalty: float = 0.0


@dataclass
class ConfigurationScore:
    """Overall score for a configuration."""
    config_id: str
    config_hash: str
    overall_score: float = 0.0
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)
    rank: int = 0
    categories: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "config_hash": self.config_hash,
            "overall_score": self.overall_score,
            "rank": self.rank,
            "categories": self.categories,
            "breakdown": {
                "peak_tps_score": self.breakdown.peak_tps_score,
                "average_tps_score": self.breakdown.average_tps_score,
                "p95_tps_score": self.breakdown.p95_tps_score,
                "min_tps_score": self.breakdown.min_tps_score,
                "ttft_score": self.breakdown.ttft_score,
                "latency_score": self.breakdown.latency_score,
                "gpu_util_score": self.breakdown.gpu_util_score,
                "power_efficiency_score": self.breakdown.power_efficiency_score,
                "vram_efficiency_score": self.breakdown.vram_efficiency_score,
                "stability_score": self.breakdown.stability_score,
                "long_context_stability_score": self.breakdown.long_context_stability_score,
                "scheduler_efficiency_score": self.breakdown.scheduler_efficiency_score,
                "graph_stability_score": self.breakdown.graph_stability_score,
                "corruption_penalty": self.breakdown.corruption_penalty,
                "oom_penalty": self.breakdown.oom_penalty,
                "crash_penalty": self.breakdown.crash_penalty,
                "memory_growth_penalty": self.breakdown.memory_growth_penalty,
            },
            "metadata": self.metadata,
        }


class ScoringEngine:
    """Compute weighted scores for benchmark configurations."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self._normalization_cache: dict[str, dict[str, float]] = {}

    def score(
        self,
        config_id: str,
        config_hash: str,
        metrics: dict[str, Any],
    ) -> ConfigurationScore:
        """Compute a complete score for a configuration."""
        breakdown = self._compute_breakdown(metrics)
        overall = self._compute_overall(breakdown)

        score = ConfigurationScore(
            config_id=config_id,
            config_hash=config_hash,
            overall_score=overall,
            breakdown=breakdown,
        )

        # Compute category scores
        score.categories = {
            "performance": self._category_performance(breakdown),
            "stability": self._category_stability(breakdown),
            "efficiency": self._category_efficiency(breakdown),
        }

        logger.info(
            f"Scored config {config_hash}: overall={overall:.4f}, "
            f"performance={score.categories['performance']:.4f}, "
            f"stability={score.categories['stability']:.4f}, "
            f"efficiency={score.categories['efficiency']:.4f}"
        )

        return score

    def rank(self, scores: list[ConfigurationScore]) -> list[ConfigurationScore]:
        """Rank configurations by overall score."""
        sorted_scores = sorted(scores, key=lambda s: s.overall_score, reverse=True)
        for i, score in enumerate(sorted_scores):
            score.rank = i + 1
        return sorted_scores

    def _compute_breakdown(self, metrics: dict[str, Any]) -> ScoreBreakdown:
        """Compute individual score components."""
        breakdown = ScoreBreakdown()

        # Performance scores
        breakdown.peak_tps_score = self._score_peak_tps(metrics)
        breakdown.average_tps_score = self._score_average_tps(metrics)
        breakdown.p95_tps_score = self._score_p95_tps(metrics)
        breakdown.min_tps_score = self._score_min_tps(metrics)

        # Latency scores
        breakdown.ttft_score = self._score_ttft(metrics)
        breakdown.latency_score = self._score_latency(metrics)

        # Efficiency scores
        breakdown.gpu_util_score = self._score_gpu_utilization(metrics)
        breakdown.power_efficiency_score = self._score_power_efficiency(metrics)
        breakdown.vram_efficiency_score = self._score_vram_efficiency(metrics)

        # Stability scores
        breakdown.stability_score = self._score_stability(metrics)
        breakdown.long_context_stability_score = self._score_long_context_stability(metrics)
        breakdown.scheduler_efficiency_score = self._score_scheduler_efficiency(metrics)
        breakdown.graph_stability_score = self._score_graph_stability(metrics)

        # Penalties
        breakdown.corruption_penalty = self._score_corruption(metrics)
        breakdown.oom_penalty = self._score_oom(metrics)
        breakdown.crash_penalty = self._score_crash(metrics)
        breakdown.memory_growth_penalty = self._score_memory_growth(metrics)

        return breakdown

    def _compute_overall(self, breakdown: ScoreBreakdown) -> float:
        """Compute weighted overall score."""
        score = (
            breakdown.peak_tps_score * self.config.peak_tps_weight
            + breakdown.average_tps_score * self.config.average_tps_weight
            + breakdown.p95_tps_score * self.config.p95_tps_weight
            + breakdown.min_tps_score * self.config.min_tps_weight
            + breakdown.ttft_score * self.config.ttft_weight
            + breakdown.latency_score * self.config.latency_weight
            + breakdown.gpu_util_score * self.config.gpu_util_weight
            + breakdown.power_efficiency_score * self.config.power_efficiency_weight
            + breakdown.vram_efficiency_score * self.config.vram_efficiency_weight
            + breakdown.stability_score * self.config.stability_weight
            + breakdown.long_context_stability_score * self.config.long_context_stability_weight
            + breakdown.scheduler_efficiency_score * self.config.scheduler_efficiency_weight
            + breakdown.graph_stability_score * self.config.graph_stability_weight
        )

        # Apply penalties
        score -= breakdown.corruption_penalty * self.config.corruption_penalty
        score -= breakdown.oom_penalty * self.config.oom_penalty
        score -= breakdown.crash_penalty * self.config.crash_penalty
        score -= breakdown.memory_growth_penalty * self.config.memory_growth_penalty

        return max(0.0, min(1.0, score))

    def _category_performance(self, breakdown: ScoreBreakdown) -> float:
        """Performance category score."""
        weights = (
            self.config.peak_tps_weight
            + self.config.average_tps_weight
            + self.config.p95_tps_weight
            + self.config.min_tps_weight
            + self.config.ttft_weight
            + self.config.latency_weight
        )
        if weights == 0:
            return 0.0
        return (
            breakdown.peak_tps_score * self.config.peak_tps_weight
            + breakdown.average_tps_score * self.config.average_tps_weight
            + breakdown.p95_tps_score * self.config.p95_tps_weight
            + breakdown.min_tps_score * self.config.min_tps_weight
            + breakdown.ttft_score * self.config.ttft_weight
            + breakdown.latency_score * self.config.latency_weight
        ) / weights

    def _category_stability(self, breakdown: ScoreBreakdown) -> float:
        """Stability category score."""
        weights = (
            self.config.stability_weight
            + self.config.long_context_stability_weight
            + self.config.scheduler_efficiency_weight
            + self.config.graph_stability_weight
        )
        if weights == 0:
            return 0.0
        return (
            breakdown.stability_score * self.config.stability_weight
            + breakdown.long_context_stability_score * self.config.long_context_stability_weight
            + breakdown.scheduler_efficiency_score * self.config.scheduler_efficiency_weight
            + breakdown.graph_stability_score * self.config.graph_stability_weight
        ) / weights

    def _category_efficiency(self, breakdown: ScoreBreakdown) -> float:
        """Efficiency category score."""
        weights = (
            self.config.gpu_util_weight
            + self.config.power_efficiency_weight
            + self.config.vram_efficiency_weight
        )
        if weights == 0:
            return 0.0
        return (
            breakdown.gpu_util_score * self.config.gpu_util_weight
            + breakdown.power_efficiency_score * self.config.power_efficiency_weight
            + breakdown.vram_efficiency_score * self.config.vram_efficiency_weight
        ) / weights

    # --- Individual scoring methods ---

    def _score_peak_tps(self, metrics: dict[str, Any]) -> float:
        peak_tps = metrics.get("peak_tps", 0)
        return min(1.0, peak_tps / 1000.0)  # Normalize: 1000 TPS = 1.0

    def _score_average_tps(self, metrics: dict[str, Any]) -> float:
        avg_tps = metrics.get("avg_tps", 0)
        return min(1.0, avg_tps / 1000.0)

    def _score_p95_tps(self, metrics: dict[str, Any]) -> float:
        p95_tps = metrics.get("p95_tps", 0)
        return min(1.0, p95_tps / 1000.0)

    def _score_min_tps(self, metrics: dict[str, Any]) -> float:
        min_tps = metrics.get("min_tps", 0)
        return min(1.0, min_tps / 1000.0)

    def _score_ttft(self, metrics: dict[str, Any]) -> float:
        """TTFT scoring: lower is better."""
        ttft = metrics.get("avg_ttft", float("inf"))
        if ttft <= 0.1:
            return 1.0
        elif ttft >= 2.0:
            return 0.0
        return max(0.0, 1.0 - (ttft - 0.1) / 1.9)

    def _score_latency(self, metrics: dict[str, Any]) -> float:
        """Latency scoring: lower is better."""
        latency = metrics.get("avg_total_time", float("inf"))
        if latency <= 0.5:
            return 1.0
        elif latency >= 5.0:
            return 0.0
        return max(0.0, 1.0 - (latency - 0.5) / 4.5)

    def _score_gpu_utilization(self, metrics: dict[str, Any]) -> float:
        gpu_util = metrics.get("avg_gpu_utilization", 0)
        return min(1.0, gpu_util / 100.0)

    def _score_power_efficiency(self, metrics: dict[str, Any]) -> float:
        """Power efficiency: higher TPS per watt is better."""
        tps = metrics.get("avg_tps", 0)
        power = metrics.get("avg_power", 1)
        if power <= 0:
            return 0.0
        efficiency = tps / power
        return min(1.0, efficiency / 10.0)  # 10 TPS/watt = 1.0

    def _score_vram_efficiency(self, metrics: dict[str, Any]) -> float:
        """VRAM efficiency: higher utilization with good performance."""
        gpu_mem = metrics.get("avg_gpu_memory", 0)
        tps = metrics.get("avg_tps", 0)
        if gpu_mem <= 0:
            return 0.0
        return min(1.0, (tps / 100.0) * (gpu_mem / 100.0))

    def _score_stability(self, metrics: dict[str, Any]) -> float:
        """Stability scoring."""
        crash_rate = metrics.get("crash_rate", 0)
        error_rate = metrics.get("error_rate", 0)
        return max(0.0, 1.0 - crash_rate - error_rate)

    def _score_long_context_stability(self, metrics: dict[str, Any]) -> float:
        """Long context stability scoring."""
        degradation = metrics.get("long_context_degradation", 0)
        return max(0.0, 1.0 - degradation)

    def _score_scheduler_efficiency(self, metrics: dict[str, Any]) -> float:
        """Scheduler efficiency scoring."""
        queue_time = metrics.get("avg_queue_time", 0)
        if queue_time <= 0.01:
            return 1.0
        elif queue_time >= 1.0:
            return 0.0
        return max(0.0, 1.0 - queue_time)

    def _score_graph_stability(self, metrics: dict[str, Any]) -> float:
        """Graph stability scoring."""
        graph_failures = metrics.get("graph_failures", 0)
        total_ops = metrics.get("total_operations", 1)
        failure_rate = graph_failures / total_ops if total_ops > 0 else 0
        return max(0.0, 1.0 - failure_rate * 10)

    def _score_corruption(self, metrics: dict[str, Any]) -> float:
        """Corruption penalty: 0 = no penalty, 1 = full penalty."""
        corruption_count = metrics.get("corruption_count", 0)
        if corruption_count == 0:
            return 0.0
        return min(1.0, corruption_count / 10.0)

    def _score_oom(self, metrics: dict[str, Any]) -> float:
        """OOM penalty."""
        oom_count = metrics.get("oom_count", 0)
        return 1.0 if oom_count > 0 else 0.0

    def _score_crash(self, metrics: dict[str, Any]) -> float:
        """Crash penalty."""
        crash_count = metrics.get("crash_count", 0)
        return 1.0 if crash_count > 0 else 0.0

    def _score_memory_growth(self, metrics: dict[str, Any]) -> float:
        """Memory growth penalty."""
        growth_rate = metrics.get("memory_growth_rate", 0)
        if growth_rate <= 0.01:
            return 0.0
        elif growth_rate >= 0.50:
            return 1.0
        return min(1.0, growth_rate / 0.50)
