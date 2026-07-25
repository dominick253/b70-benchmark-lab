"""
Textual TUI dashboard for BenchLab.

Real-time interactive terminal dashboard showing:
- Live benchmark progress
- Configuration matrix status
- GPU telemetry
- vLLM metrics
- Corruption events
- Scoring

Usage:
    benchlab dashboard-tui
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from benchlab.logging import get_logger

logger = get_logger("dashboard")

if TYPE_CHECKING:
    from textual.app import App


@dataclass
class DashboardState:
    """Shared state for the dashboard."""
    total_configs: int = 0
    completed: int = 0
    current_config: str = ""
    current_status: str = ""
    avg_tps: float = 0.0
    peak_tps: float = 0.0
    gpu_util: float = 0.0
    gpu_power: float = 0.0
    gpu_temp: float = 0.0
    corruption_events: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)


class BenchLabDashboardWrapper:
    """Console-based dashboard for environments without Textual."""

    def __init__(self, state: DashboardState | None = None) -> None:
        self.state = state or DashboardState()
        self._last_print = 0.0

    def update_result(self, result: dict[str, Any]) -> None:
        """Update display with a new result."""
        self.state.results.append(result)
        self.state.completed += 1
        if result.get("status") == "success":
            self.state.avg_tps = result.get("metrics", {}).get("avg_tps", 0)
            self.state.peak_tps = result.get("metrics", {}).get("peak_tps", 0)
        if result.get("gpu_metrics"):
            gpu = result["gpu_metrics"]
            self.state.gpu_util = gpu.get("eu_utilization", {}).get("avg", 0)
            self.state.gpu_power = gpu.get("power", {}).get("avg", 0)
            self.state.gpu_temp = gpu.get("temperature", {}).get("avg", 0)
        if result.get("corruption_events"):
            self.state.corruption_events.extend(result["corruption_events"])
        self._print_status()

    def _print_status(self) -> None:
        """Print status to console (throttled)."""
        now = time.time()
        if now - self._last_print < 0.5:
            return
        self._last_print = now

        progress = f"{self.state.completed}/{self.state.total_configs}"
        print(f"\r[BenchLab] Config: {self.state.current_config} | "
              f"Status: {self.state.current_status} | "
              f"Progress: {progress} | "
              f"TPS: {self.state.avg_tps:.1f} | "
              f"GPU: {self.state.gpu_util:.0f}%",
              end="", flush=True)

    def print_header(self) -> None:
        """Print dashboard header."""
        print("\n" + "=" * 70)
        print("  B70 Benchmark Lab — Live Dashboard")
        print("=" * 70)

    def print_summary(self) -> None:
        """Print final summary."""
        print()
        print("=" * 70)
        print("  Benchmark Complete")
        print("=" * 70)
        print(f"  Total runs:     {len(self.state.results)}")
        success = sum(1 for r in self.state.results if r.get("status") == "success")
        failed = sum(1 for r in self.state.results if r.get("status") == "failed")
        corrupted = sum(1 for r in self.state.results if r.get("status") == "corrupted")
        crashed = sum(1 for r in self.state.results if r.get("status") == "crashed")
        print(f"  Successful:     {success}")
        print(f"  Failed:         {failed}")
        print(f"  Corrupted:      {corrupted}")
        print(f"  Crashed:        {crashed}")
        if self.state.corruption_events:
            print(f"  Corruption:     {len(self.state.corruption_events)} events")
        print("=" * 70)

    def update_live(self, avg_tps: float, gpu_util: float, config_name: str, status: str) -> None:
        """Update live values without a full result."""
        self.state.avg_tps = avg_tps
        self.state.gpu_util = gpu_util
        self.state.current_config = config_name
        self.state.current_status = status
        self._print_status()
