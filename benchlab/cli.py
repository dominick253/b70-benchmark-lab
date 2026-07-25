"""
CLI entry point for B70 Benchmark Lab.

Commands:
    run        Run the full benchmark suite
    resume     Resume a previous interrupted run
    list       List all runs and configurations
    compare    Compare configurations
    report     Generate reports
    dashboard  Launch interactive dashboard
    clean      Clean up old runs and artifacts
    export     Export results
    doctor     System health check
    validate   Validate configuration
    replay     Replay a previous run

Usage:
    benchlab run
    benchlab run --model meta-llama/Llama-3.1-8B
    benchlab run --config my-config.yaml
    benchlab list
    benchlab compare --configs abc123 def456
    benchlab dashboard
    benchlab doctor
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from benchlab.config import BenchLabConfig, ModelConfig, DockerConfig, ScoringConfig
from benchlab.logging import get_logger
from benchlab.runner import BenchmarkRunner
from benchlab.validators import SystemValidator

console = Console()
app = typer.Typer(
    name="benchlab",
    help="B70 Benchmark Lab — Enterprise GPU Benchmarking Suite",
    add_completion=False,
)


def _print_header() -> None:
    console.print(Panel(
        "[bold cyan]B70 Benchmark Lab[/bold cyan]\n"
        "[dim]Enterprise GPU Benchmarking Suite for Intel Arc Pro B70[/dim]",
        border_style="cyan",
    ))


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        app.invoke_callback("run")


@app.command()
def run(
    model: str = typer.Option(
        None, "--model", "-m",
        help="Model name or path (e.g., meta-llama/Llama-3.1-8B)",
    ),
    model_path: str = typer.Option(
        None, "--model-path",
        help="Local path to model files",
    ),
    config: str = typer.Option(
        None, "--config", "-c",
        help="Path to configuration YAML file",
    ),
    output: str = typer.Option(
        "results", "--output", "-o",
        help="Output directory for results",
    ),
    max_configs: int = typer.Option(
        50, "--max-configs",
        help="Maximum number of configurations to test",
    ),
    concurrency: int = typer.Option(
        4, "--concurrency", "-j",
        help="Number of concurrent benchmark requests",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Generate configs and show what would run, without executing",
    ),
    resume: str = typer.Option(
        None, "--resume",
        help="Resume from a previous run ID",
    ),
) -> None:
    """Run the full benchmark suite."""
    _print_header()
    get_logger()

    # Build config
    if config:
        cfg = BenchLabConfig.load(config)
    else:
        default_workload = cfg.workload if (cfg := BenchLabConfig()) else None
        cfg = BenchLabConfig(
            model=ModelConfig(
                name=model or "meta-llama/Llama-3.1-8B",
                path=model_path,
            ),
            docker=DockerConfig(),
            max_configs=max_configs,
            workload=default_workload,
        )

    if model:
        cfg.model.name = model
    if model_path:
        cfg.model.path = model_path

    console.print(f"[dim]Model:[/dim] {cfg.model.name}")
    console.print(f"[dim]Docker image:[/dim] {cfg.docker.image}:{cfg.docker.image_tag}")
    console.print(f"[dim]Max configs:[/dim] {cfg.max_configs}")
    console.print(f"[dim]Concurrency:[/dim] {concurrency}")
    console.print()

    if dry_run:
        console.print("[yellow]DRY RUN — generating configurations only[/yellow]")
        from benchlab.config_generator import ConfigGenerator
        gen = ConfigGenerator(
            model_config=cfg.model,
            docker_config=cfg.docker,
            max_configs=cfg.max_configs,
        )
        configs = gen.generate_all()
        console.print(f"[green]Generated {len(configs)} configurations[/green]")
        for c in configs[:10]:
            console.print(f"  [dim]{c.config_hash}[/dim] {c.name}")
        if len(configs) > 10:
            console.print(f"  [dim]... and {len(configs) - 10} more[/dim]")
        return

    # Run benchmarks
    runner = BenchmarkRunner(cfg)
    runner.config.workload.concurrency = concurrency

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Running benchmarks...", total=None)

        try:
            results = asyncio.run(runner.run_all())
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            asyncio.run(runner.stop())
            return
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            asyncio.run(runner.stop())
            return

    # Print summary
    _print_results_summary(results)


def _print_results_summary(results: list[Any]) -> None:
    """Print a summary of benchmark results."""
    console.print()
    console.print(Panel("[bold green]Benchmark Complete[/bold green]", border_style="green"))

    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    corrupted = sum(1 for r in results if r.status == "corrupted")
    crashed = sum(1 for r in results if r.status == "crashed")

    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="white")
    table.add_row("Total Runs", str(total))
    table.add_row("Successful", str(success), style="green")
    table.add_row("Failed", str(failed), style="red")
    table.add_row("Corrupted", str(corrupted), style="yellow")
    table.add_row("Crashed", str(crashed), style="red")
    console.print(table)

    # Best config
    successful = [r for r in results if r.status == "success"]
    if successful:
        best = max(successful, key=lambda r: r.metrics.get("avg_tps", 0))
        console.print()
        console.print(Panel(
            f"[bold]Best Configuration:[/bold] {best.config_hash}\n"
            f"[dim]Avg TPS:[/dim] {best.metrics.get('avg_tps', 0):.1f}\n"
            f"[dim]Peak TPS:[/dim] {best.metrics.get('peak_tps', 0):.1f}\n"
            f"[dim]TTFT:[/dim] {best.metrics.get('avg_ttft', 0):.3f}s",
            border_style="green",
        ))

    console.print()
    console.print(f"[dim]Results saved to: {Path('results')}[/dim]")


@app.command()
def list_runs(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of runs to show"),
) -> None:
    """List all benchmark runs."""
    from benchlab.storage import Storage
    from benchlab.config import BenchLabConfig

    cfg = BenchLabConfig()
    storage = Storage(cfg.storage_path)

    runs = storage.get_runs(limit=limit)
    if not runs:
        console.print("[yellow]No runs found.[/yellow]")
        return

    table = Table(title="Benchmark Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Config", style="dim")
    table.add_column("Status", style="white")
    table.add_column("Duration", style="white")
    table.add_column("Avg TPS", style="white")
    table.add_column("Date", style="dim")

    for run in runs:
        table.add_row(
            run["run_id"][:12],
            run.get("config_hash", "")[:8],
            run.get("status", ""),
            f"{run.get('duration', 0):.0f}s",
            f"{run.get('metrics', {}).get('avg_tps', 0):.1f}",
            run.get("start_time", ""),
        )

    console.print(table)


@app.command()
def compare(
    configs: list[str] = typer.Argument(..., help="Configuration hashes to compare"),
) -> None:
    """Compare specific configurations."""
    from benchlab.storage import Storage
    from benchlab.config import BenchLabConfig

    cfg = BenchLabConfig()
    storage = Storage(cfg.storage_path)

    results = []
    for config_hash in configs:
        runs = storage.get_runs_by_config(config_hash)
        if runs:
            results.append((config_hash, runs[0]))

    if not results:
        console.print("[yellow]No results found for specified configs.[/yellow]")
        return

    table = Table(title="Configuration Comparison")
    table.add_column("Config", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Avg TPS", style="white")
    table.add_column("Peak TPS", style="white")
    table.add_column("TTFT", style="white")
    table.add_column("GPU %", style="white")
    table.add_column("Score", style="white")

    for config_hash, run in results:
        metrics = run.get("metrics", {})
        gpu = run.get("gpu_metrics", {})
        table.add_row(
            config_hash,
            run.get("status", ""),
            f"{metrics.get('avg_tps', 0):.1f}",
            f"{metrics.get('peak_tps', 0):.1f}",
            f"{metrics.get('avg_ttft', 0):.3f}s",
            f"{gpu.get('eu_utilization', {}).get('avg', 0):.0f}%",
            f"{run.get('score', 0):.4f}",
        )

    console.print(table)


@app.command()
def dashboard() -> None:
    """Launch the interactive HTML dashboard."""
    from benchlab.config import BenchLabConfig

    cfg = BenchLabConfig()
    html_path = Path(cfg.reports_dir) / "html" / "dashboard.html"

    if not html_path.exists():
        console.print("[yellow]No dashboard found. Run benchmarks first.[/yellow]")
        return

    console.print(f"[cyan]Opening dashboard:[/cyan] {html_path}")
    import webbrowser
    webbrowser.open(f"file://{html_path.absolute()}")


@app.command()
def doctor() -> None:
    """Run system health checks."""
    _print_header()
    console.print("[bold]System Health Check[/bold]\n")

    checks = SystemValidator.run_full_check()
    for name, result in checks["checks"].items():
        status = "[green]PASS[/green]" if result["passed"] else "[red]FAIL[/red]"
        console.print(f"  {status}  {name}: {result['message']}")

    if not checks["all_passed"]:
        console.print("\n[yellow]Some checks failed. Review the output above.[/yellow]")
    else:
        console.print("\n[green]All checks passed![/green]")


@app.command()
def validate(
    config: str = typer.Option(..., "--config", "-c", help="Configuration file to validate"),
) -> None:
    """Validate a configuration file."""
    from benchlab.config import BenchLabConfig
    from benchlab.validators import ConfigValidator

    cfg = BenchLabConfig.load(config)
    errors = ConfigValidator.validate_config_combination(cfg.model, cfg.docker)

    if errors:
        console.print("[red]Validation errors:[/red]")
        for error in errors:
            console.print(f"  [red]• {error}[/red]")
    else:
        console.print("[green]Configuration is valid.[/green]")


@app.command()
def clean(
    older_than_days: int = typer.Option(30, "--days", "-d", help="Remove runs older than N days"),
) -> None:
    """Clean up old runs and artifacts."""
    from benchlab.storage import Storage
    from benchlab.config import BenchLabConfig
    import time

    cfg = BenchLabConfig()
    storage = Storage(cfg.storage_path)

    cutoff = time.time() - (older_than_days * 86400)
    removed = storage.cleanup_old_runs(cutoff)
    console.print(f"[green]Removed {removed} runs older than {older_than_days} days.[/green]")


@app.command()
def export(
    format: str = typer.Option("json", "--format", "-f", help="Export format: json, csv, markdown"),
    output: str = typer.Option("results/export.json", "--output", "-o", help="Output file path"),
) -> None:
    """Export results."""
    from benchlab.config import BenchLabConfig
    from benchlab.storage import Storage
    from benchlab.report import ReportGenerator

    cfg = BenchLabConfig()
    storage = Storage(cfg.storage_path)
    report_gen = ReportGenerator(cfg)

    results = storage.get_runs()
    if not results:
        console.print("[yellow]No results to export.[/yellow]")
        return

    if format == "json":
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "csv":
        path = report_gen.generate_csv(results)
        console.print(f"[green]CSV exported to: {path}[/green]")
        return
    elif format == "markdown":
        path = report_gen.generate_markdown(results, [])
        console.print(f"[green]Markdown exported to: {path}[/green]")
        return

    console.print(f"[green]Exported {len(results)} runs to: {output}[/green]")


@app.command()
def replay(
    run_id: str = typer.Option(..., "--run", "-r", help="Run ID to replay"),
) -> None:
    """Replay a previous run with the same configuration."""
    from benchlab.config import BenchLabConfig
    from benchlab.storage import Storage

    cfg = BenchLabConfig()
    storage = Storage(cfg.storage_path)

    run = storage.get_run(run_id)
    if not run:
        console.print(f"[red]Run {run_id} not found.[/red]")
        return

    console.print(f"[cyan]Replaying run {run_id}...[/cyan]")
    # Would need to reload config from stored run data
    console.print("[yellow]Replay not yet implemented.[/yellow]")


if __name__ == "__main__":
    app()
