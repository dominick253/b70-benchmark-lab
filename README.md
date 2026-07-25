# B70 Benchmark Lab

**Enterprise-Grade GPU Benchmarking Suite for Intel Arc Pro B70**

Automatically discover the fastest and most stable vLLM configuration for Intel Arc Pro B70 GPUs.

## Features

- **Automated Configuration Matrix** — Generates and tests all GPU memory, block size, dtype, KV cache, prefix caching, tensor parallel, and graph mode combinations
- **Docker Automation** — Pulls images, creates containers with GPU support, waits for readiness, collects logs, and cleans up automatically
- **Comprehensive Workloads** — Reasoning, coding, creative writing, math, RAG, long context, conversation, tool calling, streaming, JSON generation
- **Real-time Telemetry** — Intel GPU telemetry (EU utilization, memory bandwidth, power, temperature), system metrics, Docker stats, vLLM Prometheus metrics
- **Corruption Detection** — Automatically detects repeated tokens, Unicode corruption, invalid UTF-8, broken JSON, NaN values, infinite loops, runaway generation
- **Scoring Engine** — Weighted scoring across performance, stability, and efficiency categories
- **Interactive HTML Dashboard** — Plotly charts, configuration comparison, GPU telemetry graphs, corruption viewer
- **Multiple Report Formats** — JSON, CSV, Markdown, interactive HTML
- **CLI** — Full command-line interface for running, listing, comparing, exporting, and managing benchmarks

## Quick Start

```bash
# Install
pip install -e .

# Run system health check
benchlab doctor

# Run benchmarks with default configuration
benchlab run

# Run with specific model
benchlab run --model meta-llama/Llama-3.1-8B

# Run with custom configuration file
benchlab run --config my-config.yaml

# Dry run to see what would be tested
benchlab run --dry-run

# List previous runs
benchlab list

# Compare configurations
benchlab compare abc123 def456

# Launch interactive dashboard
benchlab dashboard

# Export results
benchlab export --format json --output results.json
```

## Configuration

Create a `config.yaml`:

```yaml
model:
  name: meta-llama/Llama-3.1-8B
  path: /path/to/model
  dtype: auto
  gpu_memory_utilization: 0.90
  block_size: 16
  tensor_parallel: 1
  max_model_len: 8192
  quantization: null
  enable_prefix_caching: true
  graph_mode: piecewise
  enable_chunked_prefill: true

docker:
  image: vllm/vllm-openai
  image_tag: latest
  gpus: 1
  cpus: 8
  memory: 32G
  shm_size: 8G
  port: 8000

workload:
  concurrency: 4
  num_prompts: 10
  timeout: 60

scoring:
  peak_tps_weight: 0.15
  average_tps_weight: 0.15
  p95_tps_weight: 0.10
  min_tps_weight: 0.05
  ttft_weight: 0.10
  latency_weight: 0.10
  gpu_util_weight: 0.05
  power_efficiency_weight: 0.05
  vram_efficiency_weight: 0.05
  stability_weight: 0.10
  long_context_stability_weight: 0.05
  scheduler_efficiency_weight: 0.05
  graph_stability_weight: 0.05

corruption:
  enabled: true
  max_repeated_tokens: 10
  max_repeated_chars: 20
  max_repeated_lines: 5
  max_generation_tokens: 8192
```

## Project Structure

```
benchlab/
    __init__.py          # Package entry point
    cli.py               # CLI interface (Typer)
    config.py            # Configuration management (Pydantic)
    benchmark.py         # Core benchmark orchestration
    workload.py          # Benchmark workload execution
    scheduler.py         # Configuration matrix generation
    scoring.py           # Scoring engine
    docker_manager.py    # Docker container lifecycle
    metrics.py           # Metrics collection and aggregation
    telemetry.py         # System and GPU telemetry
    vllm_metrics.py      # vLLM Prometheus metrics scraping
    intel_gpu.py         # Intel GPU telemetry
    docker_stats.py      # Docker container statistics
    corruption.py        # Corruption detection
    prompts.py           # Prompt generation
    report.py            # Report generation (CSV, JSON, Markdown)
    html_report.py       # Interactive HTML dashboard
    storage.py           # SQLite storage layer
    logging.py           # Structured logging
    utils.py             # Utility functions
    config_generator.py  # Configuration matrix generation
    validators.py        # Input validation
    runner.py            # Full benchmark runner
configs/               # Generated configuration files
prompts/               # Prompt templates
reports/               # Generated reports
results/               # Benchmark results
logs/                  # Log files
templates/             # HTML templates
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI (Typer)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    Runner (Orchestrator)                     │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Config    │  │  Docker    │  │   Telemetry        │    │
│  │  Generator │  │  Manager   │  │   Collector        │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  Workload  │  │  Corruption│  │    Scoring         │    │
│  │  Executor  │  │  Detector  │  │    Engine          │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │  vLLM      │  │  Intel     │  │   Storage          │    │
│  │  Metrics   │  │  GPU       │  │   Manager          │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Report Generation                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐    │
│  │   JSON     │  │   CSV      │  │   HTML Dashboard   │    │
│  │  Report    │  │  Report    │  │   (Plotly)         │    │
│  └────────────┘  └────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.12+
- Intel Arc Pro B70 GPU(s)
- Docker with GPU support
- vLLM Docker image
- `intel_gpu_top` (Intel GPU tools)

## License

MIT
