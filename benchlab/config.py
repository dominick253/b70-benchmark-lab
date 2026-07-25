"""
Configuration management for BenchLab.

Pydantic-based configuration with validation, defaults, and
auto-generation of configuration matrices.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Dtype(str, Enum):
    AUTO = "auto"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"


class KVCacheDtype(str, Enum):
    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class GraphMode(str, Enum):
    PIECEWISE = "piecewise"
    FULL = "full"
    EAGER = "eager"
    DISABLED = "disabled"


class PrefixCaching(str, Enum):
    ON = "on"
    OFF = "off"


class WorkloadType(str, Enum):
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVE_WRITING = "creative_writing"
    MATH = "math"
    RAG = "rag"
    LONG_CONTEXT = "long_context"
    CONVERSATION = "conversation"
    TOOL_CALLING = "tool_calling"
    STREAMING = "streaming"
    JSON_GENERATION = "json_generation"


class WorkloadLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    CONTEXT_128K = "128k"
    CONTEXT_200K = "200k"


class ConcurrencyLevel(str, Enum):
    SINGLE = "single"
    TWO = "two"
    FOUR = "four"
    EIGHT = "eight"
    SIXTEEN = "sixteen"
    THIRTY_TWO = "thirty_two"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Model to benchmark."""
    name: str = Field(description="HuggingFace model identifier", default="meta-llama/Llama-3.1-8B-Instruct")
    path: Optional[str] = Field(default=None, description="Local model path if available")
    tensor_parallel: int = Field(default=1, ge=1, le=8, description="Tensor parallelism degree")
    dtype: Dtype = Field(default=Dtype.AUTO, description="Model dtype")
    kv_cache_dtype: KVCacheDtype = Field(default=KVCacheDtype.AUTO, description="KV cache dtype")
    trust_remote_code: bool = Field(default=False, description="Trust remote code")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.5, le=0.99, description="GPU memory utilization target")
    max_model_len: Optional[int] = Field(default=None, description="Maximum model context length")
    max_num_batched_tokens: Optional[int] = Field(default=None, description="Max batched tokens")
    max_num_seqs: int = Field(default=256, ge=1, description="Maximum number of sequences")
    block_size: int = Field(default=16, ge=8, le=256, description="KV cache block size")
    enable_prefix_caching: PrefixCaching = Field(default=PrefixCaching.ON, description="Prefix caching")
    graph_mode: GraphMode = Field(default=GraphMode.PIECEWISE, description="CUDA graph mode")
    enforce_eager: bool = Field(default=False, description="Use eager mode (no CUDA graphs)")
    enable_chunked_prefill: bool = Field(default=True, description="Chunked prefill")
    max_chunked_prefill_tokens: int = Field(default=8192, ge=1, description="Max chunked prefill tokens")
    swap_space: int = Field(default=4, ge=0, description="Swap space in GiB")
    max_logprobs: int = Field(default=5, ge=0, le=100, description="Max logprobs per token")
    disable_log_stats: bool = Field(default=False, description="Disable logging stats in vLLM")
    disable_log_requests: bool = Field(default=False, description="Disable request logging")
    quantization: Optional[str] = Field(default=None, description="Quantization method (awq, gptq, etc.)")
    load_format: str = Field(default="auto", description="Model load format")
    num_scheduler_steps: int = Field(default=16, ge=1, le=128, description="Number of scheduler steps")
    guided_decoding_backend: str = Field(default="outlines", description="Guided decoding backend")
    distributed_executor_backend: Optional[str] = Field(default=None, description="Distributed executor backend")
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(default=None, description="Multimodal processor kwargs")
    rope_scaling: Optional[dict[str, Any]] = Field(default=None, description="RoPE scaling config")
    num_attention_heads: Optional[int] = Field(default=None, description="Number of attention heads")
    num_hidden_layers: Optional[int] = Field(default=None, description="Number of hidden layers")
    num_key_value_heads: Optional[int] = Field(default=None, description="Number of KV heads")
    hidden_size: Optional[int] = Field(default=None, description="Hidden dimension size")
    head_size: Optional[int] = Field(default=None, description="Attention head size")
    intermediate_size: Optional[int] = Field(default=None, description="Intermediate FFN size")
    vocab_size: Optional[int] = Field(default=None, description="Vocabulary size")


class DockerConfig(BaseModel):
    """Docker runtime configuration."""
    image: str = Field(default="intel/llm-scaler-vllm", description="Docker image")
    image_tag: str = Field(default="0.21.0-b1", description="Docker image tag")
    runtime: str = Field(default="docker", description="Container runtime")
    gpus: int = Field(default=1, ge=1, le=8, description="Number of GPUs")
    memory: str = Field(default="0", description="Memory limit (0 = unlimited)")
    memory_swap: str = Field(default="-1", description="Memory swap limit")
    cpus: Optional[str] = Field(default=None, description="CPU limit")
    shm_size: str = Field(default="64g", description="Shared memory size")
    network: str = Field(default="host", description="Network mode")
    restart_policy: str = Field(default="no", description="Restart policy")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    volumes: list[str] = Field(default_factory=list, description="Volume mounts")
    health_check_interval: int = Field(default=5, ge=1, description="Health check interval (seconds)")
    health_check_retries: int = Field(default=10, ge=1, description="Health check max retries")
    container_name_prefix: str = Field(default="benchlab", description="Container name prefix")
    pull_image: bool = Field(default=True, description="Pull image before use")
    remove_container: bool = Field(default=True, description="Remove container after benchmark")
    remove_image: bool = Field(default=False, description="Remove image after benchmark")
    port: int = Field(default=8000, description="API server port")


class TelemetryConfig(BaseModel):
    """Telemetry collection configuration."""
    collection_interval: float = Field(default=1.0, ge=0.1, description="Collection interval (seconds)")
    collect_gpu: bool = Field(default=True, description="Collect Intel GPU telemetry")
    collect_system: bool = Field(default=True, description="Collect system telemetry")
    collect_docker: bool = Field(default=True, description="Collect Docker metrics")
    collect_vllm: bool = Field(default=True, description="Collect vLLM metrics")
    collect_network: bool = Field(default=True, description="Collect network metrics")
    collect_cpu: bool = Field(default=True, description="Collect CPU metrics")
    collect_memory: bool = Field(default=True, description="Collect memory metrics")
    collect_disk: bool = Field(default=True, description="Collect disk metrics")
    collect_numa: bool = Field(default=True, description="Collect NUMA metrics")
    collect_swap: bool = Field(default=True, description="Collect swap metrics")
    collect_page_faults: bool = Field(default=True, description="Collect page fault metrics")
    collect_context_switches: bool = Field(default=True, description="Collect context switch metrics")
    intel_gpu_json_mode: bool = Field(default=True, description="Use intel_gpu_top JSON mode")
    prometheus_endpoint: Optional[str] = Field(default=None, description="Prometheus metrics endpoint")
    prometheus_interval: float = Field(default=15.0, ge=1.0, description="Prometheus scrape interval")
    max_retention_hours: int = Field(default=720, ge=1, description="Max data retention (hours)")


class CorruptionConfig(BaseModel):
    """Corruption detection configuration."""
    enabled: bool = Field(default=True, description="Enable corruption detection")
    max_repeated_tokens: int = Field(default=10, ge=1, description="Max consecutive repeated tokens")
    max_repeated_chars: int = Field(default=50, ge=1, description="Max consecutive repeated chars")
    max_repeated_lines: int = Field(default=5, ge=1, description="Max consecutive repeated lines")
    detect_unicode_corruption: bool = Field(default=True, description="Detect Unicode corruption")
    detect_invalid_utf: bool = Field(default=True, description="Detect invalid UTF-8")
    detect_broken_json: bool = Field(default=True, description="Detect broken JSON")
    detect_nan: bool = Field(default=True, description="Detect NaN values")
    detect_empty_generation: bool = Field(default=True, description="Detect empty generations")
    detect_partial_generation: bool = Field(default=True, description="Detect partial generations")
    detect_eos_failure: bool = Field(default=True, description="Detect EOS failures")
    detect_infinite_loop: bool = Field(default=True, description="Detect infinite generation loops")
    max_generation_tokens: int = Field(default=8192, ge=1, description="Max generation tokens")
    detect_runaway_generation: bool = Field(default=True, description="Detect runaway generation")
    detect_hallucinated_binary: bool = Field(default=True, description="Detect hallucinated binary content")


class WorkloadConfig(BaseModel):
    """Benchmark workload configuration."""
    type: WorkloadType = Field(default=WorkloadType.CODING, description="Workload type")
    length: WorkloadLength = Field(default=WorkloadLength.MEDIUM, description="Workload length")
    num_prompts: int = Field(default=100, ge=1, description="Number of prompts to generate")
    prompt_template: Optional[str] = Field(default=None, description="Custom prompt template")
    min_output_tokens: int = Field(default=64, ge=1, description="Minimum output tokens")
    max_output_tokens: int = Field(default=2048, ge=1, description="Maximum output tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=50, ge=1, description="Top-k sampling")
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0, description="Presence penalty")
    stop_sequences: list[str] = Field(default_factory=list, description="Stop sequences")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    concurrency: int = Field(default=1, ge=1, description="Request concurrency")
    request_rate: Optional[float] = Field(default=None, description="Requests per second (None = unlimited)")
    timeout: int = Field(default=300, ge=1, description="Request timeout (seconds)")
    stream: bool = Field(default=False, description="Enable streaming")
    include_logprobs: bool = Field(default=False, description="Include log probabilities")
    max_tokens_per_request: int = Field(default=2048, ge=1, description="Max tokens per request")
    input_tokens_target: Optional[int] = Field(default=None, description="Target input token count")
    output_tokens_target: Optional[int] = Field(default=None, description="Target output token count")


class StabilityConfig(BaseModel):
    """Stability test configuration."""
    enabled: bool = Field(default=True, description="Enable stability tests")
    durations: list[str] = Field(
        default=["30m", "1h", "2h", "4h", "8h", "12h", "24h"],
        description="Test durations",
    )
    min_duration: str = Field(default="30m", description="Minimum test duration")
    max_duration: str = Field(default="24h", description="Maximum test duration")
    check_interval: float = Field(default=60.0, ge=1.0, description="Stability check interval (seconds)")
    measure_degradation: bool = Field(default=True, description="Measure performance degradation")
    measure_memory_leaks: bool = Field(default=True, description="Detect memory leaks")
    measure_graph_failures: bool = Field(default=True, description="Detect graph failures")
    measure_scheduler_collapse: bool = Field(default=True, description="Detect scheduler collapse")
    measure_gpu_drift: bool = Field(default=True, description="Detect GPU utilization drift")
    measure_kv_fragmentation: bool = Field(default=True, description="Detect KV cache fragmentation")
    measure_container_growth: bool = Field(default=True, description="Detect container memory growth")
    degradation_threshold: float = Field(default=0.15, ge=0.0, le=1.0, description="Degradation threshold")
    memory_leak_threshold: float = Field(default=0.10, ge=0.0, le=1.0, description="Memory leak threshold")
    gpu_drift_threshold: float = Field(default=0.20, ge=0.0, le=1.0, description="GPU utilization drift threshold")


class ThroughputConfig(BaseModel):
    """Throughput test configuration."""
    enabled: bool = Field(default=True, description="Enable throughput tests")
    concurrency_levels: list[int] = Field(default=[1, 2, 4, 8, 16, 32], description="Concurrency levels to test")
    min_concurrency: int = Field(default=1, ge=1, description="Minimum concurrency")
    max_concurrency: int = Field(default=32, ge=1, description="Maximum concurrency")
    step_concurrency: int = Field(default=1, ge=1, description="Concurrency step size")
    test_streaming: bool = Field(default=True, description="Test streaming throughput")
    test_burst: bool = Field(default=True, description="Test burst patterns")
    test_mixed: bool = Field(default=True, description="Test mixed workloads")
    test_random: bool = Field(default=True, description="Test random patterns")
    burst_multiplier: int = Field(default=5, ge=1, description="Burst multiplier")
    burst_duration: int = Field(default=10, ge=1, description="Burst duration (seconds)")
    mixed_workload_types: list[str] = Field(
        default=["coding", "reasoning", "creative_writing"],
        description="Mixed workload types",
    )


class LongContextConfig(BaseModel):
    """Long context test configuration."""
    enabled: bool = Field(default=True, description="Enable long context tests")
    context_lengths: list[int] = Field(
        default=[8192, 16384, 32768, 65536, 98304, 131072, 2097152],
        description="Context lengths to test",
    )
    min_length: int = Field(default=8192, ge=512, description="Minimum context length")
    max_length: int = Field(default=2097152, ge=512, description="Maximum context length")
    step_length: int = Field(default=8192, ge=512, description="Context length step")
    measure_tps: bool = Field(default=True, description="Measure tokens per second")
    measure_ttft: bool = Field(default=True, description="Measure time to first token")
    measure_latency: bool = Field(default=True, description="Measure end-to-end latency")
    measure_memory: bool = Field(default=True, description="Measure memory usage")
    measure_gpu_util: bool = Field(default=True, description="Measure GPU utilization")


class ScoringConfig(BaseModel):
    """Scoring configuration."""
    peak_tps_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="Peak TPS weight")
    average_tps_weight: float = Field(default=0.20, ge=0.0, le=1.0, description="Average TPS weight")
    p95_tps_weight: float = Field(default=0.15, ge=0.0, le=1.0, description="P95 TPS weight")
    min_tps_weight: float = Field(default=0.10, ge=0.0, le=1.0, description="Minimum TPS weight")
    ttft_weight: float = Field(default=0.10, ge=0.0, le=1.0, description="TTFT weight")
    latency_weight: float = Field(default=0.10, ge=0.0, le=1.0, description="Latency weight")
    corruption_penalty: float = Field(default=0.30, ge=0.0, le=1.0, description="Corruption penalty")
    oom_penalty: float = Field(default=1.0, ge=0.0, le=1.0, description="OOM penalty")
    crash_penalty: float = Field(default=1.0, ge=0.0, le=1.0, description="Crash penalty")
    gpu_util_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="GPU utilization weight")
    power_efficiency_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="Power efficiency weight")
    vram_efficiency_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="VRAM efficiency weight")
    stability_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="Stability weight")
    long_context_stability_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="Long-context stability weight")
    memory_growth_penalty: float = Field(default=0.10, ge=0.0, le=1.0, description="Memory growth penalty")
    scheduler_efficiency_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="Scheduler efficiency weight")
    graph_stability_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="Graph stability weight")


class ReportConfig(BaseModel):
    """Report generation configuration."""
    formats: list[str] = Field(default=["html", "json", "csv", "markdown", "sqlite"], description="Report formats")
    include_graphs: bool = Field(default=True, description="Include graphs")
    include_raw_data: bool = Field(default=True, description="Include raw data")
    include_logs: bool = Field(default=True, description="Include logs")
    include_config_matrix: bool = Field(default=True, description="Include configuration matrix")
    include_winner: bool = Field(default=True, description="Include winner selection")
    include_trend_analysis: bool = Field(default=True, description="Include trend analysis")
    include_corruption_viewer: bool = Field(default=True, description="Include corruption viewer")
    include_docker_logs: bool = Field(default=True, description="Include Docker logs")
    dark_mode: bool = Field(default=True, description="Use dark mode")
    output_dir: str = Field(default="reports", description="Output directory")


class BenchLabConfig(BaseModel):
    """Top-level BenchLab configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    corruption: CorruptionConfig = Field(default_factory=CorruptionConfig)
    workload: WorkloadConfig = Field(default_factory=WorkloadConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    throughput: ThroughputConfig = Field(default_factory=ThroughputConfig)
    long_context: LongContextConfig = Field(default_factory=LongContextConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # Run-level settings
    run_name: str = Field(default="", description="Run name (auto-generated if empty)")
    output_dir: str = Field(default="results", description="Output directory for results")
    log_dir: str = Field(default="logs", description="Output directory for logs")
    configs_dir: str = Field(default="configs", description="Directory for config files")
    prompts_dir: str = Field(default="prompts", description="Directory for prompt templates")
    report_dir: str = Field(default="reports", description="Directory for reports")
    dashboards_dir: str = Field(default="dashboards", description="Directory for dashboards")
    templates_dir: str = Field(default="templates", description="Directory for templates")

    # Execution settings
    parallel_runs: int = Field(default=1, ge=1, le=8, description="Number of parallel benchmarks")
    max_retries: int = Field(default=3, ge=0, le=10, description="Max retries per benchmark")
    retry_delay: float = Field(default=5.0, ge=0.0, description="Retry delay (seconds)")
    timeout: int = Field(default=7200, ge=60, description="Overall timeout (seconds)")
    dry_run: bool = Field(default=False, description="Dry run mode (no actual benchmarks)")
    resume: bool = Field(default=False, description="Resume previous run")
    skip_completed: bool = Field(default=True, description="Skip completed configurations")
    stop_on_corruption: bool = Field(default=False, description="Stop on corruption detection")
    stop_on_crash: bool = Field(default=True, description="Stop on container crash")

    # Paths
    db_path: str = Field(default="results/benchlab.db", description="SQLite database path")
    config_path: Optional[str] = Field(default=None, description="Path to config file")
    max_configs: int = Field(default=50, ge=1, le=1000, description="Max configurations to generate")
    storage_path: str = Field(default="results/benchlab.db", description="Storage path alias")
    results_dir: str = Field(default="results", description="Results directory")
    reports_dir: str = Field(default="reports", description="Reports directory")
    telemetry_collection_interval: float = Field(default=1.0, ge=0.1, description="Telemetry interval alias")

    @model_validator(mode="before")
    @classmethod
    def _infer_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("run_name"):
                data["run_name"] = f"run-{uuid.uuid4().hex[:8]}"
        return data

    def compute_config_hash(self, config: dict[str, Any]) -> str:
        """Compute a deterministic hash for a configuration."""
        content = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(exclude_none=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "BenchLabConfig":
        """Load configuration from YAML."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def default(cls) -> "BenchLabConfig":
        """Create a default configuration."""
        return cls()
