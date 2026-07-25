"""
Benchmark workload execution for BenchLab.

Handles sending requests to vLLM OpenAI-compatible API,
collecting per-request metrics, and managing concurrency.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

import aiohttp

from benchlab.config import WorkloadConfig, WorkloadLength, WorkloadType
from benchlab.logging import get_logger
from benchlab.prompts import Prompt, PromptGenerator, PromptSet
from benchlab.utils import generate_id

logger = get_logger("workload")


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    prompt: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    ttft: float
    decode_time: float
    total_time: float
    tps: float
    total_tps: float
    first_token_time: float
    last_token_time: float
    streaming: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadResult:
    """Result of a workload execution."""
    workload_type: str
    workload_length: str
    num_requests: int
    successful_requests: int
    failed_requests: int
    total_prompts_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_ttft: float
    p95_ttft: float
    p99_ttft: float
    avg_decode_time: float
    avg_total_time: float
    avg_tps: float
    p95_tps: float
    p99_tps: float
    avg_total_tps: float
    total_time: float
    requests: list[RequestMetrics] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.num_requests == 0:
            return 0.0
        return self.successful_requests / self.num_requests

    @property
    def total_tps(self) -> float:
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time


class OpenAIClient:
    """OpenAI-compatible API client for vLLM."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "none") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=300)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> bool:
        """Check if the vLLM server is ready."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return resp.status == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def wait_for_ready(self, timeout: int = 120, interval: float = 2.0) -> bool:
        """Wait for the vLLM server to be ready."""
        start = time.time()
        while time.time() - start < timeout:
            if await self.health_check():
                logger.info("vLLM server is ready")
                return True
            await asyncio.sleep(interval)
        logger.error(f"vLLM server not ready after {timeout}s")
        return False

    async def create_chat_completion(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs: Any,
    ) -> tuple[str | AsyncIterator[str], RequestMetrics]:
        """Send a chat completion request."""
        request_id = generate_id()
        start_time = time.time()

        session = await self._get_session()
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs,
        }

        try:
            if stream:
                return await self._stream_request(
                    session, url, headers, payload, request_id, start_time
                )
            else:
                return await self._non_stream_request(
                    session, url, headers, payload, request_id, start_time
                )
        except Exception as e:
            elapsed = time.time() - start_time
            metrics = RequestMetrics(
                request_id=request_id,
                prompt=prompt,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                ttft=elapsed,
                decode_time=0,
                total_time=elapsed,
                tps=0,
                total_tps=0,
                first_token_time=0,
                last_token_time=elapsed,
                error=str(e),
            )
            return "", metrics

    async def _non_stream_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        request_id: str,
        start_time: float,
    ) -> tuple[str, RequestMetrics]:
        """Non-streaming request."""
        async with session.post(url, headers=headers, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                elapsed = time.time() - start_time
                metrics = RequestMetrics(
                    request_id=request_id,
                    prompt=payload["messages"][0]["content"],
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    ttft=elapsed,
                    decode_time=0,
                    total_time=elapsed,
                    tps=0,
                    total_tps=0,
                    first_token_time=0,
                    last_token_time=elapsed,
                    error=f"HTTP {resp.status}: {error_text}",
                )
                return "", metrics

            data = await resp.json()
            elapsed = time.time() - start_time

            choice = data["choices"][0]
            completion = choice["message"]["content"]
            usage = data.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            metrics = RequestMetrics(
                request_id=request_id,
                prompt=payload["messages"][0]["content"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                ttft=elapsed,
                decode_time=elapsed,
                total_time=elapsed,
                tps=completion_tokens / elapsed if elapsed > 0 else 0,
                total_tps=total_tokens / elapsed if elapsed > 0 else 0,
                first_token_time=0,
                last_token_time=elapsed,
            )
            return completion, metrics

    async def _stream_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        request_id: str,
        start_time: float,
    ) -> tuple[AsyncIterator[str], RequestMetrics]:
        """Streaming request - returns an async iterator and metrics."""
        prompt = payload["messages"][0]["content"]
        full_response = ""
        token_count = 0
        first_token_received = False
        first_token_time = 0.0
        metrics_ref: list[RequestMetrics] = []

        async def _stream() -> AsyncIterator[str]:
            nonlocal full_response, token_count, first_token_received, first_token_time
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        elapsed = time.time() - start_time
                        metrics_ref.append(RequestMetrics(
                            request_id=request_id,
                            prompt=prompt,
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            ttft=elapsed,
                            decode_time=0,
                            total_time=elapsed,
                            tps=0,
                            total_tps=0,
                            first_token_time=0,
                            last_token_time=elapsed,
                            error=f"HTTP {resp.status}: {error_text}",
                            streaming=True,
                        ))
                        return

                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")

                            if content:
                                if not first_token_received:
                                    first_token_time = time.time()
                                    first_token_received = True

                                full_response += content
                                token_count += 1
                                yield content
                        except json.JSONDecodeError:
                            continue
            finally:
                elapsed = time.time() - start_time
                ttft = first_token_time - start_time if first_token_received else elapsed
                metrics_ref.append(RequestMetrics(
                    request_id=request_id,
                    prompt=prompt,
                    prompt_tokens=0,
                    completion_tokens=token_count,
                    total_tokens=token_count,
                    ttft=ttft,
                    decode_time=elapsed - ttft if first_token_received else elapsed,
                    total_time=elapsed,
                    tps=token_count / (elapsed - ttft) if first_token_received and (elapsed - ttft) > 0 else 0,
                    total_tps=token_count / elapsed if elapsed > 0 else 0,
                    first_token_time=ttft,
                    last_token_time=elapsed,
                    streaming=True,
                ))

        return _stream(), RequestMetrics(
            request_id=request_id,
            prompt=prompt,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            ttft=0,
            decode_time=0,
            total_time=0,
            tps=0,
            total_tps=0,
            first_token_time=0,
            last_token_time=0,
            streaming=True,
        )


class WorkloadExecutor:
    """Execute benchmark workloads against a vLLM instance."""

    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1",
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        config: WorkloadConfig | None = None,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.config = config or WorkloadConfig()
        self.client = OpenAIClient(base_url=api_url)
        self.prompt_generator = PromptGenerator(seed=self.config.seed)

    async def run_workload(
        self,
        workload_type: WorkloadType | str,
        length: WorkloadLength | str,
        num_prompts: int | None = None,
    ) -> WorkloadResult:
        """Run a complete workload benchmark."""
        if isinstance(workload_type, str):
            workload_type = WorkloadType(workload_type)
        if isinstance(length, str):
            length = WorkloadLength(length)

        num = num_prompts or self.config.num_prompts
        wt_val = workload_type.value if hasattr(workload_type, "value") else str(workload_type)
        l_val = length.value if hasattr(length, "value") else str(length)
        logger.info(
            f"Starting {wt_val} workload ({l_val}), "
            f"{num} prompts, concurrency={self.config.concurrency}"
        )

        prompt_set = self.prompt_generator.generate_prompts(
            workload_type=workload_type,
            length=length,
            num_prompts=num,
            template=self.config.prompt_template,
        )

        results = await self._execute_prompts(prompt_set)
        return self._compute_results(results, workload_type.value, length.value)

    async def _execute_prompts(self, prompt_set: PromptSet) -> list[RequestMetrics]:
        """Execute all prompts in the prompt set."""
        semaphore = asyncio.Semaphore(self.config.concurrency)
        all_metrics: list[RequestMetrics] = []

        async def _run(prompt: Prompt) -> RequestMetrics:
            async with semaphore:
                return await self._execute_single(prompt)

        tasks = [_run(p) for p in prompt_set.prompts]
        metrics_list = await asyncio.gather(*tasks, return_exceptions=True)

        for m in metrics_list:
            if isinstance(m, Exception):
                logger.error(f"Prompt execution failed: {m}")
                all_metrics.append(RequestMetrics(
                    request_id=generate_id(),
                    prompt="",
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    ttft=0,
                    decode_time=0,
                    total_time=0,
                    tps=0,
                    total_tps=0,
                    first_token_time=0,
                    last_token_time=0,
                    error=str(m),
                ))
            elif isinstance(m, RequestMetrics):
                all_metrics.append(m)

        return all_metrics

    async def _execute_single(self, prompt: Prompt) -> RequestMetrics:
        """Execute a single prompt."""
        max_tokens = min(prompt.expected_length, self.config.max_tokens_per_request)

        completion, metrics = await self.client.create_chat_completion(
            prompt=prompt.text,
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            stream=self.config.stream,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            seed=self.config.seed,
        )

        if self.config.stream and not metrics.error:
            async for _ in completion:
                pass

        return metrics

    def _compute_results(
        self,
        metrics: list[RequestMetrics],
        workload_type: str,
        workload_length: str,
    ) -> WorkloadResult:
        """Compute aggregate workload results."""
        successful = [m for m in metrics if not m.error]
        failed = [m for m in metrics if m.error]

        ttfts = [m.ttft for m in successful]
        tpss = [m.tps for m in successful]
        decode_times = [m.decode_time for m in successful]
        total_times = [m.total_time for m in successful]
        total_tpss = [m.total_tps for m in successful]

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p
            f_idx = int(k)
            c_idx = f_idx + 1
            if c_idx >= len(sorted_data):
                return sorted_data[-1]
            return sorted_data[f_idx] + (k - f_idx) * (sorted_data[c_idx] - sorted_data[f_idx])

        total_prompts_tokens = sum(m.prompt_tokens for m in successful)
        total_completion_tokens = sum(m.completion_tokens for m in successful)
        total_time = sum(total_times)

        return WorkloadResult(
            workload_type=workload_type,
            workload_length=workload_length,
            num_requests=len(metrics),
            successful_requests=len(successful),
            failed_requests=len(failed),
            total_prompts_tokens=total_prompts_tokens,
            total_completion_tokens=total_completion_tokens,
            total_tokens=total_prompts_tokens + total_completion_tokens,
            avg_ttft=sum(ttfts) / len(ttfts) if ttfts else 0,
            p95_ttft=percentile(ttfts, 0.95),
            p99_ttft=percentile(ttfts, 0.99),
            avg_decode_time=sum(decode_times) / len(decode_times) if decode_times else 0,
            avg_total_time=sum(total_times) / len(total_times) if total_times else 0,
            avg_tps=sum(tpss) / len(tpss) if tpss else 0,
            p95_tps=percentile(tpss, 0.95),
            p99_tps=percentile(tpss, 0.99),
            avg_total_tps=sum(total_tpss) / len(total_tpss) if total_tpss else 0,
            total_time=total_time,
            requests=metrics,
            errors=[{"error": m.error, "request_id": m.request_id} for m in failed],
        )

    async def run_concurrency_test(
        self,
        concurrency: int,
        num_requests: int = 10,
    ) -> WorkloadResult:
        """Run a concurrency-level throughput test."""
        original_concurrency = self.config.concurrency
        self.config.concurrency = concurrency

        prompt_set = self.prompt_generator.generate_prompts(
            workload_type=WorkloadType.CODING,
            length=WorkloadLength.MEDIUM,
            num_prompts=num_requests,
        )

        results = await self._execute_prompts(prompt_set)
        self.config.concurrency = original_concurrency

        return self._compute_results(results, "concurrency", f"level_{concurrency}")

    async def run_burst_test(
        self,
        burst_size: int = 10,
        burst_count: int = 5,
        pause_between: float = 1.0,
    ) -> WorkloadResult:
        """Run a burst workload test."""
        all_metrics: list[RequestMetrics] = []

        for i in range(burst_count):
            prompt_set = self.prompt_generator.generate_prompts(
                workload_type=WorkloadType.CODING,
                length=WorkloadLength.SHORT,
                num_prompts=burst_size,
            )
            burst_metrics = await self._execute_prompts(prompt_set)
            all_metrics.extend(burst_metrics)
            if i < burst_count - 1:
                await asyncio.sleep(pause_between)

        return self._compute_results(all_metrics, "burst", f"size_{burst_size}_count_{burst_count}")

    async def shutdown(self) -> None:
        """Shutdown the workload executor."""
        await self.client.close()
