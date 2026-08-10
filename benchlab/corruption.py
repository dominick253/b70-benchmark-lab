"""
Corruption detection for BenchLab.

Automatically detects output corruption including:
- Repeated tokens/characters/lines
- Unicode corruption
- Invalid UTF-8
- Broken JSON
- NaN values
- Empty/partial generations
- EOS failures
- Infinite generation loops
- Runaway generation
- Hallucinated binary content
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchlab.config import CorruptionConfig
from benchlab.logging import get_logger

logger = get_logger("corruption")


class CorruptionType(str, Enum):
    REPEATED_TOKENS = "repeated_tokens"
    REPEATED_CHARS = "repeated_chars"
    REPEATED_LINES = "repeated_lines"
    UNICODE_CORRUPTION = "unicode_corruption"
    INVALID_UTF = "invalid_utf"
    BROKEN_JSON = "broken_json"
    NaN_VALUES = "nan_values"
    EMPTY_GENERATION = "empty_generation"
    PARTIAL_GENERATION = "partial_generation"
    EOS_FAILURE = "eos_failure"
    INFINITE_LOOP = "infinite_loop"
    RUNAWAY_GENERATION = "runaway_generation"
    HALLUCINATED_BINARY = "hallucinated_binary"
    REPEATED_PUNCTUATION = "repeated_punctuation"


@dataclass
class CorruptionEvent:
    """A detected corruption event."""
    event_id: str
    corruption_type: CorruptionType
    severity: str  # low, medium, high, critical
    timestamp: float
    prompt: str = ""
    output: str = ""
    evidence: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "corruption_type": self.corruption_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "output": self.output,
            "evidence": self.evidence,
            "context": self.context,
        }


class CorruptionDetector:
    """Detect output corruption in model generations."""

    def __init__(self, config: CorruptionConfig) -> None:
        self.config = config
        self._events: list[CorruptionEvent] = []

    def check(
        self,
        prompt: str,
        output: str,
        context: dict[str, Any] | None = None,
    ) -> list[CorruptionEvent]:
        """Check output for corruption."""
        events: list[CorruptionEvent] = []

        if not self.config.enabled:
            return events

        # Extract prompt type from context or output
        prompt_type = (context or {}).get("workload", "unknown")

        # Run all enabled checks
        checks = [
            (self.config.detect_empty_generation, self._check_empty),
            (self.config.detect_invalid_utf, self._check_invalid_utf),
            (self.config.detect_unicode_corruption, self._check_unicode_corruption),
            (self.config.detect_broken_json, self._check_broken_json),
            (self.config.detect_nan, self._check_nan),
            (self.config.detect_partial_generation, self._check_partial),
            (self.config.detect_eos_failure, self._check_eos_failure),
            (self.config.detect_infinite_loop, self._check_infinite_loop),
            (self.config.detect_runaway_generation, self._check_runaway),
            (self.config.detect_hallucinated_binary, self._check_hallucinated_binary),
        ]

        for enabled, check_fn in checks:
            if enabled:
                result = check_fn(output, prompt, context or {})
                events.extend(result)

        # Always run token/char/line repetition checks (configurable thresholds)
        events.extend(self._check_repeated_tokens(output))
        events.extend(self._check_repeated_chars(output))
        events.extend(self._check_repeated_lines(output))
        events.extend(self._check_repeated_punctuation(output))

        self._events.extend(events)
        return events

    def _make_event(
        self,
        corruption_type: CorruptionType,
        severity: str,
        prompt: str,
        output: str,
        evidence: str,
        context: dict[str, Any],
    ) -> CorruptionEvent:
        """Create a corruption event."""
        event_id = f"corr-{len(self._events) + 1:04d}"
        return CorruptionEvent(
            event_id=event_id,
            corruption_type=corruption_type,
            severity=severity,
            timestamp=__import__("time").time(),
            prompt=prompt[:500],
            output=output[:500],
            evidence=evidence[:1000],
            context=context,
        )

    def _check_empty(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for empty generation."""
        if not output or output.strip() == "":
            return [self._make_event(
                CorruptionType.EMPTY_GENERATION, "high",
                prompt, output, "Output is empty", ctx,
            )]
        return []

    def _check_invalid_utf(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for invalid UTF-8."""
        try:
            output.encode("utf-8")
        except UnicodeEncodeError as e:
            return [self._make_event(
                CorruptionType.INVALID_UTF, "critical",
                prompt, output, f"Invalid UTF-8: {e}", ctx,
            )]
        return []

    def _check_unicode_corruption(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for Unicode corruption (replacement characters, etc.)."""
        if "\ufffd" in output:
            count = output.count("\ufffd")
            return [self._make_event(
                CorruptionType.UNICODE_CORRUPTION, "medium",
                prompt, output, f"Found {count} Unicode replacement characters", ctx,
            )]
        return []

    def _check_broken_json(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for broken JSON."""
        stripped = output.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return []

        try:
            json.loads(stripped)
        except json.JSONDecodeError as e:
            return [self._make_event(
                CorruptionType.BROKEN_JSON, "high",
                prompt, output, f"JSON parse error: {e}", ctx,
            )]
        return []

    def _check_nan(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for NaN values in output."""
        if "NaN" in output or "nan" in output or "Infinity" in output:
            return [self._make_event(
                CorruptionType.NaN_VALUES, "high",
                prompt, output, "Found NaN or Infinity in output", ctx,
            )]
        return []

    def _check_partial(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for partial/incomplete generation."""
        if output and not output.endswith((".", "!", "?", "```", "}}", "]", ")")):
            last_20 = output[-20:].strip()
            if last_20 and len(last_20) < 20:
                return [self._make_event(
                    CorruptionType.PARTIAL_GENERATION, "low",
                    prompt, output, f"Output appears truncated: ...{last_20}", ctx,
                )]
        return []

    def _check_eos_failure(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for EOS (end of sequence) failures."""
        if len(output) > 10000:
            return [self._make_event(
                CorruptionType.EOS_FAILURE, "high",
                prompt, output, f"Output exceeds 10000 chars, likely EOS failure", ctx,
            )]
        return []

    def _check_infinite_loop(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for infinite generation loops."""
        lines = output.split("\n")
        if len(lines) >= 20:
            for i in range(len(lines) - 4):
                window = " ".join(lines[i:i+4])
                if window and window in output[i+4:]:
                    return [self._make_event(
                        CorruptionType.INFINITE_LOOP, "critical",
                        prompt, output, f"Infinite loop detected at line {i}: {window[:100]}", ctx,
                    )]
        return []

    def _check_runaway(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for runaway generation."""
        if len(output) > self.config.max_generation_tokens * 4:
            return [self._make_event(
                CorruptionType.RUNAWAY_GENERATION, "critical",
                prompt, output, f"Output is {len(output)} chars (limit: {self.config.max_generation_tokens * 4})", ctx,
            )]
        return []

    def _check_hallucinated_binary(self, output: str, prompt: str, ctx: dict) -> list[CorruptionEvent]:
        """Check for hallucinated binary content."""
        if len(output) < 100:
            return []

        non_printable = sum(1 for c in output if unicodedata.category(c).startswith("C") and c not in "\n\r\t")
        ratio = non_printable / len(output) if output else 0

        if ratio > 0.1:
            return [self._make_event(
                CorruptionType.HALLUCINATED_BINARY, "critical",
                prompt, output, f"High non-printable character ratio: {ratio:.2%}", ctx,
            )]
        return []

    def _check_repeated_tokens(self, output: str) -> list[CorruptionEvent]:
        """Check for repeated tokens."""
        words = output.split()
        if len(words) < self.config.max_repeated_tokens:
            return []

        for i in range(len(words) - self.config.max_repeated_tokens + 1):
            token = words[i]
            if token and all(w == token for w in words[i:i + self.config.max_repeated_tokens]):
                return [self._make_event(
                    CorruptionType.REPEATED_TOKENS, "high",
                    "", output, f"Token '{token}' repeated {self.config.max_repeated_tokens} times",
                    {},
                )]
        return []

    def _check_repeated_chars(self, output: str) -> list[CorruptionEvent]:
        """Check for repeated characters."""
        pattern = r"(.)\1{" + str(self.config.max_repeated_chars - 1) + r",}"
        matches = re.findall(pattern, output)
        if matches:
            return [self._make_event(
                CorruptionType.REPEATED_CHARS, "medium",
                "", output, f"Repeated characters found: {matches[:3]}",
                {},
            )]
        return []

    def _check_repeated_lines(self, output: str) -> list[CorruptionEvent]:
        """Check for repeated lines."""
        lines = [l for l in output.split("\n") if l.strip()]
        if len(lines) < self.config.max_repeated_lines:
            return []

        for i in range(len(lines) - self.config.max_repeated_lines + 1):
            if all(l == lines[i] for l in lines[i:i + self.config.max_repeated_lines]):
                return [self._make_event(
                    CorruptionType.REPEATED_LINES, "high",
                    "", output, f"Line repeated {self.config.max_repeated_lines} times",
                    {},
                )]
        return []

    def _check_repeated_punctuation(self, output: str) -> list[CorruptionEvent]:
        """Check for repeated punctuation."""
        pattern = r"([.!?])\1{5,}"
        matches = re.findall(pattern, output)
        if matches:
            return [self._make_event(
                CorruptionType.REPEATED_PUNCTUATION, "medium",
                "", output, f"Repeated punctuation: {matches[:3]}",
                {},
            )]
        return []

    def get_events(self) -> list[CorruptionEvent]:
        """Get all detected corruption events."""
        return self._events

    def get_summary(self) -> dict[str, Any]:
        """Get corruption summary."""
        type_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}

        for event in self._events:
            type_counts[event.corruption_type.value] = type_counts.get(event.corruption_type.value, 0) + 1
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1

        return {
            "total_events": len(self._events),
            "by_type": type_counts,
            "by_severity": severity_counts,
            "events": [e.to_dict() for e in self._events],
        }

    def clear(self) -> None:
        """Clear all detected events."""
        self._events.clear()
