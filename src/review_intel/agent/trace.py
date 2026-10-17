"""Run trace: one JSONL file per agent run, plus a stderr log line per event.

What is recorded: run id, model, every model call (latency, token usage, stop reason, retries, request id),
every tool call (name, arguments, duration, error, result size, truncation, number of review ids seen),
errors and totals. What is NOT recorded: tool result contents (customer review text), the model's final
report text, and credentials. Every string passes through ``redact`` before it is written, which removes
Anthropic-style keys and the literal values of credential environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from review_intel.utils import relative_to_root, utc_now

logger = logging.getLogger("review_intel.agent")

SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}")
MAX_LOGGED_STRING = 500


def redact(value: Any) -> Any:
    if isinstance(value, str):
        text = _KEY_PATTERN.sub("[REDACTED]", value)
        for name in SECRET_ENV_VARS:
            secret = os.environ.get(name, "")
            if len(secret) >= 8:
                text = text.replace(secret, "[REDACTED]")
        return text if len(text) <= MAX_LOGGED_STRING else text[:MAX_LOGGED_STRING] + "…"
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


class RunTrace:
    def __init__(self, trace_dir: Path | None, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.started = time.perf_counter()
        self.path: Path | None = None
        self.events: list[dict[str, Any]] = []
        if trace_dir is not None:
            trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = trace_dir / f"{utc_now()[:19].replace(':', '').replace('-', '')}-{self.run_id}.jsonl"

    @property
    def server_log_path(self) -> Path | None:
        return self.path.with_suffix(".server.log") if self.path else None

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        record = redact({"run_id": self.run_id, "event": kind, "t": round(self.elapsed(), 3), **fields})
        self.events.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        summary = " ".join(f"{k}={v}" for k, v in record.items() if k not in ("run_id", "event", "t")
                           and not isinstance(v, (dict, list)))
        logger.info("[%s] %s %s", self.run_id, kind, summary)
        return record

    def display_path(self) -> str | None:
        return relative_to_root(self.path) if self.path else None
