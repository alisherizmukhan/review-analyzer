"""Agent configuration: model, budgets and limits, all overridable from the environment.

Environment variables (all optional except the API key, which the Anthropic SDK reads itself):
    ANTHROPIC_API_KEY                 read by the Anthropic SDK; never logged, never passed to the MCP server
    REVIEW_AGENT_MODEL                Claude model id (default claude-opus-5-5)
    REVIEW_AGENT_EFFORT               low | medium | high | xhigh | max (default medium)
    REVIEW_AGENT_FALLBACKS            "default" (server-side refusal fallback, beta) | "off" (default "default")
    REVIEW_AGENT_MAX_OUTPUT_TOKENS    max_tokens per model response (default 16000)
    REVIEW_AGENT_TOKEN_BUDGET         total input+output tokens per run (default 250000)
    REVIEW_AGENT_MAX_TURNS            model calls per run, including the final report (default 8)
    REVIEW_AGENT_MAX_TOOL_CALLS       tool executions per run (default 12)
    REVIEW_AGENT_MAX_RESULT_CHARS     characters of one tool result shown to the model (default 8000)
    REVIEW_AGENT_TOOL_TIMEOUT         seconds per MCP tool call (default 180; the first LSA search builds an index)
    REVIEW_AGENT_TRACE_DIR            where run traces are written (default runs/agent); "off" disables
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from review_intel.utils import PROJECT_ROOT

DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_MODES = ("default", "off")
DEFAULT_TRACE_DIR = PROJECT_ROOT / "runs" / "agent"
DEFAULT_SYSTEM_PROMPT = PROJECT_ROOT / "configs" / "agent" / "system_prompt.md"


class AgentConfigError(ValueError):
    """An agent setting is invalid; the message names the variable."""


def _int(env: dict[str, str], name: str, default: int, minimum: int = 1) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise AgentConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise AgentConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    fallbacks: str = "default"
    max_output_tokens: int = 16_000
    token_budget: int = 250_000
    max_turns: int = 8
    max_tool_calls: int = 12
    max_result_chars: int = 8_000
    max_duplicate_calls: int = 2  # identical (tool, arguments) calls allowed before they are refused
    max_repairs: int = 1  # extra model turns allowed to fix a report that fails validation
    tool_timeout_s: float = 180.0
    request_timeout_s: float = 300.0
    max_api_retries: int = 3
    trace_dir: Path | None = DEFAULT_TRACE_DIR
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT
    allowed_tools: frozenset[str] | None = None  # None = every read-only tool the server exposes
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.effort not in EFFORTS:
            raise AgentConfigError(f"effort must be one of {EFFORTS}, got {self.effort!r}")
        if self.fallbacks not in FALLBACK_MODES:
            raise AgentConfigError(f"fallbacks must be one of {FALLBACK_MODES}, got {self.fallbacks!r}")
        if self.max_result_chars < 500:
            raise AgentConfigError("max_result_chars must be >= 500 so ids and counts survive truncation")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None, **overrides) -> AgentConfig:
        env = os.environ if environ is None else environ
        trace = env.get("REVIEW_AGENT_TRACE_DIR", "").strip()
        if trace.lower() == "off":
            trace_dir: Path | None = None
        elif trace:
            trace_dir = Path(trace) if Path(trace).is_absolute() else PROJECT_ROOT / trace
        else:
            trace_dir = DEFAULT_TRACE_DIR
        timeout = env.get("REVIEW_AGENT_TOOL_TIMEOUT", "").strip()
        try:
            tool_timeout = float(timeout) if timeout else 180.0
        except ValueError:
            raise AgentConfigError(f"REVIEW_AGENT_TOOL_TIMEOUT must be a number, got {timeout!r}") from None
        values = dict(
            model=env.get("REVIEW_AGENT_MODEL", "").strip() or DEFAULT_MODEL,
            effort=env.get("REVIEW_AGENT_EFFORT", "").strip().lower() or "medium",
            fallbacks=env.get("REVIEW_AGENT_FALLBACKS", "").strip().lower() or "default",
            max_output_tokens=_int(env, "REVIEW_AGENT_MAX_OUTPUT_TOKENS", 16_000, minimum=1024),
            token_budget=_int(env, "REVIEW_AGENT_TOKEN_BUDGET", 250_000, minimum=10_000),
            max_turns=_int(env, "REVIEW_AGENT_MAX_TURNS", 8),
            max_tool_calls=_int(env, "REVIEW_AGENT_MAX_TOOL_CALLS", 12, minimum=0),
            max_result_chars=_int(env, "REVIEW_AGENT_MAX_RESULT_CHARS", 8_000, minimum=500),
            tool_timeout_s=tool_timeout,
            trace_dir=trace_dir,
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


# Published per-million-token prices (USD) used only for the cost *estimate* in traces. Check current pricing
# before relying on it; unknown models report no estimate rather than a guess.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int, cache_read: int = 0,
                      cache_write: int = 0) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    inp, out = prices
    total = (input_tokens * inp + cache_read * inp * CACHE_READ_MULTIPLIER
             + cache_write * inp * CACHE_WRITE_MULTIPLIER + output_tokens * out)
    return round(total / 1_000_000, 6)
