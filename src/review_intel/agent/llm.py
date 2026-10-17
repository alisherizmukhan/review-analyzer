"""Model layer: a small provider-neutral interface and its Anthropic Claude implementation.

The agent loop only uses ``LLMClient.complete`` and the message builders below; everything Anthropic-specific
(SDK calls, error classes, retries, fallbacks, prompt caching) lives in ``AnthropicLLM``. Messages use the
Messages API shape (role + content blocks), which the scripted test model also speaks.

Retries are done here, not by the SDK (``max_retries=0``), so each retry is counted and traced. Retryable:
rate limits (429, honoring ``retry-after``), overload/5xx, timeouts and connection errors. Not retryable:
authentication, permission, invalid request, not found.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from review_intel.agent.config import AgentConfig


class LLMError(RuntimeError):
    def __init__(self, message: str, *, kind: str, retryable: bool, retries: int = 0,
                 request_id: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.retries = retries
        self.request_id = request_id


@dataclass
class ToolCall:
    id: str
    name: str
    input: Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


@dataclass
class LLMResponse:
    stop_reason: str
    text: str
    tool_calls: list[ToolCall]
    content: Any  # provider content blocks, appended to the history unchanged (thinking blocks included)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    retries: int = 0
    request_id: str | None = None
    model: str | None = None
    refusal_category: str | None = None


class LLMClient(Protocol):
    model: str

    async def complete(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                       output_schema: dict[str, Any]) -> LLMResponse: ...


# -- message builders (Messages API shape) --------------------------------------------------------------


def user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant_message(response: LLMResponse) -> dict[str, Any]:
    return {"role": "assistant", "content": response.content}


def tool_results_message(results: list[dict[str, Any]], note: str | None = None) -> dict[str, Any]:
    """All tool results of one turn go back in a single user message (required for parallel calls)."""
    blocks = list(results)
    if note:
        blocks.append({"type": "text", "text": note})
    return {"role": "user", "content": blocks}


def tool_result_block(tool_use_id: str, text: str, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
    if is_error:
        block["is_error"] = True
    return block


# -- Anthropic implementation -----------------------------------------------------------------------------

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    def __init__(self, config: AgentConfig, client: Any = None, sleep: Callable[[float], Any] = asyncio.sleep):
        import anthropic  # optional dependency: pip install -e .[agent]

        self._anthropic = anthropic
        self.config = config
        self.model = config.model
        # The SDK reads ANTHROPIC_API_KEY itself; the key is never handled by this code.
        self.client = client or anthropic.AsyncAnthropic(max_retries=0, timeout=config.request_timeout_s)
        self._sleep = sleep

    def _request(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                 output_schema: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        params: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_output_tokens,
            # Stable prefix (tools + system) is cached; the growing history is cached automatically too.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "cache_control": {"type": "ephemeral"},
            "messages": messages,
            "tools": tools,
            "output_config": {"effort": self.config.effort,
                              "format": {"type": "json_schema", "schema": output_schema}},
        }
        if self.config.fallbacks == "default":
            # Server-side refusal fallback (beta): a declined request is re-run on a fallback model in-call.
            return self.client.beta.messages.create, {**params, "betas": [FALLBACK_BETA], "fallbacks": "default"}
        return self.client.messages.create, params

    def _classify(self, exc: Exception) -> tuple[str, bool, float | None]:
        a = self._anthropic
        if isinstance(exc, a.RateLimitError):
            retry_after = None
            try:
                retry_after = float(exc.response.headers.get("retry-after", ""))
            except (TypeError, ValueError, AttributeError):
                pass
            return "rate_limit", True, retry_after
        if isinstance(exc, (a.AuthenticationError, a.PermissionDeniedError)):
            return "authentication", False, None
        if isinstance(exc, (a.BadRequestError, a.NotFoundError, a.UnprocessableEntityError)):
            return "invalid_request", False, None
        if isinstance(exc, a.APITimeoutError):
            return "timeout", True, None
        if isinstance(exc, a.APIConnectionError):
            return "connection", True, None
        if isinstance(exc, a.APIStatusError):
            status = exc.status_code
            return ("server_error" if status >= 500 else f"http_{status}"), status >= 500 or status in (408, 409), None
        return "unexpected", False, None

    async def complete(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                       output_schema: dict[str, Any]) -> LLMResponse:
        create, params = self._request(system, messages, tools, output_schema)
        retries = 0
        while True:
            started = time.perf_counter()
            try:
                response = await create(**params)
                break
            except self._anthropic.AnthropicError as exc:
                kind, retryable, retry_after = self._classify(exc)
                request_id = getattr(exc, "request_id", None)
                if not retryable or retries >= self.config.max_api_retries:
                    message = getattr(exc, "message", None) or str(exc)
                    raise LLMError(f"{kind}: {message}", kind=kind, retryable=retryable, retries=retries,
                                   request_id=request_id) from exc
                delay = retry_after if retry_after is not None else min(2 ** retries + random.uniform(0, 1), 30.0)
                retries += 1
                await self._sleep(delay)
        latency = time.perf_counter() - started
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
        u = response.usage
        usage = Usage(u.input_tokens or 0, u.output_tokens or 0, getattr(u, "cache_read_input_tokens", 0) or 0,
                      getattr(u, "cache_creation_input_tokens", 0) or 0)
        details = getattr(response, "stop_details", None)
        return LLMResponse(stop_reason=response.stop_reason or "", text=text, tool_calls=calls,
                           content=response.content, usage=usage, latency_s=latency, retries=retries,
                           request_id=getattr(response, "_request_id", None), model=getattr(response, "model", None),
                           refusal_category=getattr(details, "category", None) if details else None)
