"""MCP client side of the agent: connect to the Review Analyzer server, discover its tools, execute calls.

The agent launches its *own* server subprocess over stdio for the duration of one run (an MCP stdio
connection is a private pipe, so the copy Claude Code may be running cannot be shared). ``McpToolbox`` is an
async context manager: leaving it closes the session and terminates the subprocess, also on errors.

Safety boundaries enforced here, independent of the model:
* only tools the server marks read-only (and, optionally, an explicit allowlist) are exposed or executed;
* arguments are validated against the tool's JSON Schema before anything is sent to the server;
* the child process gets a minimal environment: never the Anthropic credentials.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

import jsonschema
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

from review_intel.utils import PROJECT_ROOT

SERVER_SCRIPT = PROJECT_ROOT / "review-analyzer-mcp" / "server.py"
CORPUS_ALIASES = {
    "products": "data/processed/amazon_electronics_products/corpus.csv",
    "analytics": "data/processed/amazon_electronics/analytics_corpus.csv",
    "demo": "demo",
}
# Variables the server process may inherit. Everything else (notably ANTHROPIC_*) stays in the agent.
_PASSTHROUGH = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE",
                "LOCALAPPDATA", "APPDATA", "PROGRAMDATA", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "HF_HOME",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


class ToolExecutionError(RuntimeError):
    """The MCP connection itself failed (not a tool returning an error result)."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool


@dataclass
class ToolOutcome:
    is_error: bool
    data: Any  # structured result when the call succeeded
    error: str | None
    duration_s: float


class Toolbox(Protocol):
    tools: list[ToolSpec]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome: ...


def server_environment(corpus: str | None = None, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the server subprocess: OS essentials + REVIEW_ANALYZER_* settings, no credentials."""
    env = os.environ if environ is None else environ
    child = {k: v for k, v in env.items() if k.upper() in _PASSTHROUGH or k.startswith("REVIEW_ANALYZER_")}
    if corpus:
        child["REVIEW_ANALYZER_DATA"] = CORPUS_ALIASES.get(corpus, corpus)
    child.setdefault("PYTHONIOENCODING", "utf-8")
    return child


def server_parameters(corpus: str | None = None, python: str | None = None,
                      environ: dict[str, str] | None = None) -> StdioServerParameters:
    return StdioServerParameters(command=python or sys.executable, args=[str(SERVER_SCRIPT)], cwd=str(PROJECT_ROOT),
                                 env=server_environment(corpus, environ))


def validate_arguments(spec: ToolSpec, arguments: Any) -> str | None:
    """Return a readable error if ``arguments`` do not match the tool's input schema, else None."""
    if not isinstance(arguments, dict):
        return f"arguments must be a JSON object, got {type(arguments).__name__}"
    errors = sorted(jsonschema.Draft202012Validator(spec.input_schema).iter_errors(arguments), key=str)
    if not errors:
        return None
    return "; ".join(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5])


class McpToolbox:
    """Async context manager around one MCP client session.

    ``target`` is a ``StdioServerParameters`` (normal use: launch the server) or anything ``mcp.Client``
    accepts (tests pass an in-process server object).
    """

    def __init__(self, target: Any, *, timeout_s: float = 180.0, allowed: frozenset[str] | None = None,
                 server_log: Path | None = None):
        self.target = target
        self.timeout_s = timeout_s
        self.allowed = allowed
        self.server_log = server_log
        self.tools: list[ToolSpec] = []
        self.hidden_tools: list[str] = []
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None

    async def __aenter__(self) -> McpToolbox:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            target = self.target
            if isinstance(target, StdioServerParameters):
                errlog: TextIO = sys.stderr
                if self.server_log is not None:  # keep the server's own log out of the CLI output
                    self.server_log.parent.mkdir(parents=True, exist_ok=True)
                    errlog = stack.enter_context(self.server_log.open("w", encoding="utf-8"))
                target = stdio_client(target, errlog=errlog)
            self._client = await stack.enter_async_context(Client(target))
            listed = (await self._client.list_tools()).tools
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        for tool in listed:
            read_only = bool(tool.annotations and tool.annotations.read_only_hint)
            if not read_only or (self.allowed is not None and tool.name not in self.allowed):
                self.hidden_tools.append(tool.name)
                continue
            self.tools.append(ToolSpec(tool.name, tool.description or "", dict(tool.input_schema), read_only))
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        stack, self._stack, self._client = self._stack, None, None
        if stack is not None:
            await stack.__aexit__(*exc_info)

    @property
    def is_open(self) -> bool:
        return self._client is not None

    def spec(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        if self._client is None:
            raise ToolExecutionError("MCP connection is closed")
        started = time.perf_counter()
        try:
            result = await self._client.call_tool(name, arguments, read_timeout_seconds=self.timeout_s)
        except Exception as exc:  # transport failure or timeout: the run cannot trust this connection
            raise ToolExecutionError(f"{type(exc).__name__}: {exc}") from exc
        duration = time.perf_counter() - started
        text = " ".join(getattr(c, "text", "") for c in result.content or []).strip()
        if result.is_error:
            return ToolOutcome(True, None, text or "tool reported an error", duration)
        data = result.structured_content if result.structured_content is not None else {"text": text}
        return ToolOutcome(False, data, None, duration)
