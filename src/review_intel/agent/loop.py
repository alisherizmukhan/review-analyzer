"""The agent loop: user question -> model -> tool call -> MCP server -> tool result -> model -> validated report.

The model chooses which tools to call; this code executes them through the MCP client, feeds the results
back, and stops when the model returns a final report or a limit is reached:

* ``max_turns``       model calls per run (tool-use turns, the final answer and repair turns all count);
* ``max_tool_calls``  tool executions per run; on reaching it the model gets one instruction to write the
                      report with the evidence it has, and calling tools again aborts the run;
* ``token_budget``    total tokens across the run, checked before every model call;
* duplicate calls     the same tool with the same arguments is refused after ``max_duplicate_calls``.

Every limit ends the run with an ``AgentError`` whose ``code`` says which limit, never a silent cut-off.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from review_intel.agent.config import AgentConfig, estimate_cost_usd
from review_intel.agent.evidence import EvidenceIndex
from review_intel.agent.guard import bound_result, dumps, render_tool_output
from review_intel.agent.llm import (
    LLMClient,
    LLMError,
    ToolCall,
    Usage,
    assistant_message,
    tool_result_block,
    tool_results_message,
    user_message,
)
from review_intel.agent.report import AgentReport, ReportDraft, RunInfo, ToolUsage, finalize, strict_json_schema
from review_intel.agent.toolbox import Toolbox, ToolExecutionError, validate_arguments
from review_intel.agent.trace import RunTrace

WRAP_UP_NOTE = ("Tool-call limit reached. Do not call any more tools. Write the final JSON report now, using only "
                "the evidence above, and mention in limitations that the tool budget was exhausted.")


class AgentError(RuntimeError):
    """The run ended without a valid report. ``code`` names the reason (a limit, refusal, API or tool failure)."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def load_system_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->\s*", "", text, flags=re.S).strip()


class ReviewAgent:
    def __init__(self, llm: LLMClient, toolbox: Toolbox, config: AgentConfig, trace: RunTrace | None = None):
        self.llm = llm
        self.toolbox = toolbox
        self.config = config
        self.trace = trace or RunTrace(None)
        self.system_prompt = load_system_prompt(config.system_prompt_path)
        self.output_schema = strict_json_schema(ReportDraft)
        self.index = EvidenceIndex()
        self.usage = Usage()
        self.turns = 0
        self.tool_calls_used = 0
        self.api_retries = 0
        self.repairs = 0
        self.wrap_up_requested = False
        self._calls_by_args: Counter[str] = Counter()
        self._tool_stats: dict[str, list[float]] = {}  # name -> [calls, errors, seconds]

    # -- public -------------------------------------------------------------

    async def run(self, query: str) -> AgentReport:
        if not query or not query.strip():
            raise AgentError("The question must not be empty.", code="invalid_input")
        cfg = self.config
        self.trace.event("run_start", model=cfg.model, effort=cfg.effort, query=query.strip(),
                         limits={"max_turns": cfg.max_turns, "max_tool_calls": cfg.max_tool_calls,
                                 "token_budget": cfg.token_budget, "max_result_chars": cfg.max_result_chars},
                         tools=[t.name for t in self.toolbox.tools])
        try:
            report = await self._loop(query.strip())
        except AgentError as exc:
            self._end("error", code=exc.code, error=str(exc))
            raise
        except LLMError as exc:
            self.api_retries += exc.retries
            self._end("error", code="llm_error", error=str(exc), request_id=exc.request_id)
            raise AgentError(f"Claude API call failed ({exc}).", code="llm_error") from exc
        except ToolExecutionError as exc:
            self._end("error", code="tool_connection", error=str(exc))
            raise AgentError(f"MCP tool execution failed: {exc}", code="tool_connection") from exc
        v = report.validation
        self._end("ok", verified_review_ids=len(v.verified_review_ids), rejected_citations=len(v.rejected_citations),
                  unverified_numbers=len(v.unverified_numbers), insufficient_evidence=report.insufficient_evidence)
        return report

    # -- loop ---------------------------------------------------------------

    def _tools_param(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in self.toolbox.tools]

    async def _loop(self, query: str) -> AgentReport:
        cfg = self.config
        tools = self._tools_param()
        messages: list[dict[str, Any]] = [user_message(query)]
        for turn in range(1, cfg.max_turns + 1):
            if self.usage.total >= cfg.token_budget:
                raise AgentError(f"Token budget exhausted ({self.usage.total} of {cfg.token_budget} tokens) "
                                 "before a final report.", code="token_budget")
            response = await self.llm.complete(system=self.system_prompt, messages=messages, tools=tools,
                                               output_schema=self.output_schema)
            self.turns = turn
            self.api_retries += response.retries
            for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
                setattr(self.usage, name, getattr(self.usage, name) + getattr(response.usage, name))
            self.trace.event("model_call", turn=turn, latency_ms=round(response.latency_s * 1000, 1),
                             stop_reason=response.stop_reason, input_tokens=response.usage.input_tokens,
                             output_tokens=response.usage.output_tokens,
                             cache_read_tokens=response.usage.cache_read_tokens,
                             cache_write_tokens=response.usage.cache_write_tokens, retries=response.retries,
                             request_id=response.request_id, served_by=response.model,
                             tool_calls=[c.name for c in response.tool_calls])

            if response.stop_reason == "refusal":
                raise AgentError(f"The model declined the request (category: {response.refusal_category}).",
                                 code="refusal")
            if response.stop_reason == "max_tokens":
                raise AgentError("The model response hit max_output_tokens; raise REVIEW_AGENT_MAX_OUTPUT_TOKENS.",
                                 code="truncated")

            if response.tool_calls:
                if self.wrap_up_requested:
                    raise AgentError(f"Tool-call limit ({cfg.max_tool_calls}) reached and the model kept calling "
                                     "tools instead of writing the report.", code="limit_tool_calls")
                messages.append(assistant_message(response))
                results = []
                for call in response.tool_calls:
                    if self.tool_calls_used >= cfg.max_tool_calls:
                        self.trace.event("tool_skipped", tool=call.name, reason="tool-call limit reached")
                        results.append(tool_result_block(call.id, "Not executed: tool-call limit reached.", True))
                        continue
                    results.append(await self._execute(call))
                note = None
                if self.tool_calls_used >= cfg.max_tool_calls:
                    self.wrap_up_requested = True
                    note = WRAP_UP_NOTE
                messages.append(tool_results_message(results, note))
                continue

            try:
                draft = ReportDraft.model_validate_json(response.text)
            except ValidationError as exc:
                problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or '<root>'}: {e['msg']}"
                                     for e in exc.errors()[:8])
                self.trace.event("report_invalid", turn=turn, errors=problems, repairs_used=self.repairs)
                if self.repairs >= cfg.max_repairs:
                    raise AgentError(f"The final report failed schema validation: {problems}",
                                     code="invalid_output") from exc
                self.repairs += 1
                messages.append(assistant_message(response))
                messages.append(user_message(f"Your report did not validate against the required schema: {problems}. "
                                             "Reply with the corrected JSON report only."))
                continue
            return finalize(draft, query, self.index, self._tool_usage(), self._run_info())
        raise AgentError(f"Turn limit ({cfg.max_turns}) reached before a final report.", code="limit_turns")

    async def _execute(self, call: ToolCall) -> dict[str, Any]:
        self.tool_calls_used += 1
        spec = next((t for t in self.toolbox.tools if t.name == call.name), None)
        stats = self._tool_stats.setdefault(call.name, [0, 0, 0.0])
        stats[0] += 1

        def refuse(message: str, reason: str) -> dict[str, Any]:
            stats[1] += 1
            self.trace.event("tool_call", tool=call.name, arguments=call.input, executed=False, is_error=True,
                             error=message, reason=reason)
            return tool_result_block(call.id, message.replace("<", "\\u003c"), is_error=True)

        if spec is None:
            available = ", ".join(t.name for t in self.toolbox.tools)
            return refuse(f"Unknown or disallowed tool {call.name!r}. Available tools: {available}.", "unknown_tool")
        problem = validate_arguments(spec, call.input)
        if problem:
            return refuse(f"Invalid arguments for {call.name}: {problem}. Fix them and call again.", "invalid_args")
        key = f"{call.name}:{json.dumps(call.input, sort_keys=True, default=str)}"
        self._calls_by_args[key] += 1
        if self._calls_by_args[key] > self.config.max_duplicate_calls:
            return refuse(f"Duplicate call refused: {call.name} was already called with these arguments; reuse "
                          "the earlier result.", "duplicate")

        outcome = await self.toolbox.call(call.name, call.input)
        stats[2] += outcome.duration_s
        if outcome.is_error:
            stats[1] += 1
            self.trace.event("tool_call", tool=call.name, arguments=call.input, executed=True, is_error=True,
                             duration_ms=round(outcome.duration_s * 1000, 1), error=outcome.error)
            return tool_result_block(call.id, f"Tool error from {call.name}: {outcome.error}".replace("<", "\\u003c"),
                                     is_error=True)
        bounded, note = bound_result(outcome.data, self.config.max_result_chars)
        before = len(self.index.reviews)
        self.index.add(call.name, bounded)
        text = render_tool_output(call.name, bounded)
        self.trace.event("tool_call", tool=call.name, arguments=call.input, executed=True, is_error=False,
                         duration_ms=round(outcome.duration_s * 1000, 1), result_chars=len(dumps(outcome.data)),
                         sent_chars=len(text), truncated=note is not None,
                         new_review_ids=len(self.index.reviews) - before)
        return tool_result_block(call.id, text)

    # -- bookkeeping --------------------------------------------------------

    def _tool_usage(self) -> list[ToolUsage]:
        return [ToolUsage(name=name, calls=int(s[0]), errors=int(s[1]), total_duration_s=round(s[2], 3))
                for name, s in self._tool_stats.items()]

    def _cost(self) -> float | None:
        u = self.usage
        return estimate_cost_usd(self.config.model, u.input_tokens, u.output_tokens, u.cache_read_tokens,
                                 u.cache_write_tokens)

    def _run_info(self) -> RunInfo:
        u = self.usage
        return RunInfo(run_id=self.trace.run_id, model=self.config.model, turns=self.turns,
                       tool_calls=self.tool_calls_used, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                       cache_read_tokens=u.cache_read_tokens, cache_write_tokens=u.cache_write_tokens,
                       estimated_cost_usd=self._cost(), api_retries=self.api_retries,
                       duration_s=round(self.trace.elapsed(), 3), trace_file=self.trace.display_path())

    def _end(self, status: str, **fields: Any) -> None:
        if self.trace.events and self.trace.events[-1].get("event") == "run_end":
            return
        u = self.usage
        self.trace.event("run_end", status=status, duration_s=round(self.trace.elapsed(), 3), turns=self.turns,
                         tool_calls=self.tool_calls_used, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                         cache_read_tokens=u.cache_read_tokens, cache_write_tokens=u.cache_write_tokens,
                         api_retries=self.api_retries, estimated_cost_usd=self._cost(),
                         tools=[t.model_dump() for t in self._tool_usage()], **fields)


async def run_agent(query: str, config: AgentConfig, *, corpus: str | None = None, llm: LLMClient | None = None,
                    server_target: Any = None, trace: RunTrace | None = None) -> AgentReport:
    """Run one question end to end: start the MCP server, run the loop, always shut the server down.

    ``corpus`` is a corpus path or alias (products | analytics | demo) for the server; ``server_target``
    overrides the whole connection (tests pass an in-process server or custom stdio parameters).
    """
    from review_intel.agent.llm import AnthropicLLM
    from review_intel.agent.toolbox import McpToolbox, server_parameters

    trace = trace or RunTrace(config.trace_dir)
    llm = llm or AnthropicLLM(config)  # fails fast (missing SDK) before a server is started
    target = server_target if server_target is not None else server_parameters(corpus)
    toolbox = McpToolbox(target, timeout_s=config.tool_timeout_s, allowed=config.allowed_tools,
                         server_log=trace.server_log_path)
    try:
        await toolbox.__aenter__()
    except Exception as exc:
        trace.event("run_end", status="error", code="server_start", error=f"{type(exc).__name__}: {exc}")
        raise AgentError(f"Could not start or connect to the MCP server: {type(exc).__name__}: {exc}"
                         + (f" (server log: {trace.server_log_path})" if trace.server_log_path else ""),
                         code="server_start") from exc
    try:
        trace.event("mcp_connected", tools=[t.name for t in toolbox.tools], hidden_tools=toolbox.hidden_tools)
        if not toolbox.tools:
            raise AgentError("The MCP server exposes no usable read-only tools.", code="no_tools")
        return await ReviewAgent(llm, toolbox, config, trace).run(query)
    finally:
        await toolbox.__aexit__(None, None, None)
        trace.event("mcp_closed")
