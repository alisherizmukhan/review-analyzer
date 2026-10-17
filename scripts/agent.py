"""Ask the review-analysis agent a question (Claude API + the Review Analyzer MCP server).

The agent starts its own MCP server subprocess, lets Claude choose and call the server's tools in a bounded
loop, and prints a report whose review-id citations were checked against the actual tool results.
Requires ANTHROPIC_API_KEY in the environment and `pip install -e ".[mcp,agent]"`.

Usage (PowerShell):
    $env:ANTHROPIC_API_KEY = "..."    # or set it in your user environment; never commit it
    python scripts/agent.py --corpus products "What do customers complain about for the Echo Auto (B07VTK654B)?"
    python scripts/agent.py --corpus products --json "Compare B08F6GPRH6 and B00N2ZDXW2"
    python scripts/agent.py --corpus products --max-tool-calls 4 --model claude-sonnet-5-5 "..."

Exit codes: 0 report produced, 1 the run ended without a valid report (limit, refusal, API or tool error),
2 configuration problem (missing key or package, bad setting).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.agent import AgentConfig, AgentConfigError, AgentError, render_text, run_agent  # noqa: E402
from review_intel.agent.config import EFFORTS  # noqa: E402
from review_intel.utils import PROJECT_ROOT, setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="+", help="The review-analysis question")
    parser.add_argument("--corpus", help="Corpus for the MCP server: products | analytics | demo | a .csv/.json "
                                         "path (default: REVIEW_ANALYZER_DATA or the server default)")
    parser.add_argument("--model", help="Claude model id (default: REVIEW_AGENT_MODEL or claude-opus-5-5)")
    parser.add_argument("--effort", choices=EFFORTS)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-tool-calls", type=int)
    parser.add_argument("--max-result-chars", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--trace-dir", type=Path, help="Where to write the run trace (default runs/agent)")
    parser.add_argument("--no-trace", action="store_true", help="Do not write a trace file")
    parser.add_argument("--json", action="store_true", help="Print the validated report as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every agent event to stderr")
    args = parser.parse_args()
    setup_logging(logging.INFO if args.verbose else logging.WARNING)
    for stream in (sys.stdout, sys.stderr):  # review excerpts may not fit the console code page (e.g. cp1251)
        stream.reconfigure(errors="replace")

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("error: ANTHROPIC_API_KEY is not set. Set it in the environment (never in a file in this "
              "repository) and run again.", file=sys.stderr)
        return 2
    trace_dir = None if args.no_trace else (args.trace_dir if args.trace_dir is None or args.trace_dir.is_absolute()
                                            else PROJECT_ROOT / args.trace_dir)
    try:
        config = AgentConfig.from_env(model=args.model, effort=args.effort, max_turns=args.max_turns,
                                      max_tool_calls=args.max_tool_calls, max_result_chars=args.max_result_chars,
                                      token_budget=args.token_budget)
        if args.no_trace or args.trace_dir:
            config.trace_dir = trace_dir
    except AgentConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        report = asyncio.run(run_agent(" ".join(args.question), config, corpus=args.corpus))
    except ImportError as exc:
        print(f"error: {exc}. Install the agent extra: pip install -e \".[mcp,agent]\"", file=sys.stderr)
        return 2
    except AgentError as exc:
        print(f"agent stopped ({exc.code}): {exc}", file=sys.stderr)
        return 1
    print(report.model_dump_json(indent=2) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
