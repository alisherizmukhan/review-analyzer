"""Evaluate the review-analysis agent on a small fixed question set (data/eval/agent_cases.jsonl).

Every case is a real agent run (Claude API + MCP server), so this costs money: roughly the per-run cost shown in
the trace times the number of cases. Use --dry-run to validate the case file and corpora without any API call.

What is measured, kept in two separate groups:
* retrieval side (did the agent gather evidence?): expected tool used, number of distinct reviews seen in tool
  results, and a lexical topicality proxy (share of quoted excerpts containing one of the case's topic terms).
  This is NOT a relevance judgment; there are no human relevance labels.
* answer side (is the report grounded?): completion, citation precision (verified / (verified + rejected)),
  unverified numbers, findings by support level, insufficient-evidence flag vs expectation, injection checks.

Results are written to reports/eval/ as JSON with ids and counts only; review texts are not stored.

Usage (PowerShell):
    python scripts/eval_agent.py --dry-run
    python scripts/eval_agent.py                       # all cases, default model
    python scripts/eval_agent.py --cases injection_resistance --model claude-sonnet-5-5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from review_intel.agent import AgentConfig, AgentError, AgentReport, run_agent  # noqa: E402
from review_intel.agent.toolbox import CORPUS_ALIASES, server_parameters  # noqa: E402
from review_intel.utils import PROJECT_ROOT, REPORTS_DIR, setup_logging, utc_now, write_json  # noqa: E402

CASES = PROJECT_ROOT / "data" / "eval" / "agent_cases.jsonl"


def load_cases(path: Path, only: set[str] | None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        for key in ("id", "question", "corpus", "expect"):
            if key not in case:
                raise ValueError(f"case {case.get('id', '?')} is missing {key!r}")
    if only:
        unknown = only - {c["id"] for c in cases}
        if unknown:
            raise ValueError(f"unknown case ids: {sorted(unknown)}")
        cases = [c for c in cases if c["id"] in only]
    return cases


def corpus_path(corpus: str) -> Path | None:
    value = CORPUS_ALIASES.get(corpus, corpus)
    if value == "demo":
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def score(case: dict[str, Any], report: AgentReport) -> dict[str, Any]:
    expect = case["expect"]
    v = report.validation
    tools = [t.name for t in report.tools_used]
    verified, rejected = len(v.verified_review_ids), len(v.rejected_citations)
    terms = [t.casefold() for t in case.get("topic_terms", [])]
    excerpts = [e.excerpt.casefold() for e in report.evidence]
    checks: dict[str, bool] = {
        "expected_tool_used": any(t in tools for t in expect.get("tools_any", [])) if expect.get("tools_any") else True,
        "min_verified_citations": verified >= expect.get("min_verified_citations", 0),
        "no_rejected_citations": rejected == 0,
    }
    if expect.get("insufficient_evidence") is not None:
        checks["insufficient_evidence_flag"] = report.insufficient_evidence == expect["insufficient_evidence"]
    if expect.get("forbidden_strings"):
        dumped = report.model_dump_json()
        checks["no_forbidden_strings"] = not any(s in dumped for s in expect["forbidden_strings"])
    if expect.get("forbidden_review_ids"):
        checks["no_forbidden_review_ids"] = not set(expect["forbidden_review_ids"]) & set(v.verified_review_ids)
    if expect.get("forbidden_summary_phrases"):
        claims = (report.executive_summary + " " + " ".join(f.statement for f in report.key_findings)).casefold()
        checks["no_forbidden_claims"] = not any(p.casefold() in claims for p in expect["forbidden_summary_phrases"])
    run = report.run
    return {
        "id": case["id"], "category": case.get("category"), "status": "completed",
        "retrieval": {
            "tools_called": tools,
            "reviews_seen": v.reviews_seen,
            "excerpts_quoted": len(excerpts),
            "topicality_proxy": (round(sum(any(t in e for t in terms) for e in excerpts) / len(excerpts), 3)
                                 if terms and excerpts else None),
        },
        "answer": {
            "verified_citations": verified,
            "rejected_citations": rejected,
            "citation_precision": round(verified / (verified + rejected), 3) if verified + rejected else None,
            "unverified_numbers": len(v.unverified_numbers),
            "findings_by_support": dict(Counter(f.support for f in report.key_findings)),
            "insufficient_evidence": report.insufficient_evidence,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "cost": {"turns": run.turns, "tool_calls": run.tool_calls, "input_tokens": run.input_tokens,
                 "cache_read_tokens": run.cache_read_tokens, "output_tokens": run.output_tokens,
                 "estimated_cost_usd": run.estimated_cost_usd, "duration_s": run.duration_s, "trace": run.trace_file},
    }


async def run_case(case: dict[str, Any], config: AgentConfig) -> dict[str, Any]:
    environ = {k: v for k, v in os.environ.items() if not k.startswith("REVIEW_ANALYZER_")}
    environ.update(case.get("server_env", {}))
    target = server_parameters(corpus=case["corpus"], environ=environ)
    try:
        report = await run_agent(case["question"], config, server_target=target)
    except AgentError as exc:
        return {"id": case["id"], "category": case.get("category"), "status": "error", "error_code": exc.code,
                "error": str(exc)[:500], "passed": False}
    return score(case, report)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    done = [r for r in results if r["status"] == "completed"]
    verified = sum(r["answer"]["verified_citations"] for r in done)
    rejected = sum(r["answer"]["rejected_citations"] for r in done)
    costs = [r["cost"]["estimated_cost_usd"] for r in done if r["cost"]["estimated_cost_usd"] is not None]
    return {
        "cases": len(results),
        "completed": len(done),
        "passed_all_checks": sum(r["passed"] for r in results),
        "citation_precision": round(verified / (verified + rejected), 3) if verified + rejected else None,
        "unverified_numbers": sum(r["answer"]["unverified_numbers"] for r in done),
        "estimated_cost_usd": round(sum(costs), 4) if costs else None,
        "mean_duration_s": round(sum(r["cost"]["duration_s"] for r in done) / len(done), 1) if done else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", help="Only run these case ids")
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path, help="Output JSON (default reports/eval/agent_eval_<utc>.json)")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases and corpora; no API calls")
    args = parser.parse_args()
    setup_logging()

    cases = load_cases(CASES, set(args.cases) if args.cases else None)
    missing = [(c["id"], str(p)) for c in cases if (p := corpus_path(c["corpus"])) is not None and not p.exists()]
    for case_id, path in missing:
        print(f"case {case_id}: corpus not found: {path}", file=sys.stderr)
    if args.dry_run:
        print(f"{len(cases)} cases valid; {len(missing)} with a missing corpus.")
        return 1 if missing else 0
    if missing:
        return 2
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY is not set; the evaluation makes real (paid) API calls.", file=sys.stderr)
        return 2

    config = AgentConfig.from_env(model=args.model)
    results = []
    for case in cases:
        print(f"running {case['id']} ...", file=sys.stderr)
        results.append(asyncio.run(run_case(case, config)))
        r = results[-1]
        print(f"  {r['status']}, passed={r['passed']}" + (f", {r.get('error_code')}" if r["status"] == "error" else ""),
              file=sys.stderr)
    output = {"created_utc": utc_now(), "model": config.model, "effort": config.effort,
              "cases_file": "data/eval/agent_cases.jsonl", "summary": summarize(results), "results": results,
              "notes": ["Small fixed question set, single run per case: results vary between runs.",
                        "topicality_proxy is lexical, not a relevance judgment; no human relevance labels exist.",
                        "Review texts are not stored; ids and counts only."]}
    out = args.out or REPORTS_DIR / "eval" / f"agent_eval_{utc_now()[:19].replace(':', '').replace('-', '')}.json"
    write_json(out if out.is_absolute() else PROJECT_ROOT / out, output)
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
