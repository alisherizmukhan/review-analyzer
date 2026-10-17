"""Report schemas and the citation check.

Two stages:

* ``ReportDraft`` is what the model must return (its JSON Schema is sent as the structured-output format).
  The model labels every finding as a ``retrieved_fact`` or its own ``interpretation``.
* ``finalize`` checks the draft against the ``EvidenceIndex`` (what the model actually saw) and builds an
  ``AgentReport``: unknown review ids are removed and listed as rejected citations, excerpts must occur in
  the cited review's text, and numbers are checked against numbers in the tool results.
  ``AgentReport`` itself refuses to validate if any remaining citation is not in the verified set.

Number checks are a heuristic (a value counts as verified if some tool result contains the same number,
within rounding, or its fraction/percent form). They catch invented statistics, not every misattribution.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from review_intel.agent.evidence import EvidenceIndex

SentimentLabel = Literal["positive", "negative", "neutral", "unknown"]
SentimentSourceName = Literal["star_rating_label", "model_prediction", "not_applicable"]
Support = Literal["verified", "partially_verified", "unsupported", "interpretation"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# What the model returns
# ---------------------------------------------------------------------------


class DraftFinding(_Strict):
    statement: str = Field(description="One finding, in plain language.")
    kind: Literal["retrieved_fact", "interpretation"] = Field(
        description="retrieved_fact: directly stated by tool results. interpretation: your own reading of them.")
    supporting_review_ids: list[int] = Field(description="Ids of reviews (from tool results) that support it.")
    source_tools: list[str] = Field(description="Tools whose results the finding is based on.")


class DraftStatistic(_Strict):
    product: str = Field(description="Product id, or 'all' for the whole corpus.")
    metric: str = Field(description="What the number is, e.g. 'negative share (star-rating labels)'.")
    value: float
    unit: Literal["count", "percent", "fraction", "rating", "other"]
    sentiment_source: SentimentSourceName
    source_tool: str


class DraftAspect(_Strict):
    aspect: str
    product: str
    mentions: int | None
    negative_mentions: int | None
    negative_share: float | None
    supporting_review_ids: list[int]


class DraftEvidence(_Strict):
    review_id: int
    product: str
    excerpt: str = Field(description="Short verbatim excerpt copied from the tool result for this review.")
    sentiment: SentimentLabel
    sentiment_source: SentimentSourceName


class ReportDraft(_Strict):
    query: str
    executive_summary: str
    key_findings: list[DraftFinding]
    statistics: list[DraftStatistic]
    negative_aspects: list[DraftAspect]
    evidence: list[DraftEvidence]
    limitations: list[str]
    insufficient_evidence: bool = Field(
        description="True if the tool results do not contain enough evidence to answer the question.")


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic schema -> structured-output schema: refs inlined, every object closed and fully required."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, list):
            return [resolve(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]]))
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in ("title", "default"):
                continue
            if key == "properties":
                out[key] = {name: resolve(prop) for name, prop in value.items()}
            else:
                out[key] = resolve(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return resolve(schema)


# ---------------------------------------------------------------------------
# Validated report
# ---------------------------------------------------------------------------


class Finding(_Strict):
    statement: str
    kind: Literal["retrieved_fact", "interpretation"]
    support: Support
    supporting_review_ids: list[int]
    rejected_review_ids: list[int]
    unverified_numbers: list[str]
    source_tools: list[str]


class Statistic(DraftStatistic):
    value_verified: bool
    product_verified: bool


class Aspect(DraftAspect):
    numbers_verified: bool
    rejected_review_ids: list[int]


class Evidence(DraftEvidence):
    pass


class RejectedCitation(_Strict):
    review_id: int
    location: str
    reason: str


class Validation(_Strict):
    verified_review_ids: list[int]
    rejected_citations: list[RejectedCitation]
    unverified_numbers: list[str]
    all_citations_verified: bool
    reviews_seen: int


class ToolUsage(_Strict):
    name: str
    calls: int
    errors: int
    total_duration_s: float


class RunInfo(_Strict):
    run_id: str
    model: str
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost_usd: float | None
    api_retries: int
    duration_s: float
    trace_file: str | None


class AgentReport(_Strict):
    query: str
    executive_summary: str
    key_findings: list[Finding]
    statistics: list[Statistic]
    negative_aspects: list[Aspect]
    evidence: list[Evidence]
    limitations: list[str]
    insufficient_evidence: bool
    tools_used: list[ToolUsage]
    validation: Validation
    run: RunInfo

    @model_validator(mode="after")
    def _citations_are_verified(self) -> AgentReport:
        verified = set(self.validation.verified_review_ids)
        cited = {i for f in self.key_findings for i in f.supporting_review_ids}
        cited |= {i for a in self.negative_aspects for i in a.supporting_review_ids}
        cited |= {e.review_id for e in self.evidence}
        unverified = cited - verified
        if unverified:
            raise ValueError(f"report cites review ids that were not verified: {sorted(unverified)}")
        return self


# ---------------------------------------------------------------------------
# Citation check
# ---------------------------------------------------------------------------

# Numbers written in prose ("13.1%", "1,000 reviews", "4.5 stars"); not digits inside ids like B07VTK654B.
_NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\w)|(?<![\w.])\d+(?:\.\d+)?(?!\w)")
UNKNOWN_ID = "review id does not occur in any tool result seen in this run"
BAD_EXCERPT = "excerpt does not occur in the text the tools returned for this review"


def numbers_in(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text)]


def finalize(draft: ReportDraft, query: str, index: EvidenceIndex, tools_used: list[ToolUsage],
             run: RunInfo) -> AgentReport:
    rejected: list[RejectedCitation] = []
    unverified_numbers: list[str] = []
    verified_ids: set[int] = set()

    def check_ids(ids: list[int], location: str) -> tuple[list[int], list[int]]:
        ok, bad = [], []
        for review_id in dict.fromkeys(ids):
            (ok if index.has_review(review_id) else bad).append(review_id)
        for review_id in bad:
            rejected.append(RejectedCitation(review_id=review_id, location=location, reason=UNKNOWN_ID))
        verified_ids.update(ok)
        return ok, bad

    findings = []
    for n, item in enumerate(draft.key_findings):
        ok, bad = check_ids(item.supporting_review_ids, f"key_findings[{n}]")
        missing = [v for v in numbers_in(item.statement) if not index.number_seen(float(v.replace(",", "")))]
        unverified_numbers.extend(f"key_findings[{n}]: {v}" for v in missing)
        if item.kind == "interpretation":
            support: Support = "interpretation"
        elif bad and not ok:
            support = "unsupported"
        elif bad or missing:
            support = "partially_verified" if (ok or len(missing) < len(numbers_in(item.statement))) \
                else "unsupported"
        elif ok or numbers_in(item.statement):
            support = "verified"
        else:  # a "fact" with no review id and no checkable number has nothing to verify it against
            support = "unsupported"
        findings.append(Finding(statement=item.statement, kind=item.kind, support=support,
                                supporting_review_ids=ok, rejected_review_ids=bad, unverified_numbers=missing,
                                source_tools=item.source_tools))

    statistics = []
    for n, stat in enumerate(draft.statistics):
        value_ok = index.number_seen(stat.value)
        if not value_ok:
            unverified_numbers.append(f"statistics[{n}] {stat.metric}: {stat.value}")
        statistics.append(Statistic(**stat.model_dump(), value_verified=value_ok,
                                    product_verified=index.product_seen(stat.product)))

    aspects = []
    for n, aspect in enumerate(draft.negative_aspects):
        ok, bad = check_ids(aspect.supporting_review_ids, f"negative_aspects[{n}]")
        values = [v for v in (aspect.mentions, aspect.negative_mentions, aspect.negative_share) if v is not None]
        missing = [v for v in values if not index.number_seen(float(v))]
        unverified_numbers.extend(f"negative_aspects[{n}] {aspect.aspect}: {v}" for v in missing)
        aspects.append(Aspect(**{**aspect.model_dump(), "supporting_review_ids": ok}, numbers_verified=not missing,
                              rejected_review_ids=bad))

    evidence = []
    for n, item in enumerate(draft.evidence):
        location = f"evidence[{n}]"
        if not index.has_review(item.review_id):
            rejected.append(RejectedCitation(review_id=item.review_id, location=location, reason=UNKNOWN_ID))
        elif not index.excerpt_matches(item.review_id, item.excerpt):
            rejected.append(RejectedCitation(review_id=item.review_id, location=location, reason=BAD_EXCERPT))
        else:
            verified_ids.add(item.review_id)
            evidence.append(Evidence(**item.model_dump()))

    validation = Validation(verified_review_ids=sorted(verified_ids), rejected_citations=rejected,
                            unverified_numbers=unverified_numbers, all_citations_verified=not rejected,
                            reviews_seen=len(index.reviews))
    return AgentReport(query=query, executive_summary=draft.executive_summary, key_findings=findings,
                       statistics=statistics, negative_aspects=aspects, evidence=evidence,
                       limitations=draft.limitations, insufficient_evidence=draft.insufficient_evidence,
                       tools_used=tools_used, validation=validation, run=run)


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------

_MARK = {"verified": "[verified]", "partially_verified": "[partially verified]", "unsupported": "[UNSUPPORTED]",
         "interpretation": "[interpretation]"}


def render_text(report: AgentReport) -> str:
    lines = [f"Question: {report.query}", "", "Summary", f"  {report.executive_summary}", ""]
    if report.insufficient_evidence:
        lines += ["  NOTE: the agent reported insufficient evidence for a full answer.", ""]
    if report.key_findings:
        lines.append("Key findings")
        for f in report.key_findings:
            ids = f" (reviews {', '.join(map(str, f.supporting_review_ids))})" if f.supporting_review_ids else ""
            lines.append(f"  - {_MARK[f.support]} {f.statement}{ids}")
        lines.append("")
    if report.statistics:
        lines.append("Statistics (from tool results)")
        for s in report.statistics:
            flag = "" if s.value_verified else "  [number not found in tool results]"
            lines.append(f"  - {s.product}: {s.metric} = {s.value:g} {s.unit} [{s.sentiment_source}]{flag}")
        lines.append("")
    if report.negative_aspects:
        lines.append("Negative aspects (model predictions on sentences)")
        for a in report.negative_aspects:
            share = f", negative share {a.negative_share:g}" if a.negative_share is not None else ""
            lines.append(f"  - {a.aspect} ({a.product}): {a.negative_mentions} negative of {a.mentions} "
                         f"mentions{share}")
        lines.append("")
    if report.evidence:
        lines.append("Evidence (verified excerpts)")
        for e in report.evidence:
            lines.append(f"  - review {e.review_id} [{e.product}, {e.sentiment} via {e.sentiment_source}]: "
                         f"\"{e.excerpt}\"")
        lines.append("")
    if report.limitations:
        lines.append("Limitations")
        lines += [f"  - {item}" for item in report.limitations]
        lines.append("")
    v = report.validation
    lines.append(f"Citation check: {len(v.verified_review_ids)} review ids verified, "
                 f"{len(v.rejected_citations)} rejected, {len(v.unverified_numbers)} unverified numbers "
                 f"({v.reviews_seen} reviews seen in tool results).")
    for r in v.rejected_citations:
        lines.append(f"  rejected: review {r.review_id} at {r.location}: {r.reason}")
    for item in v.unverified_numbers:
        lines.append(f"  unverified number: {item}")
    tools = ", ".join(f"{t.name} ({t.calls} call{'s' if t.calls != 1 else ''}"
                      + (f", {t.errors} failed)" if t.errors else ")") for t in report.tools_used)
    run = report.run
    cost = f"~${run.estimated_cost_usd:.4f}" if run.estimated_cost_usd is not None else "n/a"
    lines += ["", f"Tools used: {tools or 'none'}",
              f"Run {run.run_id}: {run.model}, {run.turns} model turns, {run.tool_calls} tool calls, "
              f"{run.input_tokens + run.cache_read_tokens + run.cache_write_tokens} input / {run.output_tokens} "
              f"output tokens, est. cost {cost}, {run.duration_s:.1f}s"
              + (f", trace {run.trace_file}" if run.trace_file else "")]
    return "\n".join(lines)
