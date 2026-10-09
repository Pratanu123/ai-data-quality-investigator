"""Prepare quality evidence and invoke the managed LangChain prompt chains."""

from __future__ import annotations

import json

import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.prompt_chains import (
    finding_explanation_chain,
    report_summary_chain,
)
from src.prompt_schemas import (
    FindingExplanation,
    QualityReportSummary,
)
from src.quality_engine import QualityFinding, QualityReport

MAX_EVIDENCE_ISSUES = 5
IDENTITY_FIELDS = ("product_id", "product_name")

RULE_EVIDENCE_FIELDS = {
    "Invalid numeric values": (
        "discounted_price",
        "actual_price",
        "discount_percentage",
        "rating",
        "rating_count",
    ),
    "Ratings outside the 0-5 range": ("rating",),
    "Discount percentages outside the 0-100 range": ("discount_percentage",),
    "Zero or negative prices": ("discounted_price", "actual_price"),
    "Discounted prices exceeding actual prices": ("discounted_price", "actual_price"),
    "Inconsistent discount calculations": (
        "discounted_price",
        "actual_price",
        "discount_percentage",
    ),
    "Missing product identifiers": ("product_id",),
    "Repeated product IDs (reported separately for investigation)": ("product_id",),
}


def _json_safe(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_blank_or_missing(value: object) -> bool:
    if value is None or pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


class PromptsManager:
    """Build bounded prompt inputs and invoke versioned structured chains."""

    @staticmethod
    def explain_finding(
        llm: BaseChatModel, finding: QualityFinding
    ) -> FindingExplanation:
        relevant_fields = set(finding.columns)
        if not finding.columns:
            relevant_fields.update(IDENTITY_FIELDS)
            relevant_fields.update(RULE_EVIDENCE_FIELDS.get(finding.rule, ()))
        examples = []
        for issue in finding.issues[:MAX_EVIDENCE_ISSUES]:
            if finding.rule == "Missing and blank values" and not finding.columns:
                evidence_fields = [
                    column
                    for column, value in issue.source_values.items()
                    if _is_blank_or_missing(value)
                ]
            else:
                evidence_fields = [
                    field
                    for field in relevant_fields
                    if field in issue.source_values
                ]
            examples.append(
                {
                    "source_row": issue.source_row,
                    "reason": issue.reason,
                    "values": {
                        field: _json_safe(issue.source_values[field])
                        for field in evidence_fields
                        if field in issue.source_values
                    },
                }
            )

        finding_context = {
            "rule": finding.rule,
            "affected_rows": finding.affected_rows,
            "deterministic_details": finding.details,
            "sample_size": len(examples),
            "sample_limit": MAX_EVIDENCE_ISSUES,
            "examples": examples,
        }
        return finding_explanation_chain(llm).invoke(
            {
                "finding_json": json.dumps(
                    finding_context,
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                )
            }
        )

    @staticmethod
    def summarize_report(
        llm: BaseChatModel, report: QualityReport
    ) -> QualityReportSummary:
        report_context = {
            "dataset_rows": report.summary.rows,
            "dataset_columns": report.summary.columns,
            "findings": [
                {
                    "rule": finding.rule,
                    "affected_rows": finding.affected_rows,
                    "details": finding.details,
                }
                for finding in report.findings
            ],
        }
        return report_summary_chain(llm).invoke(
            {"report_json": json.dumps(report_context, ensure_ascii=False)}
        )
