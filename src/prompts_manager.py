"""Prepare quality evidence and invoke the managed LangChain prompt chains."""

from __future__ import annotations

import json
import math

import pandas as pd
from langchain_core.language_models import BaseChatModel

from src.prompt_chains import (
    finding_explanation_chain,
    report_summary_chain,
    validation_plan_chain,
)
from src.prompt_schemas import (
    FindingExplanation,
    QualityCheckPlan,
    QualityReportSummary,
)
from src.quality_engine import (
    SUPPORTED_CHECKS,
    QualityCheck,
    QualityFinding,
    QualityReport,
)

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
    def propose_quality_checks(
        llm: BaseChatModel, data: pd.DataFrame
    ) -> QualityCheckPlan:
        """Ask the model to select supported checks using aggregate column metadata."""
        column_profile: list[dict[str, object]] = []
        for column in data.columns:
            values = data[column]
            non_missing = values.dropna()
            numeric = pd.to_numeric(
                non_missing.astype("string").str.replace(",", "", regex=False),
                errors="coerce",
            )
            numeric_values = numeric.dropna()
            profile: dict[str, object] = {
                "name": str(column),
                "dtype": str(values.dtype),
                "row_count": len(data),
                "non_missing_count": int(values.notna().sum()),
                "distinct_non_missing_count": int(non_missing.nunique()),
                "numeric_parseable_count": len(numeric_values),
            }
            finite_values = [
                float(value)
                for value in numeric_values
                if math.isfinite(float(value))
            ]
            if finite_values:
                profile["numeric_minimum"] = min(finite_values)
                profile["numeric_maximum"] = max(finite_values)
            column_profile.append(profile)

        catalog = [
            {"rule_id": rule_id, "description": description}
            for rule_id, description in SUPPORTED_CHECKS.items()
        ]
        return validation_plan_chain(llm).invoke(
            {
                "dataset_profile_json": json.dumps(
                    {"row_count": len(data), "columns": column_profile},
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                "check_catalog_json": json.dumps(catalog, ensure_ascii=False),
            }
        )

    @staticmethod
    def fallback_quality_checks(data: pd.DataFrame) -> QualityCheckPlan:
        """Create a small local-only plan when the model output is unusable."""
        columns = [str(column) for column in data.columns]
        checks = []
        if columns:
            checks.append(
                {
                    "rule_id": "missing_values",
                    "label": "Missing or blank values",
                    "columns": columns,
                    "rationale": (
                        "Local fallback: check all columns for missing or blank "
                        "cells; this conservative check is independent of column meaning."
                    ),
                }
            )
            checks.append(
                {
                    "rule_id": "exact_duplicates",
                    "label": "Exact duplicate rows",
                    "columns": columns,
                    "rationale": (
                        "Local fallback: identify rows identical across all columns; "
                        "no rows will be removed."
                    ),
                }
            )

        return QualityCheckPlan.model_validate(
            {
                "summary": (
                    "The model response could not be used. These conservative "
                    "checks were created locally and still require your approval."
                ),
                "checks": checks,
            }
        )

    @staticmethod
    def validate_quality_plan(
        plan: QualityCheckPlan, data: pd.DataFrame
    ) -> tuple[QualityCheck, ...]:
        """Validate model-selected columns and translate proposals to engine checks."""
        available_columns = set(map(str, data.columns))
        checks: list[QualityCheck] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        seen_labels: set[str] = set()
        for proposal in plan.checks:
            unknown_columns = set(proposal.columns).difference(available_columns)
            if unknown_columns:
                raise ValueError(
                    "The proposed plan referenced columns not in this dataset: "
                    f"{', '.join(sorted(unknown_columns))}"
                )
            signature = (proposal.rule_id, tuple(proposal.columns))
            if signature in seen:
                raise ValueError(
                    f"The proposed plan contains a duplicate check for "
                    f"{', '.join(proposal.columns)}."
                )
            seen.add(signature)
            normalized_label = proposal.label.casefold()
            if normalized_label in seen_labels:
                raise ValueError(
                    f"The proposed plan contains duplicate check labels: "
                    f"{proposal.label!r}"
                )
            seen_labels.add(normalized_label)
            checks.append(
                QualityCheck(
                    rule_id=proposal.rule_id,
                    label=proposal.label,
                    columns=tuple(proposal.columns),
                    rationale=proposal.rationale,
                    minimum=proposal.minimum,
                    maximum=proposal.maximum,
                )
            )
        return tuple(checks)

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
