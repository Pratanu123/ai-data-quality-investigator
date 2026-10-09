"""Pydantic response schemas for data-quality AI workflows."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FindingExplanation(BaseModel):
    summary: str = Field(description="A concise plain-language description of this finding.")
    explanation: str = Field(description="What the rule detected and why it matters.")
    recommended_fix: str = Field(
        description="A safe, practical remediation recommendation; do not claim to edit data."
    )
    priority: Literal["Critical", "High", "Medium", "Low"] = Field(
        description="Suggested review priority based on the supplied evidence."
    )


class QualityReportSummary(BaseModel):
    overall_summary: str = Field(description="A short evidence-based quality report summary.")
    key_risks: list[str] = Field(description="Important observed risks from the finding counts.")
    recommended_priorities: list[str] = Field(
        description="Ordered review steps based only on the supplied rule counts."
    )


class QualityCheckProposal(BaseModel):
    rule_id: Literal[
        "missing_values",
        "exact_duplicates",
        "invalid_numeric",
        "numeric_range",
        "positive_numeric",
        "repeated_values",
    ]
    label: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[\w][\w .,:()/%-]{0,99}$",
        description="Short user-facing name for this check.",
    )
    columns: list[str] = Field(
        min_length=1,
        description="Exact dataset column names this check will inspect.",
    )
    rationale: str = Field(
        min_length=1,
        max_length=500,
        description="Why this deterministic check suits these columns.",
    )
    minimum: float | None = Field(
        default=None,
        description=(
            "Inclusive lower bound for numeric_range checks. positive_numeric may "
            "optionally specify 0 as its lower bound."
        ),
    )
    maximum: float | None = Field(
        default=None, description="Inclusive upper bound for numeric_range checks."
    )

    @model_validator(mode="before")
    @classmethod
    def require_distinct_columns(cls, values: object) -> object:
        if isinstance(values, dict):
            columns = values.get("columns")
            if (
                isinstance(columns, list)
                and all(isinstance(column, str) for column in columns)
                and len(columns) != len(set(columns))
            ):
                raise ValueError("columns must not contain duplicates")
        return values

    @model_validator(mode="after")
    def validate_rule_parameters(self) -> "QualityCheckProposal":
        if not self.label.strip() or not self.rationale.strip():
            raise ValueError("label and rationale must contain non-whitespace text")
        if self.rule_id == "numeric_range":
            if self.minimum is None and self.maximum is None:
                raise ValueError("numeric_range requires a minimum or maximum")
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum >= self.maximum
            ):
                raise ValueError("numeric_range minimum must be less than maximum")
        elif self.rule_id == "positive_numeric":
            if self.minimum not in (None, 0):
                raise ValueError("positive_numeric may only specify a minimum of 0")
            if self.maximum is not None:
                raise ValueError("positive_numeric does not support a maximum")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError("Only numeric_range checks may define bounds")
        if self.rule_id == "numeric_range":
            if len(self.columns) != 1:
                raise ValueError(f"{self.rule_id} requires exactly one column")
        return self


class QualityCheckPlan(BaseModel):
    summary: str = Field(description="Brief overview of the proposed validation plan.")
    checks: list[QualityCheckProposal] = Field(
        max_length=5,
        description="Specific deterministic checks selected for the dataset, possibly empty."
    )
