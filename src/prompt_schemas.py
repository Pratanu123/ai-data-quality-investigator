"""Pydantic response schemas for data-quality AI workflows."""

from typing import Literal

from pydantic import BaseModel, Field


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
