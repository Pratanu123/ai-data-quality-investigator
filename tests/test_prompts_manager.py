import json

import pandas as pd
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.prompts_manager import FindingExplanation, PromptsManager, QualityReportSummary
from src.prompt_templates import (
    PROMPT_VERSION,
    finding_explanation_prompt,
    report_summary_prompt,
)
from src.quality_engine import QualityFinding, QualityIssue, analyze_dataset


def test_finding_prompt_returns_validated_explanation():
    llm = FakeListChatModel(
        responses=[
            (
                '{"summary":"One rating is malformed.",'
                '"explanation":"The rating value is not numeric.",'
                '"recommended_fix":"Review the source rating and correct or quarantine it.",'
                '"priority":"Medium"}'
            )
        ]
    )
    finding = QualityFinding(
        rule="Invalid numeric values",
        affected_rows=1,
        details="1 invalid numeric values",
        rows=(8,),
        issues=(
            QualityIssue(
                source_row=8,
                reason="rating='five' is not a valid number",
                source_values={
                    "product_id": "P-1",
                    "product_name": "Sample",
                    "rating": "five",
                    "review_content": "Private text should not be included",
                },
            ),
        ),
    )

    result = PromptsManager.explain_finding(llm, finding)

    assert isinstance(result, FindingExplanation)
    assert result.priority == "Medium"
    assert result.recommended_fix.startswith("Review the source rating")


def test_report_prompt_returns_structured_summary_without_row_examples():
    llm = FakeListChatModel(
        responses=[
            (
                '{"overall_summary":"The dataset has a small number of findings.",'
                '"key_risks":["Some numeric values need review."],'
                '"recommended_priorities":["Correct invalid numeric values."]}'
            )
        ]
    )
    report = analyze_dataset(
        pd.DataFrame(
            {
                "product_id": ["P-1"],
                "rating": ["bad"],
                "rating_count": ["20"],
            }
        )
    )

    result = PromptsManager.summarize_report(llm, report)

    assert isinstance(result, QualityReportSummary)
    assert result.key_risks == ["Some numeric values need review."]


def test_finding_and_report_templates_are_separately_managed_and_versioned():
    finding_prompt = finding_explanation_prompt("finding response schema")
    report_prompt = report_summary_prompt("report response schema")

    assert PROMPT_VERSION == "1.4"
    assert finding_prompt.messages[1].prompt.template != report_prompt.messages[1].prompt.template
    assert "finding_json" in finding_prompt.input_variables
    assert "report_json" in report_prompt.input_variables


def test_finding_explanation_sends_only_relevant_checked_fields(monkeypatch):
    import src.prompts_manager as prompts_module

    captured: dict[str, str] = {}

    class RecordingChain:
        def invoke(self, values: dict[str, str]) -> FindingExplanation:
            captured.update(values)
            return FindingExplanation(
                summary="The score is invalid.",
                explanation="The score is not numeric.",
                recommended_fix="Check the original source.",
                priority="Medium",
            )

    def recording_chain(llm):
        assert isinstance(llm, FakeListChatModel)
        return RecordingChain()

    monkeypatch.setattr(
        prompts_module, "finding_explanation_chain", recording_chain
    )
    finding = QualityFinding(
        rule="Invalid score",
        affected_rows=1,
        details="1 invalid score",
        issues=(
            QualityIssue(
                source_row=1,
                reason="score='bad' is not numeric",
                source_values={"score": "bad", "customer_name": "PRIVATE NAME"},
            ),
        ),
        columns=("score",),
    )

    PromptsManager.explain_finding(FakeListChatModel(responses=["unused"]), finding)
    context = json.loads(captured["finding_json"])

    assert context["examples"][0]["values"] == {"score": "bad"}
    assert "PRIVATE NAME" not in captured["finding_json"]
