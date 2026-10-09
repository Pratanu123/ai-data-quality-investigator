import json
import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.prompts_manager import (
    FindingExplanation,
    PromptsManager,
    QualityReportSummary,
)
from src.prompt_schemas import QualityCheckPlan, QualityCheckProposal
from src.prompt_chains import validation_plan_chain
from src.quality_engine import QualityFinding, QualityIssue, analyze_dataset
from src.prompt_templates import (
    PROMPT_VERSION,
    dataset_quality_checks_prompt,
    finding_explanation_prompt,
    report_summary_prompt,
)


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
    plan_prompt = dataset_quality_checks_prompt("plan response schema")

    assert PROMPT_VERSION == "1.4"
    assert finding_prompt.messages[1].prompt.template != report_prompt.messages[1].prompt.template
    assert "finding_json" in finding_prompt.input_variables
    assert "report_json" in report_prompt.input_variables
    assert "dataset_profile_json" in plan_prompt.input_variables
    assert "exactly one complete JSON object" in plan_prompt.messages[0].prompt.template


def test_plan_schema_requires_supported_range_bounds():
    with pytest.raises(ValueError, match="requires a minimum or maximum"):
        QualityCheckProposal(
            rule_id="numeric_range",
            label="Rating range",
            columns=["rating"],
            rationale="Ratings usually have bounds.",
        )

    plan = QualityCheckPlan(
        summary="Check rating bounds and missing data.",
        checks=[
            QualityCheckProposal(
                rule_id="numeric_range",
                label="Rating range",
                columns=["rating"],
                rationale="The rating field is likely bounded.",
                minimum=0,
                maximum=5,
            )
        ],
    )
    assert plan.checks[0].maximum == 5
    assert QualityCheckPlan(summary="No checks fit the schema.", checks=[]).checks == []

    grouped_plan = QualityCheckPlan.model_validate(
        {
            "summary": "Check numeric fields and repeated values.",
            "checks": [
                {
                    "rule_id": "invalid_numeric",
                    "label": "Invalid numeric values",
                    "columns": ["Store", "Weekly_Sales", "Holiday_Flag"],
                    "rationale": "Check all inferred numeric fields.",
                },
                {
                    "rule_id": "positive_numeric",
                    "label": "Positive numeric values",
                    "columns": ["Weekly_Sales", "Fuel_Price"],
                    "rationale": "These measures should be positive.",
                },
                {
                    "rule_id": "repeated_values",
                    "label": "Repeated values review",
                    "columns": ["Store", "Holiday_Flag"],
                    "rationale": "Repeated values are review-only signals.",
                },
            ],
        }
    )
    assert [len(check.columns) for check in grouped_plan.checks] == [3, 2, 2]


def test_plan_schema_accepts_zero_minimum_for_positive_numeric():
    raw_plan = (
        '{"summary":"Check for positive values.",'
        '"checks":[{"rule_id":"positive_numeric",'
        '"label":"Fuel Price Must Be Positive","columns":["Fuel_Price"],'
        '"rationale":"Negative fuel prices are not realistic.",'
        '"minimum":0}]}'
    )

    plan = QualityCheckPlan.model_validate_json(raw_plan)

    assert plan.checks[0].rule_id == "positive_numeric"
    assert plan.checks[0].minimum == 0


def test_validation_plan_chain_returns_structured_supported_check():
    llm = FakeListChatModel(
        responses=[
            (
                '{"summary":"Check score values and missing entries.",'
                '"checks":[{"rule_id":"numeric_range",'
                '"label":"Scores outside expected range","columns":["score"],'
                '"rationale":"The score field is bounded from zero to ten.",'
                '"minimum":0,"maximum":10}]}'
            )
        ]
    )

    plan = PromptsManager.propose_quality_checks(
        llm,
        pd.DataFrame({"score": [3, 11], "customer": ["private", "values"]}),
    )

    assert isinstance(plan, QualityCheckPlan)
    assert plan.checks[0].rule_id == "numeric_range"
    assert plan.checks[0].columns == ["score"]
    checks = PromptsManager.validate_quality_plan(
        plan, pd.DataFrame({"score": [3, 11], "customer": ["private", "values"]})
    )
    assert checks[0].maximum == 10


def test_validation_plan_chain_requests_json_mode_and_caps_output_tokens():
    model = FakeListChatModel(responses=['{"summary":"No checks.","checks":[]}'])

    chain = validation_plan_chain(model)
    bound_model = chain.steps[1]

    assert bound_model.kwargs["response_format"] == {"type": "json_object"}
    assert bound_model.kwargs["max_tokens"] == 1200
    assert bound_model.kwargs["extra_body"] == {
        "reasoning": {"effort": "minimal"}
    }


def test_plan_validation_rejects_nonexistent_dataset_columns():
    plan = QualityCheckPlan(
        summary="Check a field.",
        checks=[
            QualityCheckProposal(
                rule_id="missing_values",
                label="Missing identifiers",
                columns=["not_present"],
                rationale="Check for gaps.",
            )
        ],
    )

    with pytest.raises(ValueError, match="not in this dataset"):
        PromptsManager.validate_quality_plan(plan, pd.DataFrame({"id": ["a"]}))


def test_plan_request_contains_only_aggregate_metadata(monkeypatch):
    import src.prompts_manager as prompts_module

    captured: dict[str, str] = {}
    expected_plan = QualityCheckPlan(
        summary="Check the score field.",
        checks=[
            QualityCheckProposal(
                rule_id="invalid_numeric",
                label="Invalid score values",
                columns=["score"],
                rationale="The score is inferred as numeric.",
            )
        ],
    )

    class RecordingChain:
        def invoke(self, values: dict[str, str]) -> QualityCheckPlan:
            captured.update(values)
            return expected_plan

    def recording_chain(llm):
        assert isinstance(llm, FakeListChatModel)
        return RecordingChain()

    monkeypatch.setattr(
        prompts_module, "validation_plan_chain", recording_chain
    )

    data = pd.DataFrame(
        {"score": ["1", "not numeric"], "private_note": ["SENSITIVE_CUSTOMER", "other"]}
    )
    result = PromptsManager.propose_quality_checks(
        FakeListChatModel(responses=["unused"]), data
    )

    assert result is expected_plan
    assert "SENSITIVE_CUSTOMER" not in captured["dataset_profile_json"]
    assert '"name": "private_note"' in captured["dataset_profile_json"]


def test_generic_finding_explanation_sends_only_approved_check_fields(monkeypatch):
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


def test_local_fallback_plan_is_small_and_uses_only_safe_checks():
    data = pd.DataFrame(
        {
            "Weekly_Sales": [1, 2, 3],
            "Store": ["A", "B", "C"],
        }
    )

    plan = PromptsManager.fallback_quality_checks(data)

    assert len(plan.checks) == 2
    assert [check.rule_id for check in plan.checks] == [
        "missing_values",
        "exact_duplicates",
    ]
    assert all(check.columns == ["Weekly_Sales", "Store"] for check in plan.checks)
