from pathlib import Path
from typing import Literal, NoReturn

import pandas as pd
from openai import LengthFinishReasonError
from openai.types.chat import ChatCompletion
from streamlit.testing.v1 import AppTest

import src.llm as llm_module
from src.prompt_schemas import QualityCheckPlan, QualityCheckProposal
from src.prompts_manager import FindingExplanation, PromptsManager
from src.quality_engine import QualityCheck, analyze_dataset


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _approved_check(
    rule_id: Literal[
        "missing_values",
        "exact_duplicates",
        "invalid_numeric",
        "numeric_range",
        "positive_numeric",
        "repeated_values",
    ],
    label: str,
    columns: tuple[str, ...],
) -> QualityCheck:
    return QualityCheck(
        rule_id=rule_id,
        label=label,
        columns=columns,
        rationale="Test-approved deterministic rule.",
    )


def _seed_approved_report(
    app: AppTest, data: pd.DataFrame, checks: tuple[QualityCheck, ...]
) -> None:
    app.session_state["quality_report"] = analyze_dataset(data, checks)
    app.session_state["approved_quality_checks"] = checks


def _plan_stub(plan: QualityCheckPlan):
    def propose(llm, data):
        assert llm is not None
        assert isinstance(data, pd.DataFrame)
        return plan

    return propose


def _click_key(app: AppTest, widget_type: str, key: str) -> None:
    widget = next(item for item in app.get(widget_type) if item.key == key)
    widget.click().run(timeout=30)


def test_dataset_picker_shows_available_csv_and_upload_option():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert "Upload a CSV…" in app.get("selectbox")[0].options
    assert any(Path(option).name == "amazon.csv" for option in app.get("selectbox")[0].options)
    assert any("amazon.csv" in text.value for text in app.caption)


def test_upload_mode_displays_csv_uploader():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("selectbox")[0].set_value("Upload a CSV…").run(timeout=30)

    assert not app.exception
    assert len(app.get("file_uploader")) == 1


def test_ai_proposed_plan_requires_explicit_human_approval(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    plan = QualityCheckPlan(
        summary="Check missing ratings.",
        checks=[
            QualityCheckProposal(
                rule_id="missing_values",
                label="Missing ratings",
                columns=["rating"],
                rationale="The rating column may be incomplete.",
            )
        ],
    )
    monkeypatch.setattr(
        PromptsManager,
        "propose_quality_checks",
        staticmethod(_plan_stub(plan)),
    )
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    _click_key(app, "button", "generate_quality_plan")
    assert app.checkbox[0].value is False
    assert "quality_report" not in app.session_state

    app.checkbox[0].set_value(True).run(timeout=30)
    _click_key(app, "button", "approve_and_run_quality_checks")

    report = app.session_state["quality_report"]
    assert not app.exception
    assert len(report.findings) == 1
    assert report.findings[0].rule == "Missing ratings"
    assert report.findings[0].columns == ("rating",)
    assert app.session_state["approved_quality_checks"][0].columns == ("rating",)


def test_truncated_llm_plan_uses_local_fallback_without_retry(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    completion = ChatCompletion(
        id="test-completion",
        choices=[
            {
                "finish_reason": "length",
                "index": 0,
                "message": {"role": "assistant", "content": None},
            }
        ],
        created=1,
        model="test-model",
        object="chat.completion",
    )

    def raise_length_error(*args: object) -> NoReturn:
        assert args[0] is not None
        assert isinstance(args[1], pd.DataFrame)
        raise LengthFinishReasonError(completion=completion)

    monkeypatch.setattr(
        PromptsManager,
        "propose_quality_checks",
        staticmethod(raise_length_error),
    )
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    _click_key(app, "button", "generate_quality_plan")

    assert not app.exception
    assert app.session_state["quality_check_plan_origin"] == "Local fallback"
    assert len(app.session_state["quality_check_candidates"]) == 2
    assert any("No retry was made" in message.value for message in app.warning)
    assert "quality_report" not in app.session_state


def test_selecting_missing_values_shows_inline_details_without_an_issue_table(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    plan = QualityCheckPlan(
        summary="Inspect missing rating counts.",
        checks=[
            QualityCheckProposal(
                rule_id="missing_values",
                label="Missing and blank values",
                columns=["rating_count"],
                rationale="This field contains missing observations.",
            )
        ],
    )
    monkeypatch.setattr(
        PromptsManager,
        "propose_quality_checks",
        staticmethod(_plan_stub(plan)),
    )
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    _click_key(app, "button", "generate_quality_plan")
    app.checkbox[0].set_value(True).run(timeout=30)
    _click_key(app, "button", "approve_and_run_quality_checks")
    app.get("segmented_control")[0].set_value("Investigate").run(timeout=30)

    assert not app.exception
    assert len(app.dataframe) == 0
    assert any(
        markdown.value == "#### Missing and blank values" for markdown in app.markdown
    )
    assert any("Field:** `rating_count`" in markdown.value for markdown in app.markdown)
    assert any("Records 1–2 of 2" in caption.value for caption in app.caption)
    assert any(
        button.label == "Explain finding and recommend a fix" for button in app.button
    )


def test_ai_explanations_stay_disabled_without_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "load_dotenv", lambda: None)
    data = pd.DataFrame({"score": ["not numeric"]})
    checks = (_approved_check("invalid_numeric", "Invalid numeric values", ("score",)),)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    _seed_approved_report(app, data, checks)
    app.session_state["workflow_stage"] = "Investigate"
    app.run(timeout=30)

    assert not app.exception
    ai_buttons = [
        button
        for button in app.button
        if button.label
        in {"Summarize report with AI", "Explain finding and recommend a fix"}
    ]
    assert ai_buttons
    assert all(button.disabled for button in ai_buttons)


def test_cleaning_uses_approved_numeric_fields_and_waits_for_cleaning_approval():
    data = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "amazon.csv")
    checks = (_approved_check("invalid_numeric", "Invalid numeric values", ("rating",)),)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    _seed_approved_report(app, data, checks)
    app.session_state["workflow_stage"] = "Clean data"
    app.run(timeout=30)

    assert not app.exception
    assert len(app.get("download_button")) == 0
    cleaning_preview = next(
        editor.value for editor in app.dataframe if editor.key == "cleaning_preview_editor"
    )
    invalid_rating = cleaning_preview.loc[
        (cleaning_preview["CSV row"] == 1280)
        & (cleaning_preview["Field"] == "rating")
    ].iloc[0]
    assert invalid_rating["Current value"] == "|"
    assert invalid_rating["Proposed value"] == "Missing (blank in CSV)"
    assert not invalid_rating["Approve"]
    _click_key(app, "button", "apply_cleaning")
    assert not app.exception
    assert len(app.get("download_button")) == 0


def test_export_is_gated_until_approved_validation_is_available():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.get("segmented_control")[0].set_value("Export").run(timeout=30)
    assert not app.exception
    assert any(
        "Generate a check plan" in message.value for message in app.info
    )
    assert len(app.get("download_button")) == 0

    data = pd.DataFrame({"id": ["A", "A"]})
    checks = (_approved_check("exact_duplicates", "Duplicate IDs", ("id",)),)
    _seed_approved_report(app, data, checks)
    app.run(timeout=30)
    assert not app.exception
    assert [button.label for button in app.get("download_button")] == [
        "Download findings report"
    ]


def test_ai_priority_and_recommendation_use_pill_badges():
    data = pd.DataFrame({"score": ["bad"]})
    checks = (_approved_check("invalid_numeric", "Invalid numeric values", ("score",)),)
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    _seed_approved_report(app, data, checks)
    app.session_state["workflow_stage"] = "Investigate"
    app.session_state["selected_quality_rule"] = "Invalid numeric values"
    app.session_state["quality_ai_finding_Invalid numeric values"] = FindingExplanation(
        summary="Invalid rating needs review.",
        explanation="The rating could not be read as a number.",
        recommended_fix="Verify the rating against the source before correcting it.",
        priority="Medium",
    )
    app.run(timeout=30)

    rendered_markdown = "\n".join(item.value for item in app.markdown)
    assert not app.exception
    assert "ai-pill-medium" in rendered_markdown
    assert "MEDIUM PRIORITY" in rendered_markdown
    assert "RECOMMENDED FIX" in rendered_markdown
