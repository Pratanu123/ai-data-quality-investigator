from pathlib import Path
from typing import Literal

import pandas as pd
from streamlit.testing.v1 import AppTest

import src.llm as llm_module
from src.prompts_manager import FindingExplanation
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


def _click_key(app: AppTest, widget_type: str, key: str) -> None:
    widget = next(item for item in app.get(widget_type) if item.key == key)
    widget.click().run(timeout=30)


def test_app_uses_only_the_bundled_amazon_dataset():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert not app.get("selectbox")
    assert not app.get("file_uploader")
    assert not any(button.label == "Generate AI check plan" for button in app.button)
    assert any("amazon.csv" in caption.value for caption in app.caption)


def test_quality_engine_runs_directly_without_an_ai_check_plan():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    _click_key(app, "button", "run_quality_checks")

    report = app.session_state["quality_report"]
    amazon_data = pd.read_csv(
        Path(__file__).resolve().parents[1] / "data" / "amazon.csv"
    )
    assert not app.exception
    assert report.summary.rows == len(amazon_data)
    assert "quality_check_plan" not in app.session_state
    assert "approved_quality_checks" not in app.session_state


def test_selecting_missing_values_shows_inline_details_without_an_issue_table():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    _click_key(app, "button", "run_quality_checks")
    app.get("segmented_control")[0].set_value("Investigate").run(timeout=30)

    assert not app.exception
    assert len(app.dataframe) == 0
    assert any(
        markdown.value == "#### Missing and blank values" for markdown in app.markdown
    )
    assert any("Field:**" in markdown.value for markdown in app.markdown)
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
        "Run the Amazon product quality engine" in message.value
        for message in app.info
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
