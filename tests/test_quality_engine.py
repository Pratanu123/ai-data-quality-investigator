import pandas as pd
import pytest

from src.quality_engine import (
    VALIDATION_RULES,
    QualityCheck,
    analyze_dataset,
    summarize_dataset,
)


def test_summary_counts_missing_blanks_and_exact_duplicates():
    data = pd.DataFrame(
        {
            "product_id": ["A", "B", "B"],
            "product_name": ["Item", " ", "Other"],
            "rating_count": [10, None, 20],
        }
    )

    summary = summarize_dataset(data)

    assert (summary.rows, summary.columns) == (3, 3)
    assert summary.missing_cells == 1
    assert summary.blank_cells == 1
    assert summary.exact_duplicate_rows == 0


def test_engine_reports_requested_rules_without_mutating_input():
    data = pd.DataFrame(
        {
            "product_id": ["A", "A", " "],
            "discounted_price": ["₹80", "₹120", "0"],
            "actual_price": ["₹100", "₹100", "₹100"],
            "discount_percentage": ["20%", "20%", "120%"],
            "rating": ["4.5", "6", "—"],
            "rating_count": ["10", "bad", "20"],
            "product_name": ["Item", None, "Other"],
        }
    )
    original = data.copy(deep=True)

    report = analyze_dataset(data)
    findings = {finding.rule: finding for finding in report.findings}

    assert tuple(findings) == VALIDATION_RULES
    assert findings["Missing and blank values"].affected_rows == 2
    assert findings["Invalid numeric values"].affected_rows == 2
    assert findings["Ratings outside the 0-5 range"].rows == (2,)
    assert findings["Discount percentages outside the 0-100 range"].rows == (3,)
    assert findings["Zero or negative prices"].rows == (3,)
    assert findings["Discounted prices exceeding actual prices"].rows == (2,)
    assert findings["Inconsistent discount calculations"].rows == (2,)
    assert findings["Missing product identifiers"].rows == (3,)
    assert findings["Repeated product IDs (reported separately for investigation)"].rows == (
        1,
        2,
    )
    invalid_numeric = findings["Invalid numeric values"]
    assert invalid_numeric.issues[0].source_row == 2
    assert invalid_numeric.issues[0].source_values["rating_count"] == "bad"
    assert invalid_numeric.issues[0].reason == "rating_count='bad' is not a valid number"
    assert invalid_numeric.issues[1].source_row == 3
    assert invalid_numeric.issues[1].reason == "rating='—' is not a valid number"
    assert findings["Discounted prices exceeding actual prices"].issues[0].reason == (
        "discounted_price='₹120' exceeds actual_price='₹100'"
    )
    assert findings[
        "Repeated product IDs (reported separately for investigation)"
    ].issues[0].reason == "product_id='A' appears in source rows 1, 2"
    pd.testing.assert_frame_equal(data, original)


def test_engine_handles_empty_dataset():
    report = analyze_dataset(pd.DataFrame())

    assert report.summary.rows == 0
    assert all(finding.affected_rows == 0 for finding in report.findings)


def test_duplicate_row_finding_exposes_both_source_records():
    duplicate = {"product_id": "A", "product_name": "Item"}
    report = analyze_dataset(pd.DataFrame([duplicate, duplicate]))
    finding = next(
        finding for finding in report.findings if finding.rule == "Exact duplicate rows"
    )

    assert finding.rows == (1, 2)
    assert finding.issues[0].source_values == duplicate
    assert finding.issues[0].reason == "All column values exactly match another source row."


def test_numeric_parser_does_not_strip_arbitrary_text_into_a_valid_number():
    report = analyze_dataset(
        pd.DataFrame(
            {
                "product_id": ["A"],
                "rating": ["5stars"],
                "actual_price": ["₹1,000"],
                "discounted_price": ["₹800"],
                "discount_percentage": ["20%"],
                "rating_count": ["100"],
            }
        )
    )
    finding = next(
        finding
        for finding in report.findings
        if finding.rule == "Invalid numeric values"
    )

    assert finding.affected_rows == 1
    assert finding.issues[0].reason == "rating='5stars' is not a valid number"


def test_selected_checks_are_column_scoped_and_do_not_mutate_data():
    data = pd.DataFrame(
        {
            "score": ["2", "7", "unknown", ""],
            "customer": ["A", "A", "B", None],
            "unreviewed": [None, " ", None, None],
        }
    )
    original = data.copy(deep=True)
    checks = (
        QualityCheck(
            rule_id="numeric_range",
            label="Score from 0 to 5",
            columns=("score",),
            rationale="Scores should use this scale.",
            minimum=0,
            maximum=5,
        ),
        QualityCheck(
            rule_id="repeated_values",
            label="Repeated customers",
            columns=("customer",),
            rationale="Review repeated names.",
        ),
    )

    report = analyze_dataset(data, checks)
    findings = {finding.rule: finding for finding in report.findings}

    assert tuple(findings) == ("Score from 0 to 5", "Repeated customers")
    assert findings["Score from 0 to 5"].rows == (2,)
    assert findings["Score from 0 to 5"].columns == ("score",)
    assert findings["Repeated customers"].rows == (1, 2)
    assert "not treated as an error" in findings["Repeated customers"].issues[0].reason
    pd.testing.assert_frame_equal(data, original)


def test_selected_check_rejects_unknown_fields_and_unsafe_ranges():
    data = pd.DataFrame({"score": [1, 2]})
    with pytest.raises(ValueError, match="unknown columns"):
        analyze_dataset(
            data,
            (
                QualityCheck(
                    rule_id="missing_values",
                    label="Missing",
                    columns=("not_a_column",),
                    rationale="test",
                ),
            ),
        )
    with pytest.raises(ValueError, match="invalid numeric bounds"):
        analyze_dataset(
            data,
            (
                QualityCheck(
                    rule_id="numeric_range",
                    label="Invalid bounds",
                    columns=("score",),
                    rationale="test",
                    minimum=5,
                    maximum=2,
                ),
            ),
        )


def test_numeric_range_supports_a_maximum_without_a_minimum():
    data = pd.DataFrame({"amount": [1, 10, 11]})
    report = analyze_dataset(
        data,
        (
            QualityCheck(
                rule_id="numeric_range",
                label="Amount maximum",
                columns=("amount",),
                rationale="Values should not exceed ten.",
                maximum=10,
            ),
        ),
    )

    assert report.findings[0].rows == (3,)


def test_positive_numeric_accepts_zero_minimum_and_rejects_other_bounds():
    data = pd.DataFrame({"amount": [0, 1, -1]})
    accepted = QualityCheck(
        rule_id="positive_numeric",
        label="Positive amount",
        columns=("amount",),
        rationale="Amounts must be positive.",
        minimum=0,
    )
    report = analyze_dataset(data, (accepted,))
    assert report.findings[0].rows == (1, 3)

    rejected = QualityCheck(
        rule_id="positive_numeric",
        label="Positive amount",
        columns=("amount",),
        rationale="Amounts must be positive.",
        minimum=1,
    )
    with pytest.raises(ValueError, match="optional minimum of 0"):
        analyze_dataset(data, (rejected,))


def test_grouped_checks_run_once_for_multiple_fields_and_explain_each_field():
    data = pd.DataFrame(
        {
            "Store": ["bad", "2", "2"],
            "Weekly_Sales": ["100", "-1", "bad"],
            "Holiday_Flag": ["0", "1", "2"],
        }
    )
    checks = (
        QualityCheck(
            rule_id="invalid_numeric",
            label="Invalid numeric values",
            columns=("Store", "Weekly_Sales", "Holiday_Flag"),
            rationale="Check all numeric fields.",
        ),
        QualityCheck(
            rule_id="positive_numeric",
            label="Positive numeric values",
            columns=("Weekly_Sales", "Holiday_Flag"),
            rationale="Check metrics for zero and negative values.",
        ),
        QualityCheck(
            rule_id="repeated_values",
            label="Repeated values review",
            columns=("Store", "Holiday_Flag"),
            rationale="Review repetition only.",
        ),
    )

    report = analyze_dataset(data, checks)
    findings = {finding.rule: finding for finding in report.findings}

    assert findings["Invalid numeric values"].rows == (1, 3)
    assert "Store='bad'" in findings["Invalid numeric values"].issues[0].reason
    assert "Weekly_Sales='bad'" in findings["Invalid numeric values"].issues[1].reason
    assert findings["Positive numeric values"].rows == (1, 2)
    assert "Holiday_Flag='0'" in findings["Positive numeric values"].issues[0].reason
    assert "Weekly_Sales='-1'" in findings["Positive numeric values"].issues[1].reason
    assert findings["Repeated values review"].rows == (2, 3)
    assert "source rows 2, 3" in findings["Repeated values review"].issues[0].reason
