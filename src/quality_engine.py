"""Deterministic, non-mutating data quality checks for tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

SUPPORTED_CHECKS = {
    "missing_values": "Find missing or blank cells in the selected columns.",
    "exact_duplicates": "Find rows identical across the selected columns.",
    "invalid_numeric": (
        "Find non-empty values in one or more selected columns that cannot be parsed "
        "as numbers. Group columns with the same numeric parsing requirement."
    ),
    "numeric_range": (
        "Find numeric values outside the proposed inclusive minimum/maximum in "
        "exactly one selected column. Propose a separate check for each field."
    ),
    "positive_numeric": (
        "Find numeric values that are zero or negative in one or more selected "
        "columns with the same positivity requirement. An optional minimum of 0 may "
        "be included; do not include a maximum."
    ),
    "repeated_values": (
        "Report repeated non-blank values in one or more selected columns for "
        "investigation; repeated values are not automatically an error."
    ),
}

NUMERIC_COLUMNS = (
    "discounted_price",
    "actual_price",
    "discount_percentage",
    "rating",
    "rating_count",
)

VALIDATION_RULES = (
    "Missing and blank values",
    "Exact duplicate rows",
    "Invalid numeric values",
    "Ratings outside the 0-5 range",
    "Discount percentages outside the 0-100 range",
    "Zero or negative prices",
    "Discounted prices exceeding actual prices",
    "Inconsistent discount calculations",
    "Missing product identifiers",
    "Repeated product IDs (reported separately for investigation)",
)


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    missing_cells: int
    blank_cells: int
    exact_duplicate_rows: int


@dataclass(frozen=True)
class QualityIssue:
    source_row: int
    reason: str
    source_values: dict[str, object]


@dataclass(frozen=True)
class QualityCheck:
    rule_id: Literal[
        "missing_values",
        "exact_duplicates",
        "invalid_numeric",
        "numeric_range",
        "positive_numeric",
        "repeated_values",
    ]
    label: str
    columns: tuple[str, ...]
    rationale: str
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class QualityFinding:
    rule: str
    affected_rows: int
    details: str
    rows: tuple[int, ...] = ()
    issues: tuple[QualityIssue, ...] = ()
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityReport:
    summary: DatasetSummary
    findings: tuple[QualityFinding, ...]


def summarize_dataset(data: pd.DataFrame) -> DatasetSummary:
    """Summarize the input without changing it."""
    text_columns = data.select_dtypes(include=["object", "string"]).columns
    blank_cells = sum(
        int(data[column].astype("string").str.strip().eq("").sum())
        for column in text_columns
    )
    return DatasetSummary(
        rows=len(data),
        columns=len(data.columns),
        missing_cells=int(data.isna().sum().sum()),
        blank_cells=blank_cells,
        exact_duplicate_rows=int(data.duplicated().sum()),
    )


def parse_numeric_values(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(float("nan"), index=data.index, dtype="float64")

    raw_values = data[column].astype("string").str.strip()
    cleaned_values = raw_values.str.replace(",", "", regex=False).str.replace(
        r"[₹$€£¥₩,%\s]", "", regex=True
    )
    return pd.to_numeric(cleaned_values, errors="coerce")


def _make_finding(
    data: pd.DataFrame,
    rule: str,
    mask: pd.Series,
    details: str,
    reasons: dict[int, str],
    columns: tuple[str, ...] = (),
) -> QualityFinding:
    issues = tuple(
        QualityIssue(
            source_row=position + 1,
            reason=reasons[position],
            source_values={
                str(column): data.iloc[position][column] for column in data.columns
            },
        )
        for position, matched in enumerate(mask)
        if matched
    )
    return QualityFinding(
        rule=rule,
        affected_rows=len(issues),
        details=details,
        rows=tuple(issue.source_row for issue in issues),
        issues=issues,
        columns=columns,
    )


def analyze_dataset(
    data: pd.DataFrame, checks: tuple[QualityCheck, ...] | None = None
) -> QualityReport:
    """Return structured findings while preserving the original DataFrame."""
    if checks is not None:
        return analyze_selected_checks(data, checks)

    summary = summarize_dataset(data)
    findings: list[QualityFinding] = []

    missing_mask = data.isna().any(axis=1)
    blank_mask = pd.Series(False, index=data.index)
    missing_or_blank_reasons: dict[int, str] = {}
    for column in data.select_dtypes(include=["object", "string"]).columns:
        column_blank_mask = data[column].astype("string").str.strip().eq("").fillna(False)
        blank_mask |= column_blank_mask
    for position in range(len(data)):
        missing_columns = [
            str(column) for column in data.columns if pd.isna(data.iloc[position][column])
        ]
        blank_columns = [
            str(column)
            for column in data.select_dtypes(include=["object", "string"]).columns
            if data.iloc[position][column] is not None
            and isinstance(data.iloc[position][column], str)
            and not data.iloc[position][column].strip()
        ]
        reasons = []
        if missing_columns:
            reasons.append(f"Missing values in: {', '.join(missing_columns)}")
        if blank_columns:
            reasons.append(f"Blank values in: {', '.join(blank_columns)}")
        missing_or_blank_reasons[position] = "; ".join(reasons)
    missing_or_blank_rows = missing_mask | blank_mask
    missing_or_blank_cells = summary.missing_cells + summary.blank_cells
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[0],
        missing_or_blank_rows,
        f"{missing_or_blank_cells} missing or blank cells",
        missing_or_blank_reasons,
    )
    )

    duplicate_mask = data.duplicated(keep=False)
    duplicate_row_count = int(duplicate_mask.sum())
    duplicate_reasons = {
        position: "All column values exactly match another source row."
        for position, matched in enumerate(duplicate_mask)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[1],
        duplicate_mask,
        (
            f"{duplicate_row_count} rows belong to exact-duplicate groups "
            f"({summary.exact_duplicate_rows} additional duplicate rows)"
        ),
        duplicate_reasons,
    )
    )

    numeric_data = {
        column: parse_numeric_values(data, column) for column in NUMERIC_COLUMNS
    }
    invalid_numeric_mask = pd.Series(False, index=data.index)
    invalid_numeric_values = 0
    invalid_numeric_columns: dict[int, list[str]] = {}
    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            continue
        raw_values = data[column].astype("string").str.strip()
        invalid_mask = raw_values.notna() & raw_values.ne("") & numeric_data[column].isna()
        invalid_numeric_values += int(invalid_mask.sum())
        invalid_numeric_mask |= invalid_mask
        for position, is_invalid in enumerate(invalid_mask):
            if is_invalid:
                invalid_numeric_columns.setdefault(position, []).append(column)
    invalid_numeric_reasons = {
        position: "; ".join(
            f"{column}={data.iloc[position][column]!r} is not a valid number"
            for column in columns
        )
        for position, columns in invalid_numeric_columns.items()
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[2],
        invalid_numeric_mask,
        f"{invalid_numeric_values} invalid numeric values",
        invalid_numeric_reasons,
    )
    )

    ratings = numeric_data["rating"]
    invalid_ratings = ratings.notna() & ~ratings.between(0, 5)
    rating_reasons = {
        position: (
            f"rating={data.iloc[position]['rating']!r} is outside the allowed 0-5 range"
        )
        for position, matched in enumerate(invalid_ratings)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[3],
        invalid_ratings,
        f"{int(invalid_ratings.sum())} ratings outside 0-5",
        rating_reasons,
    )
    )

    percentages = numeric_data["discount_percentage"]
    invalid_percentages = percentages.notna() & ~percentages.between(0, 100)
    percentage_reasons = {
        position: (
            f"discount_percentage={data.iloc[position]['discount_percentage']!r} "
            "is outside 0-100"
        )
        for position, matched in enumerate(invalid_percentages)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[4],
        invalid_percentages,
        f"{int(invalid_percentages.sum())} percentages outside 0-100",
        percentage_reasons,
    )
    )

    discounted_prices = numeric_data["discounted_price"]
    actual_prices = numeric_data["actual_price"]
    invalid_prices = (
        (discounted_prices.notna() & discounted_prices.le(0))
        | (actual_prices.notna() & actual_prices.le(0))
    )
    price_reasons = {
        position: "; ".join(
            f"{column}={data.iloc[position][column]!r} is zero or negative"
            for column, values in (
                ("discounted_price", discounted_prices),
                ("actual_price", actual_prices),
            )
            if pd.notna(values.iloc[position]) and values.iloc[position] <= 0
        )
        for position, matched in enumerate(invalid_prices)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[5],
        invalid_prices,
        f"{int(invalid_prices.sum())} rows with a zero or negative price",
        price_reasons,
    )
    )

    price_order_issues = (
        discounted_prices.notna()
        & actual_prices.notna()
        & discounted_prices.gt(actual_prices)
    )
    price_order_reasons = {
        position: (
            f"discounted_price={data.iloc[position]['discounted_price']!r} exceeds "
            f"actual_price={data.iloc[position]['actual_price']!r}"
        )
        for position, matched in enumerate(price_order_issues)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[6],
        price_order_issues,
        f"{int(price_order_issues.sum())} discounted prices exceed actual prices",
        price_order_reasons,
    )
    )

    usable_discount = (
        actual_prices.gt(0)
        & discounted_prices.ge(0)
        & percentages.between(0, 100)
    )
    calculated_percentages = (actual_prices - discounted_prices) / actual_prices * 100
    discount_issues = usable_discount & (calculated_percentages - percentages).abs().gt(1.1)
    discount_reasons = {
        position: (
            f"Listed discount is "
            f"{data.iloc[position]['discount_percentage']!r}; prices imply "
            f"{calculated_percentages.iloc[position]:.2f}% "
            f"(actual_price={data.iloc[position]['actual_price']!r}, "
            f"discounted_price={data.iloc[position]['discounted_price']!r})"
        )
        for position, matched in enumerate(discount_issues)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[7],
        discount_issues,
        (
            f"{int(discount_issues.sum())} rows where the price-derived discount "
            "differs from the listed percentage by more than 1.1 percentage points"
        ),
        discount_reasons,
    )
    )

    if "product_id" in data.columns:
        product_ids = data["product_id"].astype("string").str.strip()
        missing_ids = product_ids.isna() | product_ids.eq("").fillna(False)
        normalized_ids = product_ids.mask(missing_ids)
        repeated_ids = normalized_ids.notna() & normalized_ids.duplicated(keep=False)
        repeated_id_count = int(normalized_ids[repeated_ids].nunique())
        repeated_id_details = (
            f"{int(repeated_ids.sum())} rows across {repeated_id_count} repeated product IDs"
        )
    else:
        missing_ids = pd.Series(True, index=data.index)
        repeated_ids = pd.Series(False, index=data.index)
        repeated_id_details = "0 repeated product IDs (product_id column is absent)"
    missing_id_reasons = {
        position: (
            f"product_id={data.iloc[position]['product_id']!r} is missing or blank"
            if "product_id" in data.columns
            else "The product_id column is absent."
        )
        for position, matched in enumerate(missing_ids)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[8],
        missing_ids,
        f"{int(missing_ids.sum())} rows with a missing product ID",
        missing_id_reasons,
    )
    )
    repeated_id_rows: dict[int, tuple[int, ...]] = {}
    if "product_id" in data.columns:
        repeated_positions: dict[str, list[int]] = {}
        for position, product_id in enumerate(normalized_ids):
            if pd.notna(product_id) and repeated_ids.iloc[position]:
                repeated_positions.setdefault(str(product_id), []).append(position + 1)
        repeated_id_rows = {
            position - 1: tuple(group_rows)
            for group_rows in repeated_positions.values()
            for position in group_rows
        }
    repeated_id_reasons = {
        position: (
            f"product_id={data.iloc[position]['product_id']!r} appears in source rows "
            f"{', '.join(map(str, repeated_id_rows[position]))}"
        )
        for position, matched in enumerate(repeated_ids)
        if matched
    }
    findings.append(_make_finding(
        data,
        VALIDATION_RULES[9],
        repeated_ids,
        repeated_id_details,
        repeated_id_reasons,
    )
    )

    return QualityReport(summary=summary, findings=tuple(findings))


def analyze_selected_checks(
    data: pd.DataFrame, checks: tuple[QualityCheck, ...]
) -> QualityReport:
    """Run only approved checks from the built-in deterministic rule catalog."""
    summary = summarize_dataset(data)
    findings: list[QualityFinding] = []

    for check in checks:
        unknown_columns = set(check.columns).difference(map(str, data.columns))
        if unknown_columns:
            raise ValueError(
                f"Check {check.label!r} references unknown columns: "
                f"{', '.join(sorted(unknown_columns))}"
            )
        if check.rule_id not in SUPPORTED_CHECKS:
            raise ValueError(f"Unsupported quality check: {check.rule_id}")
        if not check.columns:
            raise ValueError(f"Check {check.label!r} must select at least one column")
        if check.rule_id == "numeric_range" and len(check.columns) != 1:
            raise ValueError(
                f"Check {check.label!r} must select exactly one column because its "
                "numeric bounds apply to a single field"
            )
        if check.rule_id == "numeric_range":
            if check.minimum is None and check.maximum is None:
                raise ValueError(f"Check {check.label!r} must define a numeric bound")
            if (
                check.minimum is not None
                and check.maximum is not None
                and check.minimum >= check.maximum
            ):
                raise ValueError(f"Check {check.label!r} has invalid numeric bounds")
        elif check.rule_id == "positive_numeric":
            if check.minimum not in (None, 0) or check.maximum is not None:
                raise ValueError(
                    f"Check {check.label!r} may only specify an optional minimum of 0"
                )

        selected = data.loc[:, list(check.columns)]
        row_mask = pd.Series(False, index=data.index)
        reasons: dict[int, str] = {}
        details = ""

        if check.rule_id == "missing_values":
            missing = selected.isna()
            blank = pd.DataFrame(False, index=data.index, columns=selected.columns)
            for column in selected.select_dtypes(include=["object", "string"]).columns:
                blank[column] = (
                    selected[column].astype("string").str.strip().eq("").fillna(False)
                )
            row_mask = (missing | blank).any(axis=1)
            cell_count = int((missing | blank).sum().sum())
            details = f"{cell_count} missing or blank cells in selected fields"
            for position in range(len(data)):
                fields = [
                    str(column)
                    for column in selected.columns
                    if missing.iloc[position][column] or blank.iloc[position][column]
                ]
                if fields:
                    reasons[position] = f"Missing or blank values in: {', '.join(fields)}"

        elif check.rule_id == "exact_duplicates":
            row_mask = selected.duplicated(keep=False)
            details = (
                f"{int(row_mask.sum())} rows match across selected fields: "
                f"{', '.join(check.columns)}"
            )
            for position, matched in enumerate(row_mask):
                if matched:
                    reasons[position] = (
                        "Selected column values exactly match another source row."
                    )

        elif check.rule_id == "invalid_numeric":
            invalid_columns: dict[int, list[str]] = {}
            invalid_value_count = 0
            for column in check.columns:
                values = parse_numeric_values(data, column)
                raw = data[column].astype("string").str.strip()
                invalid = raw.notna() & raw.ne("") & values.isna()
                invalid_value_count += int(invalid.sum())
                row_mask |= invalid
                for position, matched in enumerate(invalid):
                    if matched:
                        invalid_columns.setdefault(position, []).append(column)
            details = (
                f"{invalid_value_count} non-empty values across "
                f"{int(row_mask.sum())} rows are not valid numbers"
            )
            reasons = {
                position: "; ".join(
                    f"{column}={data.iloc[position][column]!r} is not a valid number"
                    for column in columns
                )
                for position, columns in invalid_columns.items()
            }

        elif check.rule_id in {"numeric_range", "positive_numeric"}:
            if check.rule_id == "positive_numeric":
                nonpositive_columns: dict[int, list[str]] = {}
                nonpositive_value_count = 0
                for column in check.columns:
                    values = parse_numeric_values(data, column)
                    nonpositive = values.notna() & values.le(0)
                    nonpositive_value_count += int(nonpositive.sum())
                    row_mask |= nonpositive
                    for position, matched in enumerate(nonpositive):
                        if matched:
                            nonpositive_columns.setdefault(position, []).append(column)
                details = (
                    f"{nonpositive_value_count} numeric values across "
                    f"{int(row_mask.sum())} rows are zero or negative"
                )
                reasons = {
                    position: "; ".join(
                        f"{column}={data.iloc[position][column]!r} is not positive"
                        for column in columns
                    )
                    for position, columns in nonpositive_columns.items()
                }
            else:
                column = check.columns[0]
                values = parse_numeric_values(data, column)
                row_mask = pd.Series(False, index=data.index)
                if check.minimum is not None:
                    row_mask |= values.notna() & values.lt(check.minimum)
                if check.maximum is not None:
                    row_mask |= values.notna() & values.gt(check.maximum)
                bounds = []
                if check.minimum is not None:
                    bounds.append(f"minimum {check.minimum:g}")
                if check.maximum is not None:
                    bounds.append(f"maximum {check.maximum:g}")
                details = (
                    f"{int(row_mask.sum())} numeric values outside "
                    f"{' and '.join(bounds)}"
                )
                reasons = {
                    position: (
                        f"{column}={data.iloc[position][column]!r} is outside "
                        f"the inclusive range "
                        f"[{check.minimum if check.minimum is not None else '-∞'}, "
                        f"{check.maximum if check.maximum is not None else '∞'}]"
                    )
                    for position, matched in enumerate(row_mask)
                    if matched
                }

        elif check.rule_id == "repeated_values":
            repeated_column_rows: dict[int, list[str]] = {}
            repeated_group_count = 0
            for column in check.columns:
                values = data[column].astype("string").str.strip()
                nonblank = values.notna() & values.ne("").fillna(False)
                repeated = nonblank & values.duplicated(keep=False)
                repeated_values: dict[str, list[int]] = {}
                for position, value in enumerate(values):
                    if repeated.iloc[position]:
                        repeated_values.setdefault(str(value), []).append(position + 1)
                for position, value in enumerate(values):
                    if repeated.iloc[position]:
                        repeated_column_rows.setdefault(position, []).append(
                            f"{column}={data.iloc[position][column]!r} appears in "
                            f"source rows "
                            f"{', '.join(map(str, repeated_values[str(value)]))}"
                        )
                repeated_group_count += int(values[repeated].nunique())
                row_mask |= repeated
            details = (
                f"{int(row_mask.sum())} rows have repeated values across "
                f"{repeated_group_count} column/value groups; review only"
            )
            reasons = {
                position: (
                    "; ".join(column_reasons)
                    + "; repetition is reported for investigation, not treated as an error"
                )
                for position, column_reasons in repeated_column_rows.items()
            }

        findings.append(
            _make_finding(
                data,
                check.label,
                row_mask,
                details,
                reasons,
                check.columns,
            )
        )

    return QualityReport(summary=summary, findings=tuple(findings))
