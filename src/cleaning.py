"""Conservative cleaning proposals with explicit, non-mutating approval."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.quality_engine import NUMERIC_COLUMNS, QualityCheck, parse_numeric_values


@dataclass(frozen=True)
class CleaningProposal:
    source_row: int
    column: str
    before: object
    after: object
    reason: str


def propose_cleaning(
    data: pd.DataFrame, checks: tuple[QualityCheck, ...] | None = None
) -> tuple[CleaningProposal, ...]:
    """Suggest deterministic cell-level changes without modifying the input."""
    numeric_columns = (
        set(NUMERIC_COLUMNS)
        if checks is None
        else {
            column
            for check in checks
            if check.rule_id in {"invalid_numeric", "numeric_range", "positive_numeric"}
            for column in check.columns
        }
    )
    protected_columns = (
        {"product_id"}
        if checks is None
        else {
            column
            for check in checks
            if check.rule_id == "repeated_values"
            for column in check.columns
        }
    )
    parsed_numeric = {
        column: parse_numeric_values(data, column)
        for column in numeric_columns
        if column in data.columns
    }
    proposals: list[CleaningProposal] = []

    for position in range(len(data)):
        for column_position, column in enumerate(data.columns):
            value = data.iat[position, column_position]
            if not isinstance(value, str):
                continue

            trimmed_value = value.strip()
            if not trimmed_value:
                proposals.append(
                    CleaningProposal(
                        source_row=position + 1,
                        column=str(column),
                        before=value,
                        after=None,
                        reason="Normalize a blank string to a missing value; no value is inferred.",
                    )
                )
            elif (
                column in parsed_numeric
                and pd.isna(parsed_numeric[column].iloc[position])
            ):
                proposals.append(
                    CleaningProposal(
                        source_row=position + 1,
                        column=str(column),
                        before=value,
                        after=None,
                        reason=(
                            f"{column}={value!r} is not parseable as a number; "
                            "replace it with a missing value for review."
                        ),
                    )
                )
            elif str(column) not in protected_columns and value != trimmed_value:
                proposals.append(
                    CleaningProposal(
                        source_row=position + 1,
                        column=str(column),
                        before=value,
                        after=trimmed_value,
                        reason="Remove leading and trailing whitespace.",
                    )
                )

    return tuple(proposals)


def apply_cleaning_proposals(
    data: pd.DataFrame, approved: tuple[CleaningProposal, ...]
) -> pd.DataFrame:
    """Apply only approved proposals to a copy of the original DataFrame."""
    cleaned = data.copy(deep=True)
    column_positions = {str(column): index for index, column in enumerate(data.columns)}
    columns_to_copy: set[int] = set()
    cells: set[tuple[int, int]] = set()

    for proposal in approved:
        row_position = proposal.source_row - 1
        if not 0 <= row_position < len(data):
            raise ValueError(f"Invalid source row in cleaning proposal: {proposal.source_row}")
        if proposal.column not in column_positions:
            raise ValueError(f"Unknown column in cleaning proposal: {proposal.column}")
        column_position = column_positions[proposal.column]
        cell = (row_position, column_position)
        if cell in cells:
            raise ValueError(
                f"More than one cleaning proposal targets row {proposal.source_row}, "
                f"column {proposal.column}."
            )
        cells.add(cell)

        current_value = data.iat[row_position, column_position]
        if pd.isna(current_value) or current_value != proposal.before:
            raise ValueError(
                f"Source value changed for row {proposal.source_row}, "
                f"column {proposal.column}; rerun validation before approving."
            )
        columns_to_copy.add(column_position)

    for column_position in columns_to_copy:
        cleaned.iloc[:, column_position] = cleaned.iloc[:, column_position].astype(object)

    for proposal in approved:
        cleaned.iat[proposal.source_row - 1, column_positions[proposal.column]] = (
            proposal.after
        )

    return cleaned
