import pandas as pd
import pytest

from src.cleaning import apply_cleaning_proposals, propose_cleaning


def test_proposals_are_safe_and_do_not_modify_source_data():
    data = pd.DataFrame(
        {
            "product_id": [" P-1 ", "P-1", "P-3"],
            "product_name": [" Item ", " ", "Third"],
            "rating": ["4.5", "|", "5"],
            "rating_count": [10, 20, None],
        }
    )
    original = data.copy(deep=True)

    proposals = propose_cleaning(data)

    assert [
        (proposal.source_row, proposal.column, proposal.before, proposal.after)
        for proposal in proposals
    ] == [
        (1, "product_name", " Item ", "Item"),
        (2, "product_name", " ", None),
        (2, "rating", "|", None),
    ]
    assert not any(proposal.column == "product_id" for proposal in proposals)
    pd.testing.assert_frame_equal(data, original)


def test_only_approved_proposals_are_applied_to_a_separate_copy():
    data = pd.DataFrame(
        {
            "product_id": [" P-1 "],
            "rating": ["|"],
            "product_name": [" Sample "],
        }
    )
    proposals = propose_cleaning(data)
    approved = (proposals[0],)

    cleaned = apply_cleaning_proposals(data, approved)

    assert pd.isna(cleaned.loc[0, "rating"])
    assert cleaned.loc[0, "product_id"] == " P-1 "
    assert cleaned.loc[0, "product_name"] == " Sample "
    assert data.loc[0, "rating"] == "|"


def test_apply_rejects_stale_source_values():
    data = pd.DataFrame({"rating": ["|"]})
    proposal = propose_cleaning(data)[0]
    changed_data = pd.DataFrame({"rating": ["5"]})

    with pytest.raises(ValueError, match="Source value changed"):
        apply_cleaning_proposals(changed_data, (proposal,))
