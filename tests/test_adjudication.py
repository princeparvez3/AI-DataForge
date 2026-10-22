
import pandas as pd
import pytest

from core.adjudication import (
    apply_adjudications,
    find_disagreements,
)


def test_find_disagreements_returns_only_conflicting_labels():
    df = pd.DataFrame(
        {
            "text": ["Text A", "Text B", "Text C", "Text D"],
            "annotator_a": ["Positive", "Negative", "Neutral", "Positive"],
            "annotator_b": ["positive", "Positive", "", "Negative"],
        }
    )

    result = find_disagreements(
        df,
        "annotator_a",
        "annotator_b",
        "text",
    )

    assert result["row_position"].tolist() == [1, 3]
    assert result["Text"].tolist() == ["Text B", "Text D"]
    assert len(result) == 2


def test_find_disagreements_ignores_blank_labels():
    df = pd.DataFrame(
        {
            "annotator_a": ["Positive", "", None],
            "annotator_b": ["Negative", "Positive", "Neutral"],
        }
    )

    result = find_disagreements(
        df,
        "annotator_a",
        "annotator_b",
    )

    assert result["row_position"].tolist() == [0]


def test_find_disagreements_rejects_same_column():
    df = pd.DataFrame(
        {
            "annotator_a": ["Positive"],
        }
    )

    with pytest.raises(ValueError, match="different"):
        find_disagreements(
            df,
            "annotator_a",
            "annotator_a",
        )


def test_apply_adjudications_saves_final_labels():
    df = pd.DataFrame(
        {
            "text": ["Text A", "Text B", "Text C"],
            "annotator_a": ["Positive", "Negative", "Neutral"],
            "annotator_b": ["Negative", "Negative", "Positive"],
        }
    )

    decisions = pd.DataFrame(
        {
            "row_position": [0, 2],
            "Final Label": ["Negative", "Positive"],
        }
    )

    updated, saved_count = apply_adjudications(
        df,
        decisions,
        "final_label",
        ["Positive", "Negative", "Neutral"],
    )

    assert saved_count == 2
    assert updated["final_label"].tolist() == [
        "Negative",
        "",
        "Positive",
    ]

    # Original dataset must remain unchanged.
    assert "final_label" not in df.columns


def test_apply_adjudications_rejects_invalid_label():
    df = pd.DataFrame(
        {
            "text": ["Text A"],
        }
    )

    decisions = pd.DataFrame(
        {
            "row_position": [0],
            "Final Label": ["Unknown"],
        }
    )

    with pytest.raises(ValueError, match="Invalid final label"):
        apply_adjudications(
            df,
            decisions,
            "final_label",
            ["Positive", "Negative"],
        )


def test_apply_adjudications_rejects_duplicate_positions():
    df = pd.DataFrame(
        {
            "text": ["Text A", "Text B"],
        }
    )

    decisions = pd.DataFrame(
        {
            "row_position": [0, 0],
            "Final Label": ["Positive", "Negative"],
        }
    )

    with pytest.raises(ValueError, match="Duplicate adjudication"):
        apply_adjudications(
            df,
            decisions,
            "final_label",
            ["Positive", "Negative"],
        )

