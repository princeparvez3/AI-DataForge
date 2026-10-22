
import pandas as pd
import pytest

from core.data_split import (
    get_leakage_report,
    split_dataset,
)


def test_random_split_preserves_all_records():
    df = pd.DataFrame(
        {
            "text": [f"record {i}" for i in range(10)],
            "label": ["A", "B"] * 5,
        }
    )

    train_df, test_df = split_dataset(
        df,
        test_size=0.2,
        random_state=42,
    )

    assert len(train_df) == 8
    assert len(test_df) == 2
    assert len(train_df) + len(test_df) == len(df)


def test_group_split_keeps_same_group_together():
    df = pd.DataFrame(
        {
            "text": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "id": list(range(8)),
        }
    )

    train_df, test_df = split_dataset(
        df,
        test_size=0.25,
        random_state=42,
        group_column="text",
    )

    train_groups = set(train_df["text"])
    test_groups = set(test_df["text"])

    assert train_groups.isdisjoint(test_groups)
    assert len(train_df) + len(test_df) == len(df)


def test_stratified_split_keeps_both_classes():
    df = pd.DataFrame(
        {
            "text": [f"record {i}" for i in range(20)],
            "label": ["A"] * 10 + ["B"] * 10,
        }
    )

    train_df, test_df = split_dataset(
        df,
        test_size=0.2,
        random_state=42,
        stratify_column="label",
    )

    assert set(train_df["label"]) == {"A", "B"}
    assert set(test_df["label"]) == {"A", "B"}
    assert len(test_df) == 4


def test_group_and_stratification_cannot_be_combined():
    df = pd.DataFrame(
        {
            "text": ["A", "A", "B", "B"],
            "label": ["X", "Y", "X", "Y"],
        }
    )

    with pytest.raises(ValueError):
        split_dataset(
            df,
            group_column="text",
            stratify_column="label",
        )


def test_invalid_test_size_is_rejected():
    df = pd.DataFrame({"text": ["A", "B"]})

    with pytest.raises(ValueError):
        split_dataset(df, test_size=1.0)


def test_leakage_report_detects_shared_text():
    train_df = pd.DataFrame(
        {"text": [" Hello ", "training only"]}
    )

    test_df = pd.DataFrame(
        {"text": ["hello", "testing only"]}
    )

    report = get_leakage_report(
        train_df,
        test_df,
        ["text"],
    )

    assert report["text"]["overlap_count"] == 1
    assert report["text"]["shared_values"] == ["hello"]
