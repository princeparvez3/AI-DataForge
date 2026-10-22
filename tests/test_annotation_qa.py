
import pandas as pd
import pytest

from core.annotation_qa import compare_annotations


def test_identical_annotations_have_full_agreement():
    df = pd.DataFrame(
        {
            "annotator_a": ["Positive", "Negative", "Neutral"],
            "annotator_b": ["Positive", "Negative", "Neutral"],
        }
    )

    report = compare_annotations(
        df,
        "annotator_a",
        "annotator_b",
    )

    assert report["total_records"] == 3
    assert report["compared_records"] == 3
    assert report["matching_records"] == 3
    assert report["disagreeing_records"] == 0
    assert report["agreement_percentage"] == 100.0
    assert report["cohen_kappa"] == 1.0


def test_disagreement_is_counted_correctly():
    df = pd.DataFrame(
        {
            "annotator_a": [
                "Positive",
                "Negative",
                "Positive",
                "Neutral",
            ],
            "annotator_b": [
                "Positive",
                "Negative",
                "Negative",
                "Neutral",
            ],
        }
    )

    report = compare_annotations(
        df,
        "annotator_a",
        "annotator_b",
    )

    assert report["compared_records"] == 4
    assert report["matching_records"] == 3
    assert report["disagreeing_records"] == 1
    assert report["agreement_percentage"] == 75.0


def test_labels_ignore_case_and_extra_spaces():
    df = pd.DataFrame(
        {
            "annotator_a": [" Positive ", "Negative"],
            "annotator_b": ["positive", " negative "],
        }
    )

    report = compare_annotations(
        df,
        "annotator_a",
        "annotator_b",
    )

    assert report["matching_records"] == 2
    assert report["agreement_percentage"] == 100.0


def test_missing_labels_are_excluded():
    df = pd.DataFrame(
        {
            "annotator_a": ["Positive", None, "Negative"],
            "annotator_b": ["Positive", "Negative", ""],
        }
    )

    report = compare_annotations(
        df,
        "annotator_a",
        "annotator_b",
    )

    assert report["total_records"] == 3
    assert report["compared_records"] == 1
    assert report["excluded_records"] == 2
    assert report["agreement_percentage"] == 100.0


def test_same_annotator_column_is_rejected():
    df = pd.DataFrame(
        {
            "annotator_a": ["Positive", "Negative"],
        }
    )

    with pytest.raises(ValueError, match="different"):
        compare_annotations(
            df,
            "annotator_a",
            "annotator_a",
        )


def test_missing_column_is_rejected():
    df = pd.DataFrame(
        {
            "annotator_a": ["Positive"],
        }
    )

    with pytest.raises(ValueError, match="not found"):
        compare_annotations(
            df,
            "annotator_a",
            "annotator_b",
        )
