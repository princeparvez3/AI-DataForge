
import pandas as pd

from core.quality_checks import get_quality_report


def test_report_detects_missing_and_repeated_text():
    df = pd.DataFrame(
        {
            "text": [" Hello ", "hello", "", None],
            "label": ["A", "B", "C", "D"],
        }
    )

    report = get_quality_report(df)

    assert report["missing_by_column"]["text"] == 1
    assert report["duplicate_rows"] == 0
    assert report["text_column"] == "text"
    assert report["repeated_texts"] == 1


def test_report_counts_exact_duplicate_rows():
    df = pd.DataFrame(
        {
            "text": ["same", "same"],
            "label": ["A", "A"],
        }
    )

    report = get_quality_report(df)

    assert report["duplicate_rows"] == 1
    assert report["repeated_texts"] == 1


def test_report_handles_dataset_without_text_columns():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "score": [0.5, 0.8],
        }
    )

    report = get_quality_report(df)

    assert report["text_column"] is None
    assert report["repeated_texts"] == 0
