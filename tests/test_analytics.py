import pandas as pd

from core.analytics import get_annotation_metrics, summarize_ai_review_history


def test_annotation_metrics_count_annotated_pending_and_labels():
    df = pd.DataFrame(
        {
            "text": ["Hello", "Bye", "Thanks", "Waiting"],
            "annotation": ["Positive", "Negative", "Positive", ""],
        }
    )

    metrics = get_annotation_metrics(
        df,
        annotation_column="annotation",
        text_column="text",
        expected_labels=["Positive", "Negative", "Neutral"],
    )

    assert metrics["total_records"] == 4
    assert metrics["annotated_records"] == 3
    assert metrics["pending_records"] == 1
    assert metrics["completion_rate"] == 75.0
    assert metrics["label_counts"] == {"Positive": 2, "Negative": 1}
    assert metrics["invalid_labels"] == 0


def test_annotation_metrics_find_blank_and_repeated_text_and_invalid_labels():
    df = pd.DataFrame(
        {
            "text": ["same", " SAME ", "", None],
            "annotation": ["Positive", "Unknown", "", None],
        }
    )

    metrics = get_annotation_metrics(
        df,
        annotation_column="annotation",
        text_column="text",
        expected_labels=["Positive", "Negative"],
    )

    assert metrics["blank_text_records"] == 2
    assert metrics["repeated_text_values"] == 1
    assert metrics["invalid_labels"] == 1


def test_ai_review_summary_counts_decisions_and_rates():
    history = [
        {"decision": "accepted"},
        {"decision": "accepted"},
        {"decision": "corrected"},
        {"decision": "rejected"},
        {"decision": "pending"},
    ]

    summary = summarize_ai_review_history(history)

    assert summary["total_reviewed"] == 4
    assert summary["accepted"] == 2
    assert summary["corrected"] == 1
    assert summary["rejected"] == 1
    assert summary["acceptance_rate"] == 50.0
    assert summary["correction_rate"] == 25.0
    assert summary["rejection_rate"] == 25.0


def test_ai_review_summary_empty_history_has_no_rate():
    summary = summarize_ai_review_history([])

    assert summary["total_reviewed"] == 0
    assert summary["acceptance_rate"] is None
