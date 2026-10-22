"""Pure analytics helpers for AI DataForge."""

from __future__ import annotations

import pandas as pd


REVIEW_DECISIONS = {"accepted", "corrected", "rejected"}


def get_annotation_metrics(
    dataframe: pd.DataFrame,
    annotation_column: str,
    text_column: str | None = None,
    expected_labels: list[str] | None = None,
) -> dict:
    """Calculate annotation completion, label distribution, and basic QA metrics."""
    total_records = len(dataframe)

    if annotation_column in dataframe.columns:
        labels = (
            dataframe[annotation_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        labels = pd.Series([""] * total_records, index=dataframe.index, dtype=str)

    has_label = labels.ne("")
    annotated_records = int(has_label.sum())
    label_counts = labels[has_label].value_counts().to_dict()
    pending_records = total_records - annotated_records
    completion_rate = (
        round((annotated_records / total_records) * 100, 2)
        if total_records
        else 0.0
    )

    blank_text_records = 0
    repeated_text_values = 0
    if text_column and text_column in dataframe.columns:
        text_values = (
            dataframe[text_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        blank_text_records = int(text_values.eq("").sum())
        normalized_text = text_values.str.casefold()
        nonempty_text = normalized_text[normalized_text.ne("")]
        repeated_text_values = int(nonempty_text.duplicated().sum())

    invalid_labels = 0
    normalized_expected = {
        str(label).strip().casefold()
        for label in (expected_labels or [])
        if str(label).strip()
    }
    if normalized_expected:
        invalid_labels = int(
            sum(
                1
                for label in labels[has_label].tolist()
                if label.casefold() not in normalized_expected
            )
        )

    return {
        "total_records": total_records,
        "annotated_records": annotated_records,
        "pending_records": pending_records,
        "completion_rate": completion_rate,
        "label_counts": label_counts,
        "blank_text_records": blank_text_records,
        "repeated_text_values": repeated_text_values,
        "invalid_labels": invalid_labels,
    }


def summarize_ai_review_history(history: list[dict] | None) -> dict:
    """Summarize human decisions on saved AI annotation suggestions."""
    records = [
        item
        for item in (history or [])
        if isinstance(item, dict)
        and item.get("decision") in REVIEW_DECISIONS
    ]

    accepted = sum(item["decision"] == "accepted" for item in records)
    corrected = sum(item["decision"] == "corrected" for item in records)
    rejected = sum(item["decision"] == "rejected" for item in records)
    reviewed = len(records)

    return {
        "total_reviewed": reviewed,
        "accepted": accepted,
        "corrected": corrected,
        "rejected": rejected,
        "acceptance_rate": round(accepted / reviewed * 100, 2) if reviewed else None,
        "correction_rate": round(corrected / reviewed * 100, 2) if reviewed else None,
        "rejection_rate": round(rejected / reviewed * 100, 2) if reviewed else None,
        "records": records,
    }
