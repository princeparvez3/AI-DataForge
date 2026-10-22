
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def compare_annotations(
    df: pd.DataFrame,
    annotator_a_column: str,
    annotator_b_column: str,
) -> dict:
    """Compare two annotators and calculate agreement metrics."""

    if annotator_a_column == annotator_b_column:
        raise ValueError("Choose two different annotator columns.")

    missing_columns = [
        column
        for column in (annotator_a_column, annotator_b_column)
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found: {', '.join(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    def normalize_label(value):
        if pd.isna(value):
            return None

        label = str(value).strip()

        if not label:
            return None

        return label.casefold()

    labels_a_all = df[annotator_a_column].map(normalize_label)
    labels_b_all = df[annotator_b_column].map(normalize_label)

    valid_mask = labels_a_all.notna() & labels_b_all.notna()

    labels_a = labels_a_all[valid_mask].tolist()
    labels_b = labels_b_all[valid_mask].tolist()

    if not labels_a:
        raise ValueError(
            "No records have labels from both annotators."
        )

    all_labels = sorted(set(labels_a) | set(labels_b))

    total_compared = len(labels_a)

    matching_count = sum(
        label_a == label_b
        for label_a, label_b in zip(labels_a, labels_b)
    )

    agreement_percentage = (
        matching_count / total_compared
    ) * 100

    # Kappa is undefined when only one label category is present.
    # Handle that case directly instead of calling sklearn and
    # generating avoidable warnings.
    if len(all_labels) < 2:
        kappa = None

        matrix = pd.DataFrame(
            [[total_compared]],
            index=all_labels,
            columns=all_labels,
        )
    else:
        kappa_value = cohen_kappa_score(
            labels_a,
            labels_b,
            labels=all_labels,
        )

        kappa = (
            round(float(kappa_value), 4)
            if pd.notna(kappa_value)
            else None
        )

        matrix_values = confusion_matrix(
            labels_a,
            labels_b,
            labels=all_labels,
        )

        matrix = pd.DataFrame(
            matrix_values,
            index=all_labels,
            columns=all_labels,
        )

    matrix.index.name = "Annotator A"
    matrix.columns.name = "Annotator B"

    return {
        "total_records": len(df),
        "compared_records": total_compared,
        "excluded_records": len(df) - total_compared,
        "matching_records": matching_count,
        "disagreeing_records": total_compared - matching_count,
        "agreement_percentage": round(agreement_percentage, 2),
        "cohen_kappa": kappa,
        "labels": all_labels,
        "confusion_matrix": matrix,
    }
