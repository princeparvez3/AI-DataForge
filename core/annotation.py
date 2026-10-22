
import pandas as pd


def parse_labels(labels_text: str) -> list[str]:
    """Parse comma-separated labels and remove duplicates."""
    if not isinstance(labels_text, str):
        return []

    labels = [
        label.strip()
        for label in labels_text.split(",")
        if label.strip()
    ]

    return list(dict.fromkeys(labels))


def prepare_annotation_dataframe(
    df: pd.DataFrame,
    annotation_column: str,
) -> pd.DataFrame:
    """Create the annotation column when it does not exist."""
    column_name = annotation_column.strip()

    if not column_name:
        raise ValueError("Annotation column name cannot be empty.")

    prepared = df.copy()

    if column_name not in prepared.columns:
        prepared[column_name] = ""

    return prepared


def get_annotation_summary(
    df: pd.DataFrame,
    annotation_column: str,
) -> dict:
    """Count total, annotated, and remaining records."""
    total = len(df)

    if annotation_column not in df.columns:
        return {
            "total": total,
            "annotated": 0,
            "remaining": total,
        }

    values = (
        df[annotation_column]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    annotated = int(values.ne("").sum())

    return {
        "total": total,
        "annotated": annotated,
        "remaining": total - annotated,
    }


def save_annotations(
    df: pd.DataFrame,
    edited_annotations: pd.DataFrame,
    annotation_column: str,
) -> pd.DataFrame:
    """Save edited labels while preserving other columns."""
    if annotation_column not in edited_annotations.columns:
        raise ValueError(
            "Annotation column is missing from edited data."
        )

    result = prepare_annotation_dataframe(
        df,
        annotation_column,
    )

    values = (
        edited_annotations[annotation_column]
        .reindex(result.index)
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result[annotation_column] = values

    return result
