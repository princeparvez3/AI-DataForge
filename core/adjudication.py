
import pandas as pd


def normalize_label(value) -> str | None:
    """Normalize labels for comparison."""
    if pd.isna(value):
        return None

    label = str(value).strip()

    return label.casefold() if label else None


def find_disagreements(
    df: pd.DataFrame,
    annotator_a_column: str,
    annotator_b_column: str,
    text_column: str | None = None,
) -> pd.DataFrame:
    """
    Return records where two annotators supplied different,
    non-empty labels.

    row_position identifies the record's position in the
    original DataFrame so the final decision can be saved safely.
    """

    if annotator_a_column == annotator_b_column:
        raise ValueError("Choose two different annotator columns.")

    required_columns = [
        annotator_a_column,
        annotator_b_column,
    ]

    if text_column:
        required_columns.append(text_column)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found: {', '.join(missing_columns)}"
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "row_position",
                "Text",
                "Annotator A Label",
                "Annotator B Label",
                "Final Label",
            ]
        )

    rows = []

    for position in range(len(df)):
        row = df.iloc[position]

        label_a = row[annotator_a_column]
        label_b = row[annotator_b_column]

        normalized_a = normalize_label(label_a)
        normalized_b = normalize_label(label_b)

        # Skip records where either annotator did not supply a label.
        if normalized_a is None or normalized_b is None:
            continue

        # Ignore differences in case and surrounding whitespace.
        if normalized_a == normalized_b:
            continue

        text = ""

        if text_column:
            raw_text = row[text_column]
            text = "" if pd.isna(raw_text) else str(raw_text)

        rows.append(
            {
                "row_position": position,
                "Text": text,
                "Annotator A Label": str(label_a).strip(),
                "Annotator B Label": str(label_b).strip(),
                "Final Label": "",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "row_position",
            "Text",
            "Annotator A Label",
            "Annotator B Label",
            "Final Label",
        ],
    )


def apply_adjudications(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    final_label_column: str,
    allowed_labels: list[str],
) -> tuple[pd.DataFrame, int]:
    """Apply reviewed final labels to a copy of the original dataset."""

    column_name = final_label_column.strip()

    if not column_name:
        raise ValueError("Final label column name cannot be empty.")

    if "Final Label" not in decisions.columns:
        raise ValueError("The Final Label column is missing.")

    if "row_position" not in decisions.columns:
        raise ValueError("The row_position column is missing.")

    valid_labels = {
        label.strip().casefold(): label.strip()
        for label in allowed_labels
        if isinstance(label, str) and label.strip()
    }

    if not valid_labels:
        raise ValueError("At least one allowed final label is required.")

    result = df.copy()

    if column_name not in result.columns:
        result[column_name] = ""

    saved_count = 0
    seen_positions = set()

    for _, decision in decisions.iterrows():
        raw_final_label = decision["Final Label"]

        if pd.isna(raw_final_label):
            continue

        final_label = str(raw_final_label).strip()

        # A blank decision remains unresolved.
        if not final_label:
            continue

        normalized_final = final_label.casefold()

        if normalized_final not in valid_labels:
            raise ValueError(
                f"Invalid final label: {final_label}"
            )

        try:
            position = int(decision["row_position"])
        except (TypeError, ValueError):
            raise ValueError("A row position is invalid.") from None

        if position < 0 or position >= len(result):
            raise ValueError(
                f"Row position {position} is outside the dataset."
            )

        if position in seen_positions:
            raise ValueError(
                f"Duplicate adjudication for row position {position}."
            )

        seen_positions.add(position)

        # Store the canonical form of the allowed label.
        result.iat[
            position,
            result.columns.get_loc(column_name),
        ] = valid_labels[normalized_final]

        saved_count += 1

    return result, saved_count

