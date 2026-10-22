
import pandas as pd


def get_quality_report(df: pd.DataFrame) -> dict:
    """Return basic data quality statistics."""

    missing_by_column = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }

    duplicate_rows = int(df.duplicated().sum())

    string_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    preferred_names = [
        "text",
        "message",
        "content",
        "prompt",
        "query",
    ]

    text_column = next(
        (
            column
            for name in preferred_names
            for column in string_columns
            if str(column).lower() == name
        ),
        string_columns[0] if string_columns else None,
    )

    repeated_texts = 0

    if text_column is not None:
        normalized_text = (
            df[text_column]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

        valid_text = normalized_text[
            normalized_text.notna()
            & normalized_text.ne("")
        ]

        repeated_texts = int(valid_text.duplicated().sum())

    return {
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicate_rows,
        "text_column": text_column,
        "repeated_texts": repeated_texts,
    }
