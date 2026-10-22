
import pandas as pd


def clean_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Trim text whitespace and remove exact duplicate rows."""

    cleaned_df = df.copy()

    text_columns = cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].map(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

    rows_before = len(cleaned_df)

    cleaned_df = cleaned_df.drop_duplicates().reset_index(
        drop=True
    )

    rows_removed = rows_before - len(cleaned_df)

    return cleaned_df, rows_removed
