
import pandas as pd

from core.cleaning import clean_dataframe


def test_cleaning_trims_spaces_and_removes_duplicates():
    df = pd.DataFrame(
        {
            "text": [" hello ", "hello", "world"],
            "label": ["A", "A", "B"],
        }
    )

    cleaned_df, removed = clean_dataframe(df)

    expected = pd.DataFrame(
        {
            "text": ["hello", "world"],
            "label": ["A", "B"],
        }
    )

    pd.testing.assert_frame_equal(cleaned_df, expected)
    assert removed == 1


def test_cleaning_does_not_modify_original_dataframe():
    df = pd.DataFrame(
        {
            "text": [" hello ", "world"],
            "label": ["A", "B"],
        }
    )

    original_df = df.copy()

    clean_dataframe(df)

    pd.testing.assert_frame_equal(df, original_df)


def test_cleaning_preserves_missing_values():
    df = pd.DataFrame(
        {
            "text": [" hello ", None, ""],
            "label": ["A", "B", "C"],
        }
    )

    cleaned_df, removed = clean_dataframe(df)

    assert cleaned_df.loc[0, "text"] == "hello"
    assert pd.isna(cleaned_df.loc[1, "text"])
    assert removed == 0
