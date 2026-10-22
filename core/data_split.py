
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    group_column: str | None = None,
    stratify_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into training and testing data."""

    if len(df) < 2:
        raise ValueError("At least two records are required.")

    if not 0 < test_size < 1:
        raise ValueError("Test size must be greater than 0 and less than 1.")

    if group_column and stratify_column:
        raise ValueError(
            "Choose group-aware splitting or stratification, not both."
        )

    if group_column and group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' was not found.")

    if stratify_column and stratify_column not in df.columns:
        raise ValueError(
            f"Stratification column '{stratify_column}' was not found."
        )

    positions = list(range(len(df)))

    if group_column:
        # Missing group values are treated as individual records.
        groups = []

        for position, value in enumerate(df[group_column].tolist()):
            if pd.isna(value):
                groups.append(f"missing-record-{position}")
            else:
                groups.append(
                    f"{type(value).__name__}:{value!r}"
                )

        if len(set(groups)) < 2:
            raise ValueError(
                "At least two distinct groups are required."
            )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )

        train_positions, test_positions = next(
            splitter.split(df, groups=groups)
        )

    else:
        stratify_values = None

        if stratify_column:
            labels = df[stratify_column]

            if labels.isna().any():
                raise ValueError(
                    "The stratification column contains missing labels. "
                    "Fill or review them before splitting."
                )

            label_counts = labels.value_counts()

            if len(label_counts) < 2 or (label_counts < 2).any():
                raise ValueError(
                    "Stratification requires at least two classes, "
                    "with at least two records in every class."
                )

            stratify_values = labels

        try:
            train_positions, test_positions = train_test_split(
                positions,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_values,
            )
        except ValueError as error:
            raise ValueError(
                f"Could not split the dataset: {error}"
            ) from error

    train_df = df.iloc[train_positions].reset_index(drop=True)
    test_df = df.iloc[test_positions].reset_index(drop=True)

    return train_df, test_df


def get_leakage_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
) -> dict:
    """Find normalized values shared by training and testing sets."""

    report = {}

    for column in columns:
        if column not in train_df.columns or column not in test_df.columns:
            continue

        def normalize(value):
            if pd.isna(value):
                return None

            normalized = str(value).strip().casefold()
            return normalized if normalized else None

        train_values = {
            normalize(value) for value in train_df[column].tolist()
        }
        test_values = {
            normalize(value) for value in test_df[column].tolist()
        }

        train_values.discard(None)
        test_values.discard(None)

        shared_values = sorted(train_values.intersection(test_values))

        report[column] = {
            "overlap_count": len(shared_values),
            "shared_values": shared_values[:10],
        }

    return report
