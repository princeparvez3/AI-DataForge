
import json
import sqlite3
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import pandas as pd


# Database is stored inside the project's data folder.
DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "dataforge.db"
)


@contextmanager
def get_connection():
    """Open a database connection and ensure it is closed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(DB_PATH))

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                state_key TEXT PRIMARY KEY,
                state_value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def save_state(key: str, value) -> None:
    """Save JSON-compatible application state in SQLite."""
    if not key.strip():
        raise ValueError("Storage key cannot be empty.")

    serialized_value = json.dumps(
        value,
        ensure_ascii=False,
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO app_state (state_key, state_value)
            VALUES (?, ?)
            ON CONFLICT(state_key)
            DO UPDATE SET
                state_value = excluded.state_value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, serialized_value),
        )


def load_state(key: str, default=None):
    """Load saved application state or return a default value."""
    with get_connection() as connection:
        result = connection.execute(
            """
            SELECT state_value
            FROM app_state
            WHERE state_key = ?
            """,
            (key,),
        ).fetchone()

    if result is None:
        return default

    return json.loads(result[0])


def save_dataframe(
    dataframe: pd.DataFrame,
    key: str = "main_dataset",
) -> None:
    """Save a DataFrame while preserving its tabular schema."""
    serialized_dataframe = dataframe.to_json(
        orient="table",
        date_format="iso",
        force_ascii=False,
    )

    save_state(
        key,
        {
            "storage_type": "dataframe",
            "payload": serialized_dataframe,
        },
    )


def load_dataframe(
    key: str = "main_dataset",
) -> pd.DataFrame | None:
    """Load a saved DataFrame, or return None if none exists."""
    saved_value = load_state(key)

    if saved_value is None:
        return None

    if (
        not isinstance(saved_value, dict)
        or saved_value.get("storage_type") != "dataframe"
        or "payload" not in saved_value
    ):
        raise ValueError(
            f"Saved data for '{key}' is not a valid DataFrame."
        )

    return pd.read_json(
        StringIO(saved_value["payload"]),
        orient="table",
    )
