
import pandas as pd

import core.storage as storage


def test_state_can_be_saved_and_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(
        storage,
        "DB_PATH",
        tmp_path / "test.db",
    )

    original = {
        "project": "AI DataForge",
        "records": 5,
        "status": "success",
    }

    storage.save_state("test_state", original)

    loaded = storage.load_state("test_state")

    assert loaded == original


def test_missing_state_returns_default(tmp_path, monkeypatch):
    monkeypatch.setattr(
        storage,
        "DB_PATH",
        tmp_path / "test.db",
    )

    result = storage.load_state(
        "not_found",
        default=[],
    )

    assert result == []


def test_dataframe_can_be_saved_and_loaded(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "DB_PATH",
        tmp_path / "test.db",
    )

    original = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["Hello", "AI DataForge", "Khush aamdeed"],
            "label": ["Positive", "Neutral", "Positive"],
        }
    )

    storage.save_dataframe(original)

    loaded = storage.load_dataframe()

    pd.testing.assert_frame_equal(
        loaded,
        original,
    )


def test_saving_same_key_updates_existing_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        storage,
        "DB_PATH",
        tmp_path / "test.db",
    )

    storage.save_state("counter", {"value": 1})
    storage.save_state("counter", {"value": 2})

    assert storage.load_state("counter") == {"value": 2}
