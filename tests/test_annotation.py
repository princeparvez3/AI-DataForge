import json

import pytest

from core.ai_annotation import validate_suggestion


def test_valid_suggestion_is_normalized_to_allowed_label():
    result = validate_suggestion(
        json.dumps(
            {
                "label": "positive",
                "confidence_percent": 91,
                "rationale": "The message praises the product.",
            }
        ),
        ["Positive", "Negative", "Neutral"],
    )

    assert result == {
        "label": "Positive",
        "confidence_percent": 91,
        "rationale": "The message praises the product.",
    }


def test_suggestion_rejects_label_outside_allowed_list():
    with pytest.raises(ValueError, match="outside the allowed label list"):
        validate_suggestion(
            '{"label":"Urgent","confidence_percent":80,"rationale":"test"}',
            ["Positive", "Negative", "Neutral"],
        )


def test_suggestion_rejects_invalid_confidence():
    with pytest.raises(ValueError, match="between 0 and 100"):
        validate_suggestion(
            '{"label":"Positive","confidence_percent":101,"rationale":"test"}',
            ["Positive", "Negative"],
        )


def test_suggestion_rejects_invalid_json():
    with pytest.raises(ValueError, match="valid JSON"):
        validate_suggestion("not json", ["Positive", "Negative"])
