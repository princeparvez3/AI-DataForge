
import pytest

from core.evaluation import (
    SCORE_DIMENSIONS,
    build_evaluation_record,
)


def make_scores(value=4):
    return {
        dimension: value
        for dimension in SCORE_DIMENSIONS
    }


def test_response_a_winner_creates_correct_preference_pair():
    record = build_evaluation_record(
        prompt="What is Python?",
        response_a="Python is a programming language.",
        response_b="Python is a type of car.",
        scores_a=make_scores(5),
        scores_b=make_scores(2),
        winner="Response A",
        rationale="Response A is accurate.",
    )

    assert record["winner"] == "Response A"
    assert record["chosen"] == "Python is a programming language."
    assert record["rejected"] == "Python is a type of car."
    assert record["average_score_a"] == 5.0
    assert record["average_score_b"] == 2.0


def test_response_b_winner_reverses_preference_pair():
    record = build_evaluation_record(
        prompt="Give an example.",
        response_a="Incorrect example.",
        response_b="Correct example.",
        scores_a=make_scores(2),
        scores_b=make_scores(5),
        winner="Response B",
    )

    assert record["chosen"] == "Correct example."
    assert record["rejected"] == "Incorrect example."


def test_tie_has_no_chosen_or_rejected_response():
    record = build_evaluation_record(
        prompt="Explain AI.",
        response_a="Response A.",
        response_b="Response B.",
        scores_a=make_scores(3),
        scores_b=make_scores(3),
        winner="Tie",
    )

    assert record["winner"] == "Tie"
    assert record["chosen"] is None
    assert record["rejected"] is None


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError):
        build_evaluation_record(
            prompt=" ",
            response_a="Response A",
            response_b="Response B",
            scores_a=make_scores(),
            scores_b=make_scores(),
            winner="Response A",
        )


def test_invalid_score_is_rejected():
    scores_a = make_scores()
    scores_a["correctness"] = 6

    with pytest.raises(ValueError):
        build_evaluation_record(
            prompt="Test prompt",
            response_a="Response A",
            response_b="Response B",
            scores_a=scores_a,
            scores_b=make_scores(),
            winner="Response A",
        )
