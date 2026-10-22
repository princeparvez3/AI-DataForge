
from datetime import datetime, timezone


SCORE_DIMENSIONS = (
    "correctness",
    "relevance",
    "instruction_following",
    "clarity",
)


def build_evaluation_record(
    prompt: str,
    response_a: str,
    response_b: str,
    scores_a: dict,
    scores_b: dict,
    winner: str,
    rationale: str = "",
) -> dict:
    """Validate and create one human-reviewed response comparison."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The prompt cannot be empty.")

    if not isinstance(response_a, str) or not response_a.strip():
        raise ValueError("Response A cannot be empty.")

    if not isinstance(response_b, str) or not response_b.strip():
        raise ValueError("Response B cannot be empty.")

    if winner not in ("Response A", "Response B", "Tie"):
        raise ValueError("Choose Response A, Response B, or Tie.")

    if not isinstance(scores_a, dict) or not isinstance(scores_b, dict):
        raise ValueError("Both responses must have score dictionaries.")

    validated_scores = {}

    for response_name, scores in (
        ("A", scores_a),
        ("B", scores_b),
    ):
        validated_scores[response_name] = {}

        for dimension in SCORE_DIMENSIONS:
            value = scores.get(dimension)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= 5
            ):
                raise ValueError(
                    f"Score for {response_name} / {dimension} "
                    "must be an integer from 1 to 5."
                )

            validated_scores[response_name][dimension] = value

    average_a = round(
        sum(validated_scores["A"].values()) / len(SCORE_DIMENSIONS),
        2,
    )

    average_b = round(
        sum(validated_scores["B"].values()) / len(SCORE_DIMENSIONS),
        2,
    )

    if winner == "Response A":
        chosen = response_a.strip()
        rejected = response_b.strip()
    elif winner == "Response B":
        chosen = response_b.strip()
        rejected = response_a.strip()
    else:
        chosen = None
        rejected = None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt.strip(),
        "response_a": response_a.strip(),
        "response_b": response_b.strip(),
        "score_a_correctness": validated_scores["A"]["correctness"],
        "score_a_relevance": validated_scores["A"]["relevance"],
        "score_a_instruction_following": validated_scores["A"]["instruction_following"],
        "score_a_clarity": validated_scores["A"]["clarity"],
        "score_b_correctness": validated_scores["B"]["correctness"],
        "score_b_relevance": validated_scores["B"]["relevance"],
        "score_b_instruction_following": validated_scores["B"]["instruction_following"],
        "score_b_clarity": validated_scores["B"]["clarity"],
        "average_score_a": average_a,
        "average_score_b": average_b,
        "winner": winner,
        "chosen": chosen,
        "rejected": rejected,
        "rationale": rationale.strip() if isinstance(rationale, str) else "",
    }
