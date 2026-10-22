import pandas as pd
import streamlit as st

from core.evaluation import (
    SCORE_DIMENSIONS,
    build_evaluation_record,
)
from core.storage import load_state, save_state


def render_evaluation_page():
    """Render the LLM evaluation interface with SQLite-backed history."""

    st.subheader("Compare AI Responses")
    st.write(
        "Compare two AI responses, score their quality, choose the preferred "
        "response, and export evaluation data. Evaluation history is stored "
        "in the local SQLite database."
    )

    if "evaluation_records" not in st.session_state:
        st.session_state.evaluation_records = load_state(
            "evaluation_records",
            default=[],
        )

    if st.session_state.pop("evaluation_saved", False):
        st.success("Evaluation saved successfully to the database!")

    with st.form("llm_evaluation_form"):
        prompt = st.text_area(
            "Prompt / User Question",
            height=100,
            key="eval_prompt",
            placeholder="Enter the question given to the AI models...",
        )

        response_col_a, response_col_b = st.columns(2)

        with response_col_a:
            st.markdown("### Response A")
            response_a = st.text_area(
                "Response A text",
                height=180,
                key="eval_response_a",
                placeholder="Paste the first AI response...",
            )
            st.markdown("**Score Response A**")
            scores_a = {}
            for dimension in SCORE_DIMENSIONS:
                scores_a[dimension] = st.select_slider(
                    dimension.replace("_", " ").title(),
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"eval_a_{dimension}",
                )

        with response_col_b:
            st.markdown("### Response B")
            response_b = st.text_area(
                "Response B text",
                height=180,
                key="eval_response_b",
                placeholder="Paste the second AI response...",
            )
            st.markdown("**Score Response B**")
            scores_b = {}
            for dimension in SCORE_DIMENSIONS:
                scores_b[dimension] = st.select_slider(
                    dimension.replace("_", " ").title(),
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    key=f"eval_b_{dimension}",
                )

        st.caption(
            "Scoring guide: 1 = Poor, 2 = Weak, 3 = Acceptable, "
            "4 = Good, 5 = Excellent."
        )

        winner = st.radio(
            "Which response is better overall?",
            options=["Response A", "Response B", "Tie"],
            horizontal=True,
            key="eval_winner",
        )
        rationale = st.text_area(
            "Reason for your decision (optional)",
            key="eval_rationale",
            placeholder="Explain why you preferred this response...",
        )
        submitted = st.form_submit_button(
            "Save Evaluation",
            type="primary",
        )

    if submitted:
        try:
            record = build_evaluation_record(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                scores_a=scores_a,
                scores_b=scores_b,
                winner=winner,
                rationale=rationale,
            )

            records = list(st.session_state.evaluation_records)
            records.append(record)
            save_state("evaluation_records", records)
            st.session_state.evaluation_records = records
            st.session_state.evaluation_saved = True
            st.rerun()

        except ValueError as error:
            st.error(str(error))

    st.divider()
    st.subheader("Evaluation History")
    history = st.session_state.evaluation_records

    if not history:
        st.info(
            "No evaluations saved yet. Complete the form above to create "
            "your first evaluation."
        )
        return

    history_df = pd.DataFrame(history)
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Total Evaluations", len(history_df))
    metric_b.metric(
        "Response A Wins",
        int((history_df["winner"] == "Response A").sum()),
    )
    metric_c.metric(
        "Response B Wins",
        int((history_df["winner"] == "Response B").sum()),
    )

    st.dataframe(
        history_df[
            [
                "timestamp_utc",
                "prompt",
                "average_score_a",
                "average_score_b",
                "winner",
                "rationale",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Export Evaluation Data")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Evaluation CSV",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="dataforge_evaluations.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "Download All Evaluations JSONL",
            data=history_df.to_json(
                orient="records",
                lines=True,
                force_ascii=False,
            ).encode("utf-8"),
            file_name="dataforge_evaluations.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )

    preference_records = [
        {
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
            "winner": record["winner"],
            "rationale": record["rationale"],
        }
        for record in history
        if record["winner"] != "Tie"
    ]

    st.caption(
        f"Preference pairs available: {len(preference_records)}. "
        "Tied comparisons are excluded because they have no clear winner."
    )

    if preference_records:
        preference_jsonl = pd.DataFrame(preference_records).to_json(
            orient="records",
            lines=True,
            force_ascii=False,
        ).encode("utf-8")

        st.download_button(
            "Download Preference Dataset JSONL",
            data=preference_jsonl,
            file_name="dataforge_preference_pairs.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
