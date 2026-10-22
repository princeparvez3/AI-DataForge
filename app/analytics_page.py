"""Annotation analytics dashboard for AI DataForge."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.analytics import get_annotation_metrics, summarize_ai_review_history
from core.annotation import parse_labels
from core.storage import load_state


def render_analytics_page(dataframe: pd.DataFrame) -> None:
    """Render annotation progress, label distribution, QA, and AI review metrics."""
    st.subheader("Annotation Analytics Dashboard")
    st.caption(
        "Track annotation coverage, label distribution, dataset hygiene, "
        "and human decisions on AI-assisted suggestions."
    )

    if dataframe.empty:
        st.info("The current dataset is empty. Load records to view analytics.")
        return

    text_columns = dataframe.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    preferred_text_names = ["text", "message", "content", "prompt", "query"]
    default_text_column = next(
        (
            column
            for name in preferred_text_names
            for column in text_columns
            if str(column).lower() == name
        ),
        text_columns[0] if text_columns else None,
    )

    control1, control2 = st.columns(2)
    with control1:
        if text_columns:
            text_column = st.selectbox(
                "Text column for QA checks",
                options=text_columns,
                index=text_columns.index(default_text_column),
                key="analytics_text_column",
            )
        else:
            text_column = None
            st.info("No text column found; text-specific QA checks are unavailable.")

    annotation_options = list(dataframe.columns)
    preferred_annotation_names = ["annotation", "label", "sentiment", "intent", "category"]
    default_annotation_column = next(
        (name for name in preferred_annotation_names if name in annotation_options),
        annotation_options[0],
    )
    if text_column and len(annotation_options) > 1:
        label_options = [column for column in annotation_options if column != text_column]
    else:
        label_options = annotation_options
    if not label_options:
        label_options = annotation_options

    with control2:
        annotation_column = st.selectbox(
            "Annotation / label column",
            options=label_options,
            index=(
                label_options.index(default_annotation_column)
                if default_annotation_column in label_options
                else 0
            ),
            key="analytics_annotation_column",
        )

    expected_labels_text = st.text_input(
        "Expected labels for validation (comma-separated; optional)",
        value="Positive, Negative, Neutral, Other",
        key="analytics_expected_labels",
        help="Use the labels intended for the selected annotation column. Leave blank to skip this validation.",
    )
    expected_labels = parse_labels(expected_labels_text)

    metrics = get_annotation_metrics(
        dataframe,
        annotation_column=annotation_column,
        text_column=text_column,
        expected_labels=expected_labels,
    )

    st.divider()
    st.markdown("### Annotation progress")
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total records", metrics["total_records"])
    metric2.metric("Annotated", metrics["annotated_records"])
    metric3.metric("Pending", metrics["pending_records"])
    metric4.metric("Completion", f"{metrics['completion_rate']:.1f}%")
    st.progress(metrics["completion_rate"] / 100)

    distribution_tab, pending_tab, quality_tab = st.tabs(
        ["Label distribution", "Pending records", "Data quality"]
    )

    with distribution_tab:
        st.markdown("### Label distribution")
        label_counts = metrics["label_counts"]
        if label_counts:
            distribution_df = pd.DataFrame(
                [{"Label": str(label), "Records": int(count)} for label, count in label_counts.items()]
            ).sort_values("Records", ascending=False)
            st.bar_chart(distribution_df.set_index("Label"))
            st.dataframe(distribution_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download label distribution CSV",
                data=distribution_df.to_csv(index=False).encode("utf-8"),
                file_name="dataforge_label_distribution.csv",
                mime="text/csv",
                key="download_label_distribution",
            )
        else:
            st.info(
                f"No non-empty values were found in `{annotation_column}` yet. "
                "Save annotations to see the label distribution."
            )

    with pending_tab:
        st.markdown("### Records still needing labels")
        if annotation_column in dataframe.columns:
            labels = dataframe[annotation_column].fillna("").astype(str).str.strip()
            pending_dataframe = dataframe.loc[labels.eq("")].copy()
        else:
            pending_dataframe = dataframe.copy()

        st.metric("Pending records", len(pending_dataframe))
        st.dataframe(pending_dataframe.head(200), use_container_width=True, hide_index=True)
        if not pending_dataframe.empty:
            st.download_button(
                "Download pending records CSV",
                data=pending_dataframe.to_csv(index=False).encode("utf-8"),
                file_name="dataforge_pending_annotations.csv",
                mime="text/csv",
                key="download_pending_annotations",
            )
        st.caption("The table shows up to 200 pending records to keep the page responsive.")

    with quality_tab:
        st.markdown("### Annotation quality checks")
        qa1, qa2, qa3 = st.columns(3)
        qa1.metric("Blank text records", metrics["blank_text_records"])
        qa2.metric("Repeated text values", metrics["repeated_text_values"])
        qa3.metric("Labels outside expected set", metrics["invalid_labels"])

        if expected_labels:
            st.caption("Expected labels: " + ", ".join(expected_labels))
            if metrics["invalid_labels"]:
                st.warning(
                    "Some labels are outside the expected label set. Review them before using this dataset for training."
                )
            else:
                st.success("All non-empty labels match the expected label set.")
        else:
            st.info("Enter expected labels above to enable label validation.")

        if metrics["blank_text_records"]:
            st.warning("Some records have blank text and may need to be excluded or corrected.")
        if metrics["repeated_text_values"]:
            st.info("Repeated texts may be legitimate, but review them before splitting a dataset for model evaluation.")

    st.divider()
    st.markdown("### AI-assisted annotation review")
    history = load_state("ai_annotation_audit", default=[])
    if not isinstance(history, list):
        history = []
    review = summarize_ai_review_history(history)

    review1, review2, review3, review4 = st.columns(4)
    review1.metric("Suggestions reviewed", review["total_reviewed"])
    review2.metric("Accepted unchanged", review["accepted"])
    review3.metric("Corrected by reviewer", review["corrected"])
    review4.metric("Rejected", review["rejected"])

    if review["total_reviewed"]:
        accept_rate = review["acceptance_rate"]
        correction_rate = review["correction_rate"]
        reject_rate = review["rejection_rate"]
        st.write(
            f"Acceptance rate: **{accept_rate:.1f}%** · "
            f"Correction rate: **{correction_rate:.1f}%** · "
            f"Rejection rate: **{reject_rate:.1f}%**"
        )

        history_df = pd.DataFrame(review["records"])
        visible_columns = [
            column
            for column in (
                "reviewed_at_utc",
                "model",
                "annotation_column",
                "suggested_label",
                "approved_label",
                "decision",
                "confidence_percent",
            )
            if column in history_df.columns
        ]
        st.dataframe(
            history_df[visible_columns].tail(100).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download AI review audit CSV",
            data=history_df[visible_columns].to_csv(index=False).encode("utf-8"),
            file_name="dataforge_ai_annotation_audit.csv",
            mime="text/csv",
            key="download_ai_review_audit",
        )
    else:
        st.info(
            "No saved AI review decisions are recorded yet. After you review "
            "AI suggestions and click Save Approved Annotations, this section "
            "will track accepted, corrected, and rejected labels. Older decisions "
            "cannot be reconstructed retroactively."
        )
