
import pandas as pd
import streamlit as st

from core.annotation_qa import compare_annotations


def _normalize_label(value):
    """Normalize a label for row-level disagreement display."""
    if pd.isna(value):
        return None

    label = str(value).strip()

    return label.casefold() if label else None


def render_annotation_qa_page():
    """Display inter-annotator agreement and disagreement analysis."""

    st.subheader("Annotation QA & Inter-Annotator Agreement")

    st.write(
        "Compare two sets of human annotations to measure agreement, "
        "identify disagreements, and inspect label consistency."
    )

    df = st.session_state.dataset

    if df.empty:
        st.warning("The current dataset is empty.")
        return

    st.info(
        "Choose two columns containing labels from different annotators. "
        "Do not select the text column as an annotator's label column."
    )

    all_columns = df.columns.tolist()

    label_keywords = (
        "annotator",
        "annotation",
        "label",
        "sentiment",
        "intent",
        "category",
        "class",
        "tag",
    )

    label_columns = [
        column
        for column in all_columns
        if any(
            keyword in str(column).casefold()
            for keyword in label_keywords
        )
    ]

    # Fall back to all columns if fewer than two likely label
    # columns exist, allowing the user to select custom names.
    options = (
        label_columns
        if len(label_columns) >= 2
        else all_columns
    )

    if len(options) < 2:
        st.warning(
            "At least two columns are required. Add or upload "
            "two annotator label columns before comparing."
        )
        return

    default_a = 0
    default_b = 1

    for index, column in enumerate(options):
        name = str(column).casefold()

        if "annotator_a" in name or "annotation_a" in name:
            default_a = index

        if "annotator_b" in name or "annotation_b" in name:
            default_b = index

    if default_a == default_b:
        default_b = 1 if default_a == 0 else 0

    col_a, col_b = st.columns(2)

    with col_a:
        annotator_a = st.selectbox(
            "Annotator A label column",
            options=options,
            index=default_a,
            key="qa_annotator_a",
        )

    with col_b:
        annotator_b = st.selectbox(
            "Annotator B label column",
            options=options,
            index=default_b,
            key="qa_annotator_b",
        )

    if st.button(
        "Calculate Agreement",
        type="primary",
        key="calculate_annotation_agreement",
    ):
        if annotator_a == annotator_b:
            st.error("Choose two different columns.")
        else:
            try:
                report = compare_annotations(
                    df,
                    annotator_a,
                    annotator_b,
                )

                st.session_state["annotation_qa_report"] = report
                st.session_state["annotation_qa_columns"] = (
                    annotator_a,
                    annotator_b,
                )

            except ValueError as error:
                st.error(str(error))
                st.session_state.pop("annotation_qa_report", None)
                st.session_state.pop("annotation_qa_columns", None)

    report = st.session_state.get("annotation_qa_report")
    report_columns = st.session_state.get("annotation_qa_columns")

    if report is None:
        st.caption(
            "Select two annotator columns and calculate agreement "
            "to view the report."
        )
        return

    if report_columns != (annotator_a, annotator_b):
        st.info(
            "Column selection changed. Calculate agreement again "
            "to refresh the report."
        )
        return

    st.divider()
    st.subheader("Agreement Report")

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Compared Records",
        report["compared_records"],
    )

    metric2.metric(
        "Exact Agreement",
        f"{report['agreement_percentage']:.2f}%",
    )

    metric3.metric(
        "Disagreements",
        report["disagreeing_records"],
    )

    kappa = report["cohen_kappa"]

    metric4.metric(
        "Cohen's Kappa",
        "N/A" if kappa is None else f"{kappa:.4f}",
    )

    st.caption(
        f"Total records: {report['total_records']} · "
        f"Excluded because one or both labels were blank: "
        f"{report['excluded_records']}"
    )

    if kappa is None:
        st.warning(
            "Cohen's Kappa is undefined because the compared records "
            "contain only one label category. Use a dataset with "
            "variation in labels to assess chance-adjusted agreement."
        )
    else:
        st.caption(
            "Cohen's Kappa measures agreement adjusted for chance. "
            "Interpret it alongside raw agreement and the dataset's "
            "label distribution."
        )

    st.subheader("Confusion Matrix")
    st.caption(
        "Rows represent Annotator A; columns represent Annotator B."
    )

    st.dataframe(
        report["confusion_matrix"],
        use_container_width=True,
    )

    # Show records where both annotators supplied a label but disagreed.
    labels_a = df[annotator_a].map(_normalize_label)
    labels_b = df[annotator_b].map(_normalize_label)

    valid_mask = labels_a.notna() & labels_b.notna()
    disagreement_mask = valid_mask & labels_a.ne(labels_b)

    disagreement_rows = df.loc[disagreement_mask].copy()

    st.subheader("Disagreement Review")

    if disagreement_rows.empty:
        st.success(
            "No disagreements found among records with two non-empty labels."
        )
    else:
        st.write(
            f"{len(disagreement_rows)} record(s) need review."
        )

        preferred_text_columns = [
            column
            for column in ("text", "message", "content", "prompt")
            if column in df.columns
        ]

        display_columns = list(
            dict.fromkeys(
                preferred_text_columns
                + [annotator_a, annotator_b]
            )
        )

        st.dataframe(
            disagreement_rows[display_columns],
            use_container_width=True,
        )

        st.download_button(
            "Download Disagreement Records CSV",
            data=disagreement_rows[display_columns].to_csv(
                index=False
            ).encode("utf-8"),
            file_name="annotation_disagreements.csv",
            mime="text/csv",
            use_container_width=True,
        )
