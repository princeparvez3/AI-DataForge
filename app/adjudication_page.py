"""Human review interface for resolving annotation disagreements."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.adjudication import apply_adjudications, find_disagreements
from core.storage import save_dataframe, save_state


def _fingerprint(df: pd.DataFrame) -> str:
    """Create a stable fingerprint to prevent saving stale review decisions."""
    serialized = df.to_json(
        orient="split",
        date_format="iso",
        default_handler=str,
        force_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _label_like_columns(df: pd.DataFrame) -> list[str]:
    """Suggest likely annotation columns without forcing the selection."""
    keywords = (
        "annotator",
        "annotation",
        "label",
        "sentiment",
        "intent",
        "category",
        "class",
        "tag",
    )
    suggested = [
        column
        for column in df.columns
        if any(keyword in str(column).casefold() for keyword in keywords)
    ]

    if len(suggested) >= 2:
        return suggested

    text_names = {"text", "message", "content", "prompt", "query"}
    text_columns = {
        column for column in df.columns
        if str(column).casefold() in text_names
    }
    string_columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    fallbacks = [column for column in string_columns if column not in text_columns]
    return fallbacks if len(fallbacks) >= 2 else list(df.columns)


def _unique_labels(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Collect unique, non-empty labels from selected annotation columns."""
    result: list[str] = []
    seen: set[str] = set()

    for column in columns:
        if column not in df.columns:
            continue
        for value in df[column].tolist():
            if pd.isna(value):
                continue
            label = str(value).strip()
            if label and label.casefold() not in seen:
                seen.add(label.casefold())
                result.append(label)

    return result


def render_adjudication_page() -> None:
    """Let a reviewer resolve disagreements and persist final labels."""
    st.subheader("Annotation Adjudication")
    st.write(
        "Review conflicting labels from two annotators, select a final label, "
        "and save the reviewed decision to the dataset and SQLite database."
    )

    dataset = st.session_state.get("dataset")
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        st.warning("The current dataset is empty.")
        return

    label_columns = _label_like_columns(dataset)
    if len(dataset.columns) < 2:
        st.warning("The dataset needs at least two columns to review annotations.")
        return

    default_a = 0
    default_b = 1
    for index, column in enumerate(label_columns):
        name = str(column).casefold()
        if "annotator_a" in name or "annotation_a" in name or name.endswith("_a"):
            default_a = index
        if "annotator_b" in name or "annotation_b" in name or name.endswith("_b"):
            default_b = index
    if default_a == default_b and len(label_columns) > 1:
        default_b = 1 if default_a == 0 else 0

    col_a, col_b = st.columns(2)
    with col_a:
        annotator_a_column = st.selectbox(
            "Annotator A label column",
            options=label_columns,
            index=min(default_a, len(label_columns) - 1),
            key="adjudication_annotator_a",
        )
    with col_b:
        annotator_b_column = st.selectbox(
            "Annotator B label column",
            options=label_columns,
            index=min(default_b, len(label_columns) - 1),
            key="adjudication_annotator_b",
        )

    text_candidates = [column for column in dataset.columns if str(column).casefold() in {"text", "message", "content", "prompt", "query"}]
    if not text_candidates:
        text_candidates = dataset.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

    text_options = ["No text column"] + text_candidates
    default_text_index = 1 if text_candidates else 0
    text_choice = st.selectbox(
        "Text column for review (optional)",
        options=text_options,
        index=default_text_index,
        key="adjudication_text_column",
    )
    text_column = None if text_choice == "No text column" else text_choice

    default_final_column = "final_label"
    final_label_column = st.text_input(
        "Column to save final decisions into",
        value=default_final_column,
        key="adjudication_final_column",
    ).strip()

    inferred_labels = _unique_labels(
        dataset,
        [annotator_a_column, annotator_b_column],
    )
    if not inferred_labels:
        inferred_labels = ["Positive", "Negative", "Neutral", "Other"]

    allowed_labels_text = st.text_input(
        "Allowed final labels (comma-separated)",
        value=", ".join(inferred_labels),
        key="adjudication_allowed_labels",
        help="Use labels that are valid for this dataset. Labels are matched ignoring case.",
    )
    allowed_labels = list(
        dict.fromkeys(
            label.strip()
            for label in allowed_labels_text.split(",")
            if label.strip()
        )
    )

    if annotator_a_column == annotator_b_column:
        st.warning("Choose two different annotator columns.")
        return
    if not final_label_column:
        st.warning("Enter the name of the final-label column.")
        return
    if final_label_column in {annotator_a_column, annotator_b_column}:
        st.warning("Use a new final-label column, not either annotator's input column.")
        return
    if not allowed_labels:
        st.warning("Enter at least one allowed final label.")
        return

    if st.button(
        "Find Disagreements",
        type="primary",
        key="find_adjudication_disagreements",
    ):
        try:
            decisions = find_disagreements(
                dataset,
                annotator_a_column,
                annotator_b_column,
                text_column,
            )
            st.session_state["adjudication_decisions"] = decisions
            st.session_state["adjudication_meta"] = {
                "annotator_a_column": annotator_a_column,
                "annotator_b_column": annotator_b_column,
                "text_column": text_column,
                "final_label_column": final_label_column,
                "allowed_labels": allowed_labels,
                "dataset_fingerprint": _fingerprint(dataset),
            }
        except ValueError as error:
            st.error(str(error))
            st.session_state.pop("adjudication_decisions", None)
            st.session_state.pop("adjudication_meta", None)

    decisions = st.session_state.get("adjudication_decisions")
    meta = st.session_state.get("adjudication_meta")
    if decisions is None or meta is None:
        st.caption("Select the annotator columns, then click Find Disagreements.")
        return

    settings_unchanged = (
        meta["annotator_a_column"] == annotator_a_column
        and meta["annotator_b_column"] == annotator_b_column
        and meta["text_column"] == text_column
        and meta["final_label_column"] == final_label_column
        and meta["allowed_labels"] == allowed_labels
    )
    dataset_unchanged = meta["dataset_fingerprint"] == _fingerprint(dataset)

    st.divider()
    st.subheader("Disagreement Review")
    st.caption(
        f"Found {len(decisions)} disagreements. Choose a final label for "
        "each record you want to resolve; blank decisions remain unresolved."
    )

    if not dataset_unchanged:
        st.warning(
            "The dataset changed after disagreements were found. Run Find Disagreements "
            "again before saving, to avoid applying decisions to stale rows."
        )
    if not settings_unchanged:
        st.warning("The review settings changed. Run Find Disagreements again before saving.")

    if decisions.empty:
        st.success("No disagreements found among records with two non-empty labels.")
        return

    # Use a dedicated selectbox for each disputed record instead of a
    # SelectboxColumn in st.data_editor. This makes the label choices visible
    # and reliable even when the initial final-label value is blank.
    edited_decisions = decisions.copy()
    placeholder = "-- Select final label --"

    for decision_index, decision in decisions.iterrows():
        position = int(decision["row_position"])
        st.markdown(f"**Record {position + 1}**")

        detail_col, final_col = st.columns([3, 1])
        with detail_col:
            if text_column and "Text" in decision.index:
                st.write(f"**Text:** {decision['Text']}")
            st.write(
                f"**Annotator A:** {decision['Annotator A Label']}  "
                f"| **Annotator B:** {decision['Annotator B Label']}"
            )

        select_key = (
            f"adjudication_final_label_"
            f"{meta['dataset_fingerprint'][:10]}_{position}"
        )
        with final_col:
            selected_label = st.selectbox(
                "Final Label",
                options=[placeholder] + allowed_labels,
                index=0,
                key=select_key,
            )

        edited_decisions.at[decision_index, "Final Label"] = (
            "" if selected_label == placeholder else selected_label
        )
        st.divider()

    save_col, clear_col = st.columns(2)
    with save_col:
        save_clicked = st.button(
            "Save Final Decisions",
            type="primary",
            disabled=not (dataset_unchanged and settings_unchanged),
            key="save_adjudication_decisions",
            use_container_width=True,
        )
    with clear_col:
        clear_clicked = st.button(
            "Discard Review",
            key="discard_adjudication_review",
            use_container_width=True,
        )

    if clear_clicked:
        st.session_state.pop("adjudication_decisions", None)
        st.session_state.pop("adjudication_meta", None)
        st.session_state.pop("adjudication_decisions_editor", None)
        st.rerun()

    if save_clicked:
        try:
            updated_dataset, saved_count = apply_adjudications(
                dataset,
                edited_decisions,
                final_label_column,
                allowed_labels,
            )

            if saved_count == 0:
                st.warning(
                    "No final labels were selected. Choose at least one label "
                    "before saving; nothing was written to the database."
                )
                return

            current_source = st.session_state.get(
                "current_source", "Saved dataset"
            )
            # Save the dataset and source before reporting success. If either
            # storage operation fails, display the real error rather than a
            # misleading success message.
            save_dataframe(updated_dataset, key="main_dataset")
            save_state("dataset_source", current_source)

        except (ValueError, OSError, RuntimeError, TypeError) as error:
            st.error(f"Could not save final decisions: {error}")
            return
        except Exception as error:
            st.error(
                "An unexpected error occurred while saving final decisions: "
                f"{type(error).__name__}: {error}"
            )
            return

        st.session_state.dataset = updated_dataset

        # The split was made from the previous version of the dataset.
        for key in (
            "split_train_df",
            "split_test_df",
            "split_leakage_report",
            "split_settings",
        ):
            st.session_state.pop(key, None)

        st.session_state.pop("adjudication_decisions", None)
        st.session_state.pop("adjudication_meta", None)
        st.session_state.pop("adjudication_decisions_editor", None)
        st.session_state["adjudication_saved_count"] = saved_count
        st.rerun()

