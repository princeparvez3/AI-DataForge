import hashlib
import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.ai_annotation import MODEL_NAME, suggest_label
from core.annotation import parse_labels, prepare_annotation_dataframe
from core.storage import load_state, save_dataframe, save_state


PENDING_KEY = "ai_annotation_candidates"
PENDING_META_KEY = "ai_annotation_candidate_meta"
EDITOR_KEY = "ai_annotation_review_editor"


def _is_quota_or_rate_limit_error(error: Exception) -> bool:
    """Detect quota/rate-limit errors so a batch stops promptly."""
    message = str(error).upper()
    quota_markers = (
        "EXCEEDED YOUR CURRENT QUOTA",
        "CURRENT QUOTA",
        "CHECK YOUR PLAN AND BILLING DETAILS",
        "QUOTA EXCEEDED",
        "BILLING DETAILS",
        "RESOURCE_EXHAUSTED",
    )
    return any(marker in message for marker in quota_markers) or (
        "429" in message and "QUOTA" in message
    )


def _get_api_key() -> str:
    """Read the key from the environment or local Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    serialized = df.to_json(
        orient="split",
        date_format="iso",
        default_handler=str,
        force_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _clear_pending_suggestions() -> None:
    st.session_state.pop(PENDING_KEY, None)
    st.session_state.pop(PENDING_META_KEY, None)
    st.session_state.pop(EDITOR_KEY, None)


def render_ai_annotation_page() -> None:
    """Render AI label suggestions and an explicit human approval step."""
    st.subheader("AI-Assisted Annotation")
    st.write(
        "Gemini suggests a label for each selected record. Review or correct "
        "each suggestion before saving it to your dataset."
    )

    st.warning(
        "Privacy notice: when you generate suggestions, the selected text "
        "records are sent to Google's Gemini API. Do not send confidential "
        "or sensitive data unless you are authorized to do so."
    )

    saved_message = st.session_state.pop("ai_annotation_saved_message", None)
    if saved_message:
        st.success(saved_message)
    elif st.session_state.pop("ai_annotation_saved", False):
        st.success("Approved annotations saved to the dataset and SQLite database.")

    dataset = st.session_state.dataset
    if dataset.empty:
        st.warning("The current dataset is empty.")
        return

    text_columns = dataset.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()
    if not text_columns:
        st.warning("No text columns were found in this dataset.")
        return

    preferred_names = ["text", "message", "content", "prompt", "query"]
    default_text_column = next(
        (
            column
            for name in preferred_names
            for column in text_columns
            if column.lower() == name
        ),
        text_columns[0],
    )

    text_column = st.selectbox(
        "Text column to classify",
        options=text_columns,
        index=text_columns.index(default_text_column),
        key="ai_annotation_text_column",
    )
    annotation_column = st.text_input(
        "Annotation column to write into",
        value="annotation",
        key="ai_annotation_target_column",
    ).strip()
    labels_text = st.text_input(
        "Allowed labels (comma-separated)",
        value="Positive, Negative, Neutral, Other",
        key="ai_annotation_allowed_labels",
    )
    allowed_labels = parse_labels(labels_text)
    batch_size = st.slider(
        "Maximum records per generation run",
        min_value=1,
        max_value=10,
        value=5,
        help="Each selected record makes one Gemini API request. Start with a small batch.",
        key="ai_annotation_batch_size",
    )
    only_unannotated = st.checkbox(
        "Only suggest labels for records that are not annotated yet",
        value=True,
        key="ai_annotation_only_unannotated",
    )

    api_key = _get_api_key()
    st.caption(f"Model: `{MODEL_NAME}`")
    if not api_key:
        st.error(
            "Gemini API key not found. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml or set it as an environment variable."
        )

    if not annotation_column:
        st.warning("Enter a name for the annotation column.")
    elif annotation_column == text_column:
        st.warning("The annotation column must differ from the text column.")
    elif not allowed_labels:
        st.warning("Enter at least one allowed label.")

    generate_disabled = (
        not api_key
        or not annotation_column
        or annotation_column == text_column
        or not allowed_labels
    )

    if st.button(
        "Generate AI Suggestions",
        type="primary",
        disabled=generate_disabled,
        key="generate_ai_annotation_suggestions",
    ):
        current = st.session_state.dataset
        prepared = prepare_annotation_dataframe(current, annotation_column)
        selected_positions: list[int] = []

        for position in range(len(prepared)):
            raw_text = prepared.iloc[position][text_column]
            text = "" if pd.isna(raw_text) else str(raw_text).strip()
            if not text:
                continue

            existing_value = prepared.iloc[position][annotation_column]
            existing_label = (
                "" if pd.isna(existing_value) else str(existing_value).strip()
            )
            if only_unannotated and existing_label:
                continue

            selected_positions.append(position)
            if len(selected_positions) >= batch_size:
                break

        if not selected_positions:
            st.info(
                "No eligible text records were found. Check the selected text "
                "column or turn off the option to skip annotated records."
            )
            _clear_pending_suggestions()
        else:
            _clear_pending_suggestions()
            candidates = []
            quota_or_rate_limited = False
            progress = st.progress(0, text="Generating suggestions...")

            for number, position in enumerate(selected_positions, start=1):
                text = str(prepared.iloc[position][text_column]).strip()
                try:
                    suggestion = suggest_label(
                        text=text,
                        allowed_labels=allowed_labels,
                        api_key=api_key,
                    )
                    suggested_label = suggestion["label"]
                    confidence = suggestion["confidence_percent"]
                    rationale = suggestion["rationale"]
                except Exception as error:
                    suggested_label = ""
                    confidence = 0
                    rationale = f"Suggestion failed: {str(error)[:220]}"
                    quota_or_rate_limited = _is_quota_or_rate_limit_error(error)

                candidates.append(
                    {
                        "_position": position,
                        "Row #": position + 1,
                        "Text": text,
                        "AI Suggested Label": suggested_label,
                        "Confidence (%)": confidence,
                        "AI Rationale": rationale,
                        "Approved Label": suggested_label,
                    }
                )
                progress.progress(
                    number / len(selected_positions),
                    text=f"Processed {number} of {len(selected_positions)} records",
                )

                # Do not send more records after quota/rate-limit exhaustion.
                if quota_or_rate_limited:
                    break

            progress.empty()
            if quota_or_rate_limited:
                st.error(
                    "Gemini reported a quota or rate-limit error. The batch was "
                    "stopped to avoid more failed requests. Check your active "
                    "model/project limits in Google AI Studio and billing settings. "
                    "Records not yet processed were not sent to the API."
                )
            st.session_state[PENDING_KEY] = candidates
            st.session_state[PENDING_META_KEY] = {
                "text_column": text_column,
                "annotation_column": annotation_column,
                "allowed_labels": allowed_labels,
                "dataset_fingerprint": _dataset_fingerprint(current),
            }
            st.session_state.pop(EDITOR_KEY, None)

    # Show a review table for any current suggestions.
    candidates = st.session_state.get(PENDING_KEY, [])
    meta = st.session_state.get(PENDING_META_KEY)

    if candidates and meta:
        current_dataset = st.session_state.dataset
        dataset_unchanged = (
            _dataset_fingerprint(current_dataset) == meta["dataset_fingerprint"]
        )
        settings_unchanged = (
            text_column == meta["text_column"]
            and annotation_column == meta["annotation_column"]
            and allowed_labels == meta["allowed_labels"]
        )

        st.divider()
        st.subheader("Review AI Suggestions")
        st.caption(
            "The AI suggested the labels; you are responsible for reviewing "
            "and approving the final label in the last column."
        )

        if not dataset_unchanged:
            st.warning(
                "The dataset changed after suggestions were generated. "
                "Discard these suggestions and generate them again before saving."
            )
        if not settings_unchanged:
            st.warning(
                "The text column, annotation column, or allowed labels changed. "
                "Generate suggestions again before saving."
            )

        candidate_df = pd.DataFrame(candidates)
        display_columns = [
            "_position",
            "Row #",
            "Text",
            "AI Suggested Label",
            "Confidence (%)",
            "AI Rationale",
            "Approved Label",
        ]
        candidate_df = candidate_df[display_columns]

        edited_df = st.data_editor(
            candidate_df,
            column_config={
                "_position": None,
                "Row #": st.column_config.NumberColumn("Row #", disabled=True),
                "Text": st.column_config.TextColumn("Text", disabled=True, width="large"),
                "AI Suggested Label": st.column_config.TextColumn(
                    "AI Suggested Label", disabled=True
                ),
                "Confidence (%)": st.column_config.NumberColumn(
                    "Confidence (%)", disabled=True, min_value=0, max_value=100
                ),
                "AI Rationale": st.column_config.TextColumn(
                    "AI Rationale", disabled=True, width="large"
                ),
                "Approved Label": st.column_config.SelectboxColumn(
                    "Approved Label",
                    options=meta["allowed_labels"],
                    required=False,
                ),
            },
            disabled=[
                "_position",
                "Row #",
                "Text",
                "AI Suggested Label",
                "Confidence (%)",
                "AI Rationale",
            ],
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            key=EDITOR_KEY,
        )

        apply_col, discard_col = st.columns(2)
        with apply_col:
            apply_clicked = st.button(
                "Save Approved Annotations",
                type="primary",
                disabled=not (dataset_unchanged and settings_unchanged),
                key="save_ai_approved_annotations",
                use_container_width=True,
            )
        with discard_col:
            discard_clicked = st.button(
                "Discard Suggestions",
                key="discard_ai_annotation_suggestions",
                use_container_width=True,
            )

        if discard_clicked:
            _clear_pending_suggestions()
            st.rerun()

        if apply_clicked:
            updated = prepare_annotation_dataframe(
                st.session_state.dataset,
                meta["annotation_column"],
            )
            allowed_set = set(meta["allowed_labels"])
            saved_count = 0
            audit_events = []
            reviewed_at = datetime.now(timezone.utc).isoformat()

            for _, row in edited_df.iterrows():
                suggested_value = row.get("AI Suggested Label", "")
                suggested_label = (
                    "" if pd.isna(suggested_value) else str(suggested_value).strip()
                )

                approved_value = row.get("Approved Label", "")
                approved_label = (
                    "" if pd.isna(approved_value) else str(approved_value).strip()
                )

                if approved_label and approved_label not in allowed_set:
                    st.error(
                        f"Invalid approved label on row {row['Row #']}: {approved_label}"
                    )
                    st.stop()

                # Keep an audit trail only for requests that returned a valid suggestion.
                # Raw text is deliberately not stored in the audit history.
                if suggested_label:
                    if not approved_label:
                        decision = "rejected"
                    elif approved_label.casefold() == suggested_label.casefold():
                        decision = "accepted"
                    else:
                        decision = "corrected"

                    raw_confidence = row.get("Confidence (%)", 0)
                    try:
                        confidence = int(raw_confidence)
                    except (TypeError, ValueError):
                        confidence = 0

                    audit_events.append(
                        {
                            "reviewed_at_utc": reviewed_at,
                            "model": MODEL_NAME,
                            "annotation_column": meta["annotation_column"],
                            "suggested_label": suggested_label,
                            "approved_label": approved_label,
                            "decision": decision,
                            "confidence_percent": max(0, min(100, confidence)),
                        }
                    )

                if approved_label:
                    position = int(row["_position"])
                    original_index = updated.index[position]
                    updated.at[
                        original_index,
                        meta["annotation_column"],
                    ] = approved_label
                    saved_count += 1

            if saved_count:
                save_dataframe(updated, key="main_dataset")
                save_state(
                    "dataset_source",
                    st.session_state.get("current_source", "Saved dataset"),
                )
                st.session_state.dataset = updated

                for key in (
                    "split_train_df",
                    "split_test_df",
                    "split_leakage_report",
                    "split_settings",
                ):
                    st.session_state.pop(key, None)

            if audit_events:
                existing_history = load_state("ai_annotation_audit", default=[])
                if not isinstance(existing_history, list):
                    existing_history = []
                existing_history.extend(audit_events)
                # Keep a bounded history to prevent the SQLite JSON state from growing indefinitely.
                save_state("ai_annotation_audit", existing_history[-10000:])

            if saved_count == 0 and not audit_events:
                st.warning(
                    "No valid AI suggestions or approved labels were available to save."
                )
            else:
                st.session_state.ai_annotation_saved_message = (
                    f"Saved {saved_count} annotation(s) and recorded "
                    f"{len(audit_events)} AI review decision(s) in SQLite."
                )
                _clear_pending_suggestions()
                st.rerun()

