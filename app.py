import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        for encoding in ["utf-8", "utf-8-sig", "latin1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=encoding)
            except Exception:
                continue
        raise ValueError("Unable to read CSV file.")

    if file_name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type.")

from src.config import APP_SUBTITLE, APP_TITLE, MATCH_MODES
from src.matcher import (
    combine_results_to_excel,
    convert_df_to_csv,
    reconcile_records,
    suggest_key_combinations,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        border: 1px solid #d9d9d9;
        border-radius: 12px;
        padding: 10px 12px;
        background: transparent;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] div {
        color: inherit !important;
        opacity: 1 !important;
    }

    div[data-testid="stMetricValue"] {
        color: inherit !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        color: inherit !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_sample_pair(sample_choice: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sample_choice == "System A vs B (Sample Data)":
        df_a = pd.read_csv("sample_data/client_list_system_a.csv")
        df_b = pd.read_csv("sample_data/client_list_system_b.csv")
        return df_a, df_b

    raise ValueError(f"Unknown sample selection: {sample_choice}")

def empty_match_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pair_id",
            "pair_label",
            "pair_label_a",
            "pair_label_b",
            "display_anchor_a",
            "display_anchor_b",
            "match_type",
            "match_stage",
            "match_score",
            "reason_summary",
            "change_count",
        ]
    )


def empty_field_changes_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "pair_id",
            "display_anchor_a",
            "display_anchor_b",
            "pair_label_a",
            "match_type",
            "field_name",
            "change_type",
            "change_severity",
            "old_value_display",
            "new_value_display",
        ]
    )


def _build_entity_series(df: pd.DataFrame) -> pd.Series:
    entity = pd.Series([""] * len(df), index=df.index, dtype="object")

    if "display_anchor_a" in df.columns:
        entity = df["display_anchor_a"].fillna("").astype(str).str.strip()

    if "pair_label_a" in df.columns:
        fallback_mask = entity.isin(["", "[blank]"])
        entity.loc[fallback_mask] = (
            df.loc[fallback_mask, "pair_label_a"].fillna("").astype(str).str.strip()
        )

    if "pair_id" in df.columns:
        fallback_mask = entity.isin(["", "[blank]"])
        entity.loc[fallback_mask] = df.loc[fallback_mask, "pair_id"].astype(str)

    return entity.replace("", "[unlabeled record]")


def _clean_reason(text: str) -> str:
    if pd.isna(text):
        return ""
    return (
        str(text)
        .replace(" No field-level changes were detected.", "")
        .replace(" Changed fields:", " Fields changed:")
        .strip()
    )


def _get_exact_frames(results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_unchanged = results.get("exact_unchanged")
    exact_changed = results.get("exact_changed")

    if exact_unchanged is None and exact_changed is None:
        exact_all = results.get("exact_matches")
        if isinstance(exact_all, pd.DataFrame) and not exact_all.empty:
            if "change_count" in exact_all.columns:
                exact_unchanged = exact_all.loc[exact_all["change_count"].fillna(0) == 0].copy()
                exact_changed = exact_all.loc[exact_all["change_count"].fillna(0) > 0].copy()
            else:
                exact_unchanged = exact_all.copy()
                exact_changed = empty_match_df()
        else:
            exact_unchanged = empty_match_df()
            exact_changed = empty_match_df()

    if exact_unchanged is None:
        exact_unchanged = empty_match_df()
    if exact_changed is None:
        exact_changed = empty_match_df()

    return exact_unchanged, exact_changed


def build_match_review_display(df: pd.DataFrame, reason_header: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    display_df = pd.DataFrame(index=df.index)
    display_df["Entity"] = _build_entity_series(df)

    if "display_anchor_b" in df.columns:
        matched_to = df["display_anchor_b"].fillna("").astype(str).str.strip()
    elif "pair_label_b" in df.columns:
        matched_to = df["pair_label_b"].fillna("").astype(str).str.strip()
    else:
        matched_to = pd.Series([""] * len(df), index=df.index)

    display_df["Matched To"] = matched_to.replace("", "[unlabeled record]")

    if "match_stage" in df.columns:
        display_df["Match Stage"] = df["match_stage"]

    if "match_score" in df.columns:
        display_df["Match Score"] = df["match_score"]

    if "reason_summary" in df.columns:
        display_df[reason_header] = df["reason_summary"].map(_clean_reason)

    if "change_count" in df.columns:
        display_df["Changed Fields"] = df["change_count"].fillna(0).astype(int)

    sort_cols = []
    ascending = []

    if "Match Score" in display_df.columns:
        sort_cols.append("Match Score")
        ascending.append(False)
    if "Entity" in display_df.columns:
        sort_cols.append("Entity")
        ascending.append(True)

    if sort_cols:
        display_df = display_df.sort_values(by=sort_cols, ascending=ascending)

    return display_df


def build_unmatched_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working = df.copy()
    display_anchor_column = None
    preferred_tokens = [
        "name", "client", "customer", "vendor", "company", "entity",
        "counterparty", "merchant", "payee", "beneficiary", "expense",
        "description", "invoice", "order", "transaction", "booking",
        "reference", "ref", "policy", "shipment", "product", "item",
    ]

    for col in working.columns:
        col_name = str(col).lower().replace("_", " ").strip()
        if any(token in col_name for token in preferred_tokens):
            display_anchor_column = col
            break

    if display_anchor_column is not None:
        working["Entity"] = working[display_anchor_column].fillna("").astype(str).str.strip()
    else:
        first_col = working.columns[0]
        working["Entity"] = working[first_col].fillna("").astype(str).str.strip()

    working.loc[working["Entity"].isin(["", "[blank]"]), "Entity"] = "[unlabeled record]"
    business_cols = [col for col in working.columns if col != "Entity"]
    return working[["Entity"] + business_cols]


def build_field_changes_display(field_changes: pd.DataFrame) -> pd.DataFrame:
    if field_changes.empty:
        return field_changes.copy()

    display_df = field_changes.copy()

    entity = pd.Series([""] * len(display_df), index=display_df.index, dtype="object")

    if "display_anchor_a" in display_df.columns:
        entity = display_df["display_anchor_a"].fillna("").astype(str).str.strip()

    if "display_anchor_b" in display_df.columns:
        fallback_mask = entity.eq("")
        entity.loc[fallback_mask] = (
            display_df.loc[fallback_mask, "display_anchor_b"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "pair_label_a" in display_df.columns:
        fallback_mask = entity.isin(["", "[blank]"])
        entity.loc[fallback_mask] = (
            display_df.loc[fallback_mask, "pair_label_a"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "pair_id" in display_df.columns:
        fallback_mask = entity.isin(["", "[blank]"])
        entity.loc[fallback_mask] = display_df.loc[fallback_mask, "pair_id"].astype(str)

    display_df["Entity"] = entity.replace("", "[unlabeled record]")

    if "display_anchor_b" in display_df.columns:
        matched_to = display_df["display_anchor_b"].fillna("").astype(str).str.strip()
    elif "pair_label_b" in display_df.columns:
        matched_to = display_df["pair_label_b"].fillna("").astype(str).str.strip()
    else:
        matched_to = pd.Series([""] * len(display_df), index=display_df.index, dtype="object")

    display_df["Matched To"] = matched_to.replace("", "[unlabeled record]")

    display_df = display_df.rename(
        columns={
            "match_type": "Match Type",
            "field_name": "Column",
            "change_type": "Change Type",
            "change_severity": "Severity",
            "old_value_display": "A.Value",
            "new_value_display": "B.Value",
        }
    )

    severity_order = {"high": 0, "medium": 1, "low": 2}
    if "Severity" in display_df.columns:
        display_df["_severity_sort"] = display_df["Severity"].map(severity_order).fillna(99)
        sort_by = ["_severity_sort", "Entity"]
        if "Column" in display_df.columns:
            sort_by.append("Column")
        display_df = display_df.sort_values(by=sort_by, ascending=True)

    keep_cols = [
        col
        for col in [
            "Entity",
            "Matched To",
            "Match Type",
            "Column",
            "Change Type",
            "Severity",
            "A.Value",
            "B.Value",
        ]
        if col in display_df.columns
    ]
    return display_df[keep_cols]


def metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

st.markdown(
    """
### What this tool does

Upload two files, or use sample data, to:
- reconcile records across two sources
- identify exact and likely matches
- highlight missing or unmatched records
- surface field-level differences for analyst review
"""
)

st.subheader("Choose data source")

sample_choice = st.selectbox(
    "Use sample data or upload your own files",
    options=[
        "Upload my own files",
        "System A vs B (Sample Data)",
    ],
)

use_sample_data = sample_choice != "Upload my own files"

if use_sample_data:
    try:
        df_a, df_b = load_sample_pair(sample_choice)
        st.success(f"Loaded sample dataset: {sample_choice}")
    except Exception as exc:
        st.error(f"Failed to load sample data: {exc}")
        st.stop()
else:
    st.subheader("Upload files")
    col_up_1, col_up_2 = st.columns(2)
    with col_up_1:
        file_a = st.file_uploader("Upload File A", type=["csv", "xlsx", "xls"], key="file_a")
    with col_up_2:
        file_b = st.file_uploader("Upload File B", type=["csv", "xlsx", "xls"], key="file_b")

    if not (file_a and file_b):
        st.info("Upload both files to begin, or switch to a sample dataset above.")
        st.stop()

    try:
        df_a = load_file(file_a)
        df_b = load_file(file_b)
    except Exception as exc:
        st.error(f"Failed to read files: {exc}")
        st.stop()

if df_a.empty or df_b.empty:
    st.error("One of the selected datasets is empty.")
    st.stop()

prev_a, prev_b = st.columns(2)
with prev_a:
    st.markdown("**Preview — File A**")
    st.dataframe(df_a.head(10), use_container_width=True, height=260)
with prev_b:
    st.markdown("**Preview — File B**")
    st.dataframe(df_b.head(10), use_container_width=True, height=260)

common_columns = [c for c in df_a.columns if c in df_b.columns]
if not common_columns:
    st.error("The two files do not share any common column names. Align headers first.")
    st.stop()

st.subheader("Record key")
suggestions = suggest_key_combinations(df_a, df_b)

recommended_columns: list[str] = []
if suggestions:
    top = suggestions[0]
    recommended_columns = top["columns"]
    confidence_label = (
        "High confidence"
        if top["exact_unique_both"]
        else "Medium confidence"
        if top["avg_unique"] >= 0.85
        else "Low confidence"
    )
    st.info(
        f"Suggested key: {' + '.join(recommended_columns)} | "
        f"File A uniqueness: {top['unique_a']:.1%} | "
        f"File B uniqueness: {top['unique_b']:.1%} | "
        f"Confidence: {confidence_label}"
    )

    key_choice = st.radio(
        "How should the app build the record key?",
        options=[
            f"Use suggested key ({' + '.join(recommended_columns)})",
            "Choose columns manually",
        ],
        horizontal=True,
    )

    if key_choice.startswith("Use suggested key"):
        selected_key_columns = recommended_columns
        st.multiselect(
            "Suggested key columns",
            options=common_columns,
            default=selected_key_columns,
            disabled=True,
        )
    else:
        selected_key_columns = st.multiselect(
            "Select one or more key columns",
            options=common_columns,
            default=recommended_columns,
        )
else:
    selected_key_columns = st.multiselect(
        "Select one or more key columns",
        options=common_columns,
        default=common_columns[:1],
    )

with st.expander("Other key suggestions", expanded=False):
    if suggestions:
        suggestion_rows = []
        for item in suggestions:
            suggestion_rows.append(
                {
                    "Suggested key": " + ".join(item["columns"]),
                    "File A uniqueness": item["unique_a"],
                    "File B uniqueness": item["unique_b"],
                    "Avg uniqueness": item["avg_unique"],
                    "Exact unique in both": item["exact_unique_both"],
                }
            )
        st.dataframe(pd.DataFrame(suggestion_rows), use_container_width=True, height=220)

st.subheader("Matching mode")
mode = st.selectbox("Choose matching mode", list(MATCH_MODES.keys()), index=1)

if mode == "Manual":
    high_prob_threshold = st.number_input(
        "High-probability threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
    )
    review_threshold = st.number_input(
        "Review threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
    )
    if review_threshold > high_prob_threshold:
        st.warning("Review threshold should not be higher than the high-probability threshold.")
else:
    if mode == "Tight":
        high_prob_threshold, review_threshold = 0.90, 0.75
    elif mode == "Medium":
        high_prob_threshold, review_threshold = 0.85, 0.65
    else:
        high_prob_threshold, review_threshold = 0.75, 0.55
    st.caption(f"High probability: {high_prob_threshold:.2f} | Review: {review_threshold:.2f}")

run_clicked = st.button("Run Reconciliation", type="primary", use_container_width=True)

if not run_clicked:
    st.stop()

if not selected_key_columns:
    st.warning("Pick at least one key column before running reconciliation.")
    st.stop()

if mode == "Manual" and review_threshold > high_prob_threshold:
    st.warning("Fix thresholds before running reconciliation.")
    st.stop()

with st.spinner("Reconciling records..."):
    results = reconcile_records(
        df_a=df_a,
        df_b=df_b,
        key_columns=selected_key_columns,
        high_prob_threshold=high_prob_threshold,
        review_threshold=review_threshold,
    )

summary = results.get("summary", {})
exact_unchanged, exact_changed = _get_exact_frames(results)
strong_matches = results.get("strong_matches", empty_match_df())
needs_review = results.get("needs_review", empty_match_df())
only_in_a = results.get("only_in_a", pd.DataFrame())
only_in_b = results.get("only_in_b", pd.DataFrame())
field_changes = results.get("field_changes", empty_field_changes_df())

exact_matches_total = len(exact_unchanged) + len(exact_changed)
confident_matches = exact_matches_total + len(strong_matches)
denominator = max(int(summary.get("records_a", len(df_a))), int(summary.get("records_b", len(df_b))), 1)
reconciliation_rate = confident_matches / denominator
strict_rate = exact_matches_total / denominator
extended_rate = (confident_matches + len(needs_review)) / denominator
changed_records = int(
    summary.get(
        "changed_pairs",
        len(exact_changed)
        + (strong_matches.get("change_count", pd.Series(dtype=float)).fillna(0) > 0).sum()
        + (needs_review.get("change_count", pd.Series(dtype=float)).fillna(0) > 0).sum(),
    )
)

st.subheader("Summary")
row1 = st.columns(5)
with row1[0]:
    metric("Records in A", f"{int(summary.get('records_a', len(df_a))):,}")
with row1[1]:
    metric("Records in B", f"{int(summary.get('records_b', len(df_b))):,}")
with row1[2]:
    metric("Reconciliation %", f"{reconciliation_rate:.1%}")
with row1[3]:
    metric("Exact Matches", f"{exact_matches_total:,}")
with row1[4]:
    metric("Strong Matches", f"{len(strong_matches):,}")

row2 = st.columns(5)
with row2[0]:
    metric("Needs Review", f"{len(needs_review):,}")
with row2[1]:
    metric("Only in A", f"{len(only_in_a):,}")
with row2[2]:
    metric("Only in B", f"{len(only_in_b):,}")
with row2[3]:
    metric("Changed Records", f"{changed_records:,}")
with row2[4]:
    metric("Field Changes", f"{len(field_changes):,}")

st.caption(f"Strict exact rate: {strict_rate:.1%} | Extended rate including review: {extended_rate:.1%}")

with st.expander("How matching was performed", expanded=False):
    st.markdown(
        "\n".join(
            [
                f"- Stage 1: exact match on selected key (`{' + '.join(selected_key_columns)}`)",
                "- Stage 2: probable match on unmatched leftovers using name, supporting text, numeric, and date similarity",
                "- Stage 3: possible matches moved to review when confidence is plausible but not strong enough",
            ]
        )
    )

dl1, dl2 = st.columns(2)
with dl1:
    combined_matches = pd.concat(
        [exact_unchanged, exact_changed, strong_matches, needs_review],
        ignore_index=True,
    )
    st.download_button(
        "Download matched and reviewed pairs (CSV)",
        data=convert_df_to_csv(combined_matches),
        file_name="reconciliation_pairs.csv",
        mime="text/csv",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "Download full results workbook",
        data=combine_results_to_excel(results),
        file_name="reconciliation_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

if not strong_matches.empty:
    st.subheader("Strong Matches")
    st.dataframe(
        build_match_review_display(strong_matches, "Why Matched"),
        use_container_width=True,
        height=320,
    )

if not needs_review.empty:
    st.subheader("Needs Review")
    st.dataframe(
        build_match_review_display(needs_review, "Why Review"),
        use_container_width=True,
        height=320,
    )

st.subheader("Orphans")
orph1, orph2 = st.columns(2)
with orph1:
    st.markdown(f"**Only in A ({len(only_in_a):,})**")
    if only_in_a.empty:
        st.write("No records exist only in File A.")
    else:
        st.dataframe(build_unmatched_display(only_in_a), use_container_width=True, height=320)
with orph2:
    st.markdown(f"**Only in B ({len(only_in_b):,})**")
    if only_in_b.empty:
        st.write("No records exist only in File B.")
    else:
        st.dataframe(build_unmatched_display(only_in_b), use_container_width=True, height=320)

st.subheader("Field Changes")
if not field_changes.empty:
    st.dataframe(
        build_field_changes_display(field_changes),
        use_container_width=True,
        height=420,
    )
else:
    st.write("No field-level differences detected.")

with st.expander("Upcoming enhancements", expanded=False):
    st.markdown(
        "\n".join(
            [
                "- Intent-aware matching recommendations for asymmetric use cases",
                "- Search one file against the other for screening workflows",
                "- User-configurable multi-step matching ladders",
                "- Analyst actions for review decisions and match overrides",
            ]
        )
    )