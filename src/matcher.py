import itertools
from io import BytesIO
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from rapidfuzz import fuzz


IGNORED_KEYWORDS = {
    "comment",
    "remarks",
    "status",
}

ID_LIKE_TOKENS = {
    "id",
    "identifier",
    "ref",
    "reference",
    "account",
    "account number",
    "account_no",
    "account no",
    "client id",
    "customer id",
    "vendor id",
    "code",
    "number",
}


def infer_column_roles(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, str]:
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    roles = {}

    for col in common_cols:
        name = col.lower()

        if is_id_like_column(col):
            roles[col] = "identifier"
            continue

        if any(x in name for x in ["name", "client", "company", "vendor", "customer", "entity"]):
            roles[col] = "name"
            continue

        if any(x in name for x in ["status", "state", "stage"]):
            roles[col] = "status"
            continue

        if any(x in name for x in ["date", "time"]):
            roles[col] = "date"
            continue

        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            roles[col] = "numeric"
            continue

        if any(x in name for x in ["code", "type", "category"]):
            roles[col] = "category"
            continue

        roles[col] = "text"

    return roles


def infer_display_anchor_column(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_columns: Sequence[str],
) -> str | None:
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    if not common_cols:
        return None

    candidates = []

    def add_candidate(col: str, score: int) -> None:
        if col in common_cols:
            non_null_a = df_a[col].notna().sum()
            non_null_b = df_b[col].notna().sum()
            if non_null_a > 0 and non_null_b > 0:
                candidates.append((col, score))

    for col in common_cols:
        name = col.lower().replace("_", " ").strip()

        if any(token in name for token in [
            "name", "client", "customer", "vendor", "company",
            "entity", "counterparty", "merchant", "payee", "beneficiary"
        ]):
            add_candidate(col, 100)
            continue

        if any(token in name for token in [
            "description", "narrative", "details", "memo", "remarks",
            "expense", "invoice", "order", "transaction", "booking", "reference",
            "ref", "policy", "shipment", "product", "item"
        ]):
            add_candidate(col, 80)
            continue

        if col in key_columns:
            add_candidate(col, 60)
            continue

        add_candidate(col, 10)

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: (x[1], score_column_name(x[0])),
        reverse=True,
    )
    return candidates[0][0]


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = " ".join(text.split())
    return text


def is_id_like_column(column_name: str) -> bool:
    name = column_name.lower().replace("_", " ").strip()
    return any(token in name for token in ID_LIKE_TOKENS)


def score_column_name(column_name: str) -> int:
    name = column_name.lower().replace("_", " ")
    score = 0

    positive_tokens = {
        "id": 8,
        "identifier": 8,
        "reference": 7,
        "ref": 6,
        "account": 6,
        "code": 5,
        "number": 4,
        "name": 4,
        "client": 4,
        "company": 4,
        "counterparty": 4,
        "entity": 5,
        "vendor": 3,
        "customer": 3,
        "booking": 5,
        "profit": 3,
        "desk": 2,
        "country": 2,
        "branch": 2,
        "office": 2,
        "business": 2,
        "type": 1,
    }

    negative_tokens = {
        "comment": -3,
        "remark": -3,
        "description": -1,
        "address": -1,
        "phone": -2,
        "email": -2,
        "timestamp": -3,
    }

    for token, weight in positive_tokens.items():
        if token in name:
            score += weight

    for token, weight in negative_tokens.items():
        if token in name:
            score += weight

    parts = set(name.split())
    if parts & IGNORED_KEYWORDS:
        score -= 2

    return score


def build_join_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if len(columns) == 1:
        return df[columns[0]].fillna("").astype(str).map(normalize_text)

    return (
        df[list(columns)]
        .fillna("")
        .astype(str)
        .apply(lambda row: " | ".join(normalize_text(v) for v in row), axis=1)
    )


def dataframe_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)

    for col in df.columns:
        non_null = df[col].notna().sum()
        blank_count = total - non_null
        distinct = df[col].nunique(dropna=True)
        uniqueness_ratio = distinct / non_null if non_null else 0
        blank_pct = blank_count / total if total else 0

        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null_pct": round((non_null / total) * 100, 1) if total else 0,
                "blank_pct": round(blank_pct, 4),
                "unique_values": int(distinct),
                "uniqueness_ratio": round(uniqueness_ratio, 4),
                "name_score": score_column_name(col),
                "is_id_like": is_id_like_column(col),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["is_id_like", "name_score", "uniqueness_ratio", "non_null_pct"],
        ascending=False,
    )


def suggest_key_combinations(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    max_combo_size: int = 3,
    max_candidates: int = 6,
) -> List[Dict]:
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    if not common_cols:
        return []

    profile_a = dataframe_profile(df_a)
    profile_b = dataframe_profile(df_b)

    profile_a_map = profile_a.set_index("column").to_dict("index")
    profile_b_map = profile_b.set_index("column").to_dict("index")

    ranked_cols = [
        c
        for c in profile_a["column"].tolist()
        if c in common_cols and df_a[c].notna().sum() > 0 and df_b[c].notna().sum() > 0
    ]

    ranked_cols = ranked_cols[: min(len(ranked_cols), 8)]

    suggestions: List[Dict] = []
    seen = set()

    def combo_uniqueness(df: pd.DataFrame, cols: Sequence[str]) -> float:
        if len(df) == 0:
            return 0.0

        key_series = (
            df[list(cols)]
            .fillna("")
            .astype(str)
            .apply(lambda row: " | ".join(normalize_text(v) for v in row), axis=1)
        )
        return key_series.nunique(dropna=True) / len(df)

    for size in range(1, max_combo_size + 1):
        for combo in itertools.combinations(ranked_cols, size):
            key = tuple(combo)
            if key in seen:
                continue
            seen.add(key)

            uniq_a = combo_uniqueness(df_a, combo)
            uniq_b = combo_uniqueness(df_b, combo)
            avg_uniq = (uniq_a + uniq_b) / 2
            exact_unique_both = uniq_a >= 0.98 and uniq_b >= 0.98
            name_bonus = sum(score_column_name(c) for c in combo)
            score = avg_uniq + (0.02 * name_bonus) - (0.02 * (size - 1))

            id_priority = False
            id_quality_note = ""

            if len(combo) == 1 and is_id_like_column(combo[0]):
                col = combo[0]
                a_prof = profile_a_map.get(col, {})
                b_prof = profile_b_map.get(col, {})

                uniq_a_non_null = a_prof.get("uniqueness_ratio", 0)
                uniq_b_non_null = b_prof.get("uniqueness_ratio", 0)
                blank_a = a_prof.get("blank_pct", 1)
                blank_b = b_prof.get("blank_pct", 1)

                if (
                    uniq_a_non_null >= 0.95
                    and uniq_b_non_null >= 0.95
                    and blank_a <= 0.20
                    and blank_b <= 0.20
                ):
                    score += 0.25
                    id_priority = True
                    id_quality_note = "Preferred: ID-like column is highly unique with acceptable blanks"

            suggestions.append(
                {
                    "columns": list(combo),
                    "unique_a": round(uniq_a, 3),
                    "unique_b": round(uniq_b, 3),
                    "avg_unique": round(avg_uniq, 3),
                    "exact_unique_both": exact_unique_both,
                    "score": round(score, 3),
                    "id_priority": id_priority,
                    "id_quality_note": id_quality_note,
                }
            )

    suggestions = sorted(
        suggestions,
        key=lambda x: (x["id_priority"], x["exact_unique_both"], x["score"], x["avg_unique"]),
        reverse=True,
    )

    return suggestions[:max_candidates]


def _get_best_name_like_column(df_a: pd.DataFrame, df_b: pd.DataFrame) -> str | None:
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    if not common_cols:
        return None

    ranked = sorted(common_cols, key=score_column_name, reverse=True)
    for col in ranked:
        col_name = col.lower()
        if any(
            token in col_name
            for token in ["name", "client", "company", "counterparty", "entity", "vendor", "customer"]
        ):
            return col
    return ranked[0] if ranked else None


def _find_numeric_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    numeric_cols = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(df_a[col]) and pd.api.types.is_numeric_dtype(df_b[col]):
            numeric_cols.append(col)
    return numeric_cols


def _find_date_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> List[str]:
    common_cols = [c for c in df_a.columns if c in df_b.columns]
    date_like = []
    for col in common_cols:
        col_name = col.lower()
        if "date" in col_name or "time" in col_name:
            date_like.append(col)
    return date_like


def _safe_float(value):
    try:
        if pd.isna(value) or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _score_numeric(v1, v2) -> float | None:
    n1 = _safe_float(v1)
    n2 = _safe_float(v2)

    if n1 is None or n2 is None:
        return None

    if n1 == n2:
        return 1.0

    max_abs = max(abs(n1), abs(n2), 1.0)
    diff_ratio = abs(n1 - n2) / max_abs
    return max(0.0, 1.0 - diff_ratio)


def _score_date(v1, v2) -> float | None:
    try:
        d1 = pd.to_datetime(v1, errors="coerce")
        d2 = pd.to_datetime(v2, errors="coerce")
    except Exception:
        return None

    if pd.isna(d1) or pd.isna(d2):
        return None

    diff_days = abs((d1 - d2).days)
    if diff_days == 0:
        return 1.0
    if diff_days <= 1:
        return 0.9
    if diff_days <= 3:
        return 0.75
    if diff_days <= 7:
        return 0.5
    return 0.0


def _score_candidate(
    row_a: pd.Series,
    row_b: pd.Series,
    key_columns: Sequence[str],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> Tuple[float, Dict[str, float]]:
    component_scores: Dict[str, float] = {}

    key_a = " | ".join(normalize_text(row_a[col]) for col in key_columns)
    key_b = " | ".join(normalize_text(row_b[col]) for col in key_columns)
    component_scores["key_similarity"] = fuzz.ratio(key_a, key_b) / 100

    name_col = _get_best_name_like_column(df_a, df_b)
    if name_col:
        name_a = normalize_text(row_a.get(name_col, ""))
        name_b = normalize_text(row_b.get(name_col, ""))
        if name_a or name_b:
            component_scores["name_similarity"] = fuzz.token_sort_ratio(name_a, name_b) / 100

    common_cols = [c for c in df_a.columns if c in df_b.columns]
    numeric_cols = _find_numeric_columns(df_a, df_b)
    date_cols = _find_date_columns(df_a, df_b)

    exact_text_scores = []
    for col in common_cols:
        if col in key_columns:
            continue
        if col in numeric_cols:
            continue
        if col in date_cols:
            continue

        a_val = normalize_text(row_a.get(col, ""))
        b_val = normalize_text(row_b.get(col, ""))

        if a_val == "" and b_val == "":
            continue
        exact_text_scores.append(1.0 if a_val == b_val else 0.0)

    if exact_text_scores:
        component_scores["structured_text_match"] = sum(exact_text_scores) / len(exact_text_scores)

    numeric_scores = []
    for col in numeric_cols:
        score = _score_numeric(row_a.get(col), row_b.get(col))
        if score is not None:
            numeric_scores.append(score)

    if numeric_scores:
        component_scores["numeric_similarity"] = sum(numeric_scores) / len(numeric_scores)

    date_scores = []
    for col in date_cols:
        score = _score_date(row_a.get(col), row_b.get(col))
        if score is not None:
            date_scores.append(score)

    if date_scores:
        component_scores["date_similarity"] = sum(date_scores) / len(date_scores)

    weights = {
        "key_similarity": 0.40,
        "name_similarity": 0.25,
        "structured_text_match": 0.15,
        "numeric_similarity": 0.15,
        "date_similarity": 0.05,
    }

    weighted_total = 0.0
    total_weight = 0.0
    for metric, value in component_scores.items():
        weight = weights.get(metric, 0.0)
        weighted_total += value * weight
        total_weight += weight

    final_score = weighted_total / total_weight if total_weight > 0 else 0.0
    return final_score, component_scores


def _values_equal(a, b) -> bool:
    a_blank = pd.isna(a) or str(a).strip() == ""
    b_blank = pd.isna(b) or str(b).strip() == ""

    if a_blank and b_blank:
        return True

    if a_blank != b_blank:
        return False

    a_num = _safe_float(a)
    b_num = _safe_float(b)
    if a_num is not None and b_num is not None:
        return abs(a_num - b_num) < 1e-9

    a_dt = pd.to_datetime(a, errors="coerce")
    b_dt = pd.to_datetime(b, errors="coerce")
    if not pd.isna(a_dt) and not pd.isna(b_dt):
        return a_dt == b_dt

    return normalize_text(a) == normalize_text(b)


def _format_value(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return "[blank]"
    return str(value)


def _build_pair_id(pair_number: int) -> str:
    return f"PAIR_{pair_number:06d}"


def _build_pair_label(row: pd.Series, key_columns: Sequence[str]) -> str:
    parts = []
    for col in key_columns:
        if col in row.index:
            val = _format_value(row[col])
            parts.append(str(val))
    return " | ".join(parts) if parts else "[no key context]"


def _build_display_anchor(row: pd.Series, anchor_column: str | None, key_columns: Sequence[str]) -> str:
    if anchor_column and anchor_column in row.index:
        value = _format_value(row[anchor_column])
        if value != "[blank]":
            return value
    return _build_pair_label(row, key_columns)


def _classify_change(col: str, old, new, role: str) -> Tuple[str, str]:
    old_blank = pd.isna(old) or str(old).strip() == ""
    new_blank = pd.isna(new) or str(new).strip() == ""

    if old_blank and not new_blank:
        return "missing_to_populated", "high"
    if not old_blank and new_blank:
        return "populated_to_missing", "high"

    if role == "numeric":
        old_num = _safe_float(old)
        new_num = _safe_float(new)
        if old_num is not None and new_num is not None:
            if abs(old_num - new_num) < 1e-9:
                return "formatting", "low"
            return "numeric_change", "medium"

    if role == "date":
        d1 = pd.to_datetime(old, errors="coerce")
        d2 = pd.to_datetime(new, errors="coerce")
        if not pd.isna(d1) and not pd.isna(d2):
            if d1 == d2:
                return "formatting", "low"
            return "date_change", "medium"

    if role == "status":
        return "status_change", "high"

    if role == "identifier":
        return "identifier_change", "high"

    old_txt = normalize_text(old)
    new_txt = normalize_text(new)

    if old_txt == new_txt:
        return "formatting", "low"

    similarity = fuzz.token_sort_ratio(old_txt, new_txt)

    if similarity >= 90:
        return "name_variant", "low"
    elif similarity >= 70:
        return "text_change", "medium"
    else:
        return "text_change", "high"


def _extract_field_changes(
    row_a: pd.Series,
    row_b: pd.Series,
    pair_id: str,
    match_type: str,
    key_columns: Sequence[str],
    column_roles: Dict[str, str],
    pair_label_a: str,
    pair_label_b: str,
    pair_label: str,
    display_anchor_a: str,
    display_anchor_b: str,
) -> Tuple[List[str], List[Dict]]:
    changed_fields: List[str] = []
    field_change_rows: List[Dict] = []

    candidate_cols = []
    for col in row_a.index:
        col_str = str(col)
        if col_str.startswith("__"):
            continue
        if col in row_b.index:
            candidate_cols.append(col)

    for col in candidate_cols:
        old_value = row_a[col]
        new_value = row_b[col]
        role = column_roles.get(col, "text")

        if match_type == "exact_match" and col in key_columns:
            continue

        if col in key_columns and normalize_text(old_value) == normalize_text(new_value):
            continue

        if _values_equal(old_value, new_value):
            continue

        change_type, change_severity = _classify_change(col, old_value, new_value, role)

        changed_fields.append(col)

        field_change_rows.append(
            {
                "pair_id": pair_id,
                "pair_label": pair_label,
                "pair_label_a": pair_label_a,
                "pair_label_b": pair_label_b,
                "display_anchor_a": display_anchor_a,
                "display_anchor_b": display_anchor_b,
                "match_type": match_type,
                "field_name": col,
                "field_role": role,
                "is_key_field": col in key_columns,
                "change_type": change_type,
                "change_severity": change_severity,
                "old_value": old_value,
                "new_value": new_value,
                "old_value_display": _format_value(old_value),
                "new_value_display": _format_value(new_value),
            }
        )

    return changed_fields, field_change_rows


def _build_reason_summary(
    match_type: str,
    component_scores: Dict[str, float],
    change_count: int,
    changed_fields_text: str,
) -> str:
    key_similarity = component_scores.get("key_similarity", 0.0)
    name_similarity = component_scores.get("name_similarity", 0.0)
    structured_text_match = component_scores.get("structured_text_match", 0.0)
    numeric_similarity = component_scores.get("numeric_similarity")
    date_similarity = component_scores.get("date_similarity")

    reasons = []

    if key_similarity >= 0.95:
        reasons.append("key fields align very closely")
    elif key_similarity >= 0.80:
        reasons.append("key fields are similar")
    else:
        reasons.append("key fields do not align strongly")

    if "name_similarity" in component_scores:
        if name_similarity >= 0.95:
            reasons.append("name is almost identical")
        elif name_similarity >= 0.80:
            reasons.append("name is very similar")
        elif name_similarity >= 0.65:
            reasons.append("name is somewhat similar")
        else:
            reasons.append("name similarity is weak")

    if "structured_text_match" in component_scores:
        if structured_text_match >= 0.90:
            reasons.append("supporting text fields mostly match")
        elif structured_text_match >= 0.60:
            reasons.append("some supporting text fields match")
        else:
            reasons.append("supporting text fields differ materially")

    if numeric_similarity is not None:
        if numeric_similarity >= 0.95:
            reasons.append("numeric fields are closely aligned")
        elif numeric_similarity >= 0.70:
            reasons.append("numeric fields are reasonably close")
        else:
            reasons.append("numeric fields differ materially")

    if date_similarity is not None:
        if date_similarity >= 0.95:
            reasons.append("dates match")
        elif date_similarity >= 0.70:
            reasons.append("dates are close")
        else:
            reasons.append("dates differ")

    if match_type == "exact_match":
        base = "Exact match because the inferred record key matched exactly."
    elif match_type == "strong_match":
        base = "Strong match because " + ", ".join(reasons[:3]) + "."
    else:
        if key_similarity < 0.75:
            base = "Needs review: key fields do not align strongly."
        elif name_similarity >= 0.80 and structured_text_match < 0.50:
            base = "Needs review: name is similar but supporting fields differ."
        elif name_similarity < 0.70:
            base = "Needs review: name similarity is weak."
        else:
            base = "Needs review: multiple partial similarities but no strong match."

    if change_count == 0:
        change_sentence = " No field-level changes were detected."
    elif change_count <= 3:
        change_sentence = f" Changed fields: {changed_fields_text}."
    else:
        change_sentence = f" Multiple fields changed ({change_count}): {changed_fields_text}."

    return base + change_sentence


def _merge_rows(
    row_a: pd.Series,
    row_b: pd.Series,
    score: float,
    match_type: str,
    component_scores: Dict[str, float],
    key_columns: Sequence[str],
    pair_id: str,
    column_roles: Dict[str, str],
    display_anchor_column: str | None,
) -> Tuple[Dict, List[Dict]]:
    pair_label_a = _build_pair_label(row_a, key_columns)
    pair_label_b = _build_pair_label(row_b, key_columns)

    pair_label = (
        pair_label_a if pair_label_a == pair_label_b else f"{pair_label_a} → {pair_label_b}"
    )

    display_anchor_a = _build_display_anchor(row_a, display_anchor_column, key_columns)
    display_anchor_b = _build_display_anchor(row_b, display_anchor_column, key_columns)

    if match_type == "exact_match":
        match_stage = "Stage 1: Exact Key"
    elif match_type == "strong_match":
        match_stage = "Stage 2: Probable Match"
    else:
        match_stage = "Stage 3: Needs Review"

    changed_fields, field_change_rows = _extract_field_changes(
        row_a=row_a,
        row_b=row_b,
        pair_id=pair_id,
        match_type=match_type,
        key_columns=key_columns,
        column_roles=column_roles,
        pair_label_a=pair_label_a,
        pair_label_b=pair_label_b,
        pair_label=pair_label,
        display_anchor_a=display_anchor_a,
        display_anchor_b=display_anchor_b,
    )

    changed_fields_text = ", ".join(changed_fields) if changed_fields else "No field changes"
    change_count = len(changed_fields)
    change_status = "changed" if change_count > 0 else "unchanged"

    severity_rank = {"low": 1, "medium": 2, "high": 3}

    if field_change_rows:
        highest_severity = max(
            field_change_rows,
            key=lambda x: severity_rank.get(x["change_severity"], 1)
        )["change_severity"]
        critical_count = sum(1 for x in field_change_rows if x["change_severity"] == "high")
    else:
        highest_severity = "none"
        critical_count = 0

    reason_summary = _build_reason_summary(
        match_type=match_type,
        component_scores=component_scores,
        change_count=change_count,
        changed_fields_text=changed_fields_text,
    )

    merged = {
        "pair_id": pair_id,
        "pair_label": pair_label,
        "pair_label_a": pair_label_a,
        "pair_label_b": pair_label_b,
        "display_anchor_a": display_anchor_a,
        "display_anchor_b": display_anchor_b,
        **{f"A::{col}": row_a[col] for col in row_a.index if not str(col).startswith("__")},
        **{f"B::{col}": row_b[col] for col in row_b.index if not str(col).startswith("__")},
        "match_type": match_type,
        "match_stage": match_stage,
        "match_score": round(score, 3),
        "match_key_columns": " + ".join(key_columns),
        "change_status": change_status,
        "highest_change_severity": highest_severity,
        "critical_change_count": critical_count,
        "reason_summary": reason_summary,
        "changed_fields": changed_fields_text,
        "change_count": change_count,
    }

    for metric, value in component_scores.items():
        merged[metric] = round(value, 3)

    return merged, field_change_rows


def _ensure_match_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_defaults = {
        "pair_id": "",
        "pair_label": "",
        "pair_label_a": "",
        "pair_label_b": "",
        "display_anchor_a": "",
        "display_anchor_b": "",
        "match_type": "",
        "match_stage": "",
        "match_score": 0.0,
        "match_key_columns": "",
        "change_status": "unchanged",
        "highest_change_severity": "none",
        "critical_change_count": 0,
        "reason_summary": "",
        "changed_fields": "",
        "change_count": 0,
    }

    df = df.copy()
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def _ensure_field_changes_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_defaults = {
        "pair_id": "",
        "pair_label": "",
        "pair_label_a": "",
        "pair_label_b": "",
        "display_anchor_a": "",
        "display_anchor_b": "",
        "match_type": "",
        "field_name": "",
        "field_role": "text",
        "is_key_field": False,
        "change_type": "",
        "change_severity": "low",
        "old_value": None,
        "new_value": None,
        "old_value_display": "",
        "new_value_display": "",
    }

    df = df.copy()
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def reconcile_records(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    key_columns: Sequence[str],
    exact_threshold: float = 0.999,
    high_prob_threshold: float = 0.85,
    review_threshold: float = 0.65,
) -> Dict[str, pd.DataFrame | Dict[str, int] | Dict[str, str]]:
    working_a = df_a.copy().reset_index(drop=True)
    working_b = df_b.copy().reset_index(drop=True)

    working_a["__source_row_a"] = working_a.index
    working_b["__source_row_b"] = working_b.index
    working_a["__key"] = build_join_key(working_a, key_columns)
    working_b["__key"] = build_join_key(working_b, key_columns)

    exact_unchanged: List[Dict] = []
    exact_changed: List[Dict] = []
    strong_matches: List[Dict] = []
    needs_review: List[Dict] = []
    field_changes: List[Dict] = []

    consumed_a = set()
    consumed_b = set()

    reserved_a = set()
    reserved_b = set()

    column_roles = infer_column_roles(df_a, df_b)
    display_anchor_column = infer_display_anchor_column(df_a, df_b, key_columns)
    pair_counter = 1

    key_to_b_indices: Dict[str, List[int]] = {}
    for idx_b, row_b in working_b.iterrows():
        key = row_b["__key"]
        key_to_b_indices.setdefault(key, []).append(idx_b)

    for idx_a, row_a in working_a.iterrows():
        key = row_a["__key"]
        if not key:
            continue

        available_b = [
            idx for idx in key_to_b_indices.get(key, [])
            if idx not in consumed_b
        ]

        if available_b:
            idx_b = available_b[0]
            row_b = working_b.loc[idx_b]
            pair_id = _build_pair_id(pair_counter)
            pair_counter += 1

            merged_row, change_rows = _merge_rows(
                row_a=row_a,
                row_b=row_b,
                score=1.0,
                match_type="exact_match",
                component_scores={"key_similarity": 1.0},
                key_columns=key_columns,
                pair_id=pair_id,
                column_roles=column_roles,
                display_anchor_column=display_anchor_column,
            )

            if merged_row["change_count"] == 0:
                exact_unchanged.append(merged_row)
            else:
                exact_changed.append(merged_row)

            field_changes.extend(change_rows)

            consumed_a.add(idx_a)
            consumed_b.add(idx_b)

    remaining_a = working_a.loc[~working_a.index.isin(list(consumed_a))]
    remaining_b = working_b.loc[~working_b.index.isin(list(consumed_b))]

    for idx_a, row_a in remaining_a.iterrows():
        if idx_a in reserved_a:
            continue

        best_score = -1.0
        second_best_score = -1.0
        best_idx_b = None
        best_components: Dict[str, float] = {}

        for idx_b, row_b in remaining_b.iterrows():
            if idx_b in consumed_b or idx_b in reserved_b:
                continue

            score, components = _score_candidate(
                row_a=row_a,
                row_b=row_b,
                key_columns=key_columns,
                df_a=working_a.drop(columns=["__source_row_a", "__key"], errors="ignore"),
                df_b=working_b.drop(columns=["__source_row_b", "__key"], errors="ignore"),
            )

            if score > best_score:
                second_best_score = best_score
                best_score = score
                best_idx_b = idx_b
                best_components = components
            elif score > second_best_score:
                second_best_score = score

        if best_idx_b is None:
            continue

        row_b = working_b.loc[best_idx_b]
        pair_id = _build_pair_id(pair_counter)
        pair_counter += 1

        key_similarity = best_components.get("key_similarity", 0.0)
        name_similarity = best_components.get("name_similarity", 0.0)
        structured_text_match = best_components.get("structured_text_match", 0.0)

        margin = best_score if second_best_score < 0 else (best_score - second_best_score)

        strong_eligible = (
            (
                best_score >= high_prob_threshold
                or (
                    best_score >= (high_prob_threshold - 0.03)
                    and key_similarity >= 0.85
                    and name_similarity >= 0.85
                )
            )
            and key_similarity >= 0.78
            and margin >= 0.03
        )

        review_eligible = (
            best_score >= review_threshold
            and key_similarity >= 0.72
            and margin >= 0.02
            and (
                name_similarity >= 0.65
                or structured_text_match >= 0.50
            )
        )

        if strong_eligible:
            merged_row, change_rows = _merge_rows(
                row_a=row_a,
                row_b=row_b,
                score=best_score,
                match_type="strong_match",
                component_scores=best_components,
                key_columns=key_columns,
                pair_id=pair_id,
                column_roles=column_roles,
                display_anchor_column=display_anchor_column,
            )
            strong_matches.append(merged_row)
            field_changes.extend(change_rows)

            consumed_a.add(idx_a)
            consumed_b.add(best_idx_b)

        elif review_eligible:
            merged_row, change_rows = _merge_rows(
                row_a=row_a,
                row_b=row_b,
                score=best_score,
                match_type="needs_review",
                component_scores=best_components,
                key_columns=key_columns,
                pair_id=pair_id,
                column_roles=column_roles,
                display_anchor_column=display_anchor_column,
            )
            needs_review.append(merged_row)
            field_changes.extend(change_rows)

            reserved_a.add(idx_a)
            reserved_b.add(best_idx_b)

    unmatched_a_idx = [
        idx for idx in working_a.index
        if idx not in consumed_a and idx not in reserved_a
    ]
    unmatched_b_idx = [
        idx for idx in working_b.index
        if idx not in consumed_b and idx not in reserved_b
    ]

    only_in_a = working_a.loc[unmatched_a_idx].drop(
        columns=["__source_row_a", "__key"],
        errors="ignore",
    )

    only_in_b = working_b.loc[unmatched_b_idx].drop(
        columns=["__source_row_b", "__key"],
        errors="ignore",
    )

    exact_unchanged_df = _ensure_match_dataframe_columns(pd.DataFrame(exact_unchanged))
    exact_changed_df = _ensure_match_dataframe_columns(pd.DataFrame(exact_changed))
    strong_matches_df = _ensure_match_dataframe_columns(pd.DataFrame(strong_matches))
    needs_review_df = _ensure_match_dataframe_columns(pd.DataFrame(needs_review))
    field_changes_df = _ensure_field_changes_columns(pd.DataFrame(field_changes))

    summary = {
        "records_a": len(df_a),
        "records_b": len(df_b),
        "exact_unchanged": len(exact_unchanged_df),
        "exact_changed": len(exact_changed_df),
        "strong_matches": len(strong_matches_df),
        "needs_review": len(needs_review_df),
        "only_in_a": len(only_in_a),
        "only_in_b": len(only_in_b),
        "matched_pairs": (
            len(exact_unchanged_df)
            + len(exact_changed_df)
            + len(strong_matches_df)
            + len(needs_review_df)
        ),
        "changed_pairs": int(
            (exact_changed_df["change_count"] > 0).sum()
            + (strong_matches_df["change_count"] > 0).sum()
            + (needs_review_df["change_count"] > 0).sum()
        ),
        "field_changes": len(field_changes_df),
    }

    return {
        "exact_unchanged": exact_unchanged_df,
        "exact_changed": exact_changed_df,
        "strong_matches": strong_matches_df,
        "needs_review": needs_review_df,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "field_changes": field_changes_df,
        "summary": summary,
        "column_roles": column_roles,
        "display_anchor_column": display_anchor_column or "",
    }


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def combine_results_to_excel(results: Dict[str, pd.DataFrame | Dict[str, int] | Dict[str, str]]) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, obj in results.items():
            if sheet_name == "summary" and isinstance(obj, dict):
                pd.DataFrame([obj]).to_excel(writer, sheet_name="summary", index=False)
                continue

            if sheet_name == "column_roles" and isinstance(obj, dict):
                roles_df = pd.DataFrame(
                    [{"column": column, "role": role} for column, role in obj.items()]
                )
                roles_df.to_excel(writer, sheet_name="column_roles", index=False)
                continue

            if sheet_name == "display_anchor_column" and isinstance(obj, str):
                pd.DataFrame([{"display_anchor_column": obj}]).to_excel(
                    writer, sheet_name="display_anchor", index=False
                )
                continue

            if isinstance(obj, pd.DataFrame):
                safe_name = sheet_name[:31]
                obj.to_excel(writer, sheet_name=safe_name, index=False)

    output.seek(0)
    return output.read()
