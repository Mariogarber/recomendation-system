from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TABULAR_BAND_TO_EXPERT = {
    "0": "cold_user_tabular",
    "1": "very_short_history_tabular",
    "2-20": "mid_history_tabular",
    ">20": "long_history_tabular",
}

TABULAR_HISTORY_OVERALL_COLUMNS = [
    "tabular_history_count",
    "tabular_history_mean",
    "tabular_history_std",
    "tabular_history_min",
    "tabular_history_max",
    "tabular_history_median",
    "tabular_history_positive_rate",
    "tabular_history_low_rate",
    "tabular_history_business_stars_mean",
    "tabular_history_business_star_gap",
    "tabular_history_span_days",
    "tabular_history_days_since_last",
    "tabular_history_feature_completeness",
]

TABULAR_HISTORY_AFFINITY_COLUMNS = [
    "tabular_history_same_city_count",
    "tabular_history_same_city_mean",
    "tabular_history_same_city_ratio",
    "tabular_history_same_state_count",
    "tabular_history_same_state_mean",
    "tabular_history_same_state_ratio",
    "tabular_history_same_category_count",
    "tabular_history_same_category_mean",
    "tabular_history_same_category_ratio",
    "tabular_history_same_is_open_count",
    "tabular_history_same_is_open_mean",
    "tabular_history_same_is_open_ratio",
]

TABULAR_HISTORY_COLUMNS = [*TABULAR_HISTORY_OVERALL_COLUMNS, *TABULAR_HISTORY_AFFINITY_COLUMNS]

FORBIDDEN_GLOBAL_HISTORY_COLUMNS = [
    "user_known_in_train",
    "user_train_count",
    "user_train_mean",
    "user_train_std",
    "user_train_bias",
    "user_train_span_days",
    "user_train_days_since_last_review",
    "user_business_train_gap",
    "user_business_train_bias_sum",
]

PREFIX_ALIAS_MAP = {
    "tabular_history_count": "prefix_user_count",
    "tabular_history_mean": "prefix_user_mean",
    "tabular_history_std": "prefix_user_std",
    "tabular_history_min": "prefix_user_min",
    "tabular_history_max": "prefix_user_max",
    "tabular_history_median": "prefix_user_median",
    "tabular_history_positive_rate": "prefix_user_positive_rate",
    "tabular_history_low_rate": "prefix_user_low_rate",
    "tabular_history_business_stars_mean": "prefix_user_business_stars_mean",
    "tabular_history_business_star_gap": "prefix_user_business_gap",
    "tabular_history_span_days": "prefix_user_span_days",
    "tabular_history_days_since_last": "prefix_user_days_since_last",
    "tabular_history_feature_completeness": "prefix_user_feature_completeness",
    "tabular_history_same_city_count": "prefix_same_city_count",
    "tabular_history_same_city_mean": "prefix_same_city_mean",
    "tabular_history_same_city_ratio": "prefix_same_city_ratio",
    "tabular_history_same_state_count": "prefix_same_state_count",
    "tabular_history_same_state_mean": "prefix_same_state_mean",
    "tabular_history_same_state_ratio": "prefix_same_state_ratio",
    "tabular_history_same_category_count": "prefix_same_category_count",
    "tabular_history_same_category_mean": "prefix_same_category_mean",
    "tabular_history_same_category_ratio": "prefix_same_category_ratio",
    "tabular_history_same_is_open_count": "prefix_same_is_open_count",
    "tabular_history_same_is_open_mean": "prefix_same_is_open_mean",
    "tabular_history_same_is_open_ratio": "prefix_same_is_open_ratio",
}


@dataclass(slots=True)
class TabularMoESpec:
    router_spec: Any
    feature_columns_by_expert: dict[str, list[str]]
    categorical_columns_by_expert: dict[str, list[str]]
    blend_alpha_by_band: dict[str, float]
    routing_policy: dict[str, Any]
    feature_manifest: dict[str, Any]


class _RunningHistory:
    __slots__ = (
        "count",
        "sum",
        "sum_sq",
        "min_rating",
        "max_rating",
        "first_day",
        "last_day",
        "positive_count",
        "low_count",
        "business_stars_sum",
        "ratings",
    )

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_rating = np.inf
        self.max_rating = -np.inf
        self.first_day = np.nan
        self.last_day = np.nan
        self.positive_count = 0
        self.low_count = 0
        self.business_stars_sum = 0.0
        self.ratings: list[float] = []

    def update(self, *, rating: float, day: float, business_stars: float) -> None:
        value = float(rating)
        self.count += 1
        self.sum += value
        self.sum_sq += value * value
        self.min_rating = min(self.min_rating, value)
        self.max_rating = max(self.max_rating, value)
        if np.isfinite(day):
            if not np.isfinite(self.first_day):
                self.first_day = day
            self.last_day = day
        if value >= 4.0:
            self.positive_count += 1
        if value <= 2.0:
            self.low_count += 1
        if np.isfinite(business_stars):
            self.business_stars_sum += float(business_stars)
        self.ratings.append(value)

    def base_features(self, *, current_day: float, candidate_business_stars: float, global_mean: float) -> dict[str, float]:
        if self.count <= 0:
            return {
                "tabular_history_count": 0.0,
                "tabular_history_mean": float(global_mean),
                "tabular_history_std": 0.0,
                "tabular_history_min": float(global_mean),
                "tabular_history_max": float(global_mean),
                "tabular_history_median": float(global_mean),
                "tabular_history_positive_rate": 0.0,
                "tabular_history_low_rate": 0.0,
                "tabular_history_business_stars_mean": float(global_mean),
                "tabular_history_business_star_gap": 0.0,
                "tabular_history_span_days": 0.0,
                "tabular_history_days_since_last": 0.0,
            }
        mean = self.sum / self.count
        variance = max((self.sum_sq / self.count) - (mean * mean), 0.0)
        business_stars_mean = self.business_stars_sum / self.count
        if np.isfinite(current_day) and np.isfinite(self.last_day):
            days_since_last = max(current_day - self.last_day, 0.0)
        else:
            days_since_last = 0.0
        if np.isfinite(self.first_day) and np.isfinite(self.last_day):
            span_days = max(self.last_day - self.first_day, 0.0)
        else:
            span_days = 0.0
        return {
            "tabular_history_count": float(self.count),
            "tabular_history_mean": float(mean),
            "tabular_history_std": float(np.sqrt(variance)),
            "tabular_history_min": float(self.min_rating),
            "tabular_history_max": float(self.max_rating),
            "tabular_history_median": float(np.median(np.asarray(self.ratings, dtype=np.float32))),
            "tabular_history_positive_rate": float(self.positive_count / self.count),
            "tabular_history_low_rate": float(self.low_count / self.count),
            "tabular_history_business_stars_mean": float(business_stars_mean),
            "tabular_history_business_star_gap": float(mean - candidate_business_stars) if np.isfinite(candidate_business_stars) else 0.0,
            "tabular_history_span_days": float(span_days),
            "tabular_history_days_since_last": float(days_since_last),
        }


def _safe_key(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def _timestamp_to_day(value: Any) -> float:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return np.nan
    return float(ts.value / 86_400_000_000_000.0)


def _current_row_keys(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        _safe_key(row.get("business_city_top", row.get("business_city", "__other__")), "__other__"),
        _safe_key(row.get("business_state", "__missing__"), "__missing__"),
        _safe_key(row.get("business_primary_category_family", "__other__"), "__other__"),
        _safe_key(int(pd.to_numeric(row.get("business_is_open", 0.0), errors="coerce") > 0.5), "0"),
    )


def _group_feature_block(
    *,
    stats: _RunningHistory,
    keyed_stats: dict[str, dict[str, float]],
    key: str,
    prefix: str,
    global_mean: float,
) -> dict[str, float]:
    entry = keyed_stats.get(key)
    if entry is None or entry["count"] <= 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": float(global_mean),
            f"{prefix}_ratio": 0.0,
        }
    count = float(entry["count"])
    mean = float(entry["sum"] / max(entry["count"], 1))
    ratio = float(count / stats.count) if stats.count > 0 else 0.0
    return {
        f"{prefix}_count": count,
        f"{prefix}_mean": mean,
        f"{prefix}_ratio": ratio,
    }


def _compute_feature_completeness(row_features: dict[str, float]) -> float:
    available = sum(
        1
        for name in (
            "tabular_history_same_city_count",
            "tabular_history_same_state_count",
            "tabular_history_same_category_count",
            "tabular_history_same_is_open_count",
        )
        if row_features.get(name, 0.0) > 0.0
    )
    return float(available / 4.0)


def _prefix_features_from_state(
    *,
    stats: _RunningHistory | None,
    current_day: float,
    candidate_business_stars: float,
    city_stats: dict[str, dict[str, float]],
    state_stats: dict[str, dict[str, float]],
    category_stats: dict[str, dict[str, float]],
    open_stats: dict[str, dict[str, float]],
    city_key: str,
    state_key: str,
    category_key: str,
    open_key: str,
    global_mean: float,
) -> dict[str, float]:
    history = stats if stats is not None else _RunningHistory()
    row_features = history.base_features(
        current_day=current_day,
        candidate_business_stars=candidate_business_stars,
        global_mean=global_mean,
    )
    row_features.update(
        _group_feature_block(
            stats=history,
            keyed_stats=city_stats,
            key=city_key,
            prefix="tabular_history_same_city",
            global_mean=global_mean,
        )
    )
    row_features.update(
        _group_feature_block(
            stats=history,
            keyed_stats=state_stats,
            key=state_key,
            prefix="tabular_history_same_state",
            global_mean=global_mean,
        )
    )
    row_features.update(
        _group_feature_block(
            stats=history,
            keyed_stats=category_stats,
            key=category_key,
            prefix="tabular_history_same_category",
            global_mean=global_mean,
        )
    )
    row_features.update(
        _group_feature_block(
            stats=history,
            keyed_stats=open_stats,
            key=open_key,
            prefix="tabular_history_same_is_open",
            global_mean=global_mean,
        )
    )
    row_features["tabular_history_feature_completeness"] = _compute_feature_completeness(row_features)
    return row_features


def _update_group(group_map: dict[str, dict[str, float]], key: str, rating: float) -> None:
    entry = group_map.setdefault(key, {"count": 0.0, "sum": 0.0})
    entry["count"] += 1.0
    entry["sum"] += float(rating)


def collapse_history_band(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 20:
        return "2-20"
    return ">20"


def _drop_forbidden_history_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[column for column in FORBIDDEN_GLOBAL_HISTORY_COLUMNS if column in frame.columns], errors="ignore")


def _add_prefix_alias_columns(frame: pd.DataFrame, *, global_mean: float) -> pd.DataFrame:
    out = frame.copy()
    for source_column, target_column in PREFIX_ALIAS_MAP.items():
        if source_column in out.columns and target_column not in out.columns:
            out[target_column] = out[source_column].astype(np.float32)
    if "prefix_user_mean" in out.columns:
        out["prefix_user_bias_vs_global"] = (out["prefix_user_mean"] - float(global_mean)).astype(np.float32)
    if "business_train_mean" in out.columns and "prefix_user_mean" in out.columns:
        business_anchor = pd.to_numeric(out["business_train_mean"], errors="coerce").fillna(
            pd.to_numeric(out.get("business_stars", global_mean), errors="coerce").fillna(global_mean)
        )
        out["prefix_user_business_gap"] = (out["prefix_user_mean"] - business_anchor).astype(np.float32)
    return out


def train_prefix_frame(frame: pd.DataFrame, *, global_mean: float) -> pd.DataFrame:
    return _add_prefix_alias_columns(
        build_tabular_moe_train_frame(_drop_forbidden_history_columns(frame), global_mean=global_mean),
        global_mean=global_mean,
    )


def eval_prefix_frame(target_frame: pd.DataFrame, context_frame: pd.DataFrame, *, global_mean: float) -> pd.DataFrame:
    return _add_prefix_alias_columns(
        build_tabular_moe_eval_frame(
            _drop_forbidden_history_columns(target_frame),
            _drop_forbidden_history_columns(context_frame),
            global_mean=global_mean,
        ),
        global_mean=global_mean,
    )


def build_tabular_moe_train_frame(frame: pd.DataFrame, *, global_mean: float) -> pd.DataFrame:
    ordered = frame.sort_values(["review_date", "review_id"], kind="stable").reset_index(drop=True)
    user_stats: dict[str, _RunningHistory] = {}
    by_city: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_state: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_category: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_open: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    prefix_rows: list[dict[str, float]] = []

    for _, row in ordered.iterrows():
        user_key = _safe_key(row.get("user"), "__missing_user__")
        city_key, state_key, category_key, open_key = _current_row_keys(row)
        current_day = _timestamp_to_day(row.get("review_date"))
        business_stars = float(pd.to_numeric(row.get("business_stars", global_mean), errors="coerce"))
        stats = user_stats.get(user_key)
        prefix_rows.append(
            _prefix_features_from_state(
                stats=stats,
                current_day=current_day,
                candidate_business_stars=business_stars if np.isfinite(business_stars) else float(global_mean),
                city_stats=by_city[user_key],
                state_stats=by_state[user_key],
                category_stats=by_category[user_key],
                open_stats=by_open[user_key],
                city_key=city_key,
                state_key=state_key,
                category_key=category_key,
                open_key=open_key,
                global_mean=global_mean,
            )
        )

        if stats is None:
            stats = _RunningHistory()
            user_stats[user_key] = stats
        rating = float(pd.to_numeric(row.get("rating"), errors="coerce"))
        stats.update(
            rating=rating,
            day=current_day,
            business_stars=business_stars if np.isfinite(business_stars) else float(global_mean),
        )
        _update_group(by_city[user_key], city_key, rating)
        _update_group(by_state[user_key], state_key, rating)
        _update_group(by_category[user_key], category_key, rating)
        _update_group(by_open[user_key], open_key, rating)

    prefix_frame = pd.DataFrame(prefix_rows, index=ordered.index)
    return pd.concat([ordered, prefix_frame], axis=1)


def build_tabular_moe_eval_frame(
    target_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
    *,
    global_mean: float,
) -> pd.DataFrame:
    context = context_frame.sort_values(["review_date", "review_id"], kind="stable").reset_index(drop=True)
    user_stats: dict[str, _RunningHistory] = {}
    by_city: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_state: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_category: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    by_open: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for _, row in context.iterrows():
        user_key = _safe_key(row.get("user"), "__missing_user__")
        city_key, state_key, category_key, open_key = _current_row_keys(row)
        current_day = _timestamp_to_day(row.get("review_date"))
        business_stars = float(pd.to_numeric(row.get("business_stars", global_mean), errors="coerce"))
        rating = float(pd.to_numeric(row.get("rating"), errors="coerce"))

        stats = user_stats.get(user_key)
        if stats is None:
            stats = _RunningHistory()
            user_stats[user_key] = stats
        stats.update(
            rating=rating,
            day=current_day,
            business_stars=business_stars if np.isfinite(business_stars) else float(global_mean),
        )
        _update_group(by_city[user_key], city_key, rating)
        _update_group(by_state[user_key], state_key, rating)
        _update_group(by_category[user_key], category_key, rating)
        _update_group(by_open[user_key], open_key, rating)

    ordered_target = target_frame.sort_values(["review_date", "review_id"], kind="stable").reset_index(drop=True)
    prefix_rows: list[dict[str, float]] = []
    for _, row in ordered_target.iterrows():
        user_key = _safe_key(row.get("user"), "__missing_user__")
        city_key, state_key, category_key, open_key = _current_row_keys(row)
        current_day = _timestamp_to_day(row.get("review_date"))
        business_stars = float(pd.to_numeric(row.get("business_stars", global_mean), errors="coerce"))
        prefix_rows.append(
            _prefix_features_from_state(
                stats=user_stats.get(user_key),
                current_day=current_day,
                candidate_business_stars=business_stars if np.isfinite(business_stars) else float(global_mean),
                city_stats=by_city[user_key],
                state_stats=by_state[user_key],
                category_stats=by_category[user_key],
                open_stats=by_open[user_key],
                city_key=city_key,
                state_key=state_key,
                category_key=category_key,
                open_key=open_key,
                global_mean=global_mean,
            )
        )
    prefix_frame = pd.DataFrame(prefix_rows, index=ordered_target.index)
    return pd.concat([ordered_target, prefix_frame], axis=1)


def resolve_tabular_router_branches(history_band: pd.Series | np.ndarray) -> np.ndarray:
    bands = pd.Series(history_band).astype("string")
    return bands.map(lambda band: TABULAR_BAND_TO_EXPERT.get(str(band), "long_history_tabular")).to_numpy(dtype=object)


def compute_tabular_baseline_prediction(frame: pd.DataFrame, *, global_mean: float) -> np.ndarray:
    user_mean = pd.to_numeric(frame.get("prefix_user_mean", frame.get("user_average_stars", global_mean)), errors="coerce").fillna(global_mean).to_numpy(dtype=np.float32)
    business_mean = pd.to_numeric(frame.get("business_train_mean", frame.get("business_stars", global_mean)), errors="coerce").fillna(global_mean).to_numpy(dtype=np.float32)
    user_count = pd.to_numeric(frame.get("prefix_user_count", frame.get("tabular_history_count", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    business_count = pd.to_numeric(frame.get("business_train_count", frame.get("business_review_count", 0.0)), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    user_weight = user_count / (user_count + 5.0)
    business_weight = business_count / (business_count + 10.0)
    denom = user_weight + business_weight + 1.0
    pred = ((user_weight * user_mean) + (business_weight * business_mean) + float(global_mean)) / denom
    return np.clip(pred.astype(np.float32), 1.0, 5.0)


def apply_tabular_blend(
    *,
    expert_pred: np.ndarray,
    baseline_pred: np.ndarray,
    history_band: pd.Series | np.ndarray,
    blend_alpha_by_band: dict[str, float],
) -> np.ndarray:
    bands = pd.Series(history_band).astype("string")
    alpha = bands.map(lambda band: float(blend_alpha_by_band.get(str(band), 1.0))).to_numpy(dtype=np.float32)
    return ((alpha * expert_pred) + ((1.0 - alpha) * baseline_pred)).astype(np.float32)


def build_feature_columns_by_expert(
    *,
    base_feature_columns: list[str],
    categorical_columns: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    prefix_overall = [column for column in base_feature_columns if column.startswith("prefix_user_")]
    prefix_affinity = [column for column in base_feature_columns if column.startswith("prefix_same_")]
    static_columns = [
        column
        for column in base_feature_columns
        if column not in TABULAR_HISTORY_COLUMNS
        and column not in PREFIX_ALIAS_MAP.values()
        and not column.startswith("prefix_user_")
        and not column.startswith("prefix_same_")
        and column not in FORBIDDEN_GLOBAL_HISTORY_COLUMNS
    ]

    feature_columns_by_expert = {
        "cold_user_tabular": static_columns.copy(),
        "very_short_history_tabular": [*static_columns, *prefix_overall],
        "mid_history_tabular": [*static_columns, *prefix_overall, *prefix_affinity],
        "long_history_tabular": [*static_columns, *prefix_overall, *prefix_affinity],
    }
    categorical_columns_by_expert = {
        expert: [column for column in categorical_columns if column in feature_columns]
        for expert, feature_columns in feature_columns_by_expert.items()
    }
    manifest = {
        "static_columns": static_columns,
        "prefix_safe_columns": [*prefix_overall, *prefix_affinity],
        "forbidden_global_history_columns": FORBIDDEN_GLOBAL_HISTORY_COLUMNS.copy(),
        "experts": {
            expert: {
                "n_features": len(feature_columns),
                "categorical_columns": categorical_columns_by_expert[expert],
            }
            for expert, feature_columns in feature_columns_by_expert.items()
        },
    }
    return feature_columns_by_expert, categorical_columns_by_expert, manifest
