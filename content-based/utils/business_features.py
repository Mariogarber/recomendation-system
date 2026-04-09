from __future__ import annotations

import ast
from collections import Counter
from typing import Any

import pandas as pd


def parse_categories(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(part).strip() for part in value if str(part).strip()]
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_attributes(value: Any) -> dict[str, Any]:
    parsed = _coerce_mapping(value)
    if not isinstance(parsed, dict):
        return {}
    return _flatten_mapping(parsed)


def summarize_attribute_keys(
    businesses_df: pd.DataFrame,
    *,
    column: str = "attributes",
) -> pd.DataFrame:
    if column not in businesses_df.columns:
        raise ValueError(f"Missing column: {column}")

    counts: Counter[str] = Counter()
    for raw_value in businesses_df[column]:
        parsed = parse_attributes(raw_value)
        counts.update(parsed.keys())

    rows = [
        {"attribute_key": key, "count": count}
        for key, count in counts.most_common()
    ]
    return pd.DataFrame(rows)


def extract_hours_features(value: Any) -> dict[str, float]:
    parsed = _coerce_mapping(value)
    if not isinstance(parsed, dict) or not parsed:
        return {
            "open_days_count": 0.0,
            "weekly_open_minutes": 0.0,
            "weekend_days_open": 0.0,
            "late_night_days": 0.0,
        }

    open_days_count = 0
    weekly_open_minutes = 0.0
    weekend_days_open = 0
    late_night_days = 0

    for day, interval in parsed.items():
        minutes = _interval_to_minutes(interval)
        if minutes <= 0:
            continue

        open_days_count += 1
        weekly_open_minutes += minutes
        if str(day) in {"Saturday", "Sunday"}:
            weekend_days_open += 1
        if _is_late_night(interval):
            late_night_days += 1

    return {
        "open_days_count": float(open_days_count),
        "weekly_open_minutes": float(weekly_open_minutes),
        "weekend_days_open": float(weekend_days_open),
        "late_night_days": float(late_night_days),
    }


def _coerce_mapping(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None


def _flatten_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}

    for key, value in mapping.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)

        nested = _coerce_mapping(value)
        if isinstance(nested, dict):
            flat.update(_flatten_mapping(nested, prefix=flat_key))
            continue

        flat[flat_key] = _normalize_scalar(value)

    return flat


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value.strip()
    return value


def _parse_clock(clock_text: str) -> int | None:
    if ":" not in clock_text:
        return None

    hours_text, minutes_text = clock_text.split(":", maxsplit=1)
    try:
        hours = int(hours_text)
        minutes = int(minutes_text)
    except ValueError:
        return None

    if not (0 <= hours <= 24 and 0 <= minutes < 60):
        return None
    return hours * 60 + minutes


def _interval_to_minutes(interval: Any) -> float:
    if not isinstance(interval, str) or "-" not in interval:
        return 0.0

    start_text, end_text = interval.split("-", maxsplit=1)
    start = _parse_clock(start_text)
    end = _parse_clock(end_text)
    if start is None or end is None:
        return 0.0

    if end == start:
        return 24.0 * 60.0
    if end < start:
        end += 24 * 60
    return float(end - start)


def _is_late_night(interval: Any) -> bool:
    if not isinstance(interval, str) or "-" not in interval:
        return False

    _, end_text = interval.split("-", maxsplit=1)
    end = _parse_clock(end_text)
    if end is None:
        return False

    return end >= 22 * 60 or end <= 3 * 60
