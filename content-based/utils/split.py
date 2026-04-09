from __future__ import annotations

from typing import Any

import pandas as pd


def random_train_validation_split(
    df: pd.DataFrame,
    *,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1).")
    if len(df) == 0:
        raise ValueError("Input dataframe is empty.")

    val_df = df.sample(frac=val_size, random_state=random_state)
    train_df = df.drop(index=val_df.index)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def temporal_train_validation_split(
    df: pd.DataFrame,
    *,
    val_size: float = 0.2,
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1).")
    if len(df) == 0:
        raise ValueError("Input dataframe is empty.")

    ordered = df.sort_values(timestamp_col).reset_index(drop=True)
    split_idx = max(1, int(round(len(ordered) * (1.0 - val_size))))
    split_idx = min(split_idx, len(ordered) - 1)
    train_df = ordered.iloc[:split_idx].copy()
    val_df = ordered.iloc[split_idx:].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def cold_start_breakdown(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    user_col: str = "user",
    item_col: str = "item",
) -> dict[str, Any]:
    train_users = set(train_df[user_col].dropna().unique())
    train_items = set(train_df[item_col].dropna().unique())

    user_known = eval_df[user_col].isin(train_users)
    item_known = eval_df[item_col].isin(train_items)

    total = int(len(eval_df))
    both_known = int((user_known & item_known).sum())
    new_user_known_item = int((~user_known & item_known).sum())
    known_user_new_item = int((user_known & ~item_known).sum())
    both_new = int((~user_known & ~item_known).sum())

    def pct(value: int) -> float:
        return float(value / total) if total else 0.0

    return {
        "total_rows": total,
        "both_known": both_known,
        "new_user_known_item": new_user_known_item,
        "known_user_new_item": known_user_new_item,
        "both_new": both_new,
        "both_known_pct": pct(both_known),
        "new_user_known_item_pct": pct(new_user_known_item),
        "known_user_new_item_pct": pct(known_user_new_item),
        "both_new_pct": pct(both_new),
    }
