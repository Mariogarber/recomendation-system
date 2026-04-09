from __future__ import annotations

from pathlib import Path

import pandas as pd


def get_default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def load_users(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else get_default_data_dir()
    return pd.read_csv(data_dir / "usuarios.csv", low_memory=False)


def load_businesses(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else get_default_data_dir()
    return pd.read_csv(data_dir / "negocios.csv", low_memory=False)


def load_train_reviews(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else get_default_data_dir()
    return pd.read_csv(data_dir / "train_reviews.csv", low_memory=False)


def load_test_reviews(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else get_default_data_dir()
    return pd.read_csv(data_dir / "test_reviews.csv", low_memory=False)


def canonicalize_reviews(
    df: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "business_id",
    rating_col: str = "stars",
    timestamp_col: str = "date",
) -> pd.DataFrame:
    """
    Normalize review tables to a common schema used by the utilities and models.

    Output columns:
    - user
    - item
    - rating (only if present)
    - timestamp (only if present)
    """

    rename_map: dict[str, str] = {
        user_col: "user",
        item_col: "item",
    }

    if rating_col in df.columns:
        rename_map[rating_col] = "rating"
    if timestamp_col in df.columns:
        rename_map[timestamp_col] = "timestamp"

    out = df.rename(columns=rename_map).copy()

    if "rating" in out.columns:
        out["rating"] = out["rating"].astype(float)
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    return out
