from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .frozen_embedding_regression import (
    FrozenEmbeddingBundle,
    build_review_context_only_frame,
    build_review_interaction_frame,
    compute_band_metrics,
    load_frozen_embedding_bundle,
    rmse,
)
from .gbm_features import (
    REVIEW_CONTEXT_FEATURE_NAMES,
    ScalarPriors,
    build_gbm_feature_matrix,
    compute_scalar_priors,
    fit_review_context_scaler,
)


@dataclass(slots=True)
class LGBMDeepEmbeddingConfig:
    num_leaves: int = 127
    learning_rate: float = 0.05
    n_estimators: int = 300
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    temporal_val_size: float = 0.2
    early_stopping_rounds: int = 30
    synthetic_cold_start_fraction: float = 0.3
    random_seed: int = 42


def load_deep_embedding_bundle(root: str | Path) -> FrozenEmbeddingBundle:
    return load_frozen_embedding_bundle(root)


def build_lgbm_feature_matrix(
    frame: pd.DataFrame,
    bundle: FrozenEmbeddingBundle,
    priors: ScalarPriors,
    *,
    review_context_min_timestamp: pd.Timestamp,
    review_context_means: np.ndarray,
    review_context_stds: np.ndarray,
    forced_new_user_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    return build_gbm_feature_matrix(
        frame,
        bundle,
        priors,
        review_context_min_timestamp=review_context_min_timestamp,
        review_context_means=review_context_means,
        review_context_stds=review_context_stds,
        forced_new_user_mask=forced_new_user_mask,
    )


def build_lgbm_params(config: LGBMDeepEmbeddingConfig) -> dict[str, Any]:
    return {
        "objective": "regression_l1",
        "metric": "l1",
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "subsample_freq": 1,
        "colsample_bytree": config.colsample_bytree,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "verbose": -1,
        "seed": config.random_seed,
        "feature_fraction_seed": config.random_seed,
        "bagging_seed": config.random_seed,
        "data_random_seed": config.random_seed,
    }


def round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def history_band_from_count(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return ">20"


def attach_history_band(frame: pd.DataFrame, user_counts: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    out["history_band"] = [
        history_band_from_count(int(user_counts.get(user_id, 0)))
        for user_id in out["user"].to_numpy()
    ]
    return out


def sample_cold_start_rows(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction <= 0.0:
        return frame.iloc[0:0].copy()
    sample_size = max(1, int(round(len(frame) * fraction)))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(frame), size=sample_size, replace=False)
    return frame.iloc[np.sort(sampled_idx)].reset_index(drop=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_scalar_priors(path: Path, priors: ScalarPriors) -> None:
    save_json(path, asdict(priors))


def load_scalar_priors(path: Path) -> ScalarPriors:
    data = load_json(path)
    return ScalarPriors(
        global_mean=float(data["global_mean"]),
        user_mean={k: float(v) for k, v in data["user_mean"].items()},
        user_std={k: float(v) for k, v in data["user_std"].items()},
        user_count={k: int(v) for k, v in data["user_count"].items()},
        business_mean={k: float(v) for k, v in data["business_mean"].items()},
        business_std={k: float(v) for k, v in data["business_std"].items()},
        business_count={k: int(v) for k, v in data["business_count"].items()},
    )


def save_review_context_scaler(
    path: Path,
    *,
    min_timestamp: pd.Timestamp,
    means: np.ndarray,
    stds: np.ndarray,
) -> None:
    save_json(
        path,
        {
            "min_timestamp": min_timestamp.isoformat(),
            "means": means.tolist(),
            "stds": stds.tolist(),
        },
    )


def load_review_context_scaler(path: Path) -> tuple[pd.Timestamp, np.ndarray, np.ndarray]:
    data = load_json(path)
    return (
        pd.Timestamp(data["min_timestamp"]),
        np.array(data["means"], dtype=np.float32),
        np.array(data["stds"], dtype=np.float32),
    )


def summarize_feature_join(frame: pd.DataFrame, bundle: FrozenEmbeddingBundle) -> dict[str, Any]:
    user_index = set(bundle.user_ids.to_numpy())
    business_index = set(bundle.business_ids.to_numpy())
    known_user = frame["user"].isin(user_index)
    known_business = frame["item"].isin(business_index)
    return {
        "total_rows": int(len(frame)),
        "known_user_rows": int(known_user.sum()),
        "known_business_rows": int(known_business.sum()),
        "known_both_rows": int((known_user & known_business).sum()),
        "missing_user_rows": int((~known_user).sum()),
        "missing_business_rows": int((~known_business).sum()),
        "missing_either_rows": int((~(known_user & known_business)).sum()),
    }


def compute_validation_summary(
    *,
    y_true: np.ndarray,
    raw_pred: np.ndarray,
    rounded_pred: np.ndarray,
    val_frame: pd.DataFrame,
    priors: ScalarPriors,
) -> dict[str, Any]:
    eval_frame = attach_history_band(val_frame, priors.user_count)
    raw_eval = eval_frame[["rating", "history_band"]].copy()
    raw_eval["pred"] = raw_pred
    rounded_eval = raw_eval.copy()
    rounded_eval["pred"] = rounded_pred

    return {
        "val_mae_raw": float(np.mean(np.abs(y_true - raw_pred.astype(np.float32)))),
        "val_rmse_raw": rmse(y_true, raw_pred.astype(np.float32)),
        "val_mae_rounded": float(np.mean(np.abs(y_true - rounded_pred.astype(np.float32)))),
        "val_rmse_rounded": rmse(y_true, rounded_pred.astype(np.float32)),
        "band_metrics_raw": compute_band_metrics(raw_eval).to_dict(orient="records"),
        "band_metrics_rounded": compute_band_metrics(rounded_eval).to_dict(orient="records"),
    }
