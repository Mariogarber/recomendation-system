from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.frozen_embedding_regression import (
    build_review_interaction_frame,
    compute_band_metrics,
    load_frozen_embedding_bundle,
    rmse,
)
from utils.gbm_features import build_gbm_feature_matrix, compute_scalar_priors, fit_review_context_scaler
from utils.io import get_default_data_dir, load_train_reviews
from utils.split import temporal_train_validation_split


@dataclass(slots=True)
class GBMRegressorConfig:
    num_leaves: int = 127
    learning_rate: float = 0.05
    n_estimators: int = 1000
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    temporal_val_size: float = 0.2
    early_stopping_rounds: int = 50
    synthetic_cold_start_fraction: float = 0.12
    random_seed: int = 42


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _history_band_from_count(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return ">20"


def _attach_history_band(frame: pd.DataFrame, priors_user_count: dict[str, int]) -> pd.DataFrame:
    out = frame.copy()
    out["history_band"] = [
        _history_band_from_count(int(priors_user_count.get(user_id, 0)))
        for user_id in out["user"].to_numpy()
    ]
    return out


def _sample_cold_start_rows(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction <= 0.0:
        return frame.iloc[0:0].copy()
    sample_size = max(1, int(round(len(frame) * fraction)))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(frame), size=sample_size, replace=False)
    return frame.iloc[np.sort(sampled_idx)].reset_index(drop=True)


def _compute_blend_validation(
    gbm_predictions: pd.DataFrame,
    deep_predictions_path: Path,
    known_train_users: set[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    deep_predictions = pd.read_csv(deep_predictions_path, low_memory=False)
    merged = gbm_predictions.merge(
        deep_predictions[["review_id", "pred"]].rename(columns={"pred": "deep_pred_raw"}),
        on="review_id",
        how="left",
        validate="one_to_one",
    )
    if merged["deep_pred_raw"].isna().any():
        missing = int(merged["deep_pred_raw"].isna().sum())
        raise RuntimeError(f"Missing {missing} deep validation predictions after merge.")

    deep_rounded = _round_half_up(np.clip(merged["deep_pred_raw"].to_numpy(dtype=np.float32), 1.0, 5.0))
    gbm_rounded = merged["gbm_pred_rounded"].to_numpy(dtype=np.int32)
    known_mask = merged["user"].isin(known_train_users).to_numpy(dtype=bool)
    blended = np.where(
        known_mask,
        _round_half_up((deep_rounded.astype(np.float32) + gbm_rounded.astype(np.float32)) / 2.0),
        gbm_rounded,
    ).astype(np.int32)
    merged["deep_pred_rounded"] = deep_rounded
    merged["blend_pred"] = blended

    blend_eval = merged[["rating", "history_band", "blend_pred"]].rename(columns={"blend_pred": "pred"})
    summary = {
        "blend_mae": float(np.mean(np.abs(merged["rating"].to_numpy(dtype=np.float32) - blended.astype(np.float32)))),
        "blend_rmse": rmse(merged["rating"].to_numpy(dtype=np.float32), blended.astype(np.float32)),
        "n_known_user_rows": int(known_mask.sum()),
        "n_new_user_rows": int((~known_mask).sum()),
        "band_metrics": compute_band_metrics(blend_eval).to_dict(orient="records"),
    }
    return summary, merged


def parse_args() -> argparse.Namespace:
    default_embedding_root = Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v3_iter04"
    default_deep_val = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "frozen_embedding_regressor_v1"
        / "iter04_with_review"
        / "validation_predictions.csv"
    )
    parser = argparse.ArgumentParser(description="Train a LightGBM regressor over frozen embeddings.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--embedding-root", type=Path, default=default_embedding_root)
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "gbm_regressor_v1",
    )
    parser.add_argument("--deep-validation-predictions", type=Path, default=default_deep_val)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--temporal-val-size", type=float, default=0.2)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--synthetic-cold-start-fraction", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GBMRegressorConfig(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        temporal_val_size=args.temporal_val_size,
        early_stopping_rounds=args.early_stopping_rounds,
        synthetic_cold_start_fraction=args.synthetic_cold_start_fraction,
        random_seed=args.seed,
    )

    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    _save_json(save_root / "config.json", asdict(config))

    bundle = load_frozen_embedding_bundle(args.embedding_root)
    train_reviews = load_train_reviews(args.data_dir)
    interactions = build_review_interaction_frame(train_reviews)
    train_split, val_split = temporal_train_validation_split(
        interactions,
        val_size=config.temporal_val_size,
        timestamp_col="timestamp",
    )

    priors = compute_scalar_priors(train_split)
    min_ts, rc_means, rc_stds = fit_review_context_scaler(train_split)
    known_train_users = set(priors.user_count)

    x_train_main, feature_names = build_gbm_feature_matrix(
        train_split,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_train_main = train_split["rating"].to_numpy(dtype=np.float32)

    cold_rows = _sample_cold_start_rows(train_split, config.synthetic_cold_start_fraction, config.random_seed)
    x_train_cold = np.zeros((0, x_train_main.shape[1]), dtype=np.float32)
    y_train_cold = np.zeros((0,), dtype=np.float32)
    if not cold_rows.empty:
        x_train_cold, _ = build_gbm_feature_matrix(
            cold_rows,
            bundle,
            priors,
            review_context_min_timestamp=min_ts,
            review_context_means=rc_means,
            review_context_stds=rc_stds,
            forced_new_user_mask=np.ones(len(cold_rows), dtype=bool),
        )
        y_train_cold = cold_rows["rating"].to_numpy(dtype=np.float32)

    x_train = np.vstack([x_train_main, x_train_cold])
    y_train = np.concatenate([y_train_main, y_train_cold])

    x_val, _ = build_gbm_feature_matrix(
        val_split,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_val = val_split["rating"].to_numpy(dtype=np.float32)

    lgb_train = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
    lgb_val = lgb.Dataset(x_val, label=y_val, feature_name=feature_names, reference=lgb_train)
    params = {
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
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=config.n_estimators,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iteration = int(booster.best_iteration or config.n_estimators)
    val_pred_raw = np.clip(booster.predict(x_val, num_iteration=best_iteration).astype(np.float32), 1.0, 5.0)
    val_pred_rounded = _round_half_up(val_pred_raw)

    val_eval = _attach_history_band(val_split, priors.user_count)
    validation_predictions = val_eval[
        ["review_id", "user", "item", "rating", "timestamp", "history_band", "useful", "funny", "cool"]
    ].copy()
    validation_predictions["gbm_pred_raw"] = val_pred_raw
    validation_predictions["gbm_pred_rounded"] = val_pred_rounded

    gbm_band_metrics = compute_band_metrics(
        validation_predictions[["rating", "history_band", "gbm_pred_raw"]].rename(columns={"gbm_pred_raw": "pred"})
    )
    gbm_band_metrics_rounded = compute_band_metrics(
        validation_predictions[["rating", "history_band", "gbm_pred_rounded"]].rename(
            columns={"gbm_pred_rounded": "pred"}
        )
    )

    blend_summary: dict[str, Any] | None = None
    if args.deep_validation_predictions.exists():
        blend_summary, validation_predictions = _compute_blend_validation(
            validation_predictions.copy(),
            args.deep_validation_predictions,
            known_train_users,
        )
        pd.DataFrame(blend_summary["band_metrics"]).to_csv(save_root / "band_metrics_blend.csv", index=False)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    joblib.dump(booster, save_root / "model.pkl")
    _save_json(
        save_root / "review_context_scaler.json",
        {
            "min_timestamp": min_ts.isoformat(),
            "means": rc_means.tolist(),
            "stds": rc_stds.tolist(),
        },
    )
    _save_json(
        save_root / "scalar_priors.json",
        {
            "global_mean": priors.global_mean,
            "user_mean": priors.user_mean,
            "user_std": priors.user_std,
            "user_count": priors.user_count,
            "business_mean": priors.business_mean,
            "business_std": priors.business_std,
            "business_count": priors.business_count,
        },
    )
    gbm_band_metrics.to_csv(save_root / "band_metrics_gbm_raw.csv", index=False)
    gbm_band_metrics_rounded.to_csv(save_root / "band_metrics_gbm_rounded.csv", index=False)
    validation_predictions.to_csv(save_root / "validation_predictions.csv", index=False)
    importance.to_csv(save_root / "feature_importance.csv", index=False)

    summary: dict[str, Any] = {
        "objective": "rating_regression",
        "model_type": "lightgbm_l1",
        "embedding_root": str(args.embedding_root),
        "best_iteration": best_iteration,
        "val_mae_gbm_raw": float(np.mean(np.abs(y_val - val_pred_raw))),
        "val_rmse_gbm_raw": rmse(y_val, val_pred_raw),
        "val_mae_gbm_rounded": float(np.mean(np.abs(y_val - val_pred_rounded.astype(np.float32)))),
        "val_rmse_gbm_rounded": rmse(y_val, val_pred_rounded.astype(np.float32)),
        "synthetic_cold_start_fraction": config.synthetic_cold_start_fraction,
        "n_train_rows": int(len(train_split)),
        "n_train_rows_augmented": int(len(x_train)),
        "n_val_rows": int(len(val_split)),
        "n_features": len(feature_names),
        "n_val_new_users": int((~val_split["user"].isin(list(known_train_users))).sum()),
        "config": asdict(config),
    }
    if blend_summary is not None:
        summary["blend_validation"] = blend_summary

    _save_json(save_root / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
