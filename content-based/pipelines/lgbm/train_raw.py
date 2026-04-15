from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.io import get_default_data_dir, load_businesses, load_train_reviews, load_users
from utils.lgbm_raw_features import (
    CATEGORICAL_COLUMNS,
    RAW_CORE_FEATURE_SET,
    RAW_PRIORS_FEATURE_SET,
    build_raw_feature_frame,
    fit_raw_feature_spec,
    history_band_from_count,
)
from utils.split import cold_start_breakdown, temporal_train_validation_split


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _extract_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return frame[feature_columns].copy()


def _train_booster(
    *,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame | None,
    y_val: np.ndarray | None,
    categorical_columns: list[str],
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int | None,
) -> lgb.Booster:
    train_set = lgb.Dataset(x_train, label=y_train, categorical_feature=categorical_columns, free_raw_data=False)
    valid_sets = [train_set]
    callbacks: list[Any] = [lgb.log_evaluation(period=50)]
    if x_val is not None and y_val is not None:
        valid_set = lgb.Dataset(x_val, label=y_val, categorical_feature=categorical_columns, reference=train_set, free_raw_data=False)
        valid_sets.append(valid_set)
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            callbacks.insert(0, lgb.early_stopping(early_stopping_rounds, verbose=True))

    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def _feature_summary(feature_columns: list[str]) -> dict[str, int]:
    return {
        "total": len(feature_columns),
        "user": sum(column.startswith("user_") for column in feature_columns),
        "business": sum(column.startswith("business_") for column in feature_columns),
        "review": sum(column.startswith("review_") for column in feature_columns),
        "priors": sum(
            column.startswith("user_train_")
            or column.startswith("business_train_")
            or column.endswith("_train_mean")
            or column.endswith("_train_count")
            for column in feature_columns
        ),
    }


def _build_history_band(frame: pd.DataFrame, user_counts: pd.Series) -> pd.Series:
    lookup = user_counts.to_dict()
    return pd.Series(
        [history_band_from_count(int(lookup.get(user_id, 0))) for user_id in frame["user"].astype(str)],
        index=frame.index,
        dtype="string",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM model on raw tabular content features.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_raw_features_v1",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=[RAW_CORE_FEATURE_SET, RAW_PRIORS_FEATURE_SET],
        default=RAW_PRIORS_FEATURE_SET,
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--num-leaves", type=int, default=255)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=3000)
    parser.add_argument("--min-child-samples", type=int, default=75)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    train_split, val_split = temporal_train_validation_split(
        train_reviews,
        val_size=args.validation_size,
        timestamp_col="date",
    )

    validation_spec = fit_raw_feature_spec(
        train_split,
        users_df,
        businesses_df,
        feature_set=args.feature_set,
    )
    train_frame = build_raw_feature_frame(train_split, users_df, businesses_df, validation_spec)
    val_frame = build_raw_feature_frame(val_split, users_df, businesses_df, validation_spec)

    x_train = _extract_feature_matrix(train_frame, validation_spec.feature_columns)
    y_train = train_frame["rating"].to_numpy(dtype=np.float32)
    x_val = _extract_feature_matrix(val_frame, validation_spec.feature_columns)
    y_val = val_frame["rating"].to_numpy(dtype=np.float32)

    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "boosting_type": "gbdt",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "verbose": -1,
        "seed": args.seed,
        "feature_fraction_seed": args.seed,
        "bagging_seed": args.seed,
        "data_random_seed": args.seed,
    }
    booster = _train_booster(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        categorical_columns=CATEGORICAL_COLUMNS,
        params=params,
        num_boost_round=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    best_iteration = int(booster.best_iteration or booster.current_iteration() or args.n_estimators)
    val_pred_raw = np.clip(booster.predict(x_val, num_iteration=best_iteration).astype(np.float32), 1.0, 5.0)
    val_pred_rounded = _round_half_up(val_pred_raw).clip(1, 5).astype(np.int32)

    user_counts = validation_spec.user_priors_table.set_index("user_id")["user_train_count"]
    val_eval = val_frame[["review_id", "user", "item", "rating"]].copy()
    val_eval["history_band"] = _build_history_band(val_eval, user_counts)
    val_eval["pred_raw"] = val_pred_raw
    val_eval["pred_rounded"] = val_pred_rounded
    val_eval["is_new_user"] = (val_eval["history_band"] == "0").astype(np.int32)

    cold_breakdown = cold_start_breakdown(train_split, val_split, user_col="user_id", item_col="business_id")
    band_rows = []
    for band in ["0", "1", "2-5", "6-20", ">20"]:
        subset = val_eval[val_eval["history_band"] == band]
        if subset.empty:
            continue
        band_rows.append(
            {
                "history_band": band,
                "mae": float(np.mean(np.abs(subset["rating"].to_numpy(dtype=np.float32) - subset["pred_rounded"].to_numpy(dtype=np.float32)))),
                "rmse": float(
                    np.sqrt(
                        np.mean(
                            (subset["rating"].to_numpy(dtype=np.float32) - subset["pred_rounded"].to_numpy(dtype=np.float32)) ** 2
                        )
                    )
                ),
                "n_samples": int(len(subset)),
            }
        )

    validation_summary = {
        "feature_set": args.feature_set,
        "best_iteration": best_iteration,
        "validation_mae_raw": float(np.mean(np.abs(val_eval["rating"].to_numpy(dtype=np.float32) - val_pred_raw))),
        "validation_mae_rounded": float(np.mean(np.abs(val_eval["rating"].to_numpy(dtype=np.float32) - val_pred_rounded.astype(np.float32)))),
        "validation_rmse_rounded": float(
            np.sqrt(np.mean((val_eval["rating"].to_numpy(dtype=np.float32) - val_pred_rounded.astype(np.float32)) ** 2))
        ),
        "cold_start_breakdown": cold_breakdown,
        "feature_summary": _feature_summary(validation_spec.feature_columns),
        "categorical_columns": validation_spec.categorical_columns,
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "band_metrics": band_rows,
        "config": {
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "n_estimators": args.n_estimators,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
            "validation_size": args.validation_size,
            "seed": args.seed,
        },
    }

    validation_spec_path = save_root / "validation_spec.joblib"
    validation_model_path = save_root / "validation_model.txt"
    joblib.dump(validation_spec, validation_spec_path)
    booster.save_model(str(validation_model_path))
    val_eval.to_csv(save_root / "validation_predictions.csv", index=False)
    _save_json(save_root / "validation_summary.json", validation_summary)

    feature_importance = pd.DataFrame(
        {
            "feature": validation_spec.feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)
    feature_importance.to_csv(save_root / "feature_importance.csv", index=False)

    full_spec = fit_raw_feature_spec(
        train_reviews,
        users_df,
        businesses_df,
        feature_set=args.feature_set,
    )
    full_frame = build_raw_feature_frame(train_reviews, users_df, businesses_df, full_spec)
    x_full = _extract_feature_matrix(full_frame, full_spec.feature_columns)
    y_full = full_frame["rating"].to_numpy(dtype=np.float32)

    full_booster = _train_booster(
        x_train=x_full,
        y_train=y_full,
        x_val=None,
        y_val=None,
        categorical_columns=CATEGORICAL_COLUMNS,
        params=params,
        num_boost_round=best_iteration,
        early_stopping_rounds=None,
    )

    submission_spec_path = save_root / "submission_spec.joblib"
    submission_model_path = save_root / "submission_model.txt"
    joblib.dump(full_spec, submission_spec_path)
    full_booster.save_model(str(submission_model_path))

    training_summary = {
        "feature_set": args.feature_set,
        "submission_best_iteration": best_iteration,
        "feature_summary": _feature_summary(full_spec.feature_columns),
        "categorical_columns": full_spec.categorical_columns,
        "train_rows": int(len(full_frame)),
        "validation_summary_path": str(save_root / "validation_summary.json"),
        "validation_spec_path": str(validation_spec_path),
        "validation_model_path": str(validation_model_path),
        "submission_spec_path": str(submission_spec_path),
        "submission_model_path": str(submission_model_path),
        "feature_importance_path": str(save_root / "feature_importance.csv"),
    }
    _save_json(save_root / "training_summary.json", training_summary)
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    _save_json(save_root / "config.json", config_payload)
    print(json.dumps({"validation_summary": validation_summary, "training_summary": training_summary}, indent=2))


if __name__ == "__main__":
    main()
