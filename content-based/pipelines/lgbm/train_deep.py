from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.io import get_default_data_dir, load_train_reviews
from utils.lgbm_deep_embeddings import (
    LGBMDeepEmbeddingConfig,
    attach_history_band,
    build_lgbm_feature_matrix,
    build_lgbm_params,
    build_review_interaction_frame,
    compute_scalar_priors,
    compute_validation_summary,
    fit_review_context_scaler,
    load_deep_embedding_bundle,
    round_half_up,
    save_json,
    save_review_context_scaler,
    save_scalar_priors,
    sample_cold_start_rows,
    summarize_feature_join,
)
from utils.split import temporal_train_validation_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LightGBM model over deep embeddings.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v3_iter04",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_deep_embeddings_v1",
    )
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-child-samples", type=int, default=50)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--temporal-val-size", type=float, default=0.2)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--synthetic-cold-start-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _save_feature_importance(path: Path, names: list[str], booster: lgb.Booster) -> None:
    importance = booster.feature_importance(importance_type="gain")
    frame = pd.DataFrame({"feature_name": names, "gain": importance.astype(np.float64)})
    frame.sort_values("gain", ascending=False).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = LGBMDeepEmbeddingConfig(
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
    save_json(save_root / "config.json", asdict(config))

    bundle = load_deep_embedding_bundle(args.embedding_root)
    train_reviews = load_train_reviews(args.data_dir)
    interaction_frame = build_review_interaction_frame(train_reviews)
    train_split, val_split = temporal_train_validation_split(
        interaction_frame,
        val_size=config.temporal_val_size,
        timestamp_col="timestamp",
    )

    priors = compute_scalar_priors(train_split)
    min_ts, rc_means, rc_stds = fit_review_context_scaler(train_split)
    join_summary = summarize_feature_join(train_split, bundle)

    x_train_main, feature_names = build_lgbm_feature_matrix(
        train_split,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_train_main = train_split["rating"].to_numpy(dtype=np.float32)

    cold_rows = sample_cold_start_rows(train_split, config.synthetic_cold_start_fraction, config.random_seed)
    x_train_cold = np.zeros((0, x_train_main.shape[1]), dtype=np.float32)
    y_train_cold = np.zeros((0,), dtype=np.float32)
    if not cold_rows.empty:
        x_train_cold, _ = build_lgbm_feature_matrix(
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

    x_val, _ = build_lgbm_feature_matrix(
        val_split,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_val = val_split["rating"].to_numpy(dtype=np.float32)

    params = build_lgbm_params(config)
    lgb_train = lgb.Dataset(x_train, label=y_train, feature_name=feature_names)
    lgb_val = lgb.Dataset(x_val, label=y_val, feature_name=feature_names, reference=lgb_train)
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
    val_pred_rounded = round_half_up(val_pred_raw).clip(1, 5).astype(np.int32)

    val_eval = attach_history_band(val_split, priors.user_count)
    validation_predictions = val_eval[
        ["review_id", "user", "item", "rating", "timestamp", "history_band", "useful", "funny", "cool"]
    ].copy()
    validation_predictions["lgbm_pred_raw"] = val_pred_raw
    validation_predictions["lgbm_pred_rounded"] = val_pred_rounded
    validation_predictions.to_csv(save_root / "validation_predictions.csv", index=False)

    _save_feature_importance(save_root / "feature_importance.csv", feature_names, booster)
    save_scalar_priors(save_root / "scalar_priors.json", priors)
    save_review_context_scaler(
        save_root / "review_context_scaler.json",
        min_timestamp=min_ts,
        means=rc_means,
        stds=rc_stds,
    )
    save_json(save_root / "embedding_root.json", {"embedding_root": str(args.embedding_root)})
    joblib.dump(booster, save_root / "model.pkl")

    validation_summary = {
        "objective": "rating_regression",
        "model_type": "lightgbm_l1",
        "embedding_root": str(args.embedding_root),
        "best_iteration": best_iteration,
        "synthetic_cold_start_fraction": config.synthetic_cold_start_fraction,
        "n_train_rows": int(len(train_split)),
        "n_train_rows_augmented": int(len(x_train)),
        "n_val_rows": int(len(val_split)),
        "n_features": int(len(feature_names)),
        "n_val_new_users": int((~val_split["user"].isin(priors.user_count.keys())).sum()),
        "config": asdict(config),
        "feature_join_summary": join_summary,
    }
    validation_summary.update(
        compute_validation_summary(
            y_true=y_val,
            raw_pred=val_pred_raw,
            rounded_pred=val_pred_rounded,
            val_frame=val_eval,
            priors=priors,
        )
    )
    save_json(save_root / "validation_summary.json", validation_summary)

    print(json.dumps(validation_summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

