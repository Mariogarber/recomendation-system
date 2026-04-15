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

from utils.frozen_embedding_regression import build_review_interaction_frame, load_frozen_embedding_bundle
from utils.gbm_features import build_gbm_feature_matrix, compute_scalar_priors, fit_review_context_scaler
from utils.io import get_default_data_dir, load_train_reviews


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _sample_cold_start_rows(frame, fraction: float, seed: int):
    if fraction <= 0.0:
        return frame.iloc[0:0].copy()
    sample_size = max(1, int(round(len(frame) * fraction)))
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(frame), size=sample_size, replace=False)
    return frame.iloc[np.sort(sampled_idx)].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    default_source = Path(__file__).resolve().parents[2] / "artifacts" / "gbm_regressor_v1"
    parser = argparse.ArgumentParser(description="Retrain GBM on all train data for submission.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--source-run", type=Path, default=default_source)
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "gbm_submission_v1",
    )
    parser.add_argument("--fixed-n-estimators", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_summary = _load_json(args.source_run / "validation_summary.json")
    source_config = source_summary["config"]
    best_iteration = int(args.fixed_n_estimators or source_summary["best_iteration"])
    embedding_root = Path(source_summary["embedding_root"])
    synthetic_cold_start_fraction = float(source_summary["synthetic_cold_start_fraction"])
    seed = int(source_config["random_seed"])

    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    bundle = load_frozen_embedding_bundle(embedding_root)
    train_reviews = load_train_reviews(args.data_dir)
    train_frame = build_review_interaction_frame(train_reviews)
    priors = compute_scalar_priors(train_frame)
    min_ts, rc_means, rc_stds = fit_review_context_scaler(train_frame)

    x_train_main, feature_names = build_gbm_feature_matrix(
        train_frame,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_train_main = train_frame["rating"].to_numpy(dtype=np.float32)

    cold_rows = _sample_cold_start_rows(train_frame, synthetic_cold_start_fraction, seed)
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

    params = {
        "objective": "regression_l1",
        "metric": "l1",
        "num_leaves": int(source_config["num_leaves"]),
        "learning_rate": float(source_config["learning_rate"]),
        "min_child_samples": int(source_config["min_child_samples"]),
        "subsample": float(source_config["subsample"]),
        "subsample_freq": 1,
        "colsample_bytree": float(source_config["colsample_bytree"]),
        "reg_alpha": float(source_config["reg_alpha"]),
        "reg_lambda": float(source_config["reg_lambda"]),
        "verbose": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train, feature_name=feature_names),
        num_boost_round=best_iteration,
    )

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
    _save_json(save_root / "embedding_root.json", {"embedding_root": str(embedding_root)})
    _save_json(
        save_root / "train_summary.json",
        {
            "objective": "rating_regression",
            "model_type": "lightgbm_l1",
            "training_mode": "full_train_for_competition_submission",
            "source_run": str(args.source_run),
            "embedding_root": str(embedding_root),
            "fixed_n_estimators": best_iteration,
            "synthetic_cold_start_fraction": synthetic_cold_start_fraction,
            "n_train_rows": int(len(train_frame)),
            "n_train_rows_augmented": int(len(x_train)),
            "n_features": len(feature_names),
        },
    )
    print(
        json.dumps(
            {
                "embedding_root": str(embedding_root),
                "fixed_n_estimators": best_iteration,
                "n_train_rows": int(len(train_frame)),
                "n_train_rows_augmented": int(len(x_train)),
                "n_features": len(feature_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
