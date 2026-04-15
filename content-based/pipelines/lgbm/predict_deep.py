from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.io import get_default_data_dir, load_test_reviews, load_train_reviews
from utils.lgbm_deep_embeddings import (
    LGBMDeepEmbeddingConfig,
    build_lgbm_feature_matrix,
    build_lgbm_params,
    build_review_context_only_frame,
    build_review_interaction_frame,
    compute_scalar_priors,
    fit_review_context_scaler,
    load_deep_embedding_bundle,
    load_json,
    round_half_up,
    save_json,
    save_scalar_priors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final LightGBM submission model and export predictions.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_deep_embeddings_v1",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_deep_embeddings_submission_v1",
    )
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--fixed-n-estimators", type=int, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> LGBMDeepEmbeddingConfig:
    data = load_json(path)
    return LGBMDeepEmbeddingConfig(**data)


def _save_feature_importance(path: Path, names: list[str], booster: lgb.Booster) -> None:
    importance = booster.feature_importance(importance_type="gain")
    frame = pd.DataFrame({"feature_name": names, "gain": importance.astype(np.float64)})
    frame.sort_values("gain", ascending=False).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    source_run = args.source_run
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = args.save_path or (save_root / "submission.csv")

    validation_summary = load_json(source_run / "validation_summary.json")
    config = _load_config(source_run / "config.json")
    best_iteration = int(args.fixed_n_estimators or validation_summary["best_iteration"])
    embedding_root = Path(validation_summary["embedding_root"])
    synthetic_cold_start_fraction = float(validation_summary["synthetic_cold_start_fraction"])
    seed = int(config.random_seed)

    bundle = load_deep_embedding_bundle(embedding_root)
    train_reviews = load_train_reviews(args.data_dir)
    train_frame = build_review_interaction_frame(train_reviews)
    priors = compute_scalar_priors(train_frame)
    min_ts, rc_means, rc_stds = fit_review_context_scaler(train_frame)

    x_train_main, feature_names = build_lgbm_feature_matrix(
        train_frame,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )
    y_train_main = train_frame["rating"].to_numpy(dtype=np.float32)

    cold_rows = train_frame.sample(
        frac=synthetic_cold_start_fraction,
        random_state=seed,
    )
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

    params = build_lgbm_params(config)
    booster = lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train, feature_name=feature_names),
        num_boost_round=best_iteration,
    )

    test_reviews = load_test_reviews(args.data_dir)
    test_frame = build_review_context_only_frame(test_reviews)

    x_test, _ = build_lgbm_feature_matrix(
        test_frame,
        bundle,
        priors,
        review_context_min_timestamp=min_ts,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )

    raw_pred = np.clip(booster.predict(x_test, num_iteration=best_iteration).astype(np.float32), 1.0, 5.0)
    rounded_pred = round_half_up(raw_pred).clip(1, 5).astype(np.int32)
    submission = pd.DataFrame({"review_id": test_frame["review_id"].astype(str), "stars": rounded_pred})
    submission.to_csv(save_path, index=False)

    save_scalar_priors(save_root / "scalar_priors.json", priors)
    save_json(
        save_root / "review_context_scaler.json",
        {
            "min_timestamp": min_ts.isoformat(),
            "means": rc_means.tolist(),
            "stds": rc_stds.tolist(),
        },
    )
    save_json(save_root / "embedding_root.json", {"embedding_root": str(embedding_root)})
    save_json(
        save_root / "train_summary.json",
        {
            "source_run": str(source_run),
            "embedding_root": str(embedding_root),
            "best_iteration": best_iteration,
            "synthetic_cold_start_fraction": synthetic_cold_start_fraction,
            "n_train_rows": int(len(train_frame)),
            "n_train_rows_augmented": int(len(x_train)),
            "n_features": int(len(feature_names)),
            "save_path": str(save_path),
        },
    )
    _save_feature_importance(save_root / "feature_importance.csv", feature_names, booster)
    joblib.dump(booster, save_root / "model.pkl")
    save_json(save_root / "config.json", asdict(config))

    summary = {
        "source_run": str(source_run),
        "embedding_root": str(embedding_root),
        "save_path": str(save_path),
        "n_rows": int(len(submission)),
        "n_new_user_rows": int((~test_frame["user"].isin(priors.user_count.keys())).sum()),
        "prediction_min": int(rounded_pred.min()),
        "prediction_max": int(rounded_pred.max()),
        "prediction_mean_raw": float(raw_pred.mean()),
        "prediction_mean_rounded": float(rounded_pred.mean()),
        "best_iteration": best_iteration,
    }
    save_json(save_root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
