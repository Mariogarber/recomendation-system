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
from utils.lgbm_raw_features import RAW_PRIORS_FEATURE_SET
from utils.lgbm_raw_router_features import build_router_feature_frame, fit_router_feature_spec
from utils.lgbm_tabular_moe import (
    TABULAR_BAND_TO_EXPERT,
    TabularMoESpec,
    apply_tabular_blend,
    build_feature_columns_by_expert,
    collapse_history_band,
    compute_tabular_baseline_prediction,
    eval_prefix_frame,
    resolve_tabular_router_branches,
    train_prefix_frame,
)
from utils.split import cold_start_breakdown, temporal_train_validation_split


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _extract_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return frame[feature_columns].copy()


def _build_params(*, seed: int, band: str, args: argparse.Namespace) -> dict[str, Any]:
    num_leaves = args.num_leaves
    min_child_samples = args.min_child_samples
    reg_alpha = args.reg_alpha
    reg_lambda = args.reg_lambda
    if band == "0":
        num_leaves = min(num_leaves, 127)
        min_child_samples = max(min_child_samples, 120)
        reg_alpha = max(reg_alpha, 0.15)
        reg_lambda = max(reg_lambda, 1.5)
    elif band == ">20":
        min_child_samples = max(min_child_samples, 80)
        reg_lambda = max(reg_lambda, 1.25)
    return {
        "objective": "regression_l1",
        "metric": "l1",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": args.learning_rate,
        "min_child_samples": min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": 1,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "verbose": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }


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
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        categorical_feature=categorical_columns,
        free_raw_data=False,
    )
    callbacks: list[Any] = [lgb.log_evaluation(period=50)]
    valid_sets = [train_set]
    if x_val is not None and y_val is not None and len(x_val) > 0:
        valid_set = lgb.Dataset(
            x_val,
            label=y_val,
            categorical_feature=categorical_columns,
            reference=train_set,
            free_raw_data=False,
        )
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


def _compute_band_metrics(frame: pd.DataFrame, prediction_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in ["0", "1", "2-20", ">20"]:
        subset = frame[frame["history_band"].astype(str) == band]
        if subset.empty:
            continue
        diff = subset["rating"].to_numpy(dtype=np.float32) - subset[prediction_col].to_numpy(dtype=np.float32)
        rows.append(
            {
                "history_band": band,
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff ** 2))),
                "n_samples": int(len(subset)),
            }
        )
    return rows


def _save_feature_importance(path: Path, booster: lgb.Booster, feature_columns: list[str]) -> None:
    pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False).to_csv(path, index=False)


def _read_baseline_router_summary(save_root: Path) -> dict[str, Any] | None:
    baseline_path = save_root.parent / "lgbm_raw_router_prefix_deep_v1" / "validation_summary.json"
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def _compute_baseline_comparison(current_mae: float, baseline_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if baseline_summary is None:
        return None
    baseline_mae = float(baseline_summary.get("router_validation_mae_rounded", 0.0))
    return {
        "baseline_artifact": "lgbm_raw_router_prefix_deep_v1",
        "baseline_router_validation_mae_rounded": baseline_mae,
        "current_router_validation_mae_rounded": float(current_mae),
        "mae_delta_vs_baseline": float(current_mae - baseline_mae),
    }


def _calibrate_blend_alpha(
    *,
    y_true: np.ndarray,
    expert_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> tuple[float, float]:
    best_alpha = 1.0
    best_mae = float(np.mean(np.abs(y_true - _round_half_up(expert_pred).clip(1, 5).astype(np.float32))))
    for alpha in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0]:
        blended = (alpha * expert_pred) + ((1.0 - alpha) * baseline_pred)
        rounded = _round_half_up(blended).clip(1, 5).astype(np.float32)
        mae = float(np.mean(np.abs(y_true - rounded)))
        if mae < best_mae:
            best_alpha = float(alpha)
            best_mae = mae
    return best_alpha, best_mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a prefix-safe four-expert fully tabular MoE over raw, archetype, and prefix-history features.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_tabular_moe_v1",
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--n-user-archetypes", type=int, default=64)
    parser.add_argument("--max-top-cities", type=int, default=100)
    parser.add_argument("--max-top-categories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-leaves", type=int, default=191)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--n-estimators", type=int, default=1400)
    parser.add_argument("--min-child-samples", type=int, default=60)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.05)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
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

    validation_router_spec = fit_router_feature_spec(
        train_split,
        users_df,
        businesses_df,
        feature_set=RAW_PRIORS_FEATURE_SET,
        n_user_archetypes=args.n_user_archetypes,
        max_top_cities=args.max_top_cities,
        max_top_categories=args.max_top_categories,
        random_seed=args.seed,
    )
    router_train_frame = build_router_feature_frame(train_split, users_df, businesses_df, validation_router_spec)
    router_val_frame = build_router_feature_frame(val_split, users_df, businesses_df, validation_router_spec)

    train_frame = train_prefix_frame(router_train_frame, global_mean=validation_router_spec.base_spec.global_mean)
    val_frame = eval_prefix_frame(
        router_val_frame,
        router_train_frame,
        global_mean=validation_router_spec.base_spec.global_mean,
    )
    train_frame["history_band"] = train_frame["prefix_user_count"].map(lambda value: collapse_history_band(int(value))).astype("string")
    val_frame["history_band"] = val_frame["prefix_user_count"].map(lambda value: collapse_history_band(int(value))).astype("string")

    all_feature_columns = [
        column
        for column in train_frame.columns
        if column not in {"review_id", "user", "item", "rating", "review_date", "history_band"}
        and (
            pd.api.types.is_numeric_dtype(train_frame[column])
            or isinstance(train_frame[column].dtype, pd.CategoricalDtype)
        )
    ]
    feature_columns_by_expert, categorical_columns_by_expert, feature_manifest = build_feature_columns_by_expert(
        base_feature_columns=all_feature_columns,
        categorical_columns=validation_router_spec.categorical_columns,
    )

    baseline_val = compute_tabular_baseline_prediction(val_frame, global_mean=validation_router_spec.base_spec.global_mean)
    expert_predictions_raw: dict[str, np.ndarray] = {}
    model_best_iteration: dict[str, int] = {}
    band_training_rows: dict[str, int] = {}
    band_validation_rows: dict[str, int] = {}
    blend_alpha_by_band: dict[str, float] = {}
    band_model_summary: list[dict[str, Any]] = []
    validation_boosters: dict[str, lgb.Booster] = {}

    for band, expert_name in TABULAR_BAND_TO_EXPERT.items():
        band_train = train_frame[train_frame["history_band"].astype(str) == band].copy()
        band_val = val_frame[val_frame["history_band"].astype(str) == band].copy()
        feature_columns = feature_columns_by_expert[expert_name]
        categorical_columns = categorical_columns_by_expert[expert_name]
        band_training_rows[band] = int(len(band_train))
        band_validation_rows[band] = int(len(band_val))
        if band_train.empty:
            raise ValueError(f"No training rows available for history band {band}.")

        booster = _train_booster(
            x_train=_extract_feature_matrix(band_train, feature_columns),
            y_train=band_train["rating"].to_numpy(dtype=np.float32),
            x_val=_extract_feature_matrix(band_val, feature_columns) if not band_val.empty else None,
            y_val=band_val["rating"].to_numpy(dtype=np.float32) if not band_val.empty else None,
            categorical_columns=categorical_columns,
            params=_build_params(seed=args.seed, band=band, args=args),
            num_boost_round=args.n_estimators,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        best_iteration = int(booster.best_iteration or booster.current_iteration() or args.n_estimators)
        model_best_iteration[expert_name] = best_iteration
        validation_boosters[expert_name] = booster

        band_pred = np.full(len(val_frame), np.nan, dtype=np.float32)
        if not band_val.empty:
            band_pred_subset = np.clip(
                booster.predict(
                    _extract_feature_matrix(band_val, feature_columns),
                    num_iteration=best_iteration,
                ).astype(np.float32),
                1.0,
                5.0,
            )
            band_positions = val_frame.index.get_indexer(band_val.index)
            band_pred[band_positions] = band_pred_subset
            alpha, blended_mae = _calibrate_blend_alpha(
                y_true=band_val["rating"].to_numpy(dtype=np.float32),
                expert_pred=band_pred_subset,
                baseline_pred=baseline_val[band_positions],
            )
        else:
            alpha = 1.0
            blended_mae = float("nan")
        blend_alpha_by_band[band] = alpha
        expert_predictions_raw[expert_name] = band_pred
        band_model_summary.append(
            {
                "history_band": band,
                "expert_name": expert_name,
                "best_iteration": best_iteration,
                "train_rows": int(len(band_train)),
                "val_rows": int(len(band_val)),
                "n_features": int(len(feature_columns)),
                "blend_alpha": float(alpha),
                "validation_mae_after_blend": blended_mae,
            }
        )
        _save_feature_importance(save_root / f"{expert_name}_feature_importance.csv", booster, feature_columns)

    val_expert_raw = np.full(len(val_frame), np.nan, dtype=np.float32)
    val_branches = resolve_tabular_router_branches(val_frame["history_band"])
    for _, expert_name in TABULAR_BAND_TO_EXPERT.items():
        mask = val_branches == expert_name
        if mask.any():
            val_expert_raw[mask] = expert_predictions_raw[expert_name][mask]
    val_router_raw = apply_tabular_blend(
        expert_pred=val_expert_raw,
        baseline_pred=baseline_val,
        history_band=val_frame["history_band"],
        blend_alpha_by_band=blend_alpha_by_band,
    )
    val_router_rounded = _round_half_up(val_router_raw).clip(1, 5).astype(np.int32)

    validation_eval = val_frame[["review_id", "user", "item", "rating", "history_band"]].copy()
    validation_eval["baseline_pred_raw"] = baseline_val
    validation_eval["expert_pred_raw"] = val_expert_raw
    validation_eval["router_pred_raw"] = val_router_raw
    validation_eval["router_pred_rounded"] = val_router_rounded
    validation_eval["router_branch"] = val_branches
    validation_eval.to_csv(save_root / "validation_predictions.csv", index=False)

    routing_policy = {
        "type": "deterministic_history_band",
        "bands": TABULAR_BAND_TO_EXPERT.copy(),
        "blend_alpha_by_band": blend_alpha_by_band.copy(),
    }
    validation_spec = TabularMoESpec(
        router_spec=validation_router_spec,
        feature_columns_by_expert=feature_columns_by_expert,
        categorical_columns_by_expert=categorical_columns_by_expert,
        blend_alpha_by_band=blend_alpha_by_band.copy(),
        routing_policy=routing_policy,
        feature_manifest={
            "router_feature_set": RAW_PRIORS_FEATURE_SET,
            "tabular_history_columns": feature_manifest,
        },
    )

    validation_summary = {
        "router_validation_mae_raw": float(np.mean(np.abs(validation_eval["rating"].to_numpy(dtype=np.float32) - validation_eval["router_pred_raw"].to_numpy(dtype=np.float32)))),
        "router_validation_mae_rounded": float(np.mean(np.abs(validation_eval["rating"].to_numpy(dtype=np.float32) - validation_eval["router_pred_rounded"].to_numpy(dtype=np.float32)))),
        "router_validation_rmse_rounded": float(np.sqrt(np.mean((validation_eval["rating"].to_numpy(dtype=np.float32) - validation_eval["router_pred_rounded"].to_numpy(dtype=np.float32)) ** 2))),
        "band_metrics_router": _compute_band_metrics(validation_eval, "router_pred_rounded"),
        "router_branch_rows": {
            expert_name: int((validation_eval["router_branch"] == expert_name).sum())
            for expert_name in TABULAR_BAND_TO_EXPERT.values()
        },
        "band_training_rows": band_training_rows,
        "band_validation_rows": band_validation_rows,
        "band_model_summary": band_model_summary,
        "routing_policy": routing_policy,
        "cold_start_breakdown": cold_start_breakdown(train_split, val_split, user_col="user_id", item_col="business_id"),
        "feature_manifest": validation_spec.feature_manifest,
    }
    validation_summary["baseline_comparison"] = _compute_baseline_comparison(
        validation_summary["router_validation_mae_rounded"],
        _read_baseline_router_summary(save_root),
    )

    joblib.dump(validation_spec, save_root / "validation_tabular_moe_spec.joblib")
    for expert_name, booster in validation_boosters.items():
        booster.save_model(str(save_root / f"{expert_name}_validation_model.txt"))
    _save_json(save_root / "validation_summary.json", validation_summary)
    _save_json(save_root / "feature_manifest.json", validation_spec.feature_manifest)

    submission_router_spec = fit_router_feature_spec(
        train_reviews,
        users_df,
        businesses_df,
        feature_set=RAW_PRIORS_FEATURE_SET,
        n_user_archetypes=args.n_user_archetypes,
        max_top_cities=args.max_top_cities,
        max_top_categories=args.max_top_categories,
        random_seed=args.seed,
    )
    full_router_frame = build_router_feature_frame(train_reviews, users_df, businesses_df, submission_router_spec)
    full_train_frame = train_prefix_frame(full_router_frame, global_mean=submission_router_spec.base_spec.global_mean)
    full_train_frame["history_band"] = full_train_frame["prefix_user_count"].map(lambda value: collapse_history_band(int(value))).astype("string")

    submission_spec = TabularMoESpec(
        router_spec=submission_router_spec,
        feature_columns_by_expert=feature_columns_by_expert,
        categorical_columns_by_expert=categorical_columns_by_expert,
        blend_alpha_by_band=blend_alpha_by_band.copy(),
        routing_policy=routing_policy,
        feature_manifest=validation_spec.feature_manifest,
    )
    joblib.dump(submission_spec, save_root / "submission_tabular_moe_spec.joblib")

    submission_training_summary: dict[str, Any] = {}
    for band, expert_name in TABULAR_BAND_TO_EXPERT.items():
        band_train = full_train_frame[full_train_frame["history_band"].astype(str) == band].copy()
        booster = _train_booster(
            x_train=_extract_feature_matrix(band_train, feature_columns_by_expert[expert_name]),
            y_train=band_train["rating"].to_numpy(dtype=np.float32),
            x_val=None,
            y_val=None,
            categorical_columns=categorical_columns_by_expert[expert_name],
            params=_build_params(seed=args.seed, band=band, args=args),
            num_boost_round=model_best_iteration[expert_name],
            early_stopping_rounds=None,
        )
        booster.save_model(str(save_root / f"{expert_name}_submission_model.txt"))
        submission_training_summary[expert_name] = {
            "best_iteration_from_validation": model_best_iteration[expert_name],
            "full_train_rows": int(len(band_train)),
            "n_features": int(len(feature_columns_by_expert[expert_name])),
        }

    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    training_summary = {
        "validation_spec_path": str(save_root / "validation_tabular_moe_spec.joblib"),
        "submission_spec_path": str(save_root / "submission_tabular_moe_spec.joblib"),
        "validation_predictions_path": str(save_root / "validation_predictions.csv"),
        "models": submission_training_summary,
    }
    _save_json(save_root / "training_summary.json", training_summary)
    _save_json(save_root / "config.json", config_payload)
    print(json.dumps({"validation_summary": validation_summary, "training_summary": training_summary}, indent=2))


if __name__ == "__main__":
    main()
