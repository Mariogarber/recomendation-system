from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import shutil
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model.known_user_deep_e2e import KnownUserDeepE2EModel
from utils.io import get_default_data_dir, load_businesses, load_test_reviews, load_train_reviews, load_users
from utils.known_user_deep_e2e import (
    KnownUserDeepDataConfig,
    KnownUserDeepTrainingConfig,
    build_known_user_eval_dataset,
    build_known_user_train_dataset,
    fit_known_user_deep_final_model,
    predict_known_user_dataset,
    prepare_known_user_context,
    save_known_user_checkpoint,
    train_known_user_deep_model,
)
from utils.lgbm_known_prefix_deep_features import build_known_prefix_eval_frame, load_known_prefix_embedding_bundle, resolve_router_branches
from utils.lgbm_raw_features import build_raw_feature_frame, history_band_from_count
from utils.lgbm_raw_router_features import build_router_feature_frame
from utils.split import temporal_train_validation_split


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _resolve_user_column(frame: pd.DataFrame) -> str:
    for candidate in ("user_id", "user"):
        if candidate in frame.columns:
            return candidate
    raise KeyError("Expected a user column named 'user_id' or 'user'.")


def _history_band_lookup_from_context(context_reviews: pd.DataFrame) -> dict[str, str]:
    user_column = _resolve_user_column(context_reviews)
    counts = context_reviews[user_column].astype(str).value_counts(dropna=False).to_dict()
    return {
        str(user_id): history_band_from_count(int(count))
        for user_id, count in counts.items()
    }


def _merge_known_prefix_features(base_frame: pd.DataFrame, deep_frame: pd.DataFrame) -> pd.DataFrame:
    left = base_frame.copy()
    right = deep_frame.copy()
    left["_review_id_key"] = left["review_id"].astype(str)
    right["_review_id_key"] = right["review_id"].astype(str)
    deep_columns = [
        column
        for column in right.columns
        if column.startswith("known_prefix_") and column != "known_prefix_history_band"
    ]
    merged = left.merge(right[["_review_id_key", *deep_columns]], on="_review_id_key", how="inner")
    return merged.drop(columns=["_review_id_key"])


def _apply_prediction_to_frame(
    *,
    frame: pd.DataFrame,
    subset_frame: pd.DataFrame,
    prediction: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    output = np.full(len(frame), fill_value, dtype=np.float32)
    if subset_frame.empty:
        return output
    index_lookup = pd.Series(np.arange(len(frame), dtype=np.int32), index=frame["review_id"].astype(str))
    positions = subset_frame["review_id"].astype(str).map(index_lookup).to_numpy(dtype=np.int32)
    output[positions] = prediction.astype(np.float32)
    return output


def _prediction_metrics(frame: pd.DataFrame, prediction_col: str) -> dict[str, float]:
    diff = frame["rating"].to_numpy(dtype=np.float32) - frame[prediction_col].to_numpy(dtype=np.float32)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
    }


def _band_metrics(frame: pd.DataFrame, prediction_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in ["0", "1", "2-5", "6-20", ">20"]:
        subset = frame[frame["history_band"].astype(str) == band]
        if subset.empty:
            continue
        metrics = _prediction_metrics(subset, prediction_col)
        rows.append(
            {
                "history_band": band,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "n_samples": int(len(subset)),
            }
        )
    return rows


def _short_history_metrics(frame: pd.DataFrame, prediction_col: str) -> list[dict[str, Any]]:
    if "history_count" not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    history_count = frame["history_count"].to_numpy(dtype=np.float32)
    specs = [
        ("2", history_count == 2),
        ("3", history_count == 3),
        ("4", history_count == 4),
        ("5", history_count == 5),
        ("2-3", np.isin(history_count, [2, 3])),
        ("4-5", np.isin(history_count, [4, 5])),
    ]
    for label, mask in specs:
        if not mask.any():
            continue
        subset = frame.loc[mask]
        metrics = _prediction_metrics(subset, prediction_col)
        rows.append(
            {
                "history_count_segment": label,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "n_samples": int(mask.sum()),
            }
        )
    return rows


def _branch_count_payload(frame: pd.DataFrame, branch_col: str) -> dict[str, int]:
    counts = frame[branch_col].astype(str).value_counts(dropna=False).to_dict()
    return {
        "known_model": int(counts.get("known_model", 0)),
        "known_prefix_deep_model": int(counts.get("known_prefix_deep_model", 0)),
        "known_user_deep_e2e_model": int(counts.get("known_user_deep_e2e_model", 0)),
        "cold_model": int(counts.get("cold_model", 0)),
    }


def _coverage_by_band(frame: pd.DataFrame, *, branch_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in ["0", "1", "2-5", "6-20", ">20"]:
        subset = frame[frame["history_band"].astype(str) == band]
        if subset.empty:
            continue
        available = subset[subset["deep_prediction_available"].to_numpy(dtype=bool)]
        rows.append(
            {
                "history_band": band,
                "total_rows": int(len(subset)),
                "deep_available_rows": int(len(available)),
                "deep_unavailable_rows": int(len(subset) - len(available)),
                "incumbent_branch_rows": _branch_count_payload(subset, branch_col),
                "deep_available_incumbent_branch_rows": _branch_count_payload(available, branch_col),
            }
        )
    return rows


def _deep_model_band_eval(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in ["1", "2-5", "6-20", ">20"]:
        subset = frame[
            (frame["history_band"].astype(str) == band)
            & frame["deep_prediction_available"].to_numpy(dtype=bool)
        ]
        if subset.empty:
            continue
        incumbent_metrics = _prediction_metrics(subset, "incumbent_prediction")
        deep_metrics = _prediction_metrics(subset, "deep_prediction")
        rows.append(
            {
                "history_band": band,
                "n_samples": int(len(subset)),
                "incumbent_mae": incumbent_metrics["mae"],
                "incumbent_rmse": incumbent_metrics["rmse"],
                "deep_mae": deep_metrics["mae"],
                "deep_rmse": deep_metrics["rmse"],
                "delta_mae": float(deep_metrics["mae"] - incumbent_metrics["mae"]),
            }
        )
    return rows


def _deep_model_short_history_eval(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if "history_count" not in frame.columns:
        return []
    available = frame[
        (frame["history_band"].astype(str) == "2-5")
        & frame["deep_prediction_available"].to_numpy(dtype=bool)
    ].copy()
    if available.empty:
        return []
    rows: list[dict[str, Any]] = []
    history_count = available["history_count"].to_numpy(dtype=np.float32)
    specs = [
        ("2", history_count == 2),
        ("3", history_count == 3),
        ("4", history_count == 4),
        ("5", history_count == 5),
        ("2-3", np.isin(history_count, [2, 3])),
        ("4-5", np.isin(history_count, [4, 5])),
    ]
    for label, mask in specs:
        if not mask.any():
            continue
        subset = available.loc[mask]
        incumbent_metrics = _prediction_metrics(subset, "incumbent_prediction")
        deep_metrics = _prediction_metrics(subset, "deep_prediction")
        rows.append(
            {
                "history_count_segment": label,
                "n_samples": int(mask.sum()),
                "incumbent_mae": incumbent_metrics["mae"],
                "deep_mae": deep_metrics["mae"],
                "delta_mae": float(deep_metrics["mae"] - incumbent_metrics["mae"]),
            }
        )
    return rows


def _attach_deep_predictions(incumbent_frame: pd.DataFrame, deep_frame: pd.DataFrame) -> pd.DataFrame:
    eval_frame = incumbent_frame.copy()
    eval_frame["_review_id_key"] = eval_frame["review_id"].astype(str)
    deep_lookup = deep_frame.copy()
    deep_lookup["_review_id_key"] = deep_lookup["review_id"].astype(str)
    deep_columns = [column for column in ["deep_prediction", "deep_prediction_raw", "alpha", "baseline_hat", "correction_hat", "residual_hat"] if column in deep_lookup.columns]
    eval_frame = eval_frame.merge(deep_lookup[["_review_id_key", *deep_columns]], on="_review_id_key", how="left")
    if "history_count" not in eval_frame.columns and "history_count" in deep_lookup.columns:
        eval_frame = eval_frame.merge(deep_lookup[["_review_id_key", "history_count"]], on="_review_id_key", how="left")
    eval_frame["deep_prediction_available"] = np.isfinite(eval_frame.get("deep_prediction_raw", np.nan)).astype(bool)
    return eval_frame.drop(columns=["_review_id_key"])


def _alpha_correction_stats_by_band(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in ["1", "2-5", "6-20", ">20"]:
        subset = frame[
            (frame["history_band"].astype(str) == band)
            & frame["deep_prediction_available"].to_numpy(dtype=bool)
        ]
        if subset.empty:
            continue
        if "correction_hat" in subset.columns:
            correction = subset["correction_hat"].to_numpy(dtype=np.float32)
        elif "residual_hat" in subset.columns:
            correction = subset["residual_hat"].to_numpy(dtype=np.float32)
        else:
            correction = np.zeros(len(subset), dtype=np.float32)
        incumbent = subset["incumbent_prediction"].to_numpy(dtype=np.float32)
        deep = subset["deep_prediction"].to_numpy(dtype=np.float32)
        target = subset["rating"].to_numpy(dtype=np.float32)
        incumbent_abs = np.abs(target - incumbent)
        deep_abs = np.abs(target - deep)
        rows.append(
            {
                "history_band": band,
                "n_samples": int(len(subset)),
                "alpha_mean": float(subset["alpha"].mean()) if "alpha" in subset.columns else None,
                "alpha_std": float(subset["alpha"].std(ddof=0)) if "alpha" in subset.columns else None,
                "alpha_percentiles": {
                    "p10": float(np.percentile(subset["alpha"].to_numpy(dtype=np.float32), 10)) if "alpha" in subset.columns else None,
                    "p25": float(np.percentile(subset["alpha"].to_numpy(dtype=np.float32), 25)) if "alpha" in subset.columns else None,
                    "p50": float(np.percentile(subset["alpha"].to_numpy(dtype=np.float32), 50)) if "alpha" in subset.columns else None,
                    "p75": float(np.percentile(subset["alpha"].to_numpy(dtype=np.float32), 75)) if "alpha" in subset.columns else None,
                    "p90": float(np.percentile(subset["alpha"].to_numpy(dtype=np.float32), 90)) if "alpha" in subset.columns else None,
                },
                "mean_abs_correction": float(np.abs(correction).mean()),
                "improved_vs_incumbent_pct": float(np.mean(deep_abs < incumbent_abs)),
                "worse_vs_incumbent_pct": float(np.mean(deep_abs > incumbent_abs)),
            }
        )
    return rows


def _correction_series(frame: pd.DataFrame) -> np.ndarray:
    if "correction_hat" in frame.columns:
        return frame["correction_hat"].to_numpy(dtype=np.float32)
    if "residual_hat" in frame.columns:
        return frame["residual_hat"].to_numpy(dtype=np.float32)
    return np.zeros(len(frame), dtype=np.float32)


def _apply_replace_policy(
    frame: pd.DataFrame,
    *,
    enabled_bands: list[str],
    training_config: KnownUserDeepTrainingConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    enabled_mask = frame["history_band"].astype(str).isin(enabled_bands).to_numpy(dtype=bool)
    deep_available_mask = frame["deep_prediction_available"].to_numpy(dtype=bool)
    replace_mask = enabled_mask & deep_available_mask
    policy_payload: dict[str, Any] = {
        "enabled_bands": enabled_bands.copy(),
        "selective_replace_alpha_thresholds": training_config.selective_replace_alpha_thresholds or {},
        "selective_replace_abs_correction_thresholds": training_config.selective_replace_abs_correction_thresholds or {},
        "band_thresholds_applied": {},
    }
    if not replace_mask.any():
        return replace_mask, policy_payload

    alpha_thresholds = training_config.selective_replace_alpha_thresholds or {}
    correction_thresholds = training_config.selective_replace_abs_correction_thresholds or {}
    correction = _correction_series(frame)
    alpha_values = frame["alpha"].to_numpy(dtype=np.float32) if "alpha" in frame.columns else np.zeros(len(frame), dtype=np.float32)
    band_values = frame["history_band"].astype(str).to_numpy()
    for band in enabled_bands:
        band_replace_mask = replace_mask & (band_values == band)
        if not band_replace_mask.any():
            continue
        initial_rows = int(band_replace_mask.sum())
        alpha_threshold = alpha_thresholds.get(band)
        correction_threshold = correction_thresholds.get(band)
        if alpha_threshold is not None:
            band_replace_mask &= alpha_values >= float(alpha_threshold)
        if correction_threshold is not None:
            band_replace_mask &= np.abs(correction) >= float(correction_threshold)
        replace_mask[(band_values == band) & enabled_mask & deep_available_mask] = False
        replace_mask[band_replace_mask] = True
        policy_payload["band_thresholds_applied"][band] = {
            "initial_candidate_rows": initial_rows,
            "selected_rows": int(band_replace_mask.sum()),
            "alpha_threshold": float(alpha_threshold) if alpha_threshold is not None else None,
            "abs_correction_threshold": float(correction_threshold) if correction_threshold is not None else None,
        }
    return replace_mask, policy_payload


def _threshold_subset_metrics(subset: pd.DataFrame, *, value_col: str, thresholds: list[float]) -> list[dict[str, Any]]:
    if subset.empty or value_col not in subset.columns:
        return []
    rows: list[dict[str, Any]] = []
    target = subset["rating"].to_numpy(dtype=np.float32)
    incumbent_abs = np.abs(target - subset["incumbent_prediction"].to_numpy(dtype=np.float32))
    deep_abs = np.abs(target - subset["deep_prediction"].to_numpy(dtype=np.float32))
    values = subset[value_col].to_numpy(dtype=np.float32)
    for threshold in thresholds:
        mask = values >= float(threshold)
        if not mask.any():
            rows.append(
                {
                    "threshold": float(threshold),
                    "n_samples": 0,
                    "incumbent_mae": None,
                    "deep_mae": None,
                    "delta_mae": None,
                    "improved_vs_incumbent_pct": None,
                }
            )
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "n_samples": int(mask.sum()),
                "incumbent_mae": float(np.mean(incumbent_abs[mask])),
                "deep_mae": float(np.mean(deep_abs[mask])),
                "delta_mae": float(np.mean(deep_abs[mask]) - np.mean(incumbent_abs[mask])),
                "improved_vs_incumbent_pct": float(np.mean(deep_abs[mask] < incumbent_abs[mask])),
            }
        )
    return rows


def _history_count_bin_diagnostics(subset: pd.DataFrame) -> list[dict[str, Any]]:
    if subset.empty or "history_count" not in subset.columns:
        return []
    rows: list[dict[str, Any]] = []
    target = subset["rating"].to_numpy(dtype=np.float32)
    incumbent_abs = np.abs(target - subset["incumbent_prediction"].to_numpy(dtype=np.float32))
    deep_abs = np.abs(target - subset["deep_prediction"].to_numpy(dtype=np.float32))
    history_count = subset["history_count"].to_numpy(dtype=np.float32)
    for label, mask in [
        ("2", history_count == 2),
        ("3", history_count == 3),
        ("4", history_count == 4),
        ("5", history_count == 5),
        ("2-3", np.isin(history_count, [2, 3])),
        ("4-5", np.isin(history_count, [4, 5])),
    ]:
        if not mask.any():
            continue
        rows.append(
            {
                "history_count_bin": label,
                "n_samples": int(mask.sum()),
                "incumbent_mae": float(np.mean(incumbent_abs[mask])),
                "deep_mae": float(np.mean(deep_abs[mask])),
                "delta_mae": float(np.mean(deep_abs[mask]) - np.mean(incumbent_abs[mask])),
            }
        )
    return rows


def _load_incumbent_stack(root: Path) -> dict[str, Any]:
    spec = joblib.load(root / "submission_router_spec.joblib")
    return {
        "spec": spec,
        "known_booster": lgb.Booster(model_file=str(root / "known_submission_model.txt")),
        "known_prefix_booster": lgb.Booster(model_file=str(root / "known_prefix_submission_model.txt")),
        "cold_booster": lgb.Booster(model_file=str(root / "cold_submission_model.txt")),
        "prefix_bundle": load_known_prefix_embedding_bundle(spec.known_prefix_embedding_root),
    }


def _load_run_configs(max_runs: int, *, config_family: str) -> list[dict[str, Any]]:
    if config_family == "v3_feature_injected":
        configs: list[dict[str, Any]] = [
            {
                "run_name": "runA_v3_feature_injected",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=128,
                    event_hidden_dim=128,
                    user_type_hidden_dim=128,
                    scorer_hidden_dim=256,
                    business_hidden_layers=(512, 384, 256),
                    scorer_hidden_layers=(256, 128),
                    num_attention_heads=4,
                    dropout=0.15,
                    batch_size=512,
                    learning_rate=8e-4,
                    weight_decay=2e-5,
                    max_epochs=20,
                    early_stopping_patience=4,
                    auxiliary_loss_weight=0.15,
                    band_correction_scales={"1": 0.7, "2-5": 0.95, "6-20": 1.0, ">20": 0.95},
                    band_distillation_weights={"1": 0.06, "2-5": 0.06, "6-20": 0.05, ">20": 0.04},
                ),
            },
            {
                "run_name": "runB_v3_feature_injected_capacity",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=160,
                    event_hidden_dim=160,
                    user_type_hidden_dim=160,
                    scorer_hidden_dim=320,
                    business_hidden_layers=(512, 384, 256),
                    scorer_hidden_layers=(320, 160),
                    num_attention_heads=8,
                    dropout=0.15,
                    batch_size=512,
                    learning_rate=8e-4,
                    weight_decay=2e-5,
                    max_epochs=20,
                    early_stopping_patience=4,
                    auxiliary_loss_weight=0.15,
                    band_correction_scales={"1": 0.7, "2-5": 0.95, "6-20": 1.0, ">20": 0.95},
                    band_distillation_weights={"1": 0.06, "2-5": 0.06, "6-20": 0.05, ">20": 0.04},
                    alpha_regularization_weight=0.0015,
                ),
            },
        ]
        return configs[: max(1, min(max_runs, len(configs)))]
    if config_family == "v7_mae_loss":
        configs = [
            {
                "run_name": "runD2_mae_lr_fix",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=128,
                    event_hidden_dim=128,
                    user_type_hidden_dim=128,
                    scorer_hidden_dim=256,
                    business_hidden_layers=(512, 384, 256),
                    scorer_hidden_layers=(256, 128),
                    num_attention_heads=4,
                    dropout=0.15,
                    batch_size=512,
                    learning_rate=3e-4,
                    weight_decay=2e-5,
                    max_epochs=40,
                    early_stopping_patience=10,
                    auxiliary_loss_weight=0.15,
                    band_correction_scales={"1": 0.7, "2-3": 0.9, "4-5": 0.95, "6-20": 1.0, ">20": 0.95},
                    band_distillation_weights={"1": 0.06, "2-3": 0.06, "4-5": 0.05, "6-20": 0.05, ">20": 0.04},
                    use_direct_predictor=False,
                ),
            },
            {
                "run_name": "runD1_mae_v3_clone",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=128,
                    event_hidden_dim=128,
                    user_type_hidden_dim=128,
                    scorer_hidden_dim=256,
                    business_hidden_layers=(512, 384, 256),
                    scorer_hidden_layers=(256, 128),
                    num_attention_heads=4,
                    dropout=0.15,
                    batch_size=512,
                    learning_rate=8e-4,
                    weight_decay=2e-5,
                    max_epochs=25,
                    early_stopping_patience=6,
                    auxiliary_loss_weight=0.15,
                    band_correction_scales={"1": 0.7, "2-5": 0.95, "6-20": 1.0, ">20": 0.95},
                    band_distillation_weights={"1": 0.06, "2-3": 0.06, "4-5": 0.05, "6-20": 0.05, ">20": 0.04},
                    use_direct_predictor=False,
                ),
            },
        ]
        return configs[: max(1, min(max_runs, len(configs)))]

    if config_family == "v6_regularized":
        configs = [
            {
                "run_name": "runC1_direct_l2",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=128,
                    event_hidden_dim=128,
                    user_type_hidden_dim=128,
                    scorer_hidden_dim=256,
                    business_hidden_layers=(512, 384, 256),
                    scorer_hidden_layers=(256, 128),
                    num_attention_heads=4,
                    dropout=0.20,
                    batch_size=512,
                    learning_rate=1e-4,
                    weight_decay=1e-3,
                    max_epochs=50,
                    early_stopping_patience=12,
                    auxiliary_loss_weight=0.0,
                    band_correction_scales=None,
                    band_distillation_weights={"1": 0.0, "2-3": 0.0, "4-5": 0.0, "6-20": 0.0, ">20": 0.0},
                    use_direct_predictor=True,
                ),
            },
            {
                "run_name": "runC2_gated_wider",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=128,
                    event_hidden_dim=128,
                    user_type_hidden_dim=128,
                    scorer_hidden_dim=256,
                    business_hidden_layers=(512, 384, 256),
                    scorer_hidden_layers=(256, 128),
                    num_attention_heads=4,
                    dropout=0.15,
                    batch_size=512,
                    learning_rate=8e-4,
                    weight_decay=2e-5,
                    max_epochs=25,
                    early_stopping_patience=6,
                    auxiliary_loss_weight=0.15,
                    band_correction_scales={"1": 1.2, "2-5": 1.5, "6-20": 1.5, ">20": 1.5},
                    band_distillation_weights={"1": 0.06, "2-3": 0.06, "4-5": 0.05, "6-20": 0.05, ">20": 0.04},
                    use_direct_predictor=False,
                ),
            },
        ]
        return configs[: max(1, min(max_runs, len(configs)))]

    if config_family == "v5_direct_predictor":
        configs = [
            {
                "run_name": "runA_v5_direct_predictor",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=128,
                    event_hidden_dim=128,
                    user_type_hidden_dim=128,
                    scorer_hidden_dim=256,
                    business_hidden_layers=(512, 384, 256),
                    scorer_hidden_layers=(256, 128),
                    num_attention_heads=4,
                    dropout=0.15,
                    batch_size=512,
                    learning_rate=1e-4,
                    weight_decay=2e-5,
                    max_epochs=50,
                    early_stopping_patience=10,
                    auxiliary_loss_weight=0.0,
                    band_correction_scales=None,
                    band_distillation_weights={"1": 0.0, "2-3": 0.0, "4-5": 0.0, "6-20": 0.0, ">20": 0.0},
                    use_direct_predictor=True,
                ),
            },
        ]
        return configs[: max(1, min(max_runs, len(configs)))]

    # Ultra-light corrector: embedding_dim=16, ~50k params.
    # Motivation: v_lightweight (emb=32, ~200k params) showed high val_mae oscillation
    # (±0.05 in consecutive epochs) despite monotonically decreasing train_loss.  Root cause:
    # correction_scale=1.0 allows large corrections and alpha×tanh creates a non-convex surface
    # that the optimizer overshoots with lr=1e-3.  Two combined fixes:
    #   (1) halve embedding_dim again (32→16, ~50k params, ~6.8 examples/param for band 6-20)
    #   (2) tighten correction_scales to 0.4-0.5 so the model can only make small nudges
    #   (3) reduce lr to 2e-4 and increase batch_size to 2048 for stable gradient estimates
    # The goal is a stable monotone val_mae curve, not more correction capacity.
    if config_family == "v_ultralight":
        configs = [
            {
                "run_name": "runA_ul_emb16_tight",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=16,
                    event_hidden_dim=16,
                    user_type_hidden_dim=16,
                    scorer_hidden_dim=32,
                    business_hidden_layers=(32,),
                    scorer_hidden_layers=(32,),
                    num_attention_heads=2,
                    dropout=0.30,
                    batch_size=2048,
                    learning_rate=2e-4,
                    weight_decay=2e-3,
                    max_epochs=60,
                    early_stopping_patience=12,
                    auxiliary_loss_weight=0.10,
                    band_correction_scales={"1": 0.35, "2-5": 0.45, "6-20": 0.50, ">20": 0.45},
                    band_distillation_weights={"1": 0.06, "2-5": 0.06, "6-20": 0.05, ">20": 0.04},
                ),
            },
            {
                "run_name": "runB_ul_emb16_looser",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=16,
                    event_hidden_dim=16,
                    user_type_hidden_dim=16,
                    scorer_hidden_dim=32,
                    business_hidden_layers=(32,),
                    scorer_hidden_layers=(32,),
                    num_attention_heads=2,
                    dropout=0.25,
                    batch_size=2048,
                    learning_rate=2e-4,
                    weight_decay=5e-4,
                    max_epochs=60,
                    early_stopping_patience=12,
                    auxiliary_loss_weight=0.10,
                    band_correction_scales={"1": 0.35, "2-5": 0.60, "6-20": 0.70, ">20": 0.60},
                    band_distillation_weights={"1": 0.06, "2-5": 0.06, "6-20": 0.05, ">20": 0.04},
                ),
            },
        ]
        return configs[: max(1, min(max_runs, len(configs)))]

    # Lightweight corrector: embedding_dim=32, ~200k params (vs 3.28M full model).
    # Motivation: 337k training rows / 3.28M params = 0.1 examples/param — severe underfitting
    # territory for sparse bands. Reducing to ~200k gives ~1.7 examples/param, much healthier.
    # Band-specific experts are kept but all MLPs are shallow; we expect similar or better
    # generalisation with faster training and lower variance.
    if config_family == "v_lightweight":
        configs = [
            {
                "run_name": "runA_lw_emb32_base",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=32,
                    event_hidden_dim=32,
                    user_type_hidden_dim=32,
                    scorer_hidden_dim=64,
                    business_hidden_layers=(64,),
                    scorer_hidden_layers=(64, 32),
                    num_attention_heads=2,
                    dropout=0.20,
                    batch_size=1024,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                    max_epochs=40,
                    early_stopping_patience=8,
                    auxiliary_loss_weight=0.15,
                    band_correction_scales={"1": 0.7, "2-5": 0.95, "6-20": 1.0, ">20": 0.95},
                    band_distillation_weights={"1": 0.06, "2-5": 0.06, "6-20": 0.05, ">20": 0.04},
                ),
            },
            {
                "run_name": "runB_lw_emb32_nodistill",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=32,
                    event_hidden_dim=32,
                    user_type_hidden_dim=32,
                    scorer_hidden_dim=64,
                    business_hidden_layers=(64,),
                    scorer_hidden_layers=(64, 32),
                    num_attention_heads=2,
                    dropout=0.25,
                    batch_size=1024,
                    learning_rate=1e-3,
                    weight_decay=5e-4,
                    max_epochs=40,
                    early_stopping_patience=8,
                    auxiliary_loss_weight=0.10,
                    band_correction_scales={"1": 0.7, "2-5": 0.95, "6-20": 1.0, ">20": 0.95},
                    band_distillation_weights=None,
                ),
            },
            {
                # runC: the key missing experiment.
                # LW runA (emb=32, lr=1e-3, scale 6-20=1.0) got -0.00715 on band 6-20 —
                # best of any experiment — but the val curve oscillated ±0.05.
                # UL runB (emb=16, lr=2e-4, batch=2048) proved lr=2e-4+batch=2048 gives a
                # stable monotone curve, but emb=16 has too little capacity to learn the residuals.
                # This run combines the capacity that works (emb=32) with the stability fix
                # (lr=2e-4, batch=2048). Everything else matches runA exactly.
                "run_name": "runC_lw_emb32_stable_lr",
                "training": KnownUserDeepTrainingConfig(
                    embedding_dim=32,
                    event_hidden_dim=32,
                    user_type_hidden_dim=32,
                    scorer_hidden_dim=64,
                    business_hidden_layers=(64,),
                    scorer_hidden_layers=(64, 32),
                    num_attention_heads=2,
                    dropout=0.20,
                    batch_size=2048,
                    learning_rate=2e-4,
                    weight_decay=1e-4,
                    max_epochs=60,
                    early_stopping_patience=12,
                    auxiliary_loss_weight=0.15,
                    band_correction_scales={"1": 0.7, "2-5": 0.95, "6-20": 1.0, ">20": 0.95},
                    band_distillation_weights={"1": 0.06, "2-5": 0.06, "6-20": 0.05, ">20": 0.04},
                ),
            },
        ]
        return configs[: max(1, min(max_runs, len(configs)))]

    configs: list[dict[str, Any]] = [
        {
            "run_name": "runA_short_split_base",
            "training": KnownUserDeepTrainingConfig(
                embedding_dim=128,
                event_hidden_dim=128,
                user_type_hidden_dim=128,
                scorer_hidden_dim=256,
                business_hidden_layers=(512, 384, 256),
                scorer_hidden_layers=(256, 128),
                num_attention_heads=4,
                dropout=0.15,
                batch_size=512,
                learning_rate=8e-4,
                weight_decay=2e-5,
                max_epochs=20,
                early_stopping_patience=4,
                auxiliary_loss_weight=0.15,
                band_correction_scales={"1": 0.7, "2-3": 0.9, "4-5": 0.95, "6-20": 1.0, ">20": 0.95},
                band_distillation_weights={"1": 0.06, "2-3": 0.06, "4-5": 0.05, "6-20": 0.05, ">20": 0.04},
            ),
        },
        {
            "run_name": "runB_short_split_capacity",
            "training": KnownUserDeepTrainingConfig(
                embedding_dim=160,
                event_hidden_dim=160,
                user_type_hidden_dim=160,
                scorer_hidden_dim=320,
                business_hidden_layers=(512, 384, 256),
                scorer_hidden_layers=(320, 160),
                num_attention_heads=8,
                dropout=0.15,
                batch_size=512,
                learning_rate=8e-4,
                weight_decay=2e-5,
                max_epochs=20,
                early_stopping_patience=4,
                auxiliary_loss_weight=0.15,
                band_correction_scales={"1": 0.7, "2-3": 0.9, "4-5": 0.95, "6-20": 1.0, ">20": 0.95},
                band_distillation_weights={"1": 0.06, "2-3": 0.06, "4-5": 0.05, "6-20": 0.05, ">20": 0.04},
                alpha_regularization_weight=0.0015,
            ),
        },
        {
            "run_name": "runC_short_split_conservative_2_3",
            "training": KnownUserDeepTrainingConfig(
                embedding_dim=128,
                event_hidden_dim=128,
                user_type_hidden_dim=128,
                scorer_hidden_dim=256,
                business_hidden_layers=(512, 384, 256),
                scorer_hidden_layers=(256, 128),
                num_attention_heads=4,
                dropout=0.15,
                batch_size=512,
                learning_rate=8e-4,
                weight_decay=2e-5,
                max_epochs=20,
                early_stopping_patience=4,
                auxiliary_loss_weight=0.15,
                band_correction_scales={"1": 0.7, "2-3": 0.82, "4-5": 0.95, "6-20": 1.0, ">20": 0.95},
                band_distillation_weights={"1": 0.06, "2-3": 0.08, "4-5": 0.05, "6-20": 0.05, ">20": 0.04},
            ),
        },
        {
            "run_name": "runD_short_split_aggressive_4_5",
            "training": KnownUserDeepTrainingConfig(
                embedding_dim=128,
                event_hidden_dim=128,
                user_type_hidden_dim=128,
                scorer_hidden_dim=256,
                business_hidden_layers=(512, 384, 256),
                scorer_hidden_layers=(256, 128),
                num_attention_heads=4,
                dropout=0.15,
                batch_size=512,
                learning_rate=8e-4,
                weight_decay=2e-5,
                max_epochs=20,
                early_stopping_patience=4,
                auxiliary_loss_weight=0.15,
                band_correction_scales={"1": 0.7, "2-3": 0.9, "4-5": 1.02, "6-20": 1.0, ">20": 0.95},
                band_distillation_weights={"1": 0.06, "2-3": 0.06, "4-5": 0.04, "6-20": 0.05, ">20": 0.04},
            ),
        },
    ]
    return configs[: max(1, min(max_runs, len(configs)))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mixed router with an end-to-end known-user deep branch and band fallback.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "known_user_deep_router_v4_eval_v1",
    )
    parser.add_argument(
        "--incumbent-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_raw_router_prefix_deep_v1",
    )
    parser.add_argument(
        "--business-repr-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v3_iter03" / "business_repr",
    )
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-runs", type=int, default=4)
    parser.add_argument("--known-enable-margin", type=float, default=0.002)
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument("--n-user-archetypes", type=int, default=64)
    parser.add_argument("--max-top-cities", type=int, default=100)
    parser.add_argument("--max-top-categories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config-family", type=str, default="v4", choices=["v4", "v3_feature_injected", "v5_direct_predictor", "v6_regularized", "v7_mae_loss", "v_lightweight", "v_ultralight"])
    parser.add_argument("--run-name", type=str, default=None, help="If set, only the run with this exact name is executed (skips all others).")
    return parser.parse_args()


def _predict_incumbent_router(
    *,
    spec: Any,
    known_booster: lgb.Booster,
    known_prefix_booster: lgb.Booster,
    cold_booster: lgb.Booster,
    target_reviews: pd.DataFrame,
    context_reviews: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
) -> pd.DataFrame:
    base_frame = build_raw_feature_frame(target_reviews, users_df, businesses_df, spec.base_spec)
    router_frame = build_router_feature_frame(target_reviews, users_df, businesses_df, spec)
    history_band_lookup = _history_band_lookup_from_context(context_reviews)
    base_frame["history_band"] = pd.Series(
        [history_band_lookup.get(str(user_id), "0") for user_id in base_frame["user"].astype(str)],
        index=base_frame.index,
        dtype="string",
    )
    router_frame["history_band"] = base_frame["history_band"]
    known_feature_columns = [
        column
        for column in spec.base_spec.feature_columns
        if column not in {"user_known_in_train", "business_known_in_train"}
    ]
    known_raw = np.clip(
        known_booster.predict(
            base_frame[known_feature_columns].copy(),
            num_iteration=known_booster.best_iteration or known_booster.current_iteration() or known_booster.num_trees(),
        ).astype(np.float32),
        1.0,
        5.0,
    )
    cold_raw = np.clip(
        cold_booster.predict(
            router_frame[spec.cold_feature_columns].copy(),
            num_iteration=cold_booster.best_iteration or cold_booster.current_iteration() or cold_booster.num_trees(),
        ).astype(np.float32),
        1.0,
        5.0,
    )
    known_prefix_raw = np.full(len(base_frame), np.nan, dtype=np.float32)
    if spec.enabled_known_prefix_bands:
        prefix_target = base_frame[
            (base_frame["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5)
            & (base_frame["history_band"].astype(str).isin(spec.enabled_known_prefix_bands))
        ].copy()
        prefix_bundle = load_known_prefix_embedding_bundle(spec.known_prefix_embedding_root)
        prefix_eval_raw = build_known_prefix_eval_frame(
            target_reviews[target_reviews["review_id"].astype(str).isin(prefix_target["review_id"].astype(str))].copy(),
            context_reviews,
            prefix_bundle,
            max_history_len=spec.known_prefix_max_history_len,
            target_history_bands=tuple(spec.enabled_known_prefix_bands),
        )
        prefix_eval = _merge_known_prefix_features(prefix_target, prefix_eval_raw)
        prefix_prediction = (
            np.clip(
                known_prefix_booster.predict(
                    prefix_eval[spec.known_prefix_feature_columns].copy(),
                    num_iteration=known_prefix_booster.best_iteration
                    or known_prefix_booster.current_iteration()
                    or known_prefix_booster.num_trees(),
                ).astype(np.float32),
                1.0,
                5.0,
            )
            if not prefix_eval.empty
            else np.empty(0, dtype=np.float32)
        )
        known_prefix_raw = _apply_prediction_to_frame(frame=base_frame, subset_frame=prefix_eval, prediction=prefix_prediction)

    router_branches = resolve_router_branches(
        user_known_mask=router_frame["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5,
        history_band=router_frame["history_band"],
        enabled_known_prefix_bands=tuple(spec.enabled_known_prefix_bands),
    )
    router_raw = np.where(
        router_branches == "cold_model",
        cold_raw,
        np.where(router_branches == "known_prefix_deep_model", known_prefix_raw, known_raw),
    ).astype(np.float32)
    missing_prefix_mask = (router_branches == "known_prefix_deep_model") & (~np.isfinite(known_prefix_raw))
    router_raw = np.where(missing_prefix_mask, known_raw, router_raw).astype(np.float32)
    router_branches = np.where(missing_prefix_mask, "known_model", router_branches)
    router_rounded = _round_half_up(router_raw).clip(1, 5).astype(np.int32)
    output = base_frame[["review_id", "user", "item", "rating", "history_band"]].copy()
    output["incumbent_prediction_raw"] = router_raw
    output["incumbent_prediction"] = router_rounded.astype(np.float32)
    output["incumbent_branch"] = router_branches
    return output


def _run_summary(
    *,
    run_name: str,
    training_config: KnownUserDeepTrainingConfig,
    incumbent_val: pd.DataFrame,
    deep_val: pd.DataFrame,
    training_result: Any,
    enable_margin: float,
) -> dict[str, Any]:
    eval_frame = _attach_deep_predictions(incumbent_val, deep_val)
    comparison_rows: list[dict[str, Any]] = []
    enabled_bands: list[str] = []
    for band in ["1", "2-5", "6-20", ">20"]:
        band_frame = eval_frame[eval_frame["history_band"].astype(str) == band]
        if band_frame.empty:
            continue
        available_band = band_frame[band_frame["deep_prediction_available"].to_numpy(dtype=bool)]
        if available_band.empty:
            comparison_rows.append(
                {
                    "history_band": band,
                    "full_band_rows": int(len(band_frame)),
                    "deep_available_rows": 0,
                    "replacement_candidate_rows": 0,
                    "replaced_rows_if_enabled": 0,
                    "comparison_scope": "same_band_and_deep_available_rows",
                    "incumbent_mae": None,
                    "deep_mae": None,
                    "delta_mae": None,
                    "enabled_for_router": False,
                }
            )
            continue
        incumbent_metrics = _prediction_metrics(available_band, "incumbent_prediction")
        deep_metrics = _prediction_metrics(available_band, "deep_prediction")
        delta = float(deep_metrics["mae"] - incumbent_metrics["mae"])
        replacement_candidate_rows = int(len(available_band))
        selective_policy: dict[str, Any] | None = None
        if band == "2-5":
            alpha_threshold = (training_config.selective_replace_alpha_thresholds or {}).get("2-5")
            abs_correction_threshold = (training_config.selective_replace_abs_correction_thresholds or {}).get("2-5")
            if alpha_threshold is not None or abs_correction_threshold is not None:
                candidate_mask = np.ones(len(available_band), dtype=bool)
                if alpha_threshold is not None and "alpha" in available_band.columns:
                    candidate_mask &= available_band["alpha"].to_numpy(dtype=np.float32) >= float(alpha_threshold)
                if abs_correction_threshold is not None:
                    candidate_mask &= np.abs(_correction_series(available_band)) >= float(abs_correction_threshold)
                replacement_candidate_rows = int(candidate_mask.sum())
                selective_policy = {
                    "alpha_threshold": float(alpha_threshold) if alpha_threshold is not None else None,
                    "abs_correction_threshold": float(abs_correction_threshold) if abs_correction_threshold is not None else None,
                    "selected_rows": replacement_candidate_rows,
                }
                if replacement_candidate_rows > 0:
                    candidate_subset = available_band.loc[candidate_mask]
                    incumbent_metrics = _prediction_metrics(candidate_subset, "incumbent_prediction")
                    deep_metrics = _prediction_metrics(candidate_subset, "deep_prediction")
                    delta = float(deep_metrics["mae"] - incumbent_metrics["mae"])
        band_margin = 0.0 if band == "1" else float(enable_margin)
        enabled = replacement_candidate_rows > 0 and delta <= -band_margin
        if enabled:
            enabled_bands.append(band)
        comparison_rows.append(
            {
                "history_band": band,
                "full_band_rows": int(len(band_frame)),
                "deep_available_rows": int(len(available_band)),
                "replacement_candidate_rows": replacement_candidate_rows,
                "replaced_rows_if_enabled": replacement_candidate_rows if enabled else 0,
                "comparison_scope": "same_band_and_deep_available_rows",
                "incumbent_mae": incumbent_metrics["mae"],
                "deep_mae": deep_metrics["mae"],
                "delta_mae": delta,
                "enabled_for_router": enabled,
                "selective_policy": selective_policy,
            }
        )

    final_val = eval_frame.copy()
    deep_available_mask = final_val["deep_prediction_available"].to_numpy(dtype=bool)
    replace_mask, replace_policy = _apply_replace_policy(
        final_val,
        enabled_bands=enabled_bands,
        training_config=training_config,
    )
    final_val["final_router_branch"] = final_val["incumbent_branch"].astype(str)
    final_val["final_prediction_raw"] = final_val["incumbent_prediction_raw"].astype(np.float32)
    final_val["final_prediction"] = final_val["incumbent_prediction"].astype(np.float32)
    if replace_mask.any():
        final_val.loc[replace_mask, "final_prediction"] = final_val.loc[replace_mask, "deep_prediction"].to_numpy(dtype=np.float32)
        final_val.loc[replace_mask, "final_prediction_raw"] = final_val.loc[replace_mask, "deep_prediction_raw"].to_numpy(dtype=np.float32)
        final_val.loc[replace_mask, "final_router_branch"] = "known_user_deep_e2e_model"

    incumbent_metrics = _prediction_metrics(incumbent_val, "incumbent_prediction")
    final_metrics = _prediction_metrics(final_val, "final_prediction")
    deep_eval_frame = eval_frame[eval_frame["deep_prediction_available"].to_numpy(dtype=bool)].copy()
    incumbent_known = incumbent_val[incumbent_val["history_band"].astype(str) != "0"]
    final_known = final_val[final_val["history_band"].astype(str) != "0"]
    incumbent_known_metrics = _prediction_metrics(incumbent_known, "incumbent_prediction")
    final_known_metrics = _prediction_metrics(final_known, "final_prediction")
    incumbent_band_map = {row["history_band"]: row for row in _band_metrics(incumbent_val, "incumbent_prediction")}
    final_band_map = {row["history_band"]: row for row in _band_metrics(final_val, "final_prediction")}
    deep_model_eval_band_metrics = _deep_model_band_eval(eval_frame)
    deep_model_eval_short_metrics = _deep_model_short_history_eval(eval_frame)
    if deep_eval_frame.empty:
        deep_model_eval = {
            "total_rows": int(len(incumbent_val)),
            "deep_available_rows": 0,
            "coverage_pct": 0.0,
            "incumbent_mae": None,
            "incumbent_rmse": None,
            "deep_mae": None,
            "deep_rmse": None,
            "delta_mae": None,
            "band_metrics": [],
            "short_history_metrics": [],
        }
    else:
        deep_incumbent_metrics = _prediction_metrics(deep_eval_frame, "incumbent_prediction")
        deep_metrics = _prediction_metrics(deep_eval_frame, "deep_prediction")
        deep_model_eval = {
            "total_rows": int(len(incumbent_val)),
            "deep_available_rows": int(len(deep_eval_frame)),
            "coverage_pct": float(len(deep_eval_frame) / max(len(incumbent_val), 1)),
            "incumbent_mae": deep_incumbent_metrics["mae"],
            "incumbent_rmse": deep_incumbent_metrics["rmse"],
            "deep_mae": deep_metrics["mae"],
            "deep_rmse": deep_metrics["rmse"],
            "delta_mae": float(deep_metrics["mae"] - deep_incumbent_metrics["mae"]),
            "band_metrics": deep_model_eval_band_metrics,
            "short_history_metrics": deep_model_eval_short_metrics,
        }
    short_eval_subset = eval_frame[
        (eval_frame["history_band"].astype(str) == "2-5")
        & eval_frame["deep_prediction_available"].to_numpy(dtype=bool)
    ].copy()
    router_replacement_eval = {
        "total_rows": int(len(incumbent_val)),
        "deep_available_rows": int(deep_available_mask.sum()),
        "replaced_rows": int(replace_mask.sum()),
        "enabled_bands": enabled_bands.copy(),
        "replace_policy": replace_policy,
        "band_comparison": comparison_rows,
        "coverage_by_band": _coverage_by_band(eval_frame, branch_col="incumbent_branch"),
        "alpha_correction_stats_by_band": _alpha_correction_stats_by_band(eval_frame),
        "short_history_diagnostics": {
            "alpha_threshold_eval": _threshold_subset_metrics(
                short_eval_subset,
                value_col="alpha",
                thresholds=[0.35, 0.5, 0.55, 0.65, 0.75],
            ),
            "abs_correction_threshold_eval": _threshold_subset_metrics(
                short_eval_subset.assign(abs_correction=np.abs(_correction_series(short_eval_subset))),
                value_col="abs_correction",
                thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            "history_count_bins": _history_count_bin_diagnostics(short_eval_subset),
        },
        "final_band_metrics": list(final_band_map.values()),
        "final_short_history_metrics": _short_history_metrics(final_val[final_val["history_band"].astype(str) == "2-5"], "final_prediction"),
        "final_branch_rows": _branch_count_payload(final_val, "final_router_branch"),
    }
    success = (
        final_known_metrics["mae"] < incumbent_known_metrics["mae"]
        and final_band_map.get("2-5", {}).get("mae", float("inf")) < incumbent_band_map.get("2-5", {}).get("mae", float("inf"))
        and (final_metrics["mae"] - incumbent_metrics["mae"]) <= 0.003
    )
    return {
        "run_name": run_name,
        "training_config": {
            "embedding_dim": training_config.embedding_dim,
            "event_hidden_dim": training_config.event_hidden_dim,
            "user_type_hidden_dim": training_config.user_type_hidden_dim,
            "scorer_hidden_dim": training_config.scorer_hidden_dim,
            "business_hidden_layers": list(training_config.business_hidden_layers),
            "scorer_hidden_layers": list(training_config.scorer_hidden_layers),
            "num_attention_heads": training_config.num_attention_heads,
            "dropout": training_config.dropout,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "auxiliary_loss_weight": training_config.auxiliary_loss_weight,
            "band_sample_weights": training_config.band_sample_weights or {},
            "band_correction_scales": training_config.band_correction_scales or {},
            "band_distillation_weights": training_config.band_distillation_weights or {},
            "alpha_regularization_weight": training_config.alpha_regularization_weight,
            "recency_weight_scale": training_config.recency_weight_scale,
            "selective_replace_alpha_thresholds": training_config.selective_replace_alpha_thresholds or {},
            "selective_replace_abs_correction_thresholds": training_config.selective_replace_abs_correction_thresholds or {},
        },
        "training_result": {
            "best_epoch": training_result.best_epoch,
            "best_val_mae": training_result.best_val_mae,
            "best_val_rmse": training_result.best_val_rmse,
            "train_size": training_result.train_size,
            "val_size": training_result.val_size,
        },
        "enabled_bands": enabled_bands,
        "band_comparison": comparison_rows,
        "deep_model_eval": deep_model_eval,
        "router_replacement_eval": router_replacement_eval,
        "short_history_metrics": {
            "incumbent": _short_history_metrics(eval_frame[eval_frame["history_band"].astype(str) == "2-5"], "incumbent_prediction"),
            "deep": deep_model_eval_short_metrics,
            "final": _short_history_metrics(final_val[final_val["history_band"].astype(str) == "2-5"], "final_prediction"),
        },
        "incumbent_overall_mae": incumbent_metrics["mae"],
        "final_overall_mae": final_metrics["mae"],
        "incumbent_known_mae": incumbent_known_metrics["mae"],
        "final_known_mae": final_known_metrics["mae"],
        "overall_delta": float(final_metrics["mae"] - incumbent_metrics["mae"]),
        "known_delta": float(final_known_metrics["mae"] - incumbent_known_metrics["mae"]),
        "success": bool(success),
        "final_band_metrics": list(final_band_map.values()),
        "validation_frame": final_val,
        "deep_validation_frame": deep_eval_frame,
    }


def _public_run_summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": record["run_name"],
        "training_config": record["training_config"],
        "training_result": record["training_result"],
        "enabled_bands": record["enabled_bands"],
        "band_comparison": record["band_comparison"],
        "deep_model_eval": record["deep_model_eval"],
        "router_replacement_eval": record["router_replacement_eval"],
        "short_history_metrics": record["short_history_metrics"],
        "incumbent_overall_mae": record["incumbent_overall_mae"],
        "final_overall_mae": record["final_overall_mae"],
        "incumbent_known_mae": record["incumbent_known_mae"],
        "final_known_mae": record["final_known_mae"],
        "overall_delta": record["overall_delta"],
        "known_delta": record["known_delta"],
        "success": record["success"],
        "final_band_metrics": record["final_band_metrics"],
    }


def _short_history_combined_delta(record: dict[str, Any]) -> float:
    metrics = {row["history_count_segment"]: row for row in record.get("short_history_metrics", {}).get("deep", [])}
    return float(metrics.get("2-3", {}).get("delta_mae", 0.0) + metrics.get("4-5", {}).get("delta_mae", 0.0))


def _run_sort_key(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(record["final_overall_mae"]),
        float(record["final_known_mae"]),
        _short_history_combined_delta(record),
    )


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    run_root = save_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    train_split, val_split = temporal_train_validation_split(train_reviews, val_size=args.validation_size, timestamp_col="date")

    # NOTE: user_average_stars is intentionally NOT overridden here with train-only values.
    # The incumbent LGBM (lgbm_raw_router_prefix_deep_v1) was trained WITH leaky Yelp
    # metadata stars.  Passing honest stars would feed out-of-distribution inputs to the
    # incumbent booster, corrupting its predictions and degrading the deep model.
    # The leakage fix must be applied upstream: retrain the LGBM incumbent with honest
    # stars first, then use that honest incumbent here.

    print(f"Training known-user deep router with train size: {len(train_split)}, val size: {len(val_split)}")

    incumbent_stack = _load_incumbent_stack(args.incumbent_root)
    print(f"Loaded incumbent stack with spec: {incumbent_stack['spec']}")
    incumbent_train = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=train_split,
        context_reviews=train_split,
        users_df=users_df,
        businesses_df=businesses_df,
    )
    incumbent_val = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=val_split,
        context_reviews=train_split,
        users_df=users_df,
        businesses_df=businesses_df,
    )

    deep_data_config = KnownUserDeepDataConfig(
        business_repr_root=str(args.business_repr_root),
        max_history_len=int(args.max_history_len),
        n_user_archetypes=int(args.n_user_archetypes),
        max_top_cities=int(args.max_top_cities),
        max_top_categories=int(args.max_top_categories),
        random_seed=int(args.seed),
    )

    print(f"Preparing known-user deep context with config: {asdict(deep_data_config)}")
    deep_context = prepare_known_user_context(
        context_reviews=train_split,
        users_df=users_df,
        businesses_df=businesses_df,
        data_config=deep_data_config,
    )

    run_summaries: list[dict[str, Any]] = []
    learning_curve_frames: list[pd.DataFrame] = []
    best_record: dict[str, Any] | None = None

    for run_spec in _load_run_configs(args.max_runs, config_family=args.config_family):
        if args.run_name and run_spec["run_name"] != args.run_name:
            continue
        run_name = run_spec["run_name"]
        training_config = run_spec["training"]
        print(f"Starting run {run_name} with training config: {training_config}")
        run_dir = run_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        train_data = build_known_user_train_dataset(
            train_split,
            users_df=users_df,
            businesses_df=businesses_df,
            context=deep_context,
            training_config=training_config,
            incumbent_frame=incumbent_train,
        )
        val_data = build_known_user_eval_dataset(
            val_split,
            train_split,
            users_df=users_df,
            businesses_df=businesses_df,
            context=deep_context,
            training_config=training_config,
            incumbent_frame=incumbent_val,
        )
        training_result = train_known_user_deep_model(
            train_data=train_data,
            val_data=val_data,
            context=deep_context,
            training_config=training_config,
        )
        final_state = fit_known_user_deep_final_model(
            train_data=train_data,
            context=deep_context,
            training_config=training_config,
            architecture=training_result.architecture,
            final_epochs=min(training_result.best_epoch + 1, 5),
        )
        model = KnownUserDeepE2EModel(training_result.architecture)
        model.load_state_dict(final_state)

        deep_val = predict_known_user_dataset(
            model=model,
            prepared=val_data,
            context=deep_context,
            batch_size=int(training_config.batch_size),
            device="cpu",
        ).rename(columns={"predicted_rating": "deep_prediction"})
        deep_val["deep_prediction_raw"] = deep_val["deep_prediction"].astype(np.float32)
        deep_val["deep_prediction"] = _round_half_up(deep_val["deep_prediction_raw"].to_numpy(dtype=np.float32)).clip(1, 5).astype(np.float32)

        summary = _run_summary(
            run_name=run_name,
            training_config=training_config,
            incumbent_val=incumbent_val,
            deep_val=deep_val,
            training_result=training_result,
            enable_margin=float(args.known_enable_margin),
        )

        save_known_user_checkpoint(
            path=run_dir / "known_user_deep_checkpoint.pt",
            model_state_dict=final_state,
            architecture=training_result.architecture,
            feature_contract=deep_context.feature_contract,
            data_config=deep_data_config,
            training_config=training_config,
            extra_summary=summary,
        )
        _save_json(
            run_dir / "known_user_deep_config.json",
            {
                "data_config": asdict(deep_data_config),
                "training_config": asdict(training_config),
                "feature_contract": asdict(deep_context.feature_contract),
            },
        )
        _save_json(
            run_dir / "known_user_deep_training_summary.json",
            {
                "run_name": run_name,
                "training_result": summary["training_result"],
                "enabled_bands": summary["enabled_bands"],
                "deep_model_eval": summary["deep_model_eval"],
                "router_replacement_eval": summary["router_replacement_eval"],
            },
        )
        _save_json(run_dir / "validation_summary.json", _public_run_summary(summary))
        deep_val.to_csv(run_dir / "known_user_deep_validation_predictions.csv", index=False)
        summary["learning_curves"] = training_result.learning_curves.copy()
        summary["model_state_dict"] = final_state
        summary["architecture"] = training_result.architecture
        summary["training_config_obj"] = training_config
        run_summaries.append(summary)
        learning_curve_frames.append(training_result.learning_curves.assign(run_name=run_name))

        if summary["success"] and best_record is None:
            best_record = summary

    if best_record is None:
        best_record = min(run_summaries, key=_run_sort_key)
    else:
        candidate_pool = [record for record in run_summaries if record["final_overall_mae"] <= best_record["final_overall_mae"] + 1e-9] or run_summaries
        best_record = min(candidate_pool, key=_run_sort_key)

    enabled_bands = best_record["enabled_bands"]
    selected_training_config: KnownUserDeepTrainingConfig = best_record["training_config_obj"]

    full_context = prepare_known_user_context(
        context_reviews=train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
        data_config=deep_data_config,
    )
    incumbent_full_train = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=train_reviews,
        context_reviews=train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
    )
    full_train_data = build_known_user_train_dataset(
        train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
        context=full_context,
        training_config=selected_training_config,
        incumbent_frame=incumbent_full_train,
    )
    full_final_state = fit_known_user_deep_final_model(
        train_data=full_train_data,
        context=full_context,
        training_config=selected_training_config,
        architecture=best_record["architecture"],
        final_epochs=min(int(best_record["training_result"]["best_epoch"]) + 1, 5),
    )
    selected_model = KnownUserDeepE2EModel(best_record["architecture"])
    selected_model.load_state_dict(full_final_state)

    router_spec = {
        "incumbent_router_root": str(args.incumbent_root),
        "cold_model_source": str(args.incumbent_root / "cold_submission_model.txt"),
        "known_model_source": str(args.incumbent_root / "known_submission_model.txt"),
        "known_prefix_model_source": str(args.incumbent_root / "known_prefix_submission_model.txt"),
        "known_user_deep_checkpoint_path": str(save_root / "known_user_deep_checkpoint.pt"),
        "known_user_deep_config_path": str(save_root / "known_user_deep_config.json"),
        "known_user_deep_feature_contract": asdict(full_context.feature_contract),
        "known_user_deep_training_config": asdict(selected_training_config),
        "enabled_known_deep_bands": enabled_bands,
        "selective_replace_alpha_thresholds": selected_training_config.selective_replace_alpha_thresholds or {},
        "selective_replace_abs_correction_thresholds": selected_training_config.selective_replace_abs_correction_thresholds or {},
        "fallback_band_policy": {"0": "cold_model", "1": "known_model", "2-5": "known_model", "6-20": "known_prefix_deep_model", ">20": "known_model"},
        "max_history_len": int(args.max_history_len),
        "history_summary_tokens": 4,
        "batch_size": int(selected_training_config.batch_size),
        "band_margin": float(args.known_enable_margin),
        "best_run_name": best_record["run_name"],
        "raw_spec": full_context.raw_spec,
        "router_feature_spec": full_context.router_spec,
        "business_repr_root": str(args.business_repr_root),
    }
    joblib.dump(router_spec, save_root / "router_spec.joblib")

    test_reviews = load_test_reviews(args.data_dir)
    incumbent_test = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=test_reviews,
        context_reviews=train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
    )
    test_data = build_known_user_eval_dataset(
        test_reviews,
        train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
        context=full_context,
        training_config=selected_training_config,
        incumbent_frame=incumbent_test,
    )
    deep_test = predict_known_user_dataset(
        model=selected_model,
        prepared=test_data,
        context=full_context,
        batch_size=int(selected_training_config.batch_size),
        device="cpu",
    ).rename(columns={"predicted_rating": "deep_prediction"})
    deep_test["deep_prediction_raw"] = deep_test["deep_prediction"].astype(np.float32)
    deep_test["deep_prediction"] = _round_half_up(deep_test["deep_prediction_raw"].to_numpy(dtype=np.float32)).clip(1, 5).astype(np.float32)
    deep_lookup_columns = [column for column in ["deep_prediction", "deep_prediction_raw", "alpha", "correction_hat", "residual_hat"] if column in deep_test.columns]
    deep_lookup = deep_test.set_index(deep_test["review_id"].astype(str))[deep_lookup_columns]

    final_test = incumbent_test.copy()
    final_test = _attach_deep_predictions(final_test, deep_test)
    final_test["final_prediction_raw"] = final_test["incumbent_prediction_raw"].astype(np.float32)
    final_test["final_prediction"] = final_test["incumbent_prediction"].astype(np.float32)
    final_test["final_router_branch"] = final_test["incumbent_branch"].astype(str)
    replace_mask, _ = _apply_replace_policy(
        final_test,
        enabled_bands=enabled_bands,
        training_config=selected_training_config,
    )
    if replace_mask.any():
        mapped = final_test.loc[replace_mask, "review_id"].astype(str).map(deep_lookup["deep_prediction"].to_dict()).to_numpy(dtype=np.float32)
        mapped_raw = final_test.loc[replace_mask, "review_id"].astype(str).map(deep_lookup["deep_prediction_raw"].to_dict()).to_numpy(dtype=np.float32)
        final_test.loc[replace_mask, "final_prediction"] = mapped
        final_test.loc[replace_mask, "final_prediction_raw"] = mapped_raw
        final_test.loc[replace_mask, "final_router_branch"] = "known_user_deep_e2e_model"

    submission = pd.DataFrame(
        {
            "review_id": final_test["review_id"].astype(str),
            "stars": _round_half_up(final_test["final_prediction_raw"].to_numpy(dtype=np.float32)).clip(1, 5).astype(np.int32),
        }
    )
    submission_path = save_root / "submission.csv"
    submission.to_csv(submission_path, index=False)

    final_summary = _public_run_summary(best_record)
    final_summary["best_run_name"] = best_record["run_name"]
    final_summary["enabled_bands"] = enabled_bands
    final_summary["router_spec"] = router_spec
    final_summary["submission_path"] = str(submission_path)
    final_summary["submission_summary"] = {
        "artifact_root": str(save_root),
        "submission_path": str(submission_path),
        "n_rows": int(len(submission)),
        "history_band_rows": {str(key): int(value) for key, value in final_test["history_band"].astype(str).value_counts(dropna=False).to_dict().items()},
        "known_branch_rows": int((final_test["final_router_branch"] == "known_model").sum()),
        "known_prefix_branch_rows": int((final_test["final_router_branch"] == "known_prefix_deep_model").sum()),
        "known_user_deep_branch_rows": int((final_test["final_router_branch"] == "known_user_deep_e2e_model").sum()),
        "cold_branch_rows": int((final_test["final_router_branch"] == "cold_model").sum()),
        "final_branch_rows": _branch_count_payload(final_test, "final_router_branch"),
        "enabled_known_deep_bands": enabled_bands,
        "selective_replace_alpha_thresholds": selected_training_config.selective_replace_alpha_thresholds or {},
        "selective_replace_abs_correction_thresholds": selected_training_config.selective_replace_abs_correction_thresholds or {},
        "prediction_min": int(submission["stars"].min()) if len(submission) else None,
        "prediction_max": int(submission["stars"].max()) if len(submission) else None,
        "prediction_mean": float(submission["stars"].mean()) if len(submission) else None,
    }

    _save_json(save_root / "validation_summary.json", _public_run_summary(best_record))
    _save_json(save_root / "submission_summary.json", final_summary["submission_summary"])
    _save_json(save_root / "known_user_deep_config.json", {
        "data_config": asdict(deep_data_config),
        "training_config": asdict(selected_training_config),
        "feature_contract": asdict(full_context.feature_contract),
        "best_run_name": best_record["run_name"],
    })
    _save_json(save_root / "known_user_deep_training_summary.json", {
        "best_run_name": best_record["run_name"],
        "runs": [
            {
                "run_name": record["run_name"],
                "success": record["success"],
                "final_overall_mae": record["final_overall_mae"],
                "final_known_mae": record["final_known_mae"],
                "enabled_bands": record["enabled_bands"],
                "deep_available_rows": record["deep_model_eval"]["deep_available_rows"],
                "replaced_rows": record["router_replacement_eval"]["replaced_rows"],
            }
            for record in run_summaries
        ],
        "best_run": {
            "run_name": best_record["run_name"],
            "enabled_bands": enabled_bands,
            "training_result": best_record["training_result"],
            "deep_model_eval": best_record["deep_model_eval"],
            "router_replacement_eval": best_record["router_replacement_eval"],
        },
    })
    _save_json(
        save_root / "enabled_bands.json",
        {
            "enabled_known_deep_bands": enabled_bands,
            "band_comparison_target": best_record["band_comparison"],
            "comparison_scope": "same_band_and_deep_available_rows",
            "deep_model_eval": best_record["deep_model_eval"],
            "router_replacement_eval": best_record["router_replacement_eval"],
        },
    )
    save_known_user_checkpoint(
        path=save_root / "known_user_deep_checkpoint.pt",
        model_state_dict=full_final_state,
        architecture=best_record["architecture"],
        feature_contract=full_context.feature_contract,
        data_config=deep_data_config,
        training_config=selected_training_config,
        extra_summary=_public_run_summary(best_record),
    )
    shutil.copyfile(save_root / "runs" / best_record["run_name"] / "known_user_deep_validation_predictions.csv", save_root / "known_user_deep_validation_predictions.csv")

    if learning_curve_frames:
        pd.concat(learning_curve_frames, ignore_index=True).to_csv(save_root / "learning_curves.csv", index=False)
    else:
        pd.DataFrame([{"run_name": best_record["run_name"], "epoch": 0, "metric": "unknown", "value": np.nan}]).to_csv(save_root / "learning_curves.csv", index=False)

    print(
        json.dumps(
            {
                "validation_summary": _public_run_summary(best_record),
                "submission_summary": final_summary["submission_summary"],
                "best_run": best_record["run_name"],
                "enabled_bands": enabled_bands,
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
