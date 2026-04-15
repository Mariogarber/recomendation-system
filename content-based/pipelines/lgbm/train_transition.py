from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import joblib
import numpy as np
import pandas as pd

from train_lgbm_raw_router import (
    _apply_prediction_to_frame,
    _build_history_band,
    _build_params,
    _cold_sample_weight,
    _compute_band_metrics,
    _compute_served_branch_metrics,
    _extract_feature_matrix,
    _feature_summary,
    _merge_known_prefix_features,
    _save_feature_importance,
    _save_json,
    _train_booster,
    _round_half_up,
)
from utils.io import get_default_data_dir, load_businesses, load_train_reviews, load_users
from utils.lgbm_known_prefix_deep_features import (
    DEFAULT_KNOWN_PREFIX_TARGET_BANDS,
    build_known_prefix_eval_frame,
    build_known_prefix_train_frame,
    load_known_prefix_embedding_bundle,
    parse_known_prefix_target_bands,
)
from utils.lgbm_raw_features import build_raw_feature_frame
from utils.lgbm_raw_router_features import build_router_feature_frame, fit_router_feature_spec
from utils.split import cold_start_breakdown, temporal_train_validation_split


def _build_history_count(frame: pd.DataFrame, user_counts: pd.Series) -> pd.Series:
    lookup = user_counts.to_dict()
    return pd.Series([int(lookup.get(user_id, 0)) for user_id in frame["user"].astype(str)], index=frame.index, dtype=np.int32)


def _short_weight_from_count(count: int) -> float:
    return float(np.clip((6.0 - float(count)) / 4.0, 0.25, 0.85))


def _compute_transition_count_diagnostics(
    *,
    frame: pd.DataFrame,
    known_pred: np.ndarray,
    short_pred: np.ndarray,
    medium_pred: np.ndarray,
    blend_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    known_mask = frame["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5
    history_band = frame["history_band"].astype(str).to_numpy()
    history_count = frame["history_count"].to_numpy(dtype=np.int32)
    rating = frame["rating"].to_numpy(dtype=np.float32)
    target_mask = known_mask & (history_band == "2-5")
    for count in (2, 3, 4, 5):
        mask = target_mask & (history_count == count)
        if not mask.any():
            continue
        y_true = rating[mask]
        rows.append(
            {
                "history_count": count,
                "n_samples": int(mask.sum()),
                "known_model_mae": float(np.mean(np.abs(y_true - _round_half_up(known_pred[mask]).astype(np.float32)))),
                "transition_short_mae": float(np.mean(np.abs(y_true - _round_half_up(short_pred[mask]).astype(np.float32)))),
                "transition_medium_mae": float(np.mean(np.abs(y_true - _round_half_up(medium_pred[mask]).astype(np.float32)))),
                "transition_blend_mae": float(np.mean(np.abs(y_true - _round_half_up(blend_pred[mask]).astype(np.float32)))),
                "average_short_weight": float(np.mean(np.vectorize(_short_weight_from_count)(history_count[mask]))),
            }
        )
    return rows


def _resolve_branches(
    *,
    user_known_mask: np.ndarray,
    history_band: pd.Series,
    enabled_known_prefix_bands: list[str],
    enable_transition_blend_band: bool = True,
) -> np.ndarray:
    bands = history_band.astype("string").to_numpy(dtype=object)
    out = np.full(len(bands), "known_model", dtype=object)
    out[~user_known_mask] = "cold_model"
    if enable_transition_blend_band:
        out[user_known_mask & (bands == "2-5")] = "transition_blend_model"
    prefix_mask = user_known_mask & np.isin(bands, np.asarray(enabled_known_prefix_bands, dtype=object))
    out[prefix_mask] = "known_prefix_deep_model"
    return out


def _read_baseline_summary(current_save_root: Path) -> dict[str, Any] | None:
    baseline_path = current_save_root.parent / "lgbm_hybrid_conservative_v1" / "validation_summary.json"
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hybrid router with a transition blend expert for the 2-5 band.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--save-root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_transition_blend_router_v1")
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--n-user-archetypes", type=int, default=64)
    parser.add_argument("--max-top-cities", type=int, default=100)
    parser.add_argument("--max-top-categories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--known-num-leaves", type=int, default=255)
    parser.add_argument("--known-learning-rate", type=float, default=0.03)
    parser.add_argument("--known-n-estimators", type=int, default=1500)
    parser.add_argument("--known-min-child-samples", type=int, default=75)
    parser.add_argument("--known-subsample", type=float, default=0.8)
    parser.add_argument("--known-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--known-reg-alpha", type=float, default=0.0)
    parser.add_argument("--known-reg-lambda", type=float, default=1.0)
    parser.add_argument("--known-early-stopping-rounds", type=int, default=80)
    parser.add_argument("--cold-num-leaves", type=int, default=127)
    parser.add_argument("--cold-learning-rate", type=float, default=0.03)
    parser.add_argument("--cold-n-estimators", type=int, default=2000)
    parser.add_argument("--cold-min-child-samples", type=int, default=150)
    parser.add_argument("--cold-subsample", type=float, default=0.8)
    parser.add_argument("--cold-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--cold-reg-alpha", type=float, default=0.25)
    parser.add_argument("--cold-reg-lambda", type=float, default=2.0)
    parser.add_argument("--cold-early-stopping-rounds", type=int, default=100)
    parser.add_argument("--transition-short-num-leaves", type=int, default=191)
    parser.add_argument("--transition-short-learning-rate", type=float, default=0.03)
    parser.add_argument("--transition-short-n-estimators", type=int, default=1200)
    parser.add_argument("--transition-short-min-child-samples", type=int, default=90)
    parser.add_argument("--transition-short-subsample", type=float, default=0.8)
    parser.add_argument("--transition-short-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--transition-short-reg-alpha", type=float, default=0.05)
    parser.add_argument("--transition-short-reg-lambda", type=float, default=1.25)
    parser.add_argument("--transition-short-early-stopping-rounds", type=int, default=80)
    parser.add_argument("--transition-medium-num-leaves", type=int, default=191)
    parser.add_argument("--transition-medium-learning-rate", type=float, default=0.03)
    parser.add_argument("--transition-medium-n-estimators", type=int, default=1200)
    parser.add_argument("--transition-medium-min-child-samples", type=int, default=90)
    parser.add_argument("--transition-medium-subsample", type=float, default=0.8)
    parser.add_argument("--transition-medium-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--transition-medium-reg-alpha", type=float, default=0.05)
    parser.add_argument("--transition-medium-reg-lambda", type=float, default=1.25)
    parser.add_argument("--transition-medium-early-stopping-rounds", type=int, default=80)
    parser.add_argument("--known-prefix-embedding-root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v3_iter03")
    parser.add_argument("--known-prefix-max-history-len", type=int, default=20)
    parser.add_argument("--known-prefix-target-bands", type=str, default=",".join(DEFAULT_KNOWN_PREFIX_TARGET_BANDS))
    parser.add_argument("--known-prefix-num-leaves", type=int, default=127)
    parser.add_argument("--known-prefix-learning-rate", type=float, default=0.03)
    parser.add_argument("--known-prefix-n-estimators", type=int, default=1200)
    parser.add_argument("--known-prefix-min-child-samples", type=int, default=100)
    parser.add_argument("--known-prefix-subsample", type=float, default=0.8)
    parser.add_argument("--known-prefix-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--known-prefix-reg-alpha", type=float, default=0.1)
    parser.add_argument("--known-prefix-reg-lambda", type=float, default=1.5)
    parser.add_argument("--known-prefix-early-stopping-rounds", type=int, default=80)
    parser.add_argument("--known-prefix-enable-margin", type=float, default=0.005)
    parser.add_argument("--disable-transition-blend-routing", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--gpu-platform-id", type=int, default=0)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--gpu-max-bin", type=int, default=255)
    parser.add_argument("--gpu-use-dp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    known_prefix_target_bands = parse_known_prefix_target_bands(args.known_prefix_target_bands)

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    train_split, val_split = temporal_train_validation_split(train_reviews, val_size=args.validation_size, timestamp_col="date")
    spec = fit_router_feature_spec(train_split, users_df, businesses_df, n_user_archetypes=args.n_user_archetypes, max_top_cities=args.max_top_cities, max_top_categories=args.max_top_categories, random_seed=args.seed)

    base_train = build_raw_feature_frame(train_split, users_df, businesses_df, spec.base_spec)
    base_val = build_raw_feature_frame(val_split, users_df, businesses_df, spec.base_spec)
    router_train = build_router_feature_frame(train_split, users_df, businesses_df, spec)
    router_val = build_router_feature_frame(val_split, users_df, businesses_df, spec)
    train_counts = train_split.groupby("user_id").size()
    val_counts = spec.base_spec.user_priors_table.set_index("user_id")["user_train_count"]
    for frame, counts in ((base_train, train_counts), (base_val, val_counts), (router_train, train_counts), (router_val, val_counts)):
        frame["history_band"] = _build_history_band(frame, counts)
        frame["history_count"] = _build_history_count(frame, counts)
    router_train["router_history_band"] = router_train["history_band"]

    known_features = [c for c in spec.base_spec.feature_columns if c not in {"user_known_in_train", "business_known_in_train"}]
    cold_features = spec.cold_feature_columns
    known_categoricals = [c for c in spec.base_spec.categorical_columns if c in known_features]
    cold_categoricals = [c for c in spec.categorical_columns if c in cold_features]

    def params(prefix: str) -> dict[str, Any]:
        return _build_params(
            num_leaves=getattr(args, f"{prefix}_num_leaves"),
            learning_rate=getattr(args, f"{prefix}_learning_rate"),
            min_child_samples=getattr(args, f"{prefix}_min_child_samples"),
            subsample=getattr(args, f"{prefix}_subsample"),
            colsample_bytree=getattr(args, f"{prefix}_colsample_bytree"),
            reg_alpha=getattr(args, f"{prefix}_reg_alpha"),
            reg_lambda=getattr(args, f"{prefix}_reg_lambda"),
            seed=args.seed,
            use_gpu=bool(args.use_gpu),
            gpu_platform_id=args.gpu_platform_id,
            gpu_device_id=args.gpu_device_id,
            gpu_max_bin=args.gpu_max_bin,
            gpu_use_dp=bool(args.gpu_use_dp),
        )

    known_booster = _train_booster(x_train=_extract_feature_matrix(base_train, known_features), y_train=base_train["rating"].to_numpy(dtype=np.float32), x_val=_extract_feature_matrix(base_val[base_val["user_known_in_train"] > 0.5], known_features), y_val=base_val.loc[base_val["user_known_in_train"] > 0.5, "rating"].to_numpy(dtype=np.float32), categorical_columns=known_categoricals, params=params("known"), num_boost_round=args.known_n_estimators, early_stopping_rounds=args.known_early_stopping_rounds)
    cold_booster = _train_booster(x_train=_extract_feature_matrix(router_train, cold_features), y_train=router_train["rating"].to_numpy(dtype=np.float32), x_val=_extract_feature_matrix(router_val[router_val["user_known_in_train"] < 0.5], cold_features), y_val=router_val.loc[router_val["user_known_in_train"] < 0.5, "rating"].to_numpy(dtype=np.float32), categorical_columns=cold_categoricals, params=params("cold"), num_boost_round=args.cold_n_estimators, early_stopping_rounds=args.cold_early_stopping_rounds, train_weight=router_train["router_history_band"].map(_cold_sample_weight).to_numpy(dtype=np.float32))
    short_train = base_train[(base_train["user_known_in_train"] > 0.5) & base_train["history_count"].between(1, 5, inclusive="both")]
    short_val = base_val[(base_val["user_known_in_train"] > 0.5) & base_val["history_count"].between(2, 5, inclusive="both")]
    medium_train = base_train[(base_train["user_known_in_train"] > 0.5) & base_train["history_count"].between(5, 20, inclusive="both")]
    medium_val = base_val[(base_val["user_known_in_train"] > 0.5) & base_val["history_count"].between(2, 5, inclusive="both")]
    transition_short_booster = _train_booster(x_train=_extract_feature_matrix(short_train, known_features), y_train=short_train["rating"].to_numpy(dtype=np.float32), x_val=_extract_feature_matrix(short_val, known_features), y_val=short_val["rating"].to_numpy(dtype=np.float32), categorical_columns=known_categoricals, params=params("transition_short"), num_boost_round=args.transition_short_n_estimators, early_stopping_rounds=args.transition_short_early_stopping_rounds)
    transition_medium_booster = _train_booster(x_train=_extract_feature_matrix(medium_train, known_features), y_train=medium_train["rating"].to_numpy(dtype=np.float32), x_val=_extract_feature_matrix(medium_val, known_features), y_val=medium_val["rating"].to_numpy(dtype=np.float32), categorical_columns=known_categoricals, params=params("transition_medium"), num_boost_round=args.transition_medium_n_estimators, early_stopping_rounds=args.transition_medium_early_stopping_rounds)

    prefix_bundle = load_known_prefix_embedding_bundle(args.known_prefix_embedding_root)
    kp_train_raw = build_known_prefix_train_frame(train_split, prefix_bundle, max_history_len=args.known_prefix_max_history_len, target_history_bands=known_prefix_target_bands)
    kp_val_raw = build_known_prefix_eval_frame(val_split, train_split, prefix_bundle, max_history_len=args.known_prefix_max_history_len, target_history_bands=known_prefix_target_bands)
    kp_train = _merge_known_prefix_features(base_train[base_train["review_id"].astype(str).isin(kp_train_raw["review_id"].astype(str))].copy(), kp_train_raw)
    kp_val = _merge_known_prefix_features(base_val[base_val["review_id"].astype(str).isin(kp_val_raw["review_id"].astype(str))].copy(), kp_val_raw)
    kp_cols = [c for c in kp_train.columns if c.startswith("known_prefix_") and c != "known_prefix_history_band"]
    kp_features = [*known_features, *kp_cols]
    kp_booster = _train_booster(x_train=_extract_feature_matrix(kp_train, kp_features), y_train=kp_train["rating"].to_numpy(dtype=np.float32), x_val=_extract_feature_matrix(kp_val, kp_features), y_val=kp_val["rating"].to_numpy(dtype=np.float32), categorical_columns=known_categoricals, params=params("known_prefix"), num_boost_round=args.known_prefix_n_estimators, early_stopping_rounds=args.known_prefix_early_stopping_rounds)

    known_best = int(known_booster.best_iteration or args.known_n_estimators)
    cold_best = int(cold_booster.best_iteration or args.cold_n_estimators)
    short_best = int(transition_short_booster.best_iteration or args.transition_short_n_estimators)
    medium_best = int(transition_medium_booster.best_iteration or args.transition_medium_n_estimators)
    kp_best = int(kp_booster.best_iteration or args.known_prefix_n_estimators)

    val_known = np.clip(known_booster.predict(_extract_feature_matrix(base_val, known_features), num_iteration=known_best).astype(np.float32), 1.0, 5.0)
    val_cold = np.clip(cold_booster.predict(_extract_feature_matrix(router_val, cold_features), num_iteration=cold_best).astype(np.float32), 1.0, 5.0)
    val_short = np.clip(transition_short_booster.predict(_extract_feature_matrix(base_val, known_features), num_iteration=short_best).astype(np.float32), 1.0, 5.0)
    val_medium = np.clip(transition_medium_booster.predict(_extract_feature_matrix(base_val, known_features), num_iteration=medium_best).astype(np.float32), 1.0, 5.0)
    weights = np.vectorize(_short_weight_from_count)(base_val["history_count"].to_numpy(dtype=np.int32)).astype(np.float32)
    val_transition = (weights * val_short + (1.0 - weights) * val_medium).astype(np.float32)
    val_kp_subset = np.clip(kp_booster.predict(_extract_feature_matrix(kp_val, kp_features), num_iteration=kp_best).astype(np.float32), 1.0, 5.0) if not kp_val.empty else np.empty(0, dtype=np.float32)
    val_kp = _apply_prediction_to_frame(frame=base_val, subset_frame=kp_val, prediction=val_kp_subset)

    enabled_kp_bands: list[str] = []
    band_comparison_target: list[dict[str, Any]] = []
    for band in known_prefix_target_bands:
        mask = (base_val["history_band"] == band).to_numpy() & (base_val["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5) & np.isfinite(val_kp)
        if not mask.any():
            continue
        y_true = base_val.loc[mask, "rating"].to_numpy(dtype=np.float32)
        known_mae = float(np.mean(np.abs(y_true - _round_half_up(val_known[mask]).astype(np.float32))))
        kp_mae = float(np.mean(np.abs(y_true - _round_half_up(val_kp[mask]).astype(np.float32))))
        delta = kp_mae - known_mae
        enabled = delta <= -float(args.known_prefix_enable_margin)
        if enabled:
            enabled_kp_bands.append(band)
        band_comparison_target.append({"history_band": band, "known_model_mae": known_mae, "known_prefix_deep_mae": kp_mae, "delta_mae": delta, "enabled_for_router": enabled})

    transition_routing_enabled = not bool(args.disable_transition_blend_routing)
    branches = _resolve_branches(
        user_known_mask=router_val["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5,
        history_band=base_val["history_band"],
        enabled_known_prefix_bands=enabled_kp_bands,
        enable_transition_blend_band=transition_routing_enabled,
    )
    val_router_raw = np.where(branches == "cold_model", val_cold, np.where(branches == "transition_blend_model", val_transition, np.where(branches == "known_prefix_deep_model", val_kp, val_known))).astype(np.float32)
    missing_prefix = (branches == "known_prefix_deep_model") & (~np.isfinite(val_kp))
    val_router_raw = np.where(missing_prefix, val_known, val_router_raw).astype(np.float32)
    branches = np.where(missing_prefix, "known_model", branches)
    val_router_rounded = _round_half_up(val_router_raw).clip(1, 5).astype(np.int32)

    transition_mask = (base_val["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5) & (base_val["history_band"].astype(str) == "2-5").to_numpy()
    transition_known_mae = float(np.mean(np.abs(base_val.loc[transition_mask, "rating"].to_numpy(dtype=np.float32) - _round_half_up(val_known[transition_mask]).astype(np.float32))))
    transition_blend_mae = float(np.mean(np.abs(base_val.loc[transition_mask, "rating"].to_numpy(dtype=np.float32) - _round_half_up(val_transition[transition_mask]).astype(np.float32))))

    val_eval = router_val[["review_id", "user", "item", "rating", "history_band"]].copy()
    val_eval["history_count"] = base_val["history_count"].to_numpy(dtype=np.int32)
    val_eval["pred_router_raw"] = val_router_raw
    val_eval["pred_router_rounded"] = val_router_rounded
    val_eval["router_branch"] = branches
    val_eval.to_csv(save_root / "validation_predictions.csv", index=False)

    routing_policy = {
        "cold_branch_band": "0",
        "transition_blend_band": "2-5" if transition_routing_enabled else None,
        "transition_short_train_range": "1-5",
        "transition_medium_train_range": "5-20",
        "transition_short_weight_formula": "clip((6-count)/4,0.25,0.85)",
        "enabled_known_prefix_bands": enabled_kp_bands,
    }
    spec.known_prefix_embedding_root = str(args.known_prefix_embedding_root)
    spec.known_prefix_max_history_len = int(args.known_prefix_max_history_len)
    spec.known_prefix_feature_columns = kp_features.copy()
    spec.known_prefix_target_bands = list(known_prefix_target_bands)
    spec.enabled_known_prefix_bands = enabled_kp_bands.copy()
    spec.routing_policy = routing_policy.copy()
    spec.manifest["routing_policy"] = routing_policy.copy()

    summary = {
        "router_validation_mae_raw": float(np.mean(np.abs(val_eval["rating"].to_numpy(dtype=np.float32) - val_router_raw))),
        "router_validation_mae_rounded": float(np.mean(np.abs(val_eval["rating"].to_numpy(dtype=np.float32) - val_router_rounded.astype(np.float32)))),
        "router_validation_rmse_rounded": float(np.sqrt(np.mean((val_eval["rating"].to_numpy(dtype=np.float32) - val_router_rounded.astype(np.float32)) ** 2))),
        "band_metrics_router": _compute_band_metrics(val_eval, "pred_router_rounded"),
        "band_metrics_served_by_branch": _compute_served_branch_metrics(val_eval, "pred_router_rounded"),
        "band_comparison_target": band_comparison_target,
        "transition_blend_eval": {"n_samples": int(transition_mask.sum()), "known_model_mae": transition_known_mae, "transition_blend_mae": transition_blend_mae, "delta_mae": float(transition_blend_mae - transition_known_mae), "average_short_weight": float(np.mean(weights[transition_mask]))},
        "transition_blend_eval_by_count": _compute_transition_count_diagnostics(
            frame=base_val,
            known_pred=val_known,
            short_pred=val_short,
            medium_pred=val_medium,
            blend_pred=val_transition,
        ),
        "routing_policy": routing_policy,
        "trainer_runtime": {
            "use_gpu": bool(args.use_gpu),
            "gpu_platform_id": int(args.gpu_platform_id),
            "gpu_device_id": int(args.gpu_device_id),
            "gpu_max_bin": int(args.gpu_max_bin),
            "gpu_use_dp": bool(args.gpu_use_dp),
        },
        "known_model": {"best_iteration": known_best, "train_rows": int(len(base_train)), "val_rows": int((base_val["user_known_in_train"] > 0.5).sum()), "feature_summary": _feature_summary(known_features)},
        "transition_short_model": {"best_iteration": short_best, "train_rows": int(len(short_train)), "val_rows": int(len(short_val)), "feature_summary": _feature_summary(known_features)},
        "transition_medium_model": {"best_iteration": medium_best, "train_rows": int(len(medium_train)), "val_rows": int(len(medium_val)), "feature_summary": _feature_summary(known_features)},
        "known_prefix_model": {"best_iteration": kp_best, "train_rows": int(len(kp_train)), "val_rows": int(len(kp_val)), "feature_summary": _feature_summary(kp_features), "enabled_bands": enabled_kp_bands.copy()},
        "cold_model": {"best_iteration": cold_best, "train_rows": int(len(router_train)), "val_rows": int(len(router_val[router_val["user_known_in_train"] < 0.5])), "feature_summary": _feature_summary(cold_features)},
        "router_branch_rows": {"known_model": int((branches == "known_model").sum()), "known_prefix_deep_model": int((branches == "known_prefix_deep_model").sum()), "transition_blend_model": int((branches == "transition_blend_model").sum()), "cold_model": int((branches == "cold_model").sum())},
        "cold_start_breakdown": cold_start_breakdown(train_split, val_split, user_col="user_id", item_col="business_id"),
        "spec_config": spec.config,
        "feature_manifest": spec.manifest,
    }
    baseline_summary = _read_baseline_summary(save_root)
    if baseline_summary is not None:
        summary["baseline_comparison"] = {
            "baseline_artifact": "lgbm_hybrid_conservative_v1",
            "baseline_router_validation_mae_rounded": baseline_summary.get("router_validation_mae_rounded"),
            "current_router_validation_mae_rounded": summary["router_validation_mae_rounded"],
            "mae_delta_vs_baseline": float(summary["router_validation_mae_rounded"] - float(baseline_summary.get("router_validation_mae_rounded", 0.0))),
        }

    joblib.dump(spec, save_root / "validation_router_spec.joblib")
    known_booster.save_model(str(save_root / "known_validation_model.txt"))
    cold_booster.save_model(str(save_root / "cold_validation_model.txt"))
    transition_short_booster.save_model(str(save_root / "transition_short_validation_model.txt"))
    transition_medium_booster.save_model(str(save_root / "transition_medium_validation_model.txt"))
    kp_booster.save_model(str(save_root / "known_prefix_validation_model.txt"))
    _save_feature_importance(save_root / "known_feature_importance.csv", known_booster, known_features)
    _save_feature_importance(save_root / "cold_feature_importance.csv", cold_booster, cold_features)
    _save_feature_importance(save_root / "transition_short_feature_importance.csv", transition_short_booster, known_features)
    _save_feature_importance(save_root / "transition_medium_feature_importance.csv", transition_medium_booster, known_features)
    _save_feature_importance(save_root / "known_prefix_feature_importance.csv", kp_booster, kp_features)
    _save_json(save_root / "validation_summary.json", summary)

    full_spec = fit_router_feature_spec(train_reviews, users_df, businesses_df, n_user_archetypes=args.n_user_archetypes, max_top_cities=args.max_top_cities, max_top_categories=args.max_top_categories, random_seed=args.seed)
    full_base = build_raw_feature_frame(train_reviews, users_df, businesses_df, full_spec.base_spec)
    full_router = build_router_feature_frame(train_reviews, users_df, businesses_df, full_spec)
    full_counts = train_reviews.groupby("user_id").size()
    full_base["history_band"] = _build_history_band(full_base, full_counts)
    full_base["history_count"] = _build_history_count(full_base, full_counts)
    full_router["history_band"] = _build_history_band(full_router, full_counts)
    full_router["router_history_band"] = full_router["history_band"]
    full_short = full_base[(full_base["user_known_in_train"] > 0.5) & full_base["history_count"].between(1, 5, inclusive="both")]
    full_medium = full_base[(full_base["user_known_in_train"] > 0.5) & full_base["history_count"].between(5, 20, inclusive="both")]
    full_kp_raw = build_known_prefix_train_frame(train_reviews, prefix_bundle, max_history_len=args.known_prefix_max_history_len, target_history_bands=known_prefix_target_bands)
    full_kp = _merge_known_prefix_features(full_base[full_base["review_id"].astype(str).isin(full_kp_raw["review_id"].astype(str))].copy(), full_kp_raw)
    full_spec.known_prefix_embedding_root = spec.known_prefix_embedding_root
    full_spec.known_prefix_max_history_len = spec.known_prefix_max_history_len
    full_spec.known_prefix_feature_columns = spec.known_prefix_feature_columns
    full_spec.known_prefix_target_bands = spec.known_prefix_target_bands
    full_spec.enabled_known_prefix_bands = spec.enabled_known_prefix_bands
    full_spec.routing_policy = spec.routing_policy
    full_spec.manifest["routing_policy"] = routing_policy.copy()
    joblib.dump(full_spec, save_root / "submission_router_spec.joblib")
    _train_booster(x_train=_extract_feature_matrix(full_base, known_features), y_train=full_base["rating"].to_numpy(dtype=np.float32), x_val=None, y_val=None, categorical_columns=known_categoricals, params=params("known"), num_boost_round=known_best, early_stopping_rounds=None).save_model(str(save_root / "known_submission_model.txt"))
    _train_booster(x_train=_extract_feature_matrix(full_router, cold_features), y_train=full_router["rating"].to_numpy(dtype=np.float32), x_val=None, y_val=None, categorical_columns=cold_categoricals, params=params("cold"), num_boost_round=cold_best, early_stopping_rounds=None, train_weight=full_router["router_history_band"].map(_cold_sample_weight).to_numpy(dtype=np.float32)).save_model(str(save_root / "cold_submission_model.txt"))
    _train_booster(x_train=_extract_feature_matrix(full_short, known_features), y_train=full_short["rating"].to_numpy(dtype=np.float32), x_val=None, y_val=None, categorical_columns=known_categoricals, params=params("transition_short"), num_boost_round=short_best, early_stopping_rounds=None).save_model(str(save_root / "transition_short_submission_model.txt"))
    _train_booster(x_train=_extract_feature_matrix(full_medium, known_features), y_train=full_medium["rating"].to_numpy(dtype=np.float32), x_val=None, y_val=None, categorical_columns=known_categoricals, params=params("transition_medium"), num_boost_round=medium_best, early_stopping_rounds=None).save_model(str(save_root / "transition_medium_submission_model.txt"))
    _train_booster(x_train=_extract_feature_matrix(full_kp, kp_features), y_train=full_kp["rating"].to_numpy(dtype=np.float32), x_val=None, y_val=None, categorical_columns=known_categoricals, params=params("known_prefix"), num_boost_round=kp_best, early_stopping_rounds=None).save_model(str(save_root / "known_prefix_submission_model.txt"))
    _save_json(save_root / "training_summary.json", {"validation_summary_path": str(save_root / "validation_summary.json")})
    print(json.dumps({"validation_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
