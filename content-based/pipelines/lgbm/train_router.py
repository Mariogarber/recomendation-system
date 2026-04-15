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
from scipy import sparse
from sklearn.decomposition import PCA

from utils.io import get_default_data_dir, load_businesses, load_train_reviews, load_users
from utils.lgbm_known_prefix_deep_features import (
    DEFAULT_KNOWN_PREFIX_TARGET_BANDS,
    build_known_prefix_eval_frame,
    build_known_prefix_train_frame,
    load_known_prefix_embedding_bundle,
    parse_known_prefix_target_bands,
    resolve_router_branches,
)
from utils.lgbm_raw_features import build_raw_feature_frame, history_band_from_count
from utils.lgbm_raw_router_features import build_router_feature_frame, fit_router_feature_spec
from utils.split import cold_start_breakdown, temporal_train_validation_split


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _load_business_embeddings_pca(
    embedding_root: Path, *, n_components: int = 32
) -> pd.DataFrame:
    """Load content-based business embeddings from NPZ and reduce via PCA to n_components dims."""
    repr_dir = embedding_root / "business_repr"
    business_ids = pd.read_csv(repr_dir / "business_ids.csv")["business_id"].astype(str)
    content_matrix = sparse.load_npz(repr_dir / "business_content_features.npz").astype(np.float32).toarray()
    effective_n = min(n_components, content_matrix.shape[1], content_matrix.shape[0])
    pca = PCA(n_components=effective_n, random_state=42)
    reduced = pca.fit_transform(content_matrix).astype(np.float32)
    col_names = [f"biz_emb_{i:02d}" for i in range(effective_n)]
    result = pd.DataFrame(reduced, columns=col_names)
    result.insert(0, "business_id", business_ids.to_numpy())
    return result


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
    train_weight: np.ndarray | None = None,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=train_weight,
        categorical_feature=categorical_columns,
        free_raw_data=False,
    )
    valid_sets = [train_set]
    callbacks: list[Any] = [lgb.log_evaluation(period=50)]
    if x_val is not None and y_val is not None:
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
    for band in ["0", "1", "2-5", "6-20", ">20"]:
        subset = frame[frame["history_band"] == band]
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


def _compute_served_branch_metrics(frame: pd.DataFrame, prediction_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preferred_order = ["cold_model", "known_model", "known_prefix_deep_model"]
    observed = [
        branch_name
        for branch_name in frame["router_branch"].astype(str).drop_duplicates().tolist()
        if branch_name not in preferred_order
    ]
    for branch_name in [*preferred_order, *observed]:
        subset = frame[frame["router_branch"] == branch_name]
        if subset.empty:
            continue
        diff = subset["rating"].to_numpy(dtype=np.float32) - subset[prediction_col].to_numpy(dtype=np.float32)
        rows.append(
            {
                "router_branch": branch_name,
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff ** 2))),
                "n_samples": int(len(subset)),
            }
        )
    return rows


def _feature_summary(feature_columns: list[str]) -> dict[str, int]:
    return {
        "total": len(feature_columns),
        "user": sum(column.startswith("user_") for column in feature_columns),
        "business": sum(column.startswith("business_") for column in feature_columns),
        "review": sum(column.startswith("review_") for column in feature_columns),
        "archetype": sum(
            column.startswith("archetype_")
            or column in {"business_city_top", "business_primary_category_family", "business_star_bin"}
            for column in feature_columns
        ),
        "known_prefix": sum(column.startswith("known_prefix_") for column in feature_columns),
    }


def _build_history_band(frame: pd.DataFrame, user_counts: pd.Series) -> pd.Series:
    lookup = user_counts.to_dict()
    return pd.Series(
        [history_band_from_count(int(lookup.get(user_id, 0))) for user_id in frame["user"].astype(str)],
        index=frame.index,
        dtype="string",
    )


def _build_params(
    *,
    num_leaves: int,
    learning_rate: float,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    seed: int,
    use_gpu: bool = False,
    gpu_platform_id: int = 0,
    gpu_device_id: int = 0,
    gpu_max_bin: int = 255,
    gpu_use_dp: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "regression_l1",
        "metric": "l1",
        "boosting_type": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "subsample_freq": 1,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "verbose": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
    }
    if use_gpu:
        params.update(
            {
                "device_type": "gpu",
                "gpu_platform_id": gpu_platform_id,
                "gpu_device_id": gpu_device_id,
                "max_bin": gpu_max_bin,
                "gpu_use_dp": gpu_use_dp,
            }
        )
    return params


def _cold_sample_weight(history_band: str) -> float:
    if history_band == "1":
        return 1.5
    if history_band == "2-5":
        return 1.2
    if history_band == "6-20":
        return 1.0
    return 1.0


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


def _read_baseline_router_summary(current_save_root: Path) -> dict[str, Any] | None:
    baseline_path = current_save_root.parent / "lgbm_raw_router_v1" / "validation_summary.json"
    if not baseline_path.exists():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def _compute_baseline_comparison(
    *,
    current_validation_summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if baseline_summary is None:
        return None
    return {
        "baseline_artifact": "lgbm_raw_router_v1",
        "baseline_router_validation_mae_rounded": baseline_summary.get("router_validation_mae_rounded"),
        "current_router_validation_mae_rounded": current_validation_summary["router_validation_mae_rounded"],
        "mae_delta_vs_baseline": float(
            current_validation_summary["router_validation_mae_rounded"]
            - float(baseline_summary.get("router_validation_mae_rounded", 0.0))
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a three-branch LightGBM router over raw-core, archetypes, and known-user prefix-deep features.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_train_stars_v1",
    )
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
    parser.add_argument(
        "--cold-training-max-band",
        type=str,
        default=">20",
        choices=["1", "2-5", "6-20", ">20"],
        help=(
            "Maximum history band included in cold model training. "
            "Restricting to '2-5' or '6-20' aligns the training distribution with cold-start users "
            "who have zero or limited history. Default '>20' keeps all rows (original behaviour)."
        ),
    )
    parser.add_argument(
        "--no-biz-train-stats",
        action="store_true",
        default=False,
        help="Disable the 3 business_train_stats features (mean/std/gap). Use to ablate their effect on the cold model.",
    )
    parser.add_argument(
        "--use-business-embeddings",
        action="store_true",
        default=False,
        help="Add PCA-reduced (32 components) business content embeddings as cold model features.",
    )
    parser.add_argument(
        "--cf-model-path",
        type=Path,
        default=None,
        help="Path to a trained CF model joblib (e.g. artifacts/cf_meta_model_v2/cf_model_v2.joblib). "
             "When provided, the model's item_bias_ dict is added as cf_item_bias and cf_item_prediction features.",
    )
    parser.add_argument(
        "--no-friend-features",
        action="store_true",
        default=False,
        help="Disable friend-business features (friend_business_mean, _bias, _count_log1p).",
    )

    parser.add_argument(
        "--known-prefix-embedding-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v3_iter03",
    )
    parser.add_argument("--known-prefix-max-history-len", type=int, default=20)
    parser.add_argument(
        "--known-prefix-target-bands",
        type=str,
        default=",".join(DEFAULT_KNOWN_PREFIX_TARGET_BANDS),
    )
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
    return parser.parse_args()


def _save_feature_importance(path: Path, booster: lgb.Booster, feature_columns: list[str]) -> None:
    pd.DataFrame(
        {
            "feature": feature_columns,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    known_prefix_target_bands = parse_known_prefix_target_bands(args.known_prefix_target_bands)

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    train_split, val_split = temporal_train_validation_split(
        train_reviews,
        val_size=args.validation_size,
        timestamp_col="date",
    )

    # user_average_stars from users.json is Yelp all-time metadata computed from external reviews
    # (avg 21.5 reviews outside our dataset per user). It is NOT leakage — see
    # analysis/user_average_stars_analysis.md for the full reasoning. Do not override it.

    _biz_emb_table: pd.DataFrame | None = None
    if args.use_business_embeddings:
        print("[train_router] Loading business content embeddings (PCA-32)...")
        _biz_emb_table = _load_business_embeddings_pca(args.known_prefix_embedding_root, n_components=32)
        print(f"[train_router] Business embedding table: {_biz_emb_table.shape}")

    _cf_item_bias_table: pd.DataFrame | None = None
    if args.cf_model_path is not None:
        print(f"[train_router] Loading CF model for item bias from {args.cf_model_path}...")
        # The CF model was saved when train_cf.py ran as __main__, so pickle stored the class as
        # __main__.CFSVDModel.  To unpickle successfully we must inject the class into the current
        # __main__ module before calling joblib.load().
        import sys as _sys
        import pipelines.collaborative_filtering.train_cf as _cf_train_mod
        _sys.modules["__main__"].__dict__["CFSVDModel"] = _cf_train_mod.CFSVDModel
        _cf_model = joblib.load(args.cf_model_path)
        _bias_dict = getattr(_cf_model, "item_bias_", {})
        _cf_item_bias_table = pd.DataFrame(
            {"business_id": list(_bias_dict.keys()), "cf_item_bias": list(_bias_dict.values())}
        )
        print(f"[train_router] CF item bias table: {len(_cf_item_bias_table):,} businesses")

    validation_spec = fit_router_feature_spec(
        train_split,
        users_df,
        businesses_df,
        n_user_archetypes=args.n_user_archetypes,
        max_top_cities=args.max_top_cities,
        max_top_categories=args.max_top_categories,
        random_seed=args.seed,
        include_biz_train_stats=not args.no_biz_train_stats,
        business_embedding_table=_biz_emb_table,
        cf_item_bias_table=_cf_item_bias_table,
        include_friend_features=not args.no_friend_features,
    )
    base_train_frame = build_raw_feature_frame(train_split, users_df, businesses_df, validation_spec.base_spec)
    base_val_frame = build_raw_feature_frame(val_split, users_df, businesses_df, validation_spec.base_spec)
    router_train_frame = build_router_feature_frame(train_split, users_df, businesses_df, validation_spec)
    router_val_frame = build_router_feature_frame(val_split, users_df, businesses_df, validation_spec)

    train_user_counts = train_split.groupby("user_id").size()
    val_user_counts = validation_spec.base_spec.user_priors_table.set_index("user_id")["user_train_count"]
    base_train_frame["history_band"] = _build_history_band(base_train_frame, train_user_counts)
    base_val_frame["history_band"] = _build_history_band(base_val_frame, val_user_counts)
    router_train_frame["history_band"] = _build_history_band(router_train_frame, train_user_counts)
    router_val_frame["history_band"] = _build_history_band(router_val_frame, val_user_counts)
    base_train_frame["router_history_band"] = base_train_frame["history_band"]
    router_train_frame["router_history_band"] = router_train_frame["history_band"]

    known_train = base_train_frame.copy()
    known_val = base_val_frame[base_val_frame["user_known_in_train"] > 0.5].copy()
    # Restrict cold training to low-history bands to better match cold-start test distribution
    _COLD_BAND_ORDER = ["1", "2-5", "6-20", ">20"]
    _allowed_cold_bands = _COLD_BAND_ORDER[: _COLD_BAND_ORDER.index(args.cold_training_max_band) + 1]
    cold_train = router_train_frame[router_train_frame["router_history_band"].isin(_allowed_cold_bands)].copy()
    cold_val = router_val_frame[router_val_frame["user_known_in_train"] < 0.5].copy()
    if cold_val.empty:
        cold_val = router_val_frame.copy()

    known_features = [
        column
        for column in validation_spec.base_spec.feature_columns
        if column not in {"user_known_in_train", "business_known_in_train"}
    ]
    cold_features = validation_spec.cold_feature_columns
    known_categoricals = [column for column in validation_spec.base_spec.categorical_columns if column in known_features]
    cold_categoricals = [column for column in validation_spec.categorical_columns if column in cold_features]

    known_params = _build_params(
        num_leaves=args.known_num_leaves,
        learning_rate=args.known_learning_rate,
        min_child_samples=args.known_min_child_samples,
        subsample=args.known_subsample,
        colsample_bytree=args.known_colsample_bytree,
        reg_alpha=args.known_reg_alpha,
        reg_lambda=args.known_reg_lambda,
        seed=args.seed,
    )
    cold_params = _build_params(
        num_leaves=args.cold_num_leaves,
        learning_rate=args.cold_learning_rate,
        min_child_samples=args.cold_min_child_samples,
        subsample=args.cold_subsample,
        colsample_bytree=args.cold_colsample_bytree,
        reg_alpha=args.cold_reg_alpha,
        reg_lambda=args.cold_reg_lambda,
        seed=args.seed,
    )
    known_prefix_params = _build_params(
        num_leaves=args.known_prefix_num_leaves,
        learning_rate=args.known_prefix_learning_rate,
        min_child_samples=args.known_prefix_min_child_samples,
        subsample=args.known_prefix_subsample,
        colsample_bytree=args.known_prefix_colsample_bytree,
        reg_alpha=args.known_prefix_reg_alpha,
        reg_lambda=args.known_prefix_reg_lambda,
        seed=args.seed,
    )

    known_booster = _train_booster(
        x_train=_extract_feature_matrix(known_train, known_features),
        y_train=known_train["rating"].to_numpy(dtype=np.float32),
        x_val=_extract_feature_matrix(known_val, known_features) if not known_val.empty else None,
        y_val=known_val["rating"].to_numpy(dtype=np.float32) if not known_val.empty else None,
        categorical_columns=known_categoricals,
        params=known_params,
        num_boost_round=args.known_n_estimators,
        early_stopping_rounds=args.known_early_stopping_rounds,
    )
    cold_train_weight = cold_train["router_history_band"].map(_cold_sample_weight).to_numpy(dtype=np.float32)
    cold_booster = _train_booster(
        x_train=_extract_feature_matrix(cold_train, cold_features),
        y_train=cold_train["rating"].to_numpy(dtype=np.float32),
        x_val=_extract_feature_matrix(cold_val, cold_features) if not cold_val.empty else None,
        y_val=cold_val["rating"].to_numpy(dtype=np.float32) if not cold_val.empty else None,
        categorical_columns=cold_categoricals,
        params=cold_params,
        num_boost_round=args.cold_n_estimators,
        early_stopping_rounds=args.cold_early_stopping_rounds,
        train_weight=cold_train_weight,
    )
    prefix_bundle = load_known_prefix_embedding_bundle(args.known_prefix_embedding_root)
    known_prefix_train_raw = build_known_prefix_train_frame(
        train_split,
        prefix_bundle,
        max_history_len=args.known_prefix_max_history_len,
        target_history_bands=known_prefix_target_bands,
    )
    known_prefix_val_raw = build_known_prefix_eval_frame(
        val_split,
        train_split,
        prefix_bundle,
        max_history_len=args.known_prefix_max_history_len,
        target_history_bands=known_prefix_target_bands,
    )
    known_prefix_train = _merge_known_prefix_features(
        base_train_frame[base_train_frame["review_id"].astype(str).isin(known_prefix_train_raw["review_id"].astype(str))].copy(),
        known_prefix_train_raw,
    )
    known_prefix_val = _merge_known_prefix_features(
        base_val_frame[base_val_frame["review_id"].astype(str).isin(known_prefix_val_raw["review_id"].astype(str))].copy(),
        known_prefix_val_raw,
    )
    known_prefix_deep_columns = [
        column
        for column in known_prefix_train.columns
        if column.startswith("known_prefix_") and column != "known_prefix_history_band"
    ]
    known_prefix_features = [*known_features, *known_prefix_deep_columns]
    known_prefix_booster = _train_booster(
        x_train=_extract_feature_matrix(known_prefix_train, known_prefix_features),
        y_train=known_prefix_train["rating"].to_numpy(dtype=np.float32),
        x_val=_extract_feature_matrix(known_prefix_val, known_prefix_features) if not known_prefix_val.empty else None,
        y_val=known_prefix_val["rating"].to_numpy(dtype=np.float32) if not known_prefix_val.empty else None,
        categorical_columns=known_categoricals,
        params=known_prefix_params,
        num_boost_round=args.known_prefix_n_estimators,
        early_stopping_rounds=args.known_prefix_early_stopping_rounds,
    )

    known_best_iteration = int(known_booster.best_iteration or known_booster.current_iteration() or args.known_n_estimators)
    cold_best_iteration = int(cold_booster.best_iteration or cold_booster.current_iteration() or args.cold_n_estimators)
    known_prefix_best_iteration = int(
        known_prefix_booster.best_iteration or known_prefix_booster.current_iteration() or args.known_prefix_n_estimators
    )

    val_known_pred = np.clip(
        known_booster.predict(_extract_feature_matrix(base_val_frame, known_features), num_iteration=known_best_iteration).astype(np.float32),
        1.0,
        5.0,
    )
    val_cold_pred = np.clip(
        cold_booster.predict(_extract_feature_matrix(router_val_frame, cold_features), num_iteration=cold_best_iteration).astype(np.float32),
        1.0,
        5.0,
    )
    val_known_prefix_subset_pred = (
        np.clip(
            known_prefix_booster.predict(
                _extract_feature_matrix(known_prefix_val, known_prefix_features),
                num_iteration=known_prefix_best_iteration,
            ).astype(np.float32),
            1.0,
            5.0,
        )
        if not known_prefix_val.empty
        else np.empty(0, dtype=np.float32)
    )
    val_known_prefix_pred = _apply_prediction_to_frame(
        frame=base_val_frame,
        subset_frame=known_prefix_val,
        prediction=val_known_prefix_subset_pred,
    )

    band_comparison_target: list[dict[str, Any]] = []
    enabled_known_prefix_bands: list[str] = []
    for band in known_prefix_target_bands:
        band_mask = (base_val_frame["history_band"] == band).to_numpy() & (
            base_val_frame["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5
        )
        if not band_mask.any():
            continue
        band_prefix_mask = band_mask & np.isfinite(val_known_prefix_pred)
        if not band_prefix_mask.any():
            continue
        y_true = base_val_frame.loc[band_prefix_mask, "rating"].to_numpy(dtype=np.float32)
        known_band_pred = _round_half_up(val_known_pred[band_prefix_mask]).clip(1, 5).astype(np.float32)
        prefix_band_pred = _round_half_up(val_known_prefix_pred[band_prefix_mask]).clip(1, 5).astype(np.float32)
        known_mae = float(np.mean(np.abs(y_true - known_band_pred)))
        prefix_mae = float(np.mean(np.abs(y_true - prefix_band_pred)))
        delta_mae = prefix_mae - known_mae
        enabled = delta_mae <= -float(args.known_prefix_enable_margin)
        if enabled:
            enabled_known_prefix_bands.append(band)
        band_comparison_target.append(
            {
                "history_band": band,
                "known_model_mae": known_mae,
                "known_prefix_deep_mae": prefix_mae,
                "delta_mae": delta_mae,
                "enabled_for_router": enabled,
            }
        )

    routing_policy = {
        "cold_branch_band": "0",
        "known_prefix_candidate_bands": list(known_prefix_target_bands),
        "enabled_known_prefix_bands": enabled_known_prefix_bands,
        "known_fallback_band": ">20",
        "known_prefix_enable_margin": float(args.known_prefix_enable_margin),
    }
    router_branches = resolve_router_branches(
        user_known_mask=router_val_frame["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5,
        history_band=base_val_frame["history_band"],
        enabled_known_prefix_bands=tuple(enabled_known_prefix_bands),
    )
    val_router_raw = np.where(
        router_branches == "cold_model",
        val_cold_pred,
        np.where(router_branches == "known_prefix_deep_model", val_known_prefix_pred, val_known_pred),
    ).astype(np.float32)
    missing_prefix_mask = (router_branches == "known_prefix_deep_model") & (~np.isfinite(val_known_prefix_pred))
    val_router_raw = np.where(missing_prefix_mask, val_known_pred, val_router_raw).astype(np.float32)
    router_branches = np.where(missing_prefix_mask, "known_model", router_branches)
    val_router_rounded = _round_half_up(val_router_raw).clip(1, 5).astype(np.int32)

    val_eval = router_val_frame[["review_id", "user", "item", "rating", "history_band", "user_known_in_train"]].copy()
    val_eval["pred_known_raw"] = val_known_pred
    val_eval["pred_cold_raw"] = val_cold_pred
    val_eval["pred_known_prefix_raw"] = val_known_prefix_pred
    val_eval["pred_router_raw"] = val_router_raw
    val_eval["pred_router_rounded"] = val_router_rounded
    val_eval["router_branch"] = router_branches

    validation_spec.known_prefix_embedding_root = str(args.known_prefix_embedding_root)
    validation_spec.known_prefix_max_history_len = int(args.known_prefix_max_history_len)
    validation_spec.known_prefix_feature_columns = known_prefix_features.copy()
    validation_spec.known_prefix_target_bands = list(known_prefix_target_bands)
    validation_spec.enabled_known_prefix_bands = enabled_known_prefix_bands.copy()
    validation_spec.routing_policy = routing_policy.copy()
    validation_spec.manifest["known_prefix_generated_variables"] = known_prefix_deep_columns
    validation_spec.manifest["known_prefix_embedding_root"] = str(args.known_prefix_embedding_root)
    validation_spec.manifest["routing_policy"] = routing_policy.copy()

    validation_summary = {
        "router_validation_mae_raw": float(np.mean(np.abs(val_eval["rating"].to_numpy(dtype=np.float32) - val_router_raw))),
        "router_validation_mae_rounded": float(np.mean(np.abs(val_eval["rating"].to_numpy(dtype=np.float32) - val_router_rounded.astype(np.float32)))),
        "router_validation_rmse_rounded": float(
            np.sqrt(np.mean((val_eval["rating"].to_numpy(dtype=np.float32) - val_router_rounded.astype(np.float32)) ** 2))
        ),
        "band_metrics_router": _compute_band_metrics(val_eval, "pred_router_rounded"),
        "band_metrics_served_by_branch": _compute_served_branch_metrics(val_eval, "pred_router_rounded"),
        "band_comparison_target": band_comparison_target,
        "routing_policy": routing_policy,
        "known_model": {
            "best_iteration": known_best_iteration,
            "train_rows": int(len(known_train)),
            "val_rows": int(len(known_val)),
            "feature_summary": _feature_summary(known_features),
        },
        "known_prefix_model": {
            "best_iteration": known_prefix_best_iteration,
            "train_rows": int(len(known_prefix_train)),
            "val_rows": int(len(known_prefix_val)),
            "feature_summary": _feature_summary(known_prefix_features),
            "target_bands": list(known_prefix_target_bands),
            "enabled_bands": enabled_known_prefix_bands.copy(),
            "embedding_root": str(args.known_prefix_embedding_root),
            "max_history_len": int(args.known_prefix_max_history_len),
        },
        "cold_model": {
            "best_iteration": cold_best_iteration,
            "train_rows": int(len(cold_train)),
            "val_rows": int(len(cold_val)),
            "feature_summary": _feature_summary(cold_features),
            "train_history_bands": {
                str(key): int(value)
                for key, value in cold_train["router_history_band"].value_counts(dropna=False).to_dict().items()
            },
        },
        "router_branch_rows": {
            "known_model": int((router_branches == "known_model").sum()),
            "known_prefix_deep_model": int((router_branches == "known_prefix_deep_model").sum()),
            "cold_model": int((router_branches == "cold_model").sum()),
        },
        "cold_start_breakdown": cold_start_breakdown(train_split, val_split, user_col="user_id", item_col="business_id"),
        "spec_config": validation_spec.config,
        "feature_manifest": validation_spec.manifest,
    }
    validation_summary["baseline_comparison"] = _compute_baseline_comparison(
        current_validation_summary=validation_summary,
        baseline_summary=_read_baseline_router_summary(save_root),
    )

    validation_spec_path = save_root / "validation_router_spec.joblib"
    known_validation_model_path = save_root / "known_validation_model.txt"
    known_prefix_validation_model_path = save_root / "known_prefix_validation_model.txt"
    cold_validation_model_path = save_root / "cold_validation_model.txt"
    joblib.dump(validation_spec, validation_spec_path)
    known_booster.save_model(str(known_validation_model_path))
    known_prefix_booster.save_model(str(known_prefix_validation_model_path))
    cold_booster.save_model(str(cold_validation_model_path))
    val_eval.to_csv(save_root / "validation_predictions.csv", index=False)
    _save_feature_importance(save_root / "known_feature_importance.csv", known_booster, known_features)
    _save_feature_importance(save_root / "known_prefix_feature_importance.csv", known_prefix_booster, known_prefix_features)
    _save_feature_importance(save_root / "cold_feature_importance.csv", cold_booster, cold_features)
    validation_spec.archetype_profiles.to_csv(save_root / "archetype_profiles.csv", index=False)
    _save_json(save_root / "feature_manifest.json", validation_spec.manifest)
    _save_json(save_root / "discarded_variables.json", validation_spec.manifest["discarded_variables"])
    _save_json(save_root / "validation_summary.json", validation_summary)

    # Submission model uses the same users_df (Yelp all-time average_stars) — not overridden.
    full_spec = fit_router_feature_spec(
        train_reviews,
        users_df,
        businesses_df,
        n_user_archetypes=args.n_user_archetypes,
        max_top_cities=args.max_top_cities,
        max_top_categories=args.max_top_categories,
        random_seed=args.seed,
        include_biz_train_stats=not args.no_biz_train_stats,
        business_embedding_table=_biz_emb_table,
        cf_item_bias_table=_cf_item_bias_table,
        include_friend_features=not args.no_friend_features,
    )
    full_base_frame = build_raw_feature_frame(train_reviews, users_df, businesses_df, full_spec.base_spec)
    full_router_frame = build_router_feature_frame(train_reviews, users_df, businesses_df, full_spec)
    full_user_counts = train_reviews.groupby("user_id").size()
    full_base_frame["history_band"] = _build_history_band(full_base_frame, full_user_counts)
    full_router_frame["history_band"] = _build_history_band(full_router_frame, full_user_counts)
    full_base_frame["router_history_band"] = full_base_frame["history_band"]
    full_router_frame["router_history_band"] = full_router_frame["history_band"]
    full_known_train = full_base_frame.copy()
    full_cold_train = full_router_frame[full_router_frame["router_history_band"].isin(_allowed_cold_bands)].copy()
    full_cold_weight = full_cold_train["router_history_band"].map(_cold_sample_weight).to_numpy(dtype=np.float32)
    full_known_prefix_raw = build_known_prefix_train_frame(
        train_reviews,
        prefix_bundle,
        max_history_len=args.known_prefix_max_history_len,
        target_history_bands=known_prefix_target_bands,
    )
    full_known_prefix_train = _merge_known_prefix_features(
        full_base_frame[full_base_frame["review_id"].astype(str).isin(full_known_prefix_raw["review_id"].astype(str))].copy(),
        full_known_prefix_raw,
    )

    full_known_booster = _train_booster(
        x_train=_extract_feature_matrix(full_known_train, known_features),
        y_train=full_known_train["rating"].to_numpy(dtype=np.float32),
        x_val=None,
        y_val=None,
        categorical_columns=[column for column in full_spec.base_spec.categorical_columns if column in known_features],
        params=known_params,
        num_boost_round=known_best_iteration,
        early_stopping_rounds=None,
    )
    full_known_prefix_booster = _train_booster(
        x_train=_extract_feature_matrix(full_known_prefix_train, known_prefix_features),
        y_train=full_known_prefix_train["rating"].to_numpy(dtype=np.float32),
        x_val=None,
        y_val=None,
        categorical_columns=[column for column in full_spec.base_spec.categorical_columns if column in known_prefix_features],
        params=known_prefix_params,
        num_boost_round=known_prefix_best_iteration,
        early_stopping_rounds=None,
    )
    full_cold_booster = _train_booster(
        x_train=_extract_feature_matrix(full_cold_train, full_spec.cold_feature_columns),
        y_train=full_cold_train["rating"].to_numpy(dtype=np.float32),
        x_val=None,
        y_val=None,
        categorical_columns=[column for column in full_spec.categorical_columns if column in full_spec.cold_feature_columns],
        params=cold_params,
        num_boost_round=cold_best_iteration,
        early_stopping_rounds=None,
        train_weight=full_cold_weight,
    )

    full_spec.known_prefix_embedding_root = str(args.known_prefix_embedding_root)
    full_spec.known_prefix_max_history_len = int(args.known_prefix_max_history_len)
    full_spec.known_prefix_feature_columns = known_prefix_features.copy()
    full_spec.known_prefix_target_bands = list(known_prefix_target_bands)
    full_spec.enabled_known_prefix_bands = enabled_known_prefix_bands.copy()
    full_spec.routing_policy = routing_policy.copy()
    full_spec.manifest["known_prefix_generated_variables"] = known_prefix_deep_columns
    full_spec.manifest["known_prefix_embedding_root"] = str(args.known_prefix_embedding_root)
    full_spec.manifest["routing_policy"] = routing_policy.copy()

    submission_spec_path = save_root / "submission_router_spec.joblib"
    known_submission_model_path = save_root / "known_submission_model.txt"
    known_prefix_submission_model_path = save_root / "known_prefix_submission_model.txt"
    cold_submission_model_path = save_root / "cold_submission_model.txt"
    joblib.dump(full_spec, submission_spec_path)
    full_known_booster.save_model(str(known_submission_model_path))
    full_known_prefix_booster.save_model(str(known_prefix_submission_model_path))
    full_cold_booster.save_model(str(cold_submission_model_path))

    training_summary = {
        "submission_known_best_iteration": known_best_iteration,
        "submission_known_prefix_best_iteration": known_prefix_best_iteration,
        "submission_cold_best_iteration": cold_best_iteration,
        "full_known_train_rows": int(len(full_known_train)),
        "full_known_prefix_train_rows": int(len(full_known_prefix_train)),
        "full_cold_train_rows": int(len(full_cold_train)),
        "validation_summary_path": str(save_root / "validation_summary.json"),
        "validation_spec_path": str(validation_spec_path),
        "submission_spec_path": str(submission_spec_path),
        "known_validation_model_path": str(known_validation_model_path),
        "known_prefix_validation_model_path": str(known_prefix_validation_model_path),
        "cold_validation_model_path": str(cold_validation_model_path),
        "known_submission_model_path": str(known_submission_model_path),
        "known_prefix_submission_model_path": str(known_prefix_submission_model_path),
        "cold_submission_model_path": str(cold_submission_model_path),
        "archetype_profiles_path": str(save_root / "archetype_profiles.csv"),
        "feature_manifest_path": str(save_root / "feature_manifest.json"),
    }
    config_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    _save_json(save_root / "training_summary.json", training_summary)
    _save_json(save_root / "config.json", config_payload)
    print(json.dumps({"validation_summary": validation_summary, "training_summary": training_summary}, indent=2))


if __name__ == "__main__":
    main()
