from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from predict_lgbm_raw_router_submission import _apply_prediction_to_frame, _merge_known_prefix_features, _round_half_up, _save_json
from utils.io import get_default_data_dir, load_businesses, load_test_reviews, load_train_reviews, load_users
from utils.lgbm_known_prefix_deep_features import build_known_prefix_eval_frame, load_known_prefix_embedding_bundle
from utils.lgbm_raw_features import build_raw_feature_frame, history_band_from_count
from utils.lgbm_raw_router_features import build_router_feature_frame


def _short_weight_from_count(count: int) -> float:
    return float(np.clip((6.0 - float(count)) / 4.0, 0.25, 0.85))


def _resolve_branches(
    *,
    user_known_mask: np.ndarray,
    history_band: pd.Series,
    enabled_known_prefix_bands: list[str],
    transition_blend_band: str | None = "2-5",
) -> np.ndarray:
    bands = history_band.astype("string").to_numpy(dtype=object)
    out = np.full(len(bands), "known_model", dtype=object)
    out[~user_known_mask] = "cold_model"
    if transition_blend_band is not None:
        out[user_known_mask & (bands == transition_blend_band)] = "transition_blend_model"
    prefix_mask = user_known_mask & np.isin(bands, np.asarray(enabled_known_prefix_bands, dtype=object))
    out[prefix_mask] = "known_prefix_deep_model"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission from the transition-blend hybrid router.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_transition_blend_router_v1")
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root
    save_path = args.save_path or root / "submission.csv"
    spec = joblib.load(root / "submission_router_spec.joblib")

    known_booster = lgb.Booster(model_file=str(root / "known_submission_model.txt"))
    cold_booster = lgb.Booster(model_file=str(root / "cold_submission_model.txt"))
    transition_short_booster = lgb.Booster(model_file=str(root / "transition_short_submission_model.txt"))
    transition_medium_booster = lgb.Booster(model_file=str(root / "transition_medium_submission_model.txt"))
    known_prefix_booster = lgb.Booster(model_file=str(root / "known_prefix_submission_model.txt"))

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)
    base_test = build_raw_feature_frame(test_reviews, users_df, businesses_df, spec.base_spec)
    router_test = build_router_feature_frame(test_reviews, users_df, businesses_df, spec)
    user_count_lookup = spec.base_spec.user_priors_table.set_index("user_id")["user_train_count"].to_dict()
    base_test["history_band"] = pd.Series([history_band_from_count(int(user_count_lookup.get(user_id, 0))) for user_id in base_test["user"].astype(str)], index=base_test.index, dtype="string")
    base_test["history_count"] = pd.Series([int(user_count_lookup.get(user_id, 0)) for user_id in base_test["user"].astype(str)], index=base_test.index, dtype=np.int32)
    router_test["history_band"] = base_test["history_band"]

    known_features = [c for c in spec.base_spec.feature_columns if c not in {"user_known_in_train", "business_known_in_train"}]
    known_raw = np.clip(known_booster.predict(base_test[known_features].copy(), num_iteration=known_booster.best_iteration or known_booster.num_trees()).astype(np.float32), 1.0, 5.0)
    cold_raw = np.clip(cold_booster.predict(router_test[spec.cold_feature_columns].copy(), num_iteration=cold_booster.best_iteration or cold_booster.num_trees()).astype(np.float32), 1.0, 5.0)
    short_raw = np.clip(transition_short_booster.predict(base_test[known_features].copy(), num_iteration=transition_short_booster.best_iteration or transition_short_booster.num_trees()).astype(np.float32), 1.0, 5.0)
    medium_raw = np.clip(transition_medium_booster.predict(base_test[known_features].copy(), num_iteration=transition_medium_booster.best_iteration or transition_medium_booster.num_trees()).astype(np.float32), 1.0, 5.0)
    weights = np.vectorize(_short_weight_from_count)(base_test["history_count"].to_numpy(dtype=np.int32)).astype(np.float32)
    transition_raw = (weights * short_raw + (1.0 - weights) * medium_raw).astype(np.float32)

    kp_raw = np.full(len(base_test), np.nan, dtype=np.float32)
    if spec.enabled_known_prefix_bands:
        kp_target = base_test[(base_test["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5) & base_test["history_band"].astype(str).isin(spec.enabled_known_prefix_bands)].copy()
        bundle = load_known_prefix_embedding_bundle(spec.known_prefix_embedding_root)
        kp_eval_raw = build_known_prefix_eval_frame(test_reviews[test_reviews["review_id"].astype(str).isin(kp_target["review_id"].astype(str))].copy(), train_reviews, bundle, max_history_len=spec.known_prefix_max_history_len, target_history_bands=tuple(spec.enabled_known_prefix_bands))
        kp_eval = _merge_known_prefix_features(kp_target, kp_eval_raw)
        kp_subset = np.clip(known_prefix_booster.predict(kp_eval[spec.known_prefix_feature_columns].copy(), num_iteration=known_prefix_booster.best_iteration or known_prefix_booster.num_trees()).astype(np.float32), 1.0, 5.0) if not kp_eval.empty else np.empty(0, dtype=np.float32)
        kp_raw = _apply_prediction_to_frame(frame=base_test, subset_frame=kp_eval, prediction=kp_subset)

    routing_policy = getattr(spec, "routing_policy", {}) or {}
    transition_blend_band = routing_policy.get("transition_blend_band", "2-5")
    branches = _resolve_branches(
        user_known_mask=router_test["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5,
        history_band=base_test["history_band"],
        enabled_known_prefix_bands=list(spec.enabled_known_prefix_bands),
        transition_blend_band=transition_blend_band,
    )
    router_raw = np.where(branches == "cold_model", cold_raw, np.where(branches == "transition_blend_model", transition_raw, np.where(branches == "known_prefix_deep_model", kp_raw, known_raw))).astype(np.float32)
    missing_prefix = (branches == "known_prefix_deep_model") & (~np.isfinite(kp_raw))
    router_raw = np.where(missing_prefix, known_raw, router_raw).astype(np.float32)
    branches = np.where(missing_prefix, "known_model", branches)
    router_rounded = _round_half_up(router_raw).clip(1, 5).astype(np.int32)

    submission = pd.DataFrame({"review_id": router_test["review_id"].astype(str), "stars": router_rounded})
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=False)
    summary = {
        "artifact_root": str(root),
        "save_path": str(save_path),
        "n_rows": int(len(submission)),
        "known_branch_rows": int((branches == "known_model").sum()),
        "known_prefix_branch_rows": int((branches == "known_prefix_deep_model").sum()),
        "transition_blend_branch_rows": int((branches == "transition_blend_model").sum()),
        "cold_branch_rows": int((branches == "cold_model").sum()),
        "enabled_known_prefix_bands": list(spec.enabled_known_prefix_bands),
        "transition_blend_band": transition_blend_band,
        "prediction_min": int(router_rounded.min()) if len(router_rounded) else None,
        "prediction_max": int(router_rounded.max()) if len(router_rounded) else None,
        "prediction_mean_raw": float(router_raw.mean()) if len(router_raw) else None,
        "prediction_mean_rounded": float(router_rounded.mean()) if len(router_rounded) else None,
    }
    _save_json(root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
