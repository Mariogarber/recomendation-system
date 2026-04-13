from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from utils.io import get_default_data_dir, load_businesses, load_test_reviews, load_train_reviews, load_users
from utils.known_user_deep_e2e import (
    KnownUserDeepContext,
    KnownUserDeepDataConfig,
    KnownUserDeepFeatureContract,
    KnownUserDeepTrainingConfig,
    build_known_user_eval_dataset,
    build_model_from_checkpoint,
    load_safe_business_feature_block,
    predict_known_user_dataset,
)
from utils.lgbm_known_prefix_deep_features import build_known_prefix_eval_frame, load_known_prefix_embedding_bundle, resolve_router_branches
from utils.lgbm_raw_features import build_raw_feature_frame, history_band_from_count
from utils.lgbm_raw_router_features import build_router_feature_frame


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission from the known-user deep mixed router.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts" / "known_user_deep_router_v1",
    )
    parser.add_argument("--save-path", type=Path, default=None)
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
    user_count_lookup = spec.base_spec.user_priors_table.set_index("user_id")["user_train_count"].to_dict()
    base_frame["history_band"] = pd.Series(
        [history_band_from_count(int(user_count_lookup.get(user_id, 0))) for user_id in base_frame["user"].astype(str)],
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
    output = base_frame[["review_id", "history_band"]].copy()
    output["incumbent_prediction_raw"] = router_raw
    output["incumbent_prediction"] = router_rounded.astype(np.float32)
    output["incumbent_branch"] = router_branches
    return output


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root
    save_path = args.save_path or artifact_root / "submission.csv"

    router_spec = joblib.load(artifact_root / "router_spec.joblib")
    incumbent_root = Path(router_spec["incumbent_router_root"])
    incumbent_spec = joblib.load(incumbent_root / "submission_router_spec.joblib")
    known_booster = lgb.Booster(model_file=str(incumbent_root / "known_submission_model.txt"))
    known_prefix_booster = lgb.Booster(model_file=str(incumbent_root / "known_prefix_submission_model.txt"))
    cold_booster = lgb.Booster(model_file=str(incumbent_root / "cold_submission_model.txt"))

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)

    business_ids, business_matrix, _ = load_safe_business_feature_block(router_spec["business_repr_root"])
    deep_training_config = KnownUserDeepTrainingConfig(**router_spec.get("known_user_deep_training_config", {}))
    deep_context = KnownUserDeepContext(
        data_config=KnownUserDeepDataConfig(
            business_repr_root=router_spec["business_repr_root"],
            max_history_len=int(router_spec["max_history_len"]),
        ),
        feature_contract=KnownUserDeepFeatureContract(**router_spec["known_user_deep_feature_contract"]),
        raw_spec=router_spec["raw_spec"],
        router_spec=router_spec["router_feature_spec"],
        business_ids=business_ids,
        business_matrix=business_matrix,
        business_index=pd.Series(np.arange(len(business_ids), dtype=np.int32), index=business_ids.to_numpy()),
    )
    model, _ = build_model_from_checkpoint(router_spec["known_user_deep_checkpoint_path"], device="cpu")

    incumbent_test = _predict_incumbent_router(
        spec=incumbent_spec,
        known_booster=known_booster,
        known_prefix_booster=known_prefix_booster,
        cold_booster=cold_booster,
        target_reviews=test_reviews,
        context_reviews=train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
    )
    deep_test = predict_known_user_dataset(
        model=model,
        prepared=build_known_user_eval_dataset(
            test_reviews,
            train_reviews,
            users_df=users_df,
            businesses_df=businesses_df,
            context=deep_context,
            training_config=deep_training_config,
            incumbent_frame=incumbent_test,
        ),
        context=deep_context,
        batch_size=int(router_spec.get("batch_size", 512)),
        device="cpu",
    ).rename(columns={"predicted_rating": "deep_prediction"})
    deep_test["deep_prediction_raw"] = deep_test["deep_prediction"].astype(np.float32)
    deep_test["deep_prediction"] = _round_half_up(deep_test["deep_prediction_raw"].to_numpy(dtype=np.float32)).clip(1, 5).astype(np.float32)
    deep_lookup = deep_test.set_index(deep_test["review_id"].astype(str))[["deep_prediction", "deep_prediction_raw", "alpha"]]

    final_test = incumbent_test.copy()
    final_test["final_prediction_raw"] = final_test["incumbent_prediction_raw"].astype(np.float32)
    final_test["final_prediction"] = final_test["incumbent_prediction"].astype(np.float32)
    final_test["final_router_branch"] = final_test["incumbent_branch"].astype(str)
    enabled_mask = final_test["history_band"].astype(str).isin(router_spec["enabled_known_deep_bands"])
    deep_available_mask = final_test["review_id"].astype(str).isin(deep_lookup.index.astype(str))
    replace_mask = enabled_mask & deep_available_mask
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
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=False)

    raw_pred_path = save_path.parent / "raw_predictions.csv"
    final_test[["review_id", "history_band", "incumbent_prediction_raw", "final_prediction_raw", "final_router_branch"]].rename(
        columns={"final_prediction_raw": "cb_prediction_raw"}
    ).to_csv(raw_pred_path, index=False)

    summary = {
        "artifact_root": str(artifact_root),
        "save_path": str(save_path),
        "n_rows": int(len(submission)),
        "known_branch_rows": int((final_test["final_router_branch"] == "known_model").sum()),
        "known_prefix_branch_rows": int((final_test["final_router_branch"] == "known_prefix_deep_model").sum()),
        "known_user_deep_branch_rows": int((final_test["final_router_branch"] == "known_user_deep_e2e_model").sum()),
        "cold_branch_rows": int((final_test["final_router_branch"] == "cold_model").sum()),
        "enabled_known_deep_bands": list(router_spec["enabled_known_deep_bands"]),
        "prediction_min": int(submission["stars"].min()) if len(submission) else None,
        "prediction_max": int(submission["stars"].max()) if len(submission) else None,
        "prediction_mean": float(submission["stars"].mean()) if len(submission) else None,
    }
    _save_json(artifact_root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
