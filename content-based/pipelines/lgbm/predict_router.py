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

from utils.io import get_default_data_dir, load_businesses, load_test_reviews, load_train_reviews, load_users
from utils.lgbm_known_prefix_deep_features import (
    build_known_prefix_eval_frame,
    load_known_prefix_embedding_bundle,
    resolve_router_branches,
)
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
    parser = argparse.ArgumentParser(description="Generate a submission from the routed raw-core LightGBM stack.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_raw_router_prefix_deep_v1",
    )
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root
    save_path = args.save_path or root / "submission.csv"

    spec = joblib.load(root / "submission_router_spec.joblib")
    known_booster = lgb.Booster(model_file=str(root / "known_submission_model.txt"))
    known_prefix_booster = lgb.Booster(model_file=str(root / "known_prefix_submission_model.txt"))
    cold_booster = lgb.Booster(model_file=str(root / "cold_submission_model.txt"))

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)
    base_test_frame = build_raw_feature_frame(test_reviews, users_df, businesses_df, spec.base_spec)
    router_test_frame = build_router_feature_frame(test_reviews, users_df, businesses_df, spec)
    user_count_lookup = spec.base_spec.user_priors_table.set_index("user_id")["user_train_count"].to_dict()
    base_test_frame["history_band"] = pd.Series(
        [history_band_from_count(int(user_count_lookup.get(user_id, 0))) for user_id in base_test_frame["user"].astype(str)],
        index=base_test_frame.index,
        dtype="string",
    )
    router_test_frame["history_band"] = base_test_frame["history_band"]
    known_feature_columns = [
        column
        for column in spec.base_spec.feature_columns
        if column not in {"user_known_in_train", "business_known_in_train"}
    ]

    known_raw = np.clip(
        known_booster.predict(
            base_test_frame[known_feature_columns].copy(),
            num_iteration=known_booster.best_iteration or known_booster.current_iteration() or known_booster.num_trees(),
        ).astype(np.float32),
        1.0,
        5.0,
    )
    cold_raw = np.clip(
        cold_booster.predict(
            router_test_frame[spec.cold_feature_columns].copy(),
            num_iteration=cold_booster.best_iteration or cold_booster.current_iteration() or cold_booster.num_trees(),
        ).astype(np.float32),
        1.0,
        5.0,
    )
    known_prefix_raw = np.full(len(base_test_frame), np.nan, dtype=np.float32)
    if spec.enabled_known_prefix_bands:
        known_prefix_target_frame = base_test_frame[
            (base_test_frame["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5)
            & (base_test_frame["history_band"].astype(str).isin(spec.enabled_known_prefix_bands))
        ].copy()
        prefix_bundle = load_known_prefix_embedding_bundle(spec.known_prefix_embedding_root)
        known_prefix_eval_raw = build_known_prefix_eval_frame(
            test_reviews[test_reviews["review_id"].astype(str).isin(known_prefix_target_frame["review_id"].astype(str))].copy(),
            train_reviews,
            prefix_bundle,
            max_history_len=spec.known_prefix_max_history_len,
            target_history_bands=tuple(spec.enabled_known_prefix_bands),
        )
        known_prefix_eval = _merge_known_prefix_features(known_prefix_target_frame, known_prefix_eval_raw)
        known_prefix_raw_subset = (
            np.clip(
                known_prefix_booster.predict(
                    known_prefix_eval[spec.known_prefix_feature_columns].copy(),
                    num_iteration=known_prefix_booster.best_iteration
                    or known_prefix_booster.current_iteration()
                    or known_prefix_booster.num_trees(),
                ).astype(np.float32),
                1.0,
                5.0,
            )
            if not known_prefix_eval.empty
            else np.empty(0, dtype=np.float32)
        )
        known_prefix_raw = _apply_prediction_to_frame(
            frame=base_test_frame,
            subset_frame=known_prefix_eval,
            prediction=known_prefix_raw_subset,
        )
    router_branches = resolve_router_branches(
        user_known_mask=router_test_frame["user_known_in_train"].to_numpy(dtype=np.float32) > 0.5,
        history_band=router_test_frame["history_band"],
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

    submission = pd.DataFrame(
        {
            "review_id": router_test_frame["review_id"].astype(str),
            "stars": router_rounded,
        }
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=False)

    summary = {
        "artifact_root": str(root),
        "save_path": str(save_path),
        "n_rows": int(len(submission)),
        "known_branch_rows": int((router_branches == "known_model").sum()),
        "known_prefix_branch_rows": int((router_branches == "known_prefix_deep_model").sum()),
        "cold_branch_rows": int((router_branches == "cold_model").sum()),
        "enabled_known_prefix_bands": list(spec.enabled_known_prefix_bands),
        "prediction_min": int(router_rounded.min()) if len(router_rounded) else None,
        "prediction_max": int(router_rounded.max()) if len(router_rounded) else None,
        "prediction_mean_raw": float(router_raw.mean()) if len(router_raw) else None,
        "prediction_mean_rounded": float(router_rounded.mean()) if len(router_rounded) else None,
    }
    _save_json(root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
