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
from utils.lgbm_raw_router_features import build_router_feature_frame
from utils.lgbm_tabular_moe import (
    TABULAR_BAND_TO_EXPERT,
    TabularMoESpec,
    apply_tabular_blend,
    collapse_history_band,
    compute_tabular_baseline_prediction,
    eval_prefix_frame,
    resolve_tabular_router_branches,
)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _extract_feature_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return frame[feature_columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission from the prefix-safe four-expert tabular MoE.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_tabular_moe_v1",
    )
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root
    save_path = args.save_path or root / "submission.csv"

    spec = joblib.load(root / "submission_tabular_moe_spec.joblib")
    if not isinstance(spec, TabularMoESpec):
        raise TypeError("Expected a TabularMoESpec artifact.")

    boosters = {
        expert_name: lgb.Booster(model_file=str(root / f"{expert_name}_submission_model.txt"))
        for expert_name in TABULAR_BAND_TO_EXPERT.values()
    }

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)

    context_frame = build_router_feature_frame(train_reviews, users_df, businesses_df, spec.router_spec)
    target_frame = build_router_feature_frame(test_reviews, users_df, businesses_df, spec.router_spec)
    test_frame = eval_prefix_frame(
        target_frame,
        context_frame,
        global_mean=spec.router_spec.base_spec.global_mean,
    )
    test_frame["history_band"] = test_frame["prefix_user_count"].map(lambda value: collapse_history_band(int(value))).astype("string")

    baseline_pred = compute_tabular_baseline_prediction(test_frame, global_mean=spec.router_spec.base_spec.global_mean)
    expert_branches = resolve_tabular_router_branches(test_frame["history_band"])
    expert_pred = np.full(len(test_frame), np.nan, dtype=np.float32)
    for expert_name, booster in boosters.items():
        mask = expert_branches == expert_name
        if not mask.any():
            continue
        feature_columns = spec.feature_columns_by_expert[expert_name]
        best_iteration = booster.best_iteration or booster.current_iteration() or booster.num_trees()
        expert_pred[mask] = np.clip(
            booster.predict(
                _extract_feature_matrix(test_frame.loc[mask], feature_columns),
                num_iteration=best_iteration,
            ).astype(np.float32),
            1.0,
            5.0,
        )

    router_raw = apply_tabular_blend(
        expert_pred=expert_pred,
        baseline_pred=baseline_pred,
        history_band=test_frame["history_band"],
        blend_alpha_by_band=spec.blend_alpha_by_band,
    )
    router_rounded = _round_half_up(router_raw).clip(1, 5).astype(np.int32)

    submission = pd.DataFrame(
        {
            "review_id": test_frame["review_id"].astype(str),
            "stars": router_rounded,
        }
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=False)

    summary = {
        "artifact_root": str(root),
        "save_path": str(save_path),
        "n_rows": int(len(submission)),
        "branch_rows": {
            expert_name: int((expert_branches == expert_name).sum())
            for expert_name in TABULAR_BAND_TO_EXPERT.values()
        },
        "blend_alpha_by_band": spec.blend_alpha_by_band,
        "prediction_min": int(router_rounded.min()) if len(router_rounded) else None,
        "prediction_max": int(router_rounded.max()) if len(router_rounded) else None,
        "prediction_mean_raw": float(router_raw.mean()) if len(router_raw) else None,
        "prediction_mean_rounded": float(router_rounded.mean()) if len(router_rounded) else None,
    }
    _save_json(root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
