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

from utils.io import get_default_data_dir, load_businesses, load_test_reviews, load_users
from utils.lgbm_raw_features import build_raw_feature_frame


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a competition submission from a raw-features LightGBM model.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_raw_features_v1",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root
    save_path = args.save_path or root / "submission.csv"

    spec = joblib.load(root / "submission_spec.joblib")
    booster = lgb.Booster(model_file=str(root / "submission_model.txt"))

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)
    test_frame = build_raw_feature_frame(test_reviews, users_df, businesses_df, spec)
    x_test = test_frame[spec.feature_columns].copy()

    raw_pred = np.clip(
        booster.predict(x_test, num_iteration=booster.best_iteration or booster.current_iteration() or booster.num_trees()).astype(np.float32),
        1.0,
        5.0,
    )
    rounded_pred = _round_half_up(raw_pred).clip(1, 5).astype(np.int32)

    submission = pd.DataFrame({
        "review_id": test_frame["review_id"].astype(str),
        "stars": rounded_pred,
    })
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(save_path, index=False)

    summary = {
        "artifact_root": str(root),
        "save_path": str(save_path),
        "feature_set": spec.feature_set,
        "n_rows": int(len(submission)),
        "n_new_user_rows": int((test_frame["user_known_in_train"] == 0).sum()) if "user_known_in_train" in test_frame.columns else None,
        "prediction_min": int(rounded_pred.min()) if len(rounded_pred) else None,
        "prediction_max": int(rounded_pred.max()) if len(rounded_pred) else None,
        "prediction_mean_raw": float(raw_pred.mean()) if len(raw_pred) else None,
        "prediction_mean_rounded": float(rounded_pred.mean()) if len(rounded_pred) else None,
    }
    _save_json(root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
