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

from utils.frozen_embedding_regression import build_review_context_only_frame, load_frozen_embedding_bundle
from utils.gbm_features import ScalarPriors, build_gbm_feature_matrix
from utils.io import get_default_data_dir, load_test_reviews


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def _priors_from_json(data: dict[str, Any]) -> ScalarPriors:
    return ScalarPriors(
        global_mean=float(data["global_mean"]),
        user_mean={k: float(v) for k, v in data["user_mean"].items()},
        user_std={k: float(v) for k, v in data["user_std"].items()},
        user_count={k: int(v) for k, v in data["user_count"].items()},
        business_mean={k: float(v) for k, v in data["business_mean"].items()},
        business_std={k: float(v) for k, v in data["business_std"].items()},
        business_count={k: int(v) for k, v in data["business_count"].items()},
    )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2] / "artifacts" / "gbm_submission_v1"
    parser = argparse.ArgumentParser(description="Generate a submission from the trained GBM model.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--artifact-root", type=Path, default=default_root)
    parser.add_argument("--save-path", type=Path, default=default_root / "submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root

    booster = joblib.load(root / "model.pkl")
    priors = _priors_from_json(_load_json(root / "scalar_priors.json"))
    scaler = _load_json(root / "review_context_scaler.json")
    embedding_root = Path(_load_json(root / "embedding_root.json")["embedding_root"])

    bundle = load_frozen_embedding_bundle(embedding_root)
    min_timestamp = pd.Timestamp(scaler["min_timestamp"])
    rc_means = np.array(scaler["means"], dtype=np.float32)
    rc_stds = np.array(scaler["stds"], dtype=np.float32)

    test_reviews = load_test_reviews(args.data_dir)
    test_frame = build_review_context_only_frame(test_reviews)
    x_test, _ = build_gbm_feature_matrix(
        test_frame,
        bundle,
        priors,
        review_context_min_timestamp=min_timestamp,
        review_context_means=rc_means,
        review_context_stds=rc_stds,
    )

    raw_pred = np.clip(booster.predict(x_test).astype(np.float32), 1.0, 5.0)
    rounded_pred = _round_half_up(raw_pred).clip(1, 5).astype(np.int32)

    submission = pd.DataFrame({"review_id": test_frame["review_id"].astype(str), "stars": rounded_pred})
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.save_path, index=False)

    summary = {
        "artifact_root": str(root),
        "embedding_root": str(embedding_root),
        "save_path": str(args.save_path),
        "n_rows": int(len(submission)),
        "n_new_user_rows": int((~test_frame["user"].isin(list(priors.user_count))).sum()),
        "prediction_min": int(rounded_pred.min()),
        "prediction_max": int(rounded_pred.max()),
        "prediction_mean_raw": float(raw_pred.mean()),
        "prediction_mean_rounded": float(rounded_pred.mean()),
    }
    _save_json(root / "submission_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
