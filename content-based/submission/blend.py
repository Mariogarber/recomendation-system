from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils.io import get_default_data_dir, load_test_reviews, load_train_reviews


def _round_half_up(values: np.ndarray) -> np.ndarray:
    return np.floor(values + 0.5).astype(np.int32)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1] / "artifacts"
    parser = argparse.ArgumentParser(description="Blend deep and GBM submissions with GBM fallback for cold start.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--deep-submission",
        type=Path,
        default=root / "frozen_embedding_submission_v1" / "submission.csv",
    )
    parser.add_argument(
        "--gbm-submission",
        type=Path,
        default=root / "gbm_submission_v1" / "submission.csv",
    )
    parser.add_argument("--save-root", type=Path, default=root / "blended_submission_v1")
    parser.add_argument("--save-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    save_path = args.save_path or (save_root / "submission.csv")

    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)
    known_train_users = set(train_reviews["user_id"].astype(str))
    test_frame = test_reviews[["review_id", "user_id"]].copy()
    test_frame["user_id"] = test_frame["user_id"].astype(str)

    deep_submission = pd.read_csv(args.deep_submission, low_memory=False).rename(columns={"stars": "deep_stars"})
    gbm_submission = pd.read_csv(args.gbm_submission, low_memory=False).rename(columns={"stars": "gbm_stars"})

    merged = test_frame.merge(deep_submission, on="review_id", how="left", validate="one_to_one")
    merged = merged.merge(gbm_submission, on="review_id", how="left", validate="one_to_one")
    if merged["deep_stars"].isna().any() or merged["gbm_stars"].isna().any():
        raise RuntimeError("Deep or GBM submission is missing predictions for some review_id values.")

    deep_stars = merged["deep_stars"].to_numpy(dtype=np.float32)
    gbm_stars = merged["gbm_stars"].to_numpy(dtype=np.float32)
    known_mask = merged["user_id"].isin(known_train_users).to_numpy(dtype=bool)

    blended = np.where(
        known_mask,
        _round_half_up((deep_stars + gbm_stars) / 2.0),
        gbm_stars.astype(np.int32),
    ).clip(1, 5)

    submission = pd.DataFrame({"review_id": merged["review_id"], "stars": blended.astype(np.int32)})
    submission.to_csv(save_path, index=False)

    debug = merged.copy()
    debug["known_train_user"] = known_mask
    debug["final_stars"] = blended.astype(np.int32)
    debug.to_csv(save_root / "submission_debug.csv", index=False)

    summary = {
        "deep_submission": str(args.deep_submission),
        "gbm_submission": str(args.gbm_submission),
        "save_path": str(save_path),
        "n_rows": int(len(submission)),
        "n_known_user_rows": int(known_mask.sum()),
        "n_fallback_rows": int((~known_mask).sum()),
        "final_min": int(submission["stars"].min()),
        "final_max": int(submission["stars"].max()),
        "final_mean": float(submission["stars"].mean()),
    }
    (save_root / "submission_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
