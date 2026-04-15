from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from model.frozen_embedding_regressor import FrozenEmbeddingRegressor, FrozenEmbeddingRegressorArchitecture
from train_frozen_embedding_regressor import _resolve_device
from utils.frozen_embedding_regression import (
    attach_embedding_indices,
    build_review_context_only_frame,
    load_frozen_embedding_bundle,
    transform_review_context_features,
)
from utils.io import get_default_data_dir, load_test_reviews


class FrozenInferenceDataset(Dataset):
    def __init__(
        self,
        *,
        user_idx: np.ndarray,
        business_idx: np.ndarray,
        review_context: np.ndarray,
    ) -> None:
        self.user_idx = user_idx.astype(np.int64, copy=False)
        self.business_idx = business_idx.astype(np.int64, copy=False)
        self.review_context = review_context.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self.user_idx))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "user_idx": torch.tensor(self.user_idx[index], dtype=torch.long),
            "business_idx": torch.tensor(self.business_idx[index], dtype=torch.long),
            "review_context": torch.from_numpy(self.review_context[index]),
        }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    default_artifact_root = Path(__file__).resolve().parents[2] / "artifacts" / "frozen_embedding_submission_v1"
    parser = argparse.ArgumentParser(
        description="Generate a rounded competition submission from the final frozen embedding regressor."
    )
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--artifact-root", type=Path, default=default_artifact_root)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=default_artifact_root / "submission.csv",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root
    checkpoint_path = artifact_root / "checkpoint.pt"
    review_context_summary = _load_json(artifact_root / "review_context_summary.json")
    train_summary = _load_json(artifact_root / "train_summary.json")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    bundle = load_frozen_embedding_bundle(train_summary["embedding_root"])

    test_reviews = load_test_reviews(args.data_dir)
    test_frame = build_review_context_only_frame(test_reviews)
    indexed_frame, join_summary = attach_embedding_indices(test_frame, bundle)
    if int(join_summary["kept_rows"]) != int(join_summary["total_rows"]):
        raise RuntimeError(
            "Submission inference requires full coverage, but some test rows are missing user/business embeddings: "
            f"{join_summary}"
        )

    review_context = transform_review_context_features(indexed_frame, context_summary=review_context_summary)
    dataset = FrozenInferenceDataset(
        user_idx=indexed_frame["user_idx"].to_numpy(dtype=np.int32),
        business_idx=indexed_frame["business_idx"].to_numpy(dtype=np.int32),
        review_context=review_context,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    architecture = FrozenEmbeddingRegressorArchitecture(**checkpoint["architecture"])
    device = _resolve_device(args.device)
    model = FrozenEmbeddingRegressor(architecture).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    user_embeddings = torch.tensor(bundle.user_embeddings, dtype=torch.float32, device=device)
    business_embeddings = torch.tensor(bundle.business_embeddings, dtype=torch.float32, device=device)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            user_idx = batch["user_idx"].to(device)
            business_idx = batch["business_idx"].to(device)
            batch_review_context = batch["review_context"].to(device)
            batch_pred = model(
                user_embedding=user_embeddings[user_idx],
                business_embedding=business_embeddings[business_idx],
                review_context=batch_review_context if batch_review_context.shape[1] > 0 else None,
            )
            batch_pred = torch.clamp(batch_pred, 1.0, 5.0)
            preds.append(batch_pred.detach().cpu().numpy())

    raw_prediction = np.concatenate(preds).astype(np.float32)
    rounded_prediction = np.rint(raw_prediction).clip(1, 5).astype(np.int32)

    submission = pd.DataFrame(
        {
            "ids": indexed_frame["review_id"].astype(str).to_numpy(),
            "prediction": rounded_prediction,
        }
    )
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.save_path, index=False)

    payload = {
        "artifact_root": str(artifact_root),
        "embedding_root": train_summary["embedding_root"],
        "checkpoint_path": str(checkpoint_path),
        "save_path": str(args.save_path),
        "n_rows": int(len(submission)),
        "join_summary": join_summary,
        "prediction_min": int(rounded_prediction.min()) if len(rounded_prediction) else None,
        "prediction_max": int(rounded_prediction.max()) if len(rounded_prediction) else None,
        "prediction_mean_raw": float(raw_prediction.mean()) if len(raw_prediction) else None,
        "prediction_mean_rounded": float(rounded_prediction.mean()) if len(rounded_prediction) else None,
    }
    with (artifact_root / "submission_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
