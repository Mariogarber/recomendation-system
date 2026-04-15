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
from torch import nn
from torch.utils.data import DataLoader

from pipelines.deep.train_regressor import (
    FrozenInteractionDataset,
    FrozenRegressorConfig,
    _build_model,
    _resolve_device,
    _save_json,
    _seed_everything,
)
from utils.frozen_embedding_regression import (
    attach_embedding_indices,
    build_review_context_features,
    build_review_context_only_frame,
    build_review_interaction_frame,
    load_frozen_embedding_bundle,
    summarize_embedding_join,
)
from utils.io import get_default_data_dir, load_test_reviews, load_train_reviews


def _load_validation_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _config_from_summary(summary: dict[str, Any]) -> FrozenRegressorConfig:
    config = summary["config"]
    return FrozenRegressorConfig(
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        max_epochs=int(config["max_epochs"]),
        early_stopping_patience=int(config["early_stopping_patience"]),
        temporal_val_size=float(config["temporal_val_size"]),
        projection_dim=int(config["projection_dim"]),
        review_projection_dim=int(config["review_projection_dim"]),
        user_hidden_layers=tuple(int(value) for value in config["user_hidden_layers"]),
        business_hidden_layers=tuple(int(value) for value in config["business_hidden_layers"]),
        interaction_hidden_layers=tuple(int(value) for value in config["interaction_hidden_layers"]),
        review_hidden_layers=tuple(int(value) for value in config["review_hidden_layers"]),
        head_hidden_layers=tuple(int(value) for value in config["head_hidden_layers"]),
        dropout=float(config["dropout"]),
        random_seed=int(config["random_seed"]),
        device=str(config["device"]),
    )


def parse_args() -> argparse.Namespace:
    default_source_run = (
        Path(__file__).resolve().parents[2]
        / "artifacts"
        / "frozen_embedding_regressor_v1"
        / "iter04_with_review"
        / "validation_summary.json"
    )
    parser = argparse.ArgumentParser(
        description="Train the final competition regressor over frozen deep embeddings using all original train reviews."
    )
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--source-run-summary", type=Path, default=default_source_run)
    parser.add_argument(
        "--embedding-root",
        type=Path,
        default=None,
        help="Override the embedding root from the selected validation summary.",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "frozen_embedding_submission_v1",
    )
    parser.add_argument(
        "--fixed-epochs",
        type=int,
        default=None,
        help="Override the number of full-train epochs. Defaults to the best_epoch of the source run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override the device from the source run config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_summary = _load_validation_summary(args.source_run_summary)
    config = _config_from_summary(source_summary)
    if args.device is not None:
        config.device = args.device
    _seed_everything(config.random_seed)

    fixed_epochs = int(args.fixed_epochs or source_summary["best_epoch"])
    embedding_root = Path(args.embedding_root) if args.embedding_root is not None else Path(source_summary["embedding_root"])
    include_review_context = bool(source_summary["include_review_context"])

    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    bundle = load_frozen_embedding_bundle(embedding_root)
    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)
    train_frame = build_review_interaction_frame(train_reviews)
    train_frame, train_join_summary = attach_embedding_indices(train_frame, bundle)
    test_join_summary = summarize_embedding_join(build_review_context_only_frame(test_reviews), bundle)

    train_context, _, context_summary = build_review_context_features(
        train_frame=train_frame,
        eval_frame=train_frame,
        include_review_context=include_review_context,
    )
    train_targets = train_frame["rating"].to_numpy(dtype=np.float32)

    dataset = FrozenInteractionDataset(
        user_idx=train_frame["user_idx"].to_numpy(dtype=np.int32),
        business_idx=train_frame["business_idx"].to_numpy(dtype=np.int32),
        review_context=train_context,
        target=train_targets,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    device = _resolve_device(config.device)
    model = _build_model(bundle=bundle, review_context_dim=train_context.shape[1], config=config).to(device)
    user_embeddings = torch.tensor(bundle.user_embeddings, dtype=torch.float32, device=device)
    business_embeddings = torch.tensor(bundle.business_embeddings, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()

    epoch_history: list[dict[str, float | int]] = []
    for epoch in range(1, fixed_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad()
            user_idx = batch["user_idx"].to(device)
            business_idx = batch["business_idx"].to(device)
            review_context = batch["review_context"].to(device)
            target = batch["target"].to(device)

            pred = model(
                user_embedding=user_embeddings[user_idx],
                business_embedding=business_embeddings[business_idx],
                review_context=review_context if review_context.shape[1] > 0 else None,
            )
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        epoch_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            }
        )

    checkpoint_payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": source_summary["architecture"],
        "embedding_root": str(embedding_root),
        "source_run_summary": str(args.source_run_summary),
        "source_experiment_name": source_summary["experiment_name"],
        "include_review_context": include_review_context,
        "fixed_epochs": int(fixed_epochs),
        "config": json.loads(json.dumps(source_summary["config"])),
    }
    torch.save(checkpoint_payload, save_root / "checkpoint.pt")

    train_summary = {
        "objective": "rating_regression",
        "embedding_regime": "frozen",
        "training_mode": "full_train_for_competition_submission",
        "source_run_summary": str(args.source_run_summary),
        "source_experiment_name": source_summary["experiment_name"],
        "embedding_root": str(embedding_root),
        "include_review_context": include_review_context,
        "fixed_epochs": int(fixed_epochs),
        "n_train_rows": int(len(train_frame)),
        "device": str(device),
        "train_join_summary": train_join_summary,
        "test_join_summary": test_join_summary,
        "review_context_summary": context_summary,
        "epoch_history": epoch_history,
    }

    _save_json(save_root / "config.json", source_summary["config"])
    _save_json(save_root / "review_context_summary.json", context_summary)
    _save_json(save_root / "train_summary.json", train_summary)

    print(json.dumps(train_summary, indent=2))


if __name__ == "__main__":
    main()
