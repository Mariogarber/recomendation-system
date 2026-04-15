from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.frozen_embedding_regressor import (
    FrozenEmbeddingRegressor,
    FrozenEmbeddingRegressorArchitecture,
)
from utils.frozen_embedding_regression import (
    FrozenEmbeddingBundle,
    attach_embedding_indices,
    build_review_context_features,
    build_review_context_only_frame,
    build_review_interaction_frame,
    compute_band_metrics,
    compute_preference_auc,
    fit_ridge_embedding_baseline,
    load_frozen_embedding_bundle,
    rmse,
    summarize_embedding_join,
)
from utils.io import get_default_data_dir, load_test_reviews, load_train_reviews
from utils.split import temporal_train_validation_split


@dataclass(slots=True)
class FrozenRegressorConfig:
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 5e-5
    max_epochs: int = 20
    early_stopping_patience: int = 4
    temporal_val_size: float = 0.2
    projection_dim: int = 64
    review_projection_dim: int = 16
    user_hidden_layers: tuple[int, ...] = (128,)
    business_hidden_layers: tuple[int, ...] = (128,)
    interaction_hidden_layers: tuple[int, ...] = (128, 64)
    review_hidden_layers: tuple[int, ...] = (16,)
    head_hidden_layers: tuple[int, ...] = (64, 32)
    dropout: float = 0.10
    random_seed: int = 42
    device: str = "auto"


class FrozenInteractionDataset(Dataset):
    def __init__(
        self,
        *,
        user_idx: np.ndarray,
        business_idx: np.ndarray,
        review_context: np.ndarray,
        target: np.ndarray,
    ) -> None:
        self.user_idx = user_idx.astype(np.int64, copy=False)
        self.business_idx = business_idx.astype(np.int64, copy=False)
        self.review_context = review_context.astype(np.float32, copy=False)
        self.target = target.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self.target))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "user_idx": torch.tensor(self.user_idx[index], dtype=torch.long),
            "business_idx": torch.tensor(self.business_idx[index], dtype=torch.long),
            "review_context": torch.from_numpy(self.review_context[index]),
            "target": torch.tensor(self.target[index], dtype=torch.float32),
        }


def _parse_hidden_layers(raw_value: str) -> tuple[int, ...]:
    text = raw_value.strip()
    if not text or text.lower() in {"default", "none", "auto"}:
        return ()
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            raise argparse.ArgumentTypeError("Hidden layer sizes must be positive integers.")
        values.append(dim)
    return tuple(values)


def _resolve_device(raw_value: str) -> torch.device:
    if raw_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_value)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _compute_selection_score(summary: dict[str, Any]) -> tuple[float, float]:
    band_metrics = summary["band_metrics"]
    band_one = next((row["mae"] for row in band_metrics if row["history_band"] == "1"), float("inf"))
    band_two_five = next((row["mae"] for row in band_metrics if row["history_band"] == "2-5"), float("inf"))
    return float(band_one), float(band_two_five)


def _select_best_experiment(candidates: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    ordered = sorted(
        candidates,
        key=lambda summary: (
            float(summary[metric_key]),
            *_compute_selection_score(summary),
        ),
    )
    selected = ordered[0]
    if len(ordered) > 1:
        best = ordered[0]
        runner_up = ordered[1]
        if abs(float(best[metric_key]) - float(runner_up[metric_key])) <= 0.005:
            best_short = _compute_selection_score(best)
            runner_up_short = _compute_selection_score(runner_up)
            if runner_up_short < best_short:
                selected = runner_up
    return selected


def _build_model(
    *,
    bundle: FrozenEmbeddingBundle,
    review_context_dim: int,
    config: FrozenRegressorConfig,
) -> FrozenEmbeddingRegressor:
    architecture = FrozenEmbeddingRegressorArchitecture(
        user_input_dim=int(bundle.user_embeddings.shape[1]),
        business_input_dim=int(bundle.business_embeddings.shape[1]),
        review_context_dim=int(review_context_dim),
        projection_dim=config.projection_dim,
        review_projection_dim=config.review_projection_dim,
        user_hidden_layers=config.user_hidden_layers,
        business_hidden_layers=config.business_hidden_layers,
        interaction_hidden_layers=config.interaction_hidden_layers,
        review_hidden_layers=config.review_hidden_layers,
        head_hidden_layers=config.head_hidden_layers,
        dropout=config.dropout,
    )
    return FrozenEmbeddingRegressor(architecture)


def _evaluate_model(
    *,
    model: FrozenEmbeddingRegressor,
    loader: DataLoader,
    user_embeddings: torch.Tensor,
    business_embeddings: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            user_idx = batch["user_idx"].to(device)
            business_idx = batch["business_idx"].to(device)
            review_context = batch["review_context"].to(device)
            target = batch["target"].to(device)
            pred = model(
                user_embedding=user_embeddings[user_idx],
                business_embedding=business_embeddings[business_idx],
                review_context=review_context if review_context.shape[1] > 0 else None,
            )
            pred = torch.clamp(pred, 1.0, 5.0)
            preds.append(pred.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _train_single_experiment(
    *,
    name: str,
    bundle: FrozenEmbeddingBundle,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    include_review_context: bool,
    config: FrozenRegressorConfig,
    save_dir: Path,
) -> dict[str, Any]:
    device = _resolve_device(config.device)
    train_context, val_context, context_summary = build_review_context_features(
        train_frame=train_frame,
        eval_frame=val_frame,
        include_review_context=include_review_context,
    )

    train_dataset = FrozenInteractionDataset(
        user_idx=train_frame["user_idx"].to_numpy(dtype=np.int32),
        business_idx=train_frame["business_idx"].to_numpy(dtype=np.int32),
        review_context=train_context,
        target=train_frame["rating"].to_numpy(dtype=np.float32),
    )
    val_dataset = FrozenInteractionDataset(
        user_idx=val_frame["user_idx"].to_numpy(dtype=np.int32),
        business_idx=val_frame["business_idx"].to_numpy(dtype=np.int32),
        review_context=val_context,
        target=val_frame["rating"].to_numpy(dtype=np.float32),
    )

    model = _build_model(bundle=bundle, review_context_dim=train_context.shape[1], config=config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.SmoothL1Loss()
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    user_embeddings = torch.tensor(bundle.user_embeddings, dtype=torch.float32, device=device)
    business_embeddings = torch.tensor(bundle.business_embeddings, dtype=torch.float32, device=device)

    best_state = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "architecture": asdict(model.architecture),
    }
    best_epoch = 1
    best_val_mae = float("inf")
    patience_left = config.early_stopping_patience
    epoch_history: list[dict[str, float | int]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
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

        val_pred, val_target = _evaluate_model(
            model=model,
            loader=val_loader,
            user_embeddings=user_embeddings,
            business_embeddings=business_embeddings,
            device=device,
        )
        val_mae = float(np.mean(np.abs(val_target - val_pred)))
        val_rmse = rmse(val_target, val_pred)
        epoch_history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = int(epoch)
            best_state = {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "architecture": asdict(model.architecture),
            }
            patience_left = config.early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state["model_state_dict"])
    val_pred, val_target = _evaluate_model(
        model=model,
        loader=val_loader,
        user_embeddings=user_embeddings,
        business_embeddings=business_embeddings,
        device=device,
    )

    val_eval = val_frame[["review_id", "user", "item", "rating", "timestamp", "history_band", "useful", "funny", "cool"]].copy()
    val_eval["pred"] = val_pred
    pairwise_auc, auc_users = compute_preference_auc(val_eval[["user", "rating", "history_band", "pred"]].copy())
    band_metrics = compute_band_metrics(val_eval[["rating", "history_band", "pred"]].copy())

    checkpoint_path = save_dir / "checkpoint.pt"
    _save_json(save_dir / "config.json", asdict(config))
    torch.save(best_state, checkpoint_path)

    reload_model = _build_model(bundle=bundle, review_context_dim=train_context.shape[1], config=config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reload_model.load_state_dict(checkpoint["model_state_dict"])
    reload_pred, _ = _evaluate_model(
        model=reload_model,
        loader=val_loader,
        user_embeddings=user_embeddings,
        business_embeddings=business_embeddings,
        device=device,
    )
    reload_matches = bool(np.allclose(val_pred, reload_pred, atol=1e-5))

    summary = {
        "experiment_name": name,
        "embedding_root": str(bundle.root),
        "include_review_context": bool(include_review_context),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(np.mean(np.abs(val_target - val_pred))),
        "best_val_rmse": rmse(val_target, val_pred),
        "pairwise_auc": float(pairwise_auc),
        "pairwise_auc_users": int(auc_users),
        "n_train_rows": int(len(train_frame)),
        "n_val_rows": int(len(val_frame)),
        "checkpoint_reload_match": reload_matches,
        "epoch_history": epoch_history,
        "band_metrics": band_metrics.to_dict(orient="records"),
        "config": asdict(config),
        "review_context_summary": context_summary,
        "architecture": best_state["architecture"],
    }
    _save_json(save_dir / "validation_summary.json", summary)
    band_metrics.to_csv(save_dir / "band_metrics.csv", index=False)
    val_eval.to_csv(save_dir / "validation_predictions.csv", index=False)
    _save_json(save_dir / "review_context_summary.json", context_summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a downstream regressor over frozen deep user/business embeddings.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument(
        "--use-test-reviews-as-validation",
        action="store_true",
        help="Use test_reviews.csv as the explicit validation set instead of splitting train_reviews temporally again.",
    )
    parser.add_argument(
        "--primary-embedding-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v3_iter03",
    )
    parser.add_argument(
        "--compare-embedding-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v3_iter04",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "frozen_embedding_regressor_v1",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--temporal-val-size", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--review-projection-dim", type=int, default=16)
    parser.add_argument("--user-hidden-layers", type=_parse_hidden_layers, default=(128,))
    parser.add_argument("--business-hidden-layers", type=_parse_hidden_layers, default=(128,))
    parser.add_argument("--interaction-hidden-layers", type=_parse_hidden_layers, default=(128, 64))
    parser.add_argument("--review-hidden-layers", type=_parse_hidden_layers, default=(16,))
    parser.add_argument("--head-hidden-layers", type=_parse_hidden_layers, default=(64, 32))
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FrozenRegressorConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        temporal_val_size=args.temporal_val_size,
        projection_dim=args.projection_dim,
        review_projection_dim=args.review_projection_dim,
        user_hidden_layers=args.user_hidden_layers,
        business_hidden_layers=args.business_hidden_layers,
        interaction_hidden_layers=args.interaction_hidden_layers,
        review_hidden_layers=args.review_hidden_layers,
        head_hidden_layers=args.head_hidden_layers,
        dropout=args.dropout,
        random_seed=args.seed,
        device=args.device,
    )
    _seed_everything(config.random_seed)

    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)
    _save_json(save_root / "config.json", asdict(config))

    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)
    interactions = build_review_interaction_frame(train_reviews)
    if args.use_test_reviews_as_validation:
        train_split = interactions
        val_split = build_review_interaction_frame(test_reviews)
        split_summary = {
            "mode": "explicit_train_and_validation_files",
            "total_rows": int(len(train_split) + len(val_split)),
            "train_rows": int(len(train_split)),
            "val_rows": int(len(val_split)),
            "train_min_timestamp": pd.Timestamp(train_split["timestamp"].min()).isoformat(),
            "train_max_timestamp": pd.Timestamp(train_split["timestamp"].max()).isoformat(),
            "val_min_timestamp": pd.Timestamp(val_split["timestamp"].min()).isoformat(),
            "val_max_timestamp": pd.Timestamp(val_split["timestamp"].max()).isoformat(),
        }
    else:
        train_split, val_split = temporal_train_validation_split(
            interactions,
            val_size=config.temporal_val_size,
            timestamp_col="timestamp",
        )
        split_summary = {
            "mode": "temporal_split_from_train_reviews",
            "total_rows": int(len(interactions)),
            "train_rows": int(len(train_split)),
            "val_rows": int(len(val_split)),
            "train_min_timestamp": pd.Timestamp(train_split["timestamp"].min()).isoformat(),
            "train_max_timestamp": pd.Timestamp(train_split["timestamp"].max()).isoformat(),
            "val_min_timestamp": pd.Timestamp(val_split["timestamp"].min()).isoformat(),
            "val_max_timestamp": pd.Timestamp(val_split["timestamp"].max()).isoformat(),
        }
    _save_json(save_root / "split_summary.json", split_summary)

    primary_bundle = load_frozen_embedding_bundle(args.primary_embedding_root)
    compare_bundle = load_frozen_embedding_bundle(args.compare_embedding_root)
    test_context_frame = build_review_context_only_frame(test_reviews)

    experiments: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    primary_train, primary_train_join = attach_embedding_indices(train_split, primary_bundle)
    primary_val, primary_val_join = attach_embedding_indices(val_split, primary_bundle)
    primary_test_join = summarize_embedding_join(test_context_frame, primary_bundle)

    ridge_dir = save_root / "ridge_iter03_baseline"
    ridge_dir.mkdir(parents=True, exist_ok=True)
    ridge_summary, ridge_predictions = fit_ridge_embedding_baseline(
        bundle=primary_bundle,
        train_frame=primary_train,
        val_frame=primary_val,
    )
    ridge_predictions.to_csv(ridge_dir / "validation_predictions.csv", index=False)
    ridge_band_metrics = compute_band_metrics(ridge_predictions[["rating", "history_band", "pred"]].copy())
    ridge_band_metrics.to_csv(ridge_dir / "band_metrics.csv", index=False)
    ridge_payload = {
        **ridge_summary,
        "embedding_root": str(primary_bundle.root),
        "train_join_summary": primary_train_join,
        "val_join_summary": primary_val_join,
        "test_join_summary": primary_test_join,
        "band_metrics": ridge_band_metrics.to_dict(orient="records"),
    }
    _save_json(ridge_dir / "validation_summary.json", ridge_payload)
    ranking_rows.append(
        {
            "experiment_name": "ridge_iter03_baseline",
            "embedding_root": str(primary_bundle.root),
            "include_review_context": False,
            "model_type": "ridge_baseline",
            "val_mae": ridge_summary["mae"],
            "val_rmse": ridge_summary["rmse"],
            "pairwise_auc": ridge_summary["pairwise_auc"],
        }
    )

    compare_train, compare_train_join = attach_embedding_indices(train_split, compare_bundle)
    compare_val, compare_val_join = attach_embedding_indices(val_split, compare_bundle)
    compare_test_join = summarize_embedding_join(test_context_frame, compare_bundle)

    experiment_specs = [
        ("iter03_with_review", primary_bundle, True, primary_train, primary_val, primary_train_join, primary_val_join, primary_test_join),
        ("iter03_no_review", primary_bundle, False, primary_train, primary_val, primary_train_join, primary_val_join, primary_test_join),
        ("iter04_with_review", compare_bundle, True, compare_train, compare_val, compare_train_join, compare_val_join, compare_test_join),
    ]

    for name, bundle, include_review_context, train_frame, val_frame, train_join, val_join, test_join in experiment_specs:
        experiment_dir = save_root / name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        summary = _train_single_experiment(
            name=name,
            bundle=bundle,
            train_frame=train_frame,
            val_frame=val_frame,
            include_review_context=include_review_context,
            config=config,
            save_dir=experiment_dir,
        )
        summary["train_join_summary"] = train_join
        summary["val_join_summary"] = val_join
        summary["test_join_summary"] = test_join
        _save_json(experiment_dir / "validation_summary.json", summary)
        experiments.append(summary)
        ranking_rows.append(
            {
                "experiment_name": name,
                "embedding_root": str(bundle.root),
                "include_review_context": bool(include_review_context),
                "model_type": "frozen_mlp_regressor",
                "val_mae": float(summary["best_val_mae"]),
                "val_rmse": float(summary["best_val_rmse"]),
                "pairwise_auc": float(summary["pairwise_auc"]),
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values("val_mae", ascending=True).reset_index(drop=True)
    ranking_df.to_csv(save_root / "experiment_ranking.csv", index=False)

    selected_trainable = _select_best_experiment(experiments, "best_val_mae")
    overall_candidates = [
        {
            "experiment_name": "ridge_iter03_baseline",
            "best_val_mae": ridge_payload["mae"],
            "band_metrics": ridge_payload["band_metrics"],
        },
        *experiments,
    ]
    selected_overall = _select_best_experiment(overall_candidates, "best_val_mae")

    overall_summary = {
        "objective": "rating_regression",
        "embedding_regime": "frozen",
        "primary_embedding_root": str(args.primary_embedding_root),
        "compare_embedding_root": str(args.compare_embedding_root),
        "selected_experiment": selected_overall["experiment_name"],
        "selected_trainable_experiment": selected_trainable["experiment_name"],
        "selection_rule": "lowest val_mae; if within 0.005, prefer lower history-band mae for 1 and 2-5",
        "split_summary": split_summary,
        "ridge_baseline": ridge_payload,
        "experiments": {
            experiment["experiment_name"]: {
                "embedding_root": experiment["embedding_root"],
                "include_review_context": experiment["include_review_context"],
                "best_val_mae": experiment["best_val_mae"],
                "best_val_rmse": experiment["best_val_rmse"],
                "pairwise_auc": experiment["pairwise_auc"],
                "best_epoch": experiment["best_epoch"],
            }
            for experiment in experiments
        },
    }
    _save_json(save_root / "run_summary.json", overall_summary)

    print(json.dumps(overall_summary, indent=2))


if __name__ == "__main__":
    main()
