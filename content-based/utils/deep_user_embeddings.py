from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.deep_user_encoder import DeepUserEncoderArchitecture, DeepUserRatingModel

from .business_representation import BusinessRepresentationBundle
from .io import canonicalize_reviews
from .split import temporal_train_validation_split
from .user_representation import build_safe_user_metadata_block


@dataclass(slots=True)
class DeepUserEmbeddingConfig:
    business_view: str = "full"
    business_blocks: list[str] | None = None
    include_metadata: bool = True
    temporal_val_size: float = 0.2
    max_history_len: int = 20
    embedding_dim: int = 128
    business_hidden_dim: int = 384
    rating_hidden_dim: int = 32
    metadata_hidden_dim: int = 64
    scorer_hidden_dim: int = 256
    business_hidden_layers: tuple[int, ...] = (512, 384, 256)
    rating_hidden_layers: tuple[int, ...] = (64, 32)
    metadata_hidden_layers: tuple[int, ...] = (128, 64)
    scorer_hidden_layers: tuple[int, ...] = (256, 128)
    dropout: float = 0.15
    batch_size: int = 768
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    max_epochs: int = 20
    early_stopping_patience: int = 4
    random_seed: int = 42
    device: str = "auto"


@dataclass(slots=True)
class DeepUserEmbeddingBundle:
    user_ids: pd.Series
    business_ids: pd.Series
    user_embedding_matrix: np.ndarray
    business_embedding_matrix: np.ndarray
    clean_user_table: pd.DataFrame
    clean_business_table: pd.DataFrame
    user_feature_metadata: pd.DataFrame
    business_feature_metadata: pd.DataFrame
    training_summary: dict[str, Any]
    user_feature_names: list[str]
    business_feature_names: list[str]
    model_state_dict: dict[str, Any]

    def save(self, save_dir: str | Path) -> dict[str, str]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, str] = {}

        user_ids_path = save_dir / "user_deep_ids.csv"
        self.user_ids.to_frame(name="user_id").to_csv(user_ids_path, index=False)
        saved_paths["user_ids"] = str(user_ids_path)

        business_ids_path = save_dir / "business_deep_ids.csv"
        self.business_ids.to_frame(name="business_id").to_csv(business_ids_path, index=False)
        saved_paths["business_ids"] = str(business_ids_path)

        user_embedding_path = save_dir / "user_deep_features.npz"
        np.savez_compressed(user_embedding_path, embeddings=self.user_embedding_matrix.astype(np.float32))
        saved_paths["user_embeddings"] = str(user_embedding_path)

        business_embedding_path = save_dir / "business_deep_features.npz"
        np.savez_compressed(business_embedding_path, embeddings=self.business_embedding_matrix.astype(np.float32))
        saved_paths["business_embeddings"] = str(business_embedding_path)

        feature_names_path = save_dir / "user_deep_feature_names.json"
        feature_names_path.write_text(
            json.dumps(
                {
                    "user_deep_features": self.user_feature_names,
                    "business_deep_features": self.business_feature_names,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        saved_paths["feature_names"] = str(feature_names_path)

        user_feature_metadata_path = save_dir / "user_deep_feature_metadata.csv"
        self.user_feature_metadata.to_csv(user_feature_metadata_path, index=False)
        saved_paths["user_feature_metadata"] = str(user_feature_metadata_path)

        business_feature_metadata_path = save_dir / "business_deep_feature_metadata.csv"
        self.business_feature_metadata.to_csv(business_feature_metadata_path, index=False)
        saved_paths["business_feature_metadata"] = str(business_feature_metadata_path)

        summary_path = save_dir / "user_deep_summary.json"
        summary_path.write_text(
            json.dumps(self.training_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        saved_paths["summary"] = str(summary_path)

        clean_user_path = _save_clean_table(self.clean_user_table, save_dir / "user_deep_clean_table")
        clean_business_path = _save_clean_table(self.clean_business_table, save_dir / "business_deep_clean_table")
        saved_paths["clean_user_table"] = str(clean_user_path)
        saved_paths["clean_business_table"] = str(clean_business_path)

        checkpoint_path = save_dir / "deep_user_encoder_checkpoint.pt"
        torch.save(self.model_state_dict, checkpoint_path)
        saved_paths["checkpoint"] = str(checkpoint_path)

        return saved_paths


class _PrefixSampleDataset(Dataset):
    def __init__(
        self,
        *,
        history_item_idx: np.ndarray,
        history_ratings: np.ndarray,
        candidate_item_idx: np.ndarray,
        user_idx: np.ndarray,
        target_ratings: np.ndarray,
    ) -> None:
        self.history_item_idx = history_item_idx.astype(np.int64, copy=False)
        self.history_ratings = history_ratings.astype(np.float32, copy=False)
        self.candidate_item_idx = candidate_item_idx.astype(np.int64, copy=False)
        self.user_idx = user_idx.astype(np.int64, copy=False)
        self.target_ratings = target_ratings.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return int(len(self.candidate_item_idx))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history_item_idx": torch.from_numpy(self.history_item_idx[index]),
            "history_ratings": torch.from_numpy(self.history_ratings[index]),
            "candidate_item_idx": torch.tensor(self.candidate_item_idx[index], dtype=torch.long),
            "user_idx": torch.tensor(self.user_idx[index], dtype=torch.long),
            "target_rating": torch.tensor(self.target_ratings[index], dtype=torch.float32),
        }


class DeepUserEmbeddingBuilder:
    def __init__(self, config: DeepUserEmbeddingConfig | None = None) -> None:
        self.config = config or DeepUserEmbeddingConfig()

    def fit_transform(
        self,
        *,
        train_reviews: pd.DataFrame,
        business_bundle: BusinessRepresentationBundle,
        users_df: pd.DataFrame | None = None,
        test_reviews: pd.DataFrame | None = None,
        target_user_ids: pd.Series | list[str] | np.ndarray | None = None,
    ) -> DeepUserEmbeddingBundle:
        torch.manual_seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        train_canonical = canonicalize_reviews(train_reviews)
        if {"user", "item", "rating", "timestamp"} - set(train_canonical.columns):
            raise ValueError("train_reviews must contain user, item, rating, and timestamp columns")

        business_matrix, _ = self._select_business_view(business_bundle)
        business_dense = business_matrix.toarray().astype(np.float32, copy=False)
        business_index = pd.Series(
            np.arange(len(business_bundle.business_ids), dtype=np.int32),
            index=business_bundle.business_ids.to_numpy(),
        )

        target_user_series = build_target_user_ids(
            users_df=users_df,
            train_reviews=train_reviews,
            test_reviews=test_reviews,
            target_user_ids=target_user_ids,
        )
        user_index = pd.Series(
            np.arange(len(target_user_series), dtype=np.int32),
            index=target_user_series.to_numpy(),
        )
        metadata_matrix, clean_metadata_table, metadata_feature_names = build_safe_user_metadata_block(
            unique_users=target_user_series,
            users_df=users_df,
            train_reviews=train_reviews,
            include_metadata=self.config.include_metadata,
        )
        metadata_dense = metadata_matrix.toarray().astype(np.float32, copy=False)

        interactions = train_canonical[train_canonical["item"].isin(business_index.index)].copy()
        interactions["item_idx"] = interactions["item"].map(business_index).astype(np.int32)
        interactions["user_idx"] = interactions["user"].map(user_index).astype(np.int32)
        interactions = interactions.sort_values(["user", "timestamp", "item"]).reset_index(drop=True)

        train_split, val_split = temporal_train_validation_split(
            interactions,
            val_size=self.config.temporal_val_size,
            timestamp_col="timestamp",
        )

        train_arrays = _build_prefix_training_arrays(
            interactions=train_split,
            max_history_len=self.config.max_history_len,
        )
        val_arrays = _build_fixed_context_arrays(
            target_interactions=val_split,
            context_interactions=train_split,
            max_history_len=self.config.max_history_len,
        )

        rating_min = float(interactions["rating"].min())
        rating_max = float(interactions["rating"].max())

        training_result = self._train_with_validation(
            train_arrays=train_arrays,
            val_arrays=val_arrays,
            business_dense=business_dense,
            metadata_dense=metadata_dense,
            metadata_feature_names=metadata_feature_names,
            business_input_dim=business_dense.shape[1],
            rating_min=rating_min,
            rating_max=rating_max,
        )

        final_epochs = max(1, int(training_result["best_epoch"]))
        full_train_arrays = _build_prefix_training_arrays(
            interactions=interactions,
            max_history_len=self.config.max_history_len,
        )
        final_model = self._train_final_model(
            train_arrays=full_train_arrays,
            business_dense=business_dense,
            metadata_dense=metadata_dense,
            metadata_feature_names=metadata_feature_names,
            business_input_dim=business_dense.shape[1],
            rating_min=rating_min,
            rating_max=rating_max,
            final_epochs=final_epochs,
        )

        export_histories = _build_export_history_arrays(
            target_user_ids=target_user_series,
            interactions=interactions,
            user_index=user_index,
            max_history_len=self.config.max_history_len,
        )
        device = _resolve_device(self.config.device)
        final_model.to(device)

        user_embedding_matrix = self._encode_all_users(
            model=final_model,
            history_item_idx=export_histories["history_item_idx"],
            history_ratings=export_histories["history_ratings"],
            business_dense=business_dense,
            metadata_dense=metadata_dense,
            device=device,
            rating_min=rating_min,
            rating_max=rating_max,
        )
        business_embedding_matrix = self._encode_all_businesses(
            model=final_model,
            business_dense=business_dense,
            device=device,
        )

        clean_user_table, clean_business_table = self._build_clean_tables(
            target_user_series=target_user_series,
            users_df=users_df,
            clean_metadata_table=clean_metadata_table,
            export_histories=export_histories,
            business_ids=business_bundle.business_ids,
            business_embedding_matrix=business_embedding_matrix,
        )
        user_feature_names = [f"user_deep__{index:03d}" for index in range(user_embedding_matrix.shape[1])]
        business_feature_names = [f"business_deep__{index:03d}" for index in range(business_embedding_matrix.shape[1])]
        user_feature_metadata = pd.DataFrame(
            {
                "feature_index": np.arange(len(user_feature_names), dtype=int),
                "feature_name": user_feature_names,
                "source": "deep_user_encoder",
            }
        )
        business_feature_metadata = pd.DataFrame(
            {
                "feature_index": np.arange(len(business_feature_names), dtype=int),
                "feature_name": business_feature_names,
                "source": "business_tower",
            }
        )

        architecture_kwargs = self._build_architecture_kwargs(
            business_input_dim=business_dense.shape[1],
            metadata_input_dim=len(metadata_feature_names),
        )
        architecture_field_names = self._architecture_field_names()
        model_parameter_count = int(sum(param.numel() for param in final_model.parameters()))
        trainable_parameter_count = int(sum(param.numel() for param in final_model.parameters() if param.requires_grad))

        training_summary = {
            "config": _jsonable_value(asdict(self.config)),
            "business_view": self.config.business_view,
            "business_blocks": self.config.business_blocks,
            "architecture_field_names": architecture_field_names,
            "architecture_kwargs": _jsonable_value(architecture_kwargs),
            "model_parameter_count": model_parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "n_train_samples": int(len(train_arrays["candidate_item_idx"])),
            "n_validation_samples": int(len(val_arrays["candidate_item_idx"])),
            "n_full_train_samples": int(len(full_train_arrays["candidate_item_idx"])),
            "best_epoch": int(training_result["best_epoch"]),
            "best_val_mae": float(training_result["best_val_mae"]),
            "best_val_rmse": float(training_result["best_val_rmse"]),
            "final_epochs": int(final_epochs),
            "history_max_len": int(self.config.max_history_len),
            "n_export_users": int(len(target_user_series)),
            "n_businesses": int(len(business_bundle.business_ids)),
            "user_embedding_shape": [int(user_embedding_matrix.shape[0]), int(user_embedding_matrix.shape[1])],
            "business_embedding_shape": [int(business_embedding_matrix.shape[0]), int(business_embedding_matrix.shape[1])],
            "embedding_source_counts": {
                str(key): int(value)
                for key, value in clean_user_table["embedding_source"].value_counts(dropna=False).to_dict().items()
            },
            "history_band_counts": {
                str(key): int(value)
                for key, value in clean_user_table["history_band"].value_counts(dropna=False).to_dict().items()
            },
            "epoch_history": training_result["epoch_history"],
        }

        return DeepUserEmbeddingBundle(
            user_ids=target_user_series.reset_index(drop=True),
            business_ids=business_bundle.business_ids.reset_index(drop=True),
            user_embedding_matrix=user_embedding_matrix.astype(np.float32),
            business_embedding_matrix=business_embedding_matrix.astype(np.float32),
            clean_user_table=clean_user_table,
            clean_business_table=clean_business_table,
            user_feature_metadata=user_feature_metadata,
            business_feature_metadata=business_feature_metadata,
            training_summary=training_summary,
            user_feature_names=user_feature_names,
            business_feature_names=business_feature_names,
            model_state_dict=deepcopy(final_model.state_dict()),
        )

    def _select_business_view(
        self,
        business_bundle: BusinessRepresentationBundle,
    ) -> tuple[sparse.csr_matrix, list[str]]:
        matrix = business_bundle.get_matrix(
            view=self.config.business_view,
            blocks=self.config.business_blocks,
        ).tocsr()
        if self.config.business_blocks is not None:
            mask = business_bundle.feature_metadata["block_name"].isin(self.config.business_blocks)
            feature_names = business_bundle.feature_metadata.loc[mask, "feature_name"].tolist()
        elif self.config.business_view == "content":
            feature_names = business_bundle.content_feature_names
        elif self.config.business_view == "prior":
            feature_names = business_bundle.prior_feature_names
        elif self.config.business_view == "full":
            feature_names = business_bundle.full_feature_names
        else:
            raise ValueError("Unsupported business_view")
        return matrix, feature_names

    def _train_with_validation(
        self,
        *,
        train_arrays: dict[str, np.ndarray],
        val_arrays: dict[str, np.ndarray],
        business_dense: np.ndarray,
        metadata_dense: np.ndarray,
        metadata_feature_names: list[str],
        business_input_dim: int,
        rating_min: float,
        rating_max: float,
    ) -> dict[str, Any]:
        device = _resolve_device(self.config.device)
        model = self._build_model(
            business_input_dim=business_input_dim,
            metadata_input_dim=len(metadata_feature_names),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.SmoothL1Loss()

        train_loader = DataLoader(
            _PrefixSampleDataset(**train_arrays),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            _PrefixSampleDataset(**val_arrays),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        business_tensor = torch.tensor(business_dense, dtype=torch.float32, device=device)
        metadata_tensor = torch.tensor(metadata_dense, dtype=torch.float32, device=device)

        best_state = deepcopy(model.state_dict())
        best_epoch = 1
        best_val_mae = float("inf")
        best_val_rmse = float("inf")
        patience_left = self.config.early_stopping_patience
        epoch_history: list[dict[str, float | int]] = []

        for epoch in range(1, self.config.max_epochs + 1):
            model.train()
            train_losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad()
                batch_tensors = _prepare_batch_tensors(
                    batch=batch,
                    business_tensor=business_tensor,
                    metadata_tensor=metadata_tensor,
                    rating_min=rating_min,
                    rating_max=rating_max,
                )
                predictions, _, _ = model(
                    history_business_features=batch_tensors["history_business_features"],
                    history_rating_features=batch_tensors["history_rating_features"],
                    history_mask=batch_tensors["history_mask"],
                    user_metadata=batch_tensors["user_metadata"],
                    candidate_business_features=batch_tensors["candidate_business_features"],
                )
                loss = loss_fn(predictions, batch_tensors["target_ratings"])
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))

            val_mae, val_rmse = _evaluate_model(
                model=model,
                loader=val_loader,
                business_tensor=business_tensor,
                metadata_tensor=metadata_tensor,
                rating_min=rating_min,
                rating_max=rating_max,
            )
            epoch_history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                    "val_mae": float(val_mae),
                    "val_rmse": float(val_rmse),
                }
            )

            if val_mae < best_val_mae:
                best_val_mae = float(val_mae)
                best_val_rmse = float(val_rmse)
                best_epoch = int(epoch)
                best_state = deepcopy(model.state_dict())
                patience_left = self.config.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        return {
            "best_state": best_state,
            "best_epoch": best_epoch,
            "best_val_mae": best_val_mae,
            "best_val_rmse": best_val_rmse,
            "epoch_history": epoch_history,
        }

    def _train_final_model(
        self,
        *,
        train_arrays: dict[str, np.ndarray],
        business_dense: np.ndarray,
        metadata_dense: np.ndarray,
        metadata_feature_names: list[str],
        business_input_dim: int,
        rating_min: float,
        rating_max: float,
        final_epochs: int,
    ) -> DeepUserRatingModel:
        device = _resolve_device(self.config.device)
        model = self._build_model(
            business_input_dim=business_input_dim,
            metadata_input_dim=len(metadata_feature_names),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.SmoothL1Loss()

        train_loader = DataLoader(
            _PrefixSampleDataset(**train_arrays),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        business_tensor = torch.tensor(business_dense, dtype=torch.float32, device=device)
        metadata_tensor = torch.tensor(metadata_dense, dtype=torch.float32, device=device)

        for _ in range(final_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                batch_tensors = _prepare_batch_tensors(
                    batch=batch,
                    business_tensor=business_tensor,
                    metadata_tensor=metadata_tensor,
                    rating_min=rating_min,
                    rating_max=rating_max,
                )
                predictions, _, _ = model(
                    history_business_features=batch_tensors["history_business_features"],
                    history_rating_features=batch_tensors["history_rating_features"],
                    history_mask=batch_tensors["history_mask"],
                    user_metadata=batch_tensors["user_metadata"],
                    candidate_business_features=batch_tensors["candidate_business_features"],
                )
                loss = loss_fn(predictions, batch_tensors["target_ratings"])
                loss.backward()
                optimizer.step()
        return model

    def _encode_all_users(
        self,
        *,
        model: DeepUserRatingModel,
        history_item_idx: np.ndarray,
        history_ratings: np.ndarray,
        business_dense: np.ndarray,
        metadata_dense: np.ndarray,
        device: torch.device,
        rating_min: float,
        rating_max: float,
    ) -> np.ndarray:
        model.eval()
        business_tensor = torch.tensor(business_dense, dtype=torch.float32, device=device)
        metadata_tensor = torch.tensor(metadata_dense, dtype=torch.float32, device=device)
        outputs: list[np.ndarray] = []
        batch_size = max(self.config.batch_size, 1024)

        with torch.no_grad():
            for start in range(0, len(history_item_idx), batch_size):
                end = min(start + batch_size, len(history_item_idx))
                batch_history_idx = torch.tensor(history_item_idx[start:end], dtype=torch.long, device=device)
                batch_history_ratings = torch.tensor(history_ratings[start:end], dtype=torch.float32, device=device)
                batch_history_mask = batch_history_idx.ge(0)
                history_features = business_tensor[batch_history_idx.clamp_min(0)] * batch_history_mask.unsqueeze(-1)
                rating_features = _build_history_rating_features(
                    history_ratings=batch_history_ratings,
                    history_mask=batch_history_mask,
                    rating_min=rating_min,
                    rating_max=rating_max,
                )
                user_embedding = model.encode_user(
                    history_business_features=history_features,
                    history_rating_features=rating_features,
                    history_mask=batch_history_mask,
                    user_metadata=metadata_tensor[start:end],
                )
                outputs.append(user_embedding.cpu().numpy().astype(np.float32))

        return np.vstack(outputs) if outputs else np.zeros((0, self.config.embedding_dim), dtype=np.float32)

    def _encode_all_businesses(
        self,
        *,
        model: DeepUserRatingModel,
        business_dense: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        model.eval()
        outputs: list[np.ndarray] = []
        batch_size = max(self.config.batch_size, 2048)
        with torch.no_grad():
            for start in range(0, len(business_dense), batch_size):
                end = min(start + batch_size, len(business_dense))
                batch_business = torch.tensor(business_dense[start:end], dtype=torch.float32, device=device)
                business_embedding = model.encode_business(batch_business)
                outputs.append(business_embedding.cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, self.config.embedding_dim), dtype=np.float32)

    def _build_clean_tables(
        self,
        *,
        target_user_series: pd.Series,
        users_df: pd.DataFrame | None,
        clean_metadata_table: pd.DataFrame,
        export_histories: dict[str, np.ndarray],
        business_ids: pd.Series,
        business_embedding_matrix: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        metadata_lookup = clean_metadata_table.copy()
        has_metadata_record = (
            metadata_lookup["user_id"].isin(set(users_df["user_id"]))
            if users_df is not None and "user_id" in users_df.columns
            else False
        )
        metadata_lookup["has_metadata_record"] = has_metadata_record

        clean_user_table = pd.DataFrame(
            {
                "user_id": target_user_series,
                "history_count_train": export_histories["history_count"].astype(np.int32),
            }
        )
        clean_user_table["history_band"] = [
            _history_band_from_count(int(count))
            for count in clean_user_table["history_count_train"].to_numpy(dtype=np.int32)
        ]
        clean_user_table = clean_user_table.merge(metadata_lookup, on="user_id", how="left")
        clean_user_table["embedding_source"] = np.where(
            clean_user_table["history_count_train"].to_numpy(dtype=np.int32) > 0,
            "history",
            np.where(clean_user_table["has_metadata_record"].fillna(False), "metadata_only", "default_only"),
        )

        clean_business_table = pd.DataFrame(
            {
                "business_id": business_ids.to_numpy(),
                "deep_embedding_norm": np.linalg.norm(business_embedding_matrix, axis=1).astype(np.float32),
            }
        )
        return clean_user_table, clean_business_table

    def _build_model(self, *, business_input_dim: int, metadata_input_dim: int) -> DeepUserRatingModel:
        architecture = DeepUserEncoderArchitecture(
            **self._build_architecture_kwargs(
                business_input_dim=business_input_dim,
                metadata_input_dim=metadata_input_dim,
            )
        )
        return DeepUserRatingModel(architecture)

    def _build_architecture_kwargs(self, *, business_input_dim: int, metadata_input_dim: int) -> dict[str, Any]:
        config = asdict(self.config)
        config["business_input_dim"] = business_input_dim
        config["metadata_input_dim"] = metadata_input_dim
        field_names = self._architecture_field_names()
        return {field_name: config[field_name] for field_name in field_names if field_name in config}

    @staticmethod
    def _architecture_field_names() -> list[str]:
        architecture_fields = getattr(DeepUserEncoderArchitecture, "__dataclass_fields__", None)
        if architecture_fields:
            return list(architecture_fields.keys())

        signature = inspect.signature(DeepUserEncoderArchitecture)
        return [
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]


def build_target_user_ids(
    *,
    users_df: pd.DataFrame | None,
    train_reviews: pd.DataFrame | None = None,
    test_reviews: pd.DataFrame | None = None,
    target_user_ids: pd.Series | list[str] | np.ndarray | None = None,
) -> pd.Series:
    if target_user_ids is not None:
        raw = pd.Series(target_user_ids, dtype="object").dropna().drop_duplicates().reset_index(drop=True)
        raw.name = "user_id"
        return raw

    parts: list[pd.Series] = []
    if users_df is not None and "user_id" in users_df.columns:
        parts.append(users_df["user_id"].dropna())
    if train_reviews is not None:
        parts.append(canonicalize_reviews(train_reviews)["user"].dropna())
    if test_reviews is not None:
        parts.append(canonicalize_reviews(test_reviews)["user"].dropna())

    if not parts:
        return pd.Series(dtype="object", name="user_id")

    combined = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    combined.name = "user_id"
    return combined


def _build_prefix_training_arrays(
    *,
    interactions: pd.DataFrame,
    max_history_len: int,
) -> dict[str, np.ndarray]:
    ordered = interactions.sort_values(["user", "timestamp", "item"]).reset_index(drop=True)
    n_samples = len(ordered)
    history_item_idx = np.full((n_samples, max_history_len), -1, dtype=np.int32)
    history_ratings = np.zeros((n_samples, max_history_len), dtype=np.float32)
    candidate_item_idx = ordered["item_idx"].to_numpy(dtype=np.int32)
    user_idx = ordered["user_idx"].to_numpy(dtype=np.int32)
    target_ratings = ordered["rating"].to_numpy(dtype=np.float32)

    for _, group in ordered.groupby("user", sort=False):
        group_indices = group.index.to_numpy(dtype=np.int32)
        item_values = group["item_idx"].to_numpy(dtype=np.int32)
        rating_values = group["rating"].to_numpy(dtype=np.float32)
        for local_idx, global_idx in enumerate(group_indices):
            start = max(0, local_idx - max_history_len)
            history_slice_items = item_values[start:local_idx]
            history_slice_ratings = rating_values[start:local_idx]
            if len(history_slice_items) == 0:
                continue
            history_item_idx[global_idx, : len(history_slice_items)] = history_slice_items
            history_ratings[global_idx, : len(history_slice_ratings)] = history_slice_ratings

    return {
        "history_item_idx": history_item_idx,
        "history_ratings": history_ratings,
        "candidate_item_idx": candidate_item_idx,
        "user_idx": user_idx,
        "target_ratings": target_ratings,
    }


def _build_fixed_context_arrays(
    *,
    target_interactions: pd.DataFrame,
    context_interactions: pd.DataFrame,
    max_history_len: int,
) -> dict[str, np.ndarray]:
    ordered_targets = target_interactions.sort_values(["user", "timestamp", "item"]).reset_index(drop=True)
    n_samples = len(ordered_targets)
    history_item_idx = np.full((n_samples, max_history_len), -1, dtype=np.int32)
    history_ratings = np.zeros((n_samples, max_history_len), dtype=np.float32)
    candidate_item_idx = ordered_targets["item_idx"].to_numpy(dtype=np.int32)
    user_idx = ordered_targets["user_idx"].to_numpy(dtype=np.int32)
    target_ratings = ordered_targets["rating"].to_numpy(dtype=np.float32)

    context_lookup = _build_history_lookup(
        interactions=context_interactions,
        max_history_len=max_history_len,
    )
    for row_idx, user in enumerate(ordered_targets["user"].tolist()):
        items, ratings, _ = context_lookup.get(user, (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0))
        if len(items) == 0:
            continue
        history_item_idx[row_idx, : len(items)] = items
        history_ratings[row_idx, : len(ratings)] = ratings

    return {
        "history_item_idx": history_item_idx,
        "history_ratings": history_ratings,
        "candidate_item_idx": candidate_item_idx,
        "user_idx": user_idx,
        "target_ratings": target_ratings,
    }


def _build_export_history_arrays(
    *,
    target_user_ids: pd.Series,
    interactions: pd.DataFrame,
    user_index: pd.Series,
    max_history_len: int,
) -> dict[str, np.ndarray]:
    history_lookup = _build_history_lookup(
        interactions=interactions,
        max_history_len=max_history_len,
    )
    history_item_idx = np.full((len(target_user_ids), max_history_len), -1, dtype=np.int32)
    history_ratings = np.zeros((len(target_user_ids), max_history_len), dtype=np.float32)
    history_count = np.zeros(len(target_user_ids), dtype=np.int32)

    for row_idx, user_id in enumerate(target_user_ids.to_numpy()):
        items, ratings, count = history_lookup.get(user_id, (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), 0))
        history_count[row_idx] = int(count)
        if len(items) == 0:
            continue
        history_item_idx[row_idx, : len(items)] = items
        history_ratings[row_idx, : len(ratings)] = ratings

    return {
        "history_item_idx": history_item_idx,
        "history_ratings": history_ratings,
        "history_count": history_count,
        "user_idx": target_user_ids.map(user_index).to_numpy(dtype=np.int32),
    }


def _build_history_lookup(
    *,
    interactions: pd.DataFrame,
    max_history_len: int,
) -> dict[Any, tuple[np.ndarray, np.ndarray, int]]:
    ordered = interactions.sort_values(["user", "timestamp", "item"]).reset_index(drop=True)
    lookup: dict[Any, tuple[np.ndarray, np.ndarray, int]] = {}
    for user, group in ordered.groupby("user", sort=False):
        items = group["item_idx"].to_numpy(dtype=np.int32)
        ratings = group["rating"].to_numpy(dtype=np.float32)
        if len(items) > max_history_len:
            items = items[-max_history_len:]
            ratings = ratings[-max_history_len:]
        lookup[user] = (items, ratings, int(len(group)))
    return lookup


def _prepare_batch_tensors(
    *,
    batch: dict[str, torch.Tensor],
    business_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor,
    rating_min: float,
    rating_max: float,
) -> dict[str, torch.Tensor]:
    history_item_idx = batch["history_item_idx"].to(device=business_tensor.device, dtype=torch.long)
    history_ratings = batch["history_ratings"].to(device=business_tensor.device, dtype=torch.float32)
    candidate_item_idx = batch["candidate_item_idx"].to(device=business_tensor.device, dtype=torch.long)
    user_idx = batch["user_idx"].to(device=business_tensor.device, dtype=torch.long)
    target_ratings = batch["target_rating"].to(device=business_tensor.device, dtype=torch.float32)

    history_mask = history_item_idx.ge(0)
    history_business_features = business_tensor[history_item_idx.clamp_min(0)] * history_mask.unsqueeze(-1)
    candidate_business_features = business_tensor[candidate_item_idx]
    history_rating_features = _build_history_rating_features(
        history_ratings=history_ratings,
        history_mask=history_mask,
        rating_min=rating_min,
        rating_max=rating_max,
    )
    user_metadata = metadata_tensor[user_idx] if metadata_tensor.shape[1] > 0 else None

    return {
        "history_business_features": history_business_features,
        "history_rating_features": history_rating_features,
        "history_mask": history_mask,
        "candidate_business_features": candidate_business_features,
        "user_metadata": user_metadata,
        "target_ratings": target_ratings,
    }


def _build_history_rating_features(
    *,
    history_ratings: torch.Tensor,
    history_mask: torch.Tensor,
    rating_min: float,
    rating_max: float,
) -> torch.Tensor:
    rating_range = max(rating_max - rating_min, 1e-6)
    rating_norm = (history_ratings - rating_min) / rating_range
    masked_sum = (history_ratings * history_mask.float()).sum(dim=1, keepdim=True)
    masked_count = history_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    history_mean = masked_sum / masked_count
    rating_centered = history_ratings - history_mean
    rating_features = torch.stack([rating_norm, rating_centered], dim=-1)
    return rating_features * history_mask.unsqueeze(-1)


def _evaluate_model(
    *,
    model: DeepUserRatingModel,
    loader: DataLoader,
    business_tensor: torch.Tensor,
    metadata_tensor: torch.Tensor,
    rating_min: float,
    rating_max: float,
) -> tuple[float, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_tensors = _prepare_batch_tensors(
                batch=batch,
                business_tensor=business_tensor,
                metadata_tensor=metadata_tensor,
                rating_min=rating_min,
                rating_max=rating_max,
            )
            pred, _, _ = model(
                history_business_features=batch_tensors["history_business_features"],
                history_rating_features=batch_tensors["history_rating_features"],
                history_mask=batch_tensors["history_mask"],
                user_metadata=batch_tensors["user_metadata"],
                candidate_business_features=batch_tensors["candidate_business_features"],
            )
            pred = pred.clamp(min=rating_min, max=rating_max)
            predictions.append(pred.cpu().numpy())
            targets.append(batch_tensors["target_ratings"].cpu().numpy())

    y_pred = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    y_true = np.concatenate(targets) if targets else np.array([], dtype=np.float32)
    if len(y_true) == 0:
        return float("nan"), float("nan")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, rmse


def _resolve_device(raw_device: str) -> torch.device:
    if raw_device != "auto":
        return torch.device(raw_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _history_band_from_count(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 20:
        return "6-20"
    return ">20"


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_clean_table(df: pd.DataFrame, filepath_without_suffix: Path) -> Path:
    parquet_path = filepath_without_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError, ValueError):
        csv_path = filepath_without_suffix.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
