from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class KnownUserTwoTowerConfig:
    max_history_len: int = 20
    embedding_dim: int = 128
    business_hidden_layers: tuple[int, ...] = (512, 256)
    event_hidden_layers: tuple[int, ...] = (128,)
    user_hidden_layers: tuple[int, ...] = (128,)
    fusion_hidden_layers: tuple[int, ...] = (256, 128)
    baseline_hidden_layers: tuple[int, ...] = (128, 64)
    cross_hidden_layers: tuple[int, ...] = (512, 256)
    cross_depth: int = 3
    alpha_hidden_dim: int = 64
    categorical_embedding_dim: int = 8
    history_band_embedding_dim: int = 8
    num_attention_heads: int = 4
    dropout: float = 0.15
    correction_loss_weight: float = 0.2
    baseline_loss_weight: float = 0.05
    alpha_regularization_weight: float = 0.01
    band_correction_scales: dict[str, float] | None = None
    band_distillation_weights: dict[str, float] | None = None
    batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    max_epochs: int = 20
    early_stopping_patience: int = 4
    random_seed: int = 42
    device: str = "auto"


@dataclass(slots=True)
class KnownUserTwoTowerArchitecture:
    business_input_dim: int
    user_numeric_input_dim: int
    user_aux_input_dim: int
    baseline_input_dim: int
    event_scalar_dim: int
    history_band_vocab_size: int
    user_categorical_vocab_sizes: tuple[int, ...]
    band_correction_scale_by_id: dict[int, float]
    embedding_dim: int = 128
    business_hidden_layers: tuple[int, ...] = (512, 256)
    event_hidden_layers: tuple[int, ...] = (128,)
    user_hidden_layers: tuple[int, ...] = (128,)
    fusion_hidden_layers: tuple[int, ...] = (256, 128)
    baseline_hidden_layers: tuple[int, ...] = (128, 64)
    cross_hidden_layers: tuple[int, ...] = (512, 256)
    cross_depth: int = 3
    alpha_hidden_dim: int = 64
    categorical_embedding_dim: int = 8
    history_band_embedding_dim: int = 8
    num_attention_heads: int = 4
    dropout: float = 0.15
    global_mean: float = 3.5


def _build_mlp(
    *,
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    dropout: float,
    output_activation: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    if output_activation:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.weight(xl) + xl


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _contract_get(contract: Any, name: str) -> Any:
    if isinstance(contract, dict):
        return contract[name]
    return getattr(contract, name)


def _to_level_vocab(levels: Iterable[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for index, value in enumerate(levels, start=1):
        text = str(value)
        if text not in vocab:
            vocab[text] = index
    return vocab


def build_known_user_two_tower_architecture(contract: Any, config: KnownUserTwoTowerConfig) -> KnownUserTwoTowerArchitecture:
    user_categorical_feature_names = list(_contract_get(contract, "user_categorical_feature_names"))
    user_categorical_levels = _contract_get(contract, "user_categorical_levels")
    user_categorical_vocab_sizes = tuple(len(list(user_categorical_levels[name])) + 1 for name in user_categorical_feature_names)
    history_band_levels = tuple(_contract_get(contract, "history_band_levels"))
    history_band_vocab = _to_level_vocab(history_band_levels)
    band_scale_cfg = config.band_correction_scales or {"1": 0.45, "2-5": 0.75, "6-20": 1.0, ">20": 1.0}
    band_correction_scale_by_id = {
        int(history_band_vocab.get("1", 0)): float(band_scale_cfg.get("1", 0.45)),
        int(history_band_vocab.get("2-5", 0)): float(band_scale_cfg.get("2-5", 0.75)),
        int(history_band_vocab.get("6-20", 0)): float(band_scale_cfg.get("6-20", 1.0)),
        int(history_band_vocab.get(">20", 0)): float(band_scale_cfg.get(">20", 1.0)),
    }
    return KnownUserTwoTowerArchitecture(
        business_input_dim=len(list(_contract_get(contract, "business_feature_names"))),
        user_numeric_input_dim=len(list(_contract_get(contract, "user_numeric_feature_names"))),
        user_aux_input_dim=len(list(_contract_get(contract, "user_aux_feature_names"))),
        baseline_input_dim=len(list(_contract_get(contract, "baseline_feature_names"))),
        event_scalar_dim=len(list(_contract_get(contract, "event_scalar_feature_names"))),
        history_band_vocab_size=len(history_band_levels) + 1,
        user_categorical_vocab_sizes=user_categorical_vocab_sizes,
        band_correction_scale_by_id=band_correction_scale_by_id,
        embedding_dim=config.embedding_dim,
        business_hidden_layers=config.business_hidden_layers,
        event_hidden_layers=config.event_hidden_layers,
        user_hidden_layers=config.user_hidden_layers,
        fusion_hidden_layers=config.fusion_hidden_layers,
        baseline_hidden_layers=config.baseline_hidden_layers,
        cross_hidden_layers=config.cross_hidden_layers,
        cross_depth=config.cross_depth,
        alpha_hidden_dim=config.alpha_hidden_dim,
        categorical_embedding_dim=config.categorical_embedding_dim,
        history_band_embedding_dim=config.history_band_embedding_dim,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout,
        global_mean=float(_contract_get(contract, "global_mean")),
    )


class KnownUserTwoTowerCrossModel(nn.Module):
    def __init__(self, architecture: KnownUserTwoTowerArchitecture) -> None:
        super().__init__()
        self.architecture = architecture

        self.business_tower = _build_mlp(
            input_dim=architecture.business_input_dim,
            hidden_dims=architecture.business_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=False,
        )
        self.event_encoder = _build_mlp(
            input_dim=architecture.embedding_dim + architecture.event_scalar_dim,
            hidden_dims=architecture.event_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.user_categorical_embeddings = nn.ModuleList(
            nn.Embedding(vocab_size, architecture.categorical_embedding_dim)
            for vocab_size in architecture.user_categorical_vocab_sizes
        )
        self.history_band_embedding = nn.Embedding(
            architecture.history_band_vocab_size,
            architecture.history_band_embedding_dim,
        )
        self.user_context_encoder = _build_mlp(
            input_dim=(
                architecture.user_numeric_input_dim
                + architecture.user_aux_input_dim
                + (len(architecture.user_categorical_vocab_sizes) * architecture.categorical_embedding_dim)
                + architecture.history_band_embedding_dim
            ),
            hidden_dims=architecture.user_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.query_projection = _build_mlp(
            input_dim=architecture.embedding_dim * 2,
            hidden_dims=(architecture.embedding_dim,),
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.history_attention = nn.MultiheadAttention(
            embed_dim=architecture.embedding_dim,
            num_heads=architecture.num_attention_heads,
            dropout=architecture.dropout,
            batch_first=True,
        )
        self.user_fusion = _build_mlp(
            input_dim=architecture.embedding_dim * 8,
            hidden_dims=architecture.fusion_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.baseline_head = _build_mlp(
            input_dim=architecture.baseline_input_dim,
            hidden_dims=architecture.baseline_hidden_layers,
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )
        self.cross_input_dim = (architecture.embedding_dim * 9) + architecture.baseline_input_dim + 2 + 15
        self.cross_layers = nn.ModuleList(CrossLayer(self.cross_input_dim) for _ in range(max(architecture.cross_depth, 1)))
        self.cross_projection = _build_mlp(
            input_dim=self.cross_input_dim,
            hidden_dims=architecture.cross_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.correction_head = nn.Linear(architecture.embedding_dim, 1)
        self.alpha_head = _build_mlp(
            input_dim=architecture.user_aux_input_dim + architecture.history_band_embedding_dim + 6,
            hidden_dims=(architecture.alpha_hidden_dim,),
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )

    @classmethod
    def from_contract(
        cls,
        contract: Any,
        config: KnownUserTwoTowerConfig | None = None,
    ) -> "KnownUserTwoTowerCrossModel":
        config = config or KnownUserTwoTowerConfig()
        architecture = build_known_user_two_tower_architecture(contract, config)
        return cls(architecture)

    @staticmethod
    def _safe_attention(
        attention: nn.MultiheadAttention,
        *,
        query: torch.Tensor,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, hidden_dim = tokens.shape
        output = torch.zeros(batch_size, hidden_dim, device=tokens.device, dtype=tokens.dtype)
        if tokens.shape[1] == 0:
            return output
        valid_rows = valid_mask.any(dim=1)
        if not valid_rows.any():
            return output
        attended, _ = attention(
            query[valid_rows].unsqueeze(1),
            tokens[valid_rows],
            tokens[valid_rows],
            key_padding_mask=~valid_mask[valid_rows],
            need_weights=False,
        )
        output[valid_rows] = attended.squeeze(1)
        return output

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        float_mask = mask.to(values.dtype)
        denom = float_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * float_mask.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _last_valid(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, history_len, hidden_dim = values.shape
        count = mask.long().sum(dim=1).clamp_min(1)
        last_idx = (count - 1).clamp(min=0, max=max(history_len - 1, 0))
        gathered = values[torch.arange(batch_size, device=values.device), last_idx]
        return torch.where(mask.any(dim=1).unsqueeze(-1), gathered, torch.zeros(batch_size, hidden_dim, device=values.device, dtype=values.dtype))

    @staticmethod
    def _recency_mean(values: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = weights * mask.to(values.dtype)
        denom = weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (values * weight.unsqueeze(-1)).sum(dim=1) / denom

    def encode_business(self, business_features: torch.Tensor) -> torch.Tensor:
        return self.business_tower(business_features)

    def encode_user_context(
        self,
        *,
        user_numeric_features: torch.Tensor,
        user_aux_features: torch.Tensor,
        user_categorical_ids: torch.Tensor,
        history_band_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        categorical_parts: list[torch.Tensor] = []
        for column_index, embedding in enumerate(self.user_categorical_embeddings):
            categorical_parts.append(embedding(user_categorical_ids[:, column_index].clamp_min(0)))
        band_embedding = self.history_band_embedding(history_band_ids.clamp_min(0))
        encoder_input = torch.cat([user_numeric_features, user_aux_features, *categorical_parts, band_embedding], dim=1)
        return self.user_context_encoder(encoder_input), band_embedding

    def _encode_history(
        self,
        *,
        candidate_vec: torch.Tensor,
        user_context_vec: torch.Tensor,
        history_business_features: torch.Tensor,
        history_rating_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, history_len, business_dim = history_business_features.shape
        history_business_vec = self.business_tower(history_business_features.reshape(batch_size * history_len, business_dim))
        history_business_vec = history_business_vec.reshape(batch_size, history_len, self.architecture.embedding_dim)
        event_input = torch.cat([history_business_vec, history_rating_features], dim=-1)
        history_tokens = self.event_encoder(event_input.reshape(batch_size * history_len, -1)).reshape(batch_size, history_len, self.architecture.embedding_dim)
        mean_memory = self._masked_mean(history_tokens, history_mask)
        recency_memory = self._recency_mean(history_tokens, history_rating_features[:, :, 8].clamp_min(0.0), history_mask)
        positive_mask = history_mask & history_rating_features[:, :, 3].gt(0.0)
        negative_mask = history_mask & history_rating_features[:, :, 4].gt(0.0)
        positive_memory = self._masked_mean(history_tokens, positive_mask)
        negative_memory = self._masked_mean(history_tokens, negative_mask)
        last_memory = self._last_valid(history_tokens, history_mask)
        query = self.query_projection(torch.cat([candidate_vec, user_context_vec], dim=1))
        attention_memory = self._safe_attention(
            self.history_attention,
            query=query,
            tokens=history_tokens,
            valid_mask=history_mask,
        )
        return {
            "history_tokens": history_tokens,
            "mean_memory": mean_memory,
            "recency_memory": recency_memory,
            "attention_memory": attention_memory,
            "positive_memory": positive_memory,
            "negative_memory": negative_memory,
            "last_memory": last_memory,
        }

    @staticmethod
    def _pair_features(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dot = (a * b).sum(dim=1, keepdim=True)
        cosine = F.cosine_similarity(a, b, dim=1, eps=1e-8).unsqueeze(-1)
        l2 = torch.norm(a - b, dim=1, keepdim=True)
        return cosine, dot, l2

    def _band_scale(self, history_band_ids: torch.Tensor) -> torch.Tensor:
        scale = torch.ones_like(history_band_ids, dtype=torch.float32)
        for band_id, band_scale in self.architecture.band_correction_scale_by_id.items():
            if band_id <= 0:
                continue
            scale = torch.where(history_band_ids == int(band_id), torch.full_like(scale, float(band_scale)), scale)
        return scale

    def forward(
        self,
        *,
        candidate_business_features: torch.Tensor,
        history_business_features: torch.Tensor,
        history_rating_features: torch.Tensor,
        history_mask: torch.Tensor,
        user_numeric_features: torch.Tensor,
        user_aux_features: torch.Tensor,
        user_categorical_ids: torch.Tensor,
        history_band_ids: torch.Tensor,
        baseline_features: torch.Tensor,
        incumbent_prediction_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidate_vec = self.encode_business(candidate_business_features)
        user_context_vec, band_embedding = self.encode_user_context(
            user_numeric_features=user_numeric_features,
            user_aux_features=user_aux_features,
            user_categorical_ids=user_categorical_ids,
            history_band_ids=history_band_ids,
        )
        history = self._encode_history(
            candidate_vec=candidate_vec,
            user_context_vec=user_context_vec,
            history_business_features=history_business_features,
            history_rating_features=history_rating_features,
            history_mask=history_mask,
        )
        user_vec = self.user_fusion(
            torch.cat(
                [
                    user_context_vec,
                    history["mean_memory"],
                    history["recency_memory"],
                    history["attention_memory"],
                    history["positive_memory"],
                    history["negative_memory"],
                    history["last_memory"],
                    torch.abs(candidate_vec - history["attention_memory"]),
                ],
                dim=1,
            )
        )
        baseline_raw = self.baseline_head(baseline_features).squeeze(-1)
        baseline_hat = self.architecture.global_mean + (2.0 * torch.tanh(baseline_raw))

        mean_cos, mean_dot, mean_l2 = self._pair_features(candidate_vec, history["mean_memory"])
        recency_cos, recency_dot, recency_l2 = self._pair_features(candidate_vec, history["recency_memory"])
        attn_cos, attn_dot, attn_l2 = self._pair_features(candidate_vec, history["attention_memory"])
        pos_cos, _, pos_l2 = self._pair_features(candidate_vec, history["positive_memory"])
        neg_cos, _, neg_l2 = self._pair_features(candidate_vec, history["negative_memory"])
        similarity_features = torch.cat(
            [
                mean_cos,
                mean_dot,
                mean_l2,
                recency_cos,
                recency_dot,
                recency_l2,
                attn_cos,
                attn_dot,
                attn_l2,
                pos_cos,
                pos_l2,
                neg_cos,
                neg_l2,
                (baseline_hat - incumbent_prediction_raw).unsqueeze(-1),
                incumbent_prediction_raw.unsqueeze(-1),
            ],
            dim=1,
        )
        cross_base = torch.cat(
            [
                user_vec,
                candidate_vec,
                user_context_vec,
                history["mean_memory"],
                history["recency_memory"],
                history["attention_memory"],
                history["positive_memory"],
                history["negative_memory"],
                history["last_memory"],
                baseline_features,
                baseline_hat.unsqueeze(-1),
                incumbent_prediction_raw.unsqueeze(-1),
                similarity_features,
            ],
            dim=1,
        )
        cross = cross_base
        for layer in self.cross_layers:
            cross = layer(cross_base, cross)
        cross_hidden = self.cross_projection(cross)
        correction_hat = torch.tanh(self.correction_head(cross_hidden).squeeze(-1))
        alpha_input = torch.cat(
            [
                user_aux_features,
                band_embedding,
                baseline_hat.unsqueeze(-1),
                incumbent_prediction_raw.unsqueeze(-1),
                attn_cos,
                attn_l2,
                mean_cos,
                recency_cos,
            ],
            dim=1,
        )
        alpha = torch.sigmoid(self.alpha_head(alpha_input).squeeze(-1))
        correction_scale = self._band_scale(history_band_ids).to(correction_hat.device, correction_hat.dtype)
        scaled_correction = correction_scale * correction_hat
        predicted_rating = torch.clamp(incumbent_prediction_raw + (alpha * scaled_correction), 1.0, 5.0)
        return {
            "predicted_rating": predicted_rating,
            "baseline_hat": baseline_hat,
            "correction_hat": scaled_correction,
            "residual_hat": scaled_correction,
            "alpha": alpha,
            "user_vec": user_vec,
            "user_context_vec": user_context_vec,
            "candidate_business_vec": candidate_vec,
            "history_attention_vec": history["attention_memory"],
            "incumbent_prediction_raw": incumbent_prediction_raw,
            "correction_scale": correction_scale,
        }


def compute_known_user_two_tower_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_rating: torch.Tensor,
    incumbent_prediction_raw: torch.Tensor,
    history_band_ids: torch.Tensor,
    config: KnownUserTwoTowerConfig,
) -> torch.Tensor:
    mask = torch.isfinite(target_rating)
    rating_target = torch.nan_to_num(target_rating, nan=0.0)
    rating_pred = outputs["predicted_rating"]
    correction_pred = outputs.get("correction_hat", outputs["residual_hat"])
    correction_target = torch.clamp(target_rating - incumbent_prediction_raw, min=-1.5, max=1.5)
    if mask.any():
        main_loss = F.smooth_l1_loss(rating_pred[mask], rating_target[mask])
        baseline_loss = F.smooth_l1_loss(outputs["baseline_hat"][mask], rating_target[mask])
        correction_loss = F.smooth_l1_loss(correction_pred[mask], correction_target[mask])
    else:
        zero = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)
        main_loss = zero
        baseline_loss = zero
        correction_loss = zero

    distill_weights = torch.ones_like(incumbent_prediction_raw, dtype=rating_pred.dtype)
    distill_config = config.band_distillation_weights or {"1": 0.20, "2-5": 0.10, "6-20": 0.05, ">20": 0.05}
    band_weight_map = {
        2: float(distill_config.get("1", 0.20)),
        3: float(distill_config.get("2-5", 0.10)),
        4: float(distill_config.get("6-20", 0.05)),
        5: float(distill_config.get(">20", 0.05)),
        6: float(distill_config.get("__unknown__", 0.05)),
    }
    for band_id, weight in band_weight_map.items():
        distill_weights = torch.where(history_band_ids == band_id, torch.full_like(distill_weights, weight), distill_weights)
    if mask.any():
        distill_loss = ((rating_pred[mask] - incumbent_prediction_raw[mask]).abs() * distill_weights[mask]).mean()
        alpha_penalty = (outputs["alpha"][mask] * outputs["correction_hat"][mask].abs()).mean()
    else:
        distill_loss = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)
        alpha_penalty = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)
    return (
        main_loss
        + (config.correction_loss_weight * correction_loss)
        + (config.baseline_loss_weight * baseline_loss)
        + distill_loss
        + (config.alpha_regularization_weight * alpha_penalty)
    )
