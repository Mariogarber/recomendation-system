from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(slots=True)
class DeepUserEncoderArchitecture:
    business_input_dim: int
    metadata_input_dim: int
    embedding_dim: int = 128
    business_hidden_dim: int = 256
    rating_hidden_dim: int = 16
    metadata_hidden_dim: int = 32
    scorer_hidden_dim: int = 128
    business_hidden_layers: tuple[int, ...] = ()
    rating_hidden_layers: tuple[int, ...] = ()
    metadata_hidden_layers: tuple[int, ...] = ()
    scorer_hidden_layers: tuple[int, ...] = ()
    dropout: float = 0.1
    history_shrinkage_temperature: float = 3.0
    rating_modulation_scale: float = 0.35


def _build_mlp(
    *,
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    dropout: float,
    output_activation: bool = False,
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


class DeepUserRatingModel(nn.Module):
    """
    Rating-aware user encoder trained to predict ratings for candidate businesses.

    The model learns:
    - a dense business projection ("business tower")
    - a metadata-driven base user embedding
    - a history residual that is shrunk when user history is short
    - a set-based aggregation of historical item embeddings where ratings act as a light modulation
    - a final rating regressor on top of user and candidate business embeddings
    """

    def __init__(self, architecture: DeepUserEncoderArchitecture) -> None:
        super().__init__()
        self.architecture = architecture

        business_hidden_layers = (
            architecture.business_hidden_layers
            if architecture.business_hidden_layers
            else (architecture.business_hidden_dim, architecture.business_hidden_dim)
        )
        self.business_tower = _build_mlp(
            input_dim=architecture.business_input_dim,
            hidden_dims=business_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=False,
        )
        rating_hidden_layers = (
            architecture.rating_hidden_layers
            if architecture.rating_hidden_layers
            else (architecture.rating_hidden_dim,)
        )
        self.rating_encoder = _build_mlp(
            input_dim=2,
            hidden_dims=rating_hidden_layers,
            output_dim=architecture.rating_hidden_dim,
            dropout=0.0,
            output_activation=True,
        )
        self.history_content_gate = nn.Sequential(
            nn.Linear(architecture.embedding_dim, architecture.business_hidden_dim),
            nn.ReLU(),
            nn.Linear(architecture.business_hidden_dim, 1),
        )
        self.history_rating_gate = nn.Sequential(
            nn.Linear(architecture.rating_hidden_dim, max(architecture.rating_hidden_dim, 8)),
            nn.ReLU(),
            nn.Linear(max(architecture.rating_hidden_dim, 8), 1),
        )
        self.history_residual_encoder = nn.Sequential(
            nn.Linear(architecture.embedding_dim + architecture.rating_hidden_dim + 2, architecture.business_hidden_dim),
            nn.ReLU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(architecture.business_hidden_dim, architecture.embedding_dim),
            nn.ReLU(),
        )

        if architecture.metadata_input_dim > 0:
            metadata_hidden_layers = (
                architecture.metadata_hidden_layers
                if architecture.metadata_hidden_layers
                else (max(architecture.metadata_hidden_dim * 2, 32),)
            )
            self.metadata_encoder = _build_mlp(
                input_dim=architecture.metadata_input_dim,
                hidden_dims=metadata_hidden_layers,
                output_dim=architecture.metadata_hidden_dim,
                dropout=architecture.dropout,
                output_activation=True,
            )
            metadata_out_dim = architecture.metadata_hidden_dim
        else:
            self.metadata_encoder = None
            metadata_out_dim = 0

        if metadata_out_dim > 0:
            self.base_user_encoder = nn.Sequential(
                nn.Linear(metadata_out_dim, architecture.embedding_dim),
                nn.ReLU(),
                nn.Dropout(architecture.dropout),
                nn.Linear(architecture.embedding_dim, architecture.embedding_dim),
                nn.ReLU(),
            )
        else:
            self.base_user_encoder = None

        self.default_user_embedding = nn.Parameter(torch.zeros(architecture.embedding_dim))
        self.history_shrinkage_gate = nn.Sequential(
            nn.Linear((architecture.embedding_dim * 2) + 2, architecture.scorer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(architecture.scorer_hidden_dim, 1),
        )
        self.user_fusion = nn.Sequential(
            nn.Linear(architecture.embedding_dim * 2, architecture.scorer_hidden_dim),
            nn.ReLU(),
            nn.Dropout(architecture.dropout),
            nn.Linear(architecture.scorer_hidden_dim, architecture.embedding_dim),
            nn.ReLU(),
        )
        scorer_hidden_layers = (
            architecture.scorer_hidden_layers
            if architecture.scorer_hidden_layers
            else (architecture.scorer_hidden_dim, architecture.scorer_hidden_dim)
        )
        self.scorer = _build_mlp(
            input_dim=(architecture.embedding_dim * 3) + 1,
            hidden_dims=scorer_hidden_layers,
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )

    def encode_business(self, business_features: torch.Tensor) -> torch.Tensor:
        return self.business_tower(business_features)

    def encode_user(
        self,
        history_business_features: torch.Tensor,
        history_rating_features: torch.Tensor,
        history_mask: torch.Tensor,
        user_metadata: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, history_len, feature_dim = history_business_features.shape
        history_emb = self.encode_business(history_business_features.reshape(batch_size * history_len, feature_dim))
        history_emb = history_emb.reshape(batch_size, history_len, self.architecture.embedding_dim)

        rating_emb = self.rating_encoder(history_rating_features)
        history_mask_float = history_mask.float()
        history_count = history_mask_float.sum(dim=1, keepdim=True)
        history_presence = history_count.gt(0).float()

        content_gate_logits = self.history_content_gate(history_emb).squeeze(-1)
        rating_gate_logits = self.history_rating_gate(rating_emb).squeeze(-1) * self.architecture.rating_modulation_scale
        gate_weights = torch.sigmoid(content_gate_logits + rating_gate_logits) * history_mask_float
        denominators = gate_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        history_context = (history_emb * gate_weights.unsqueeze(-1)).sum(dim=1) / denominators

        masked_history_count = history_count.clamp_min(1.0)
        rating_summary = (rating_emb * history_mask_float.unsqueeze(-1)).sum(dim=1) / masked_history_count
        history_fraction = history_count / max(float(history_len), 1.0)
        history_log_count = torch.log1p(history_count)
        history_count_features = torch.cat([history_fraction, history_log_count], dim=1)

        history_residual_input = torch.cat(
            [history_context, rating_summary, history_count_features],
            dim=1,
        )
        history_residual = self.history_residual_encoder(history_residual_input) * history_presence

        if (
            self.metadata_encoder is not None
            and self.base_user_encoder is not None
            and user_metadata is not None
            and user_metadata.shape[1] > 0
        ):
            metadata_context = self.metadata_encoder(user_metadata)
            base_user_embedding = self.base_user_encoder(metadata_context)
        else:
            base_user_embedding = torch.zeros(
                batch_size,
                self.architecture.embedding_dim,
                device=history_business_features.device,
                dtype=history_business_features.dtype,
            )

        base_user_embedding = base_user_embedding + self.default_user_embedding.unsqueeze(0)

        learned_history_mix = torch.sigmoid(
            self.history_shrinkage_gate(
                torch.cat([base_user_embedding, history_residual, history_count_features], dim=1)
            )
        )
        count_shrinkage = history_count / (history_count + self.architecture.history_shrinkage_temperature)
        history_mix = learned_history_mix * count_shrinkage
        scaled_history_residual = history_mix * history_residual

        return self.user_fusion(torch.cat([base_user_embedding, scaled_history_residual], dim=1))

    def forward(
        self,
        history_business_features: torch.Tensor,
        history_rating_features: torch.Tensor,
        history_mask: torch.Tensor,
        user_metadata: torch.Tensor | None,
        candidate_business_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_embedding = self.encode_user(
            history_business_features=history_business_features,
            history_rating_features=history_rating_features,
            history_mask=history_mask,
            user_metadata=user_metadata,
        )
        candidate_embedding = self.encode_business(candidate_business_features)
        interaction_dot = (user_embedding * candidate_embedding).sum(dim=1, keepdim=True)
        scorer_input = torch.cat(
            [
                user_embedding,
                candidate_embedding,
                torch.abs(user_embedding - candidate_embedding),
                interaction_dot,
            ],
            dim=1,
        )
        rating_prediction = self.scorer(scorer_input).squeeze(-1)
        return rating_prediction, user_embedding, candidate_embedding
