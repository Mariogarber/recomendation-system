from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(slots=True)
class FrozenEmbeddingRegressorArchitecture:
    user_input_dim: int
    business_input_dim: int
    review_context_dim: int
    projection_dim: int = 64
    review_projection_dim: int = 16
    user_hidden_layers: tuple[int, ...] = (128,)
    business_hidden_layers: tuple[int, ...] = (128,)
    interaction_hidden_layers: tuple[int, ...] = (128, 64)
    review_hidden_layers: tuple[int, ...] = (16,)
    head_hidden_layers: tuple[int, ...] = (64, 32)
    dropout: float = 0.10


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


class FrozenEmbeddingRegressor(nn.Module):
    """
    Rating regressor over frozen user/business deep embeddings plus small review context.
    """

    def __init__(self, architecture: FrozenEmbeddingRegressorArchitecture) -> None:
        super().__init__()
        self.architecture = architecture

        self.user_encoder = _build_mlp(
            input_dim=architecture.user_input_dim,
            hidden_dims=architecture.user_hidden_layers,
            output_dim=architecture.projection_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.business_encoder = _build_mlp(
            input_dim=architecture.business_input_dim,
            hidden_dims=architecture.business_hidden_layers,
            output_dim=architecture.projection_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )

        interaction_input_dim = (architecture.projection_dim * 4) + 2
        interaction_hidden_layers = architecture.interaction_hidden_layers[:-1]
        interaction_output_dim = (
            architecture.interaction_hidden_layers[-1]
            if architecture.interaction_hidden_layers
            else architecture.projection_dim
        )
        self.interaction_encoder = _build_mlp(
            input_dim=interaction_input_dim,
            hidden_dims=interaction_hidden_layers,
            output_dim=interaction_output_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.interaction_output_dim = interaction_output_dim

        if architecture.review_context_dim > 0:
            review_hidden_layers = architecture.review_hidden_layers[:-1]
            review_output_dim = (
                architecture.review_hidden_layers[-1]
                if architecture.review_hidden_layers
                else architecture.review_projection_dim
            )
            self.review_encoder = _build_mlp(
                input_dim=architecture.review_context_dim,
                hidden_dims=review_hidden_layers,
                output_dim=review_output_dim,
                dropout=architecture.dropout,
                output_activation=True,
            )
            self.review_output_dim = review_output_dim
        else:
            self.review_encoder = None
            self.review_output_dim = 0

        head_input_dim = self.interaction_output_dim + self.review_output_dim
        self.head = _build_mlp(
            input_dim=head_input_dim,
            hidden_dims=architecture.head_hidden_layers,
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )

    def forward(
        self,
        *,
        user_embedding: torch.Tensor,
        business_embedding: torch.Tensor,
        review_context: torch.Tensor | None,
    ) -> torch.Tensor:
        user_projected = self.user_encoder(user_embedding)
        business_projected = self.business_encoder(business_embedding)

        dot = (user_projected * business_projected).sum(dim=1, keepdim=True)
        user_norm = torch.linalg.norm(user_projected, dim=1, keepdim=True).clamp_min(1e-8)
        business_norm = torch.linalg.norm(business_projected, dim=1, keepdim=True).clamp_min(1e-8)
        cosine = dot / (user_norm * business_norm)

        interaction_features = torch.cat(
            [
                user_projected,
                business_projected,
                torch.abs(user_projected - business_projected),
                user_projected * business_projected,
                dot,
                cosine,
            ],
            dim=1,
        )
        interaction_representation = self.interaction_encoder(interaction_features)

        if self.review_encoder is not None and review_context is not None and review_context.shape[1] > 0:
            review_representation = self.review_encoder(review_context)
            head_input = torch.cat([interaction_representation, review_representation], dim=1)
        else:
            head_input = interaction_representation

        return self.head(head_input).squeeze(-1)
