from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class KnownUserDeepE2EConfig:
    max_history_len: int = 20
    history_summary_tokens: int = 4
    embedding_dim: int = 128
    business_hidden_layers: tuple[int, ...] = (512, 384, 256)
    event_hidden_layers: tuple[int, ...] = (128,)
    user_hidden_layers: tuple[int, ...] = (128,)
    taste_hidden_layers: tuple[int, ...] = (256, 128)
    baseline_hidden_layers: tuple[int, ...] = (128, 64)
    gate_hidden_dim: int = 64
    categorical_embedding_dim: int = 8
    history_band_embedding_dim: int = 8
    num_attention_heads: int = 4
    dropout: float = 0.15
    aux_like_weight: float = 0.15
    aux_dislike_weight: float = 0.15
    correction_loss_weight: float = 0.2
    baseline_loss_weight: float = 0.05
    expert_strategy: str = "banded_moe_v1"
    band_correction_scales: dict[str, float] | None = None
    band_distillation_weights: dict[str, float] | None = None
    alpha_regularization_weight: float = 0.0
    use_direct_predictor: bool = False
    batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    max_epochs: int = 20
    early_stopping_patience: int = 4
    random_seed: int = 42
    device: str = "auto"


@dataclass(slots=True)
class KnownUserDeepE2EArchitecture:
    business_input_dim: int
    user_numeric_input_dim: int
    user_aux_input_dim: int
    baseline_input_dim: int
    event_scalar_dim: int
    history_band_vocab_size: int
    user_categorical_vocab_sizes: tuple[int, ...]
    expert_band_ids: dict[str, tuple[int, ...]]
    expert_correction_scales: dict[str, float]
    embedding_dim: int = 128
    business_hidden_layers: tuple[int, ...] = (512, 384, 256)
    event_hidden_layers: tuple[int, ...] = (128,)
    user_hidden_layers: tuple[int, ...] = (128,)
    taste_hidden_layers: tuple[int, ...] = (256, 128)
    baseline_hidden_layers: tuple[int, ...] = (128, 64)
    gate_hidden_dim: int = 64
    categorical_embedding_dim: int = 8
    history_band_embedding_dim: int = 8
    num_attention_heads: int = 4
    dropout: float = 0.15
    history_summary_tokens: int = 4
    global_mean: float = 3.5
    use_direct_predictor: bool = False


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


def build_known_user_deep_e2e_architecture(contract: Any, config: KnownUserDeepE2EConfig) -> KnownUserDeepE2EArchitecture:
    user_categorical_feature_names = list(_contract_get(contract, "user_categorical_feature_names"))
    user_categorical_levels = _contract_get(contract, "user_categorical_levels")
    user_categorical_vocab_sizes = tuple(len(list(user_categorical_levels[name])) + 1 for name in user_categorical_feature_names)
    history_band_levels = tuple(_contract_get(contract, "history_band_levels"))
    history_band_vocab = _to_level_vocab(history_band_levels)
    return KnownUserDeepE2EArchitecture(
        business_input_dim=len(list(_contract_get(contract, "business_feature_names"))),
        user_numeric_input_dim=len(list(_contract_get(contract, "user_numeric_feature_names"))),
        user_aux_input_dim=len(list(_contract_get(contract, "user_aux_feature_names"))),
        baseline_input_dim=len(list(_contract_get(contract, "baseline_feature_names"))),
        event_scalar_dim=len(list(_contract_get(contract, "event_scalar_feature_names"))),
        history_band_vocab_size=len(history_band_levels) + 1,
        user_categorical_vocab_sizes=user_categorical_vocab_sizes,
        expert_band_ids={
            "band_1": (int(history_band_vocab.get("1", 0)),),
            "band_2_3": (int(history_band_vocab.get("2-5", 0)),),
            "band_4_5": (int(history_band_vocab.get("2-5", 0)),),
            "band_6_20": tuple(int(value) for value in (history_band_vocab.get("6-20", 0),) if int(value) > 0),
            "band_gt_20": tuple(int(value) for value in (history_band_vocab.get(">20", 0),) if int(value) > 0),
        },
        expert_correction_scales={
            "band_1": float((config.band_correction_scales or {}).get("1", 0.7)),
            "band_2_3": float((config.band_correction_scales or {}).get("2-3", (config.band_correction_scales or {}).get("2-5", 0.9))),
            "band_4_5": float((config.band_correction_scales or {}).get("4-5", (config.band_correction_scales or {}).get("2-5", 0.95))),
            "band_6_20": float((config.band_correction_scales or {}).get("6-20", 1.0)),
            "band_gt_20": float((config.band_correction_scales or {}).get(">20", 0.95)),
        },
        embedding_dim=config.embedding_dim,
        business_hidden_layers=config.business_hidden_layers,
        event_hidden_layers=config.event_hidden_layers,
        user_hidden_layers=config.user_hidden_layers,
        taste_hidden_layers=config.taste_hidden_layers,
        baseline_hidden_layers=config.baseline_hidden_layers,
        gate_hidden_dim=config.gate_hidden_dim,
        categorical_embedding_dim=config.categorical_embedding_dim,
        history_band_embedding_dim=config.history_band_embedding_dim,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout,
        history_summary_tokens=config.history_summary_tokens,
        global_mean=float(_contract_get(contract, "global_mean")),
        use_direct_predictor=config.use_direct_predictor,
    )


class KnownUserDeepE2EModel(nn.Module):
    def __init__(self, architecture: KnownUserDeepE2EArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.max_history_len = 20

        business_hidden_layers = architecture.business_hidden_layers or (architecture.embedding_dim, architecture.embedding_dim)
        self.business_tower = _build_mlp(
            input_dim=architecture.business_input_dim,
            hidden_dims=business_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=False,
        )

        event_hidden_layers = architecture.event_hidden_layers or (architecture.embedding_dim,)
        self.event_encoder = _build_mlp(
            input_dim=architecture.embedding_dim + architecture.event_scalar_dim,
            hidden_dims=event_hidden_layers,
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
        self.user_type_encoder = _build_mlp(
            input_dim=
            architecture.user_numeric_input_dim
            + architecture.user_aux_input_dim
            + (len(architecture.user_categorical_vocab_sizes) * architecture.categorical_embedding_dim)
            + architecture.history_band_embedding_dim,
            hidden_dims=architecture.user_hidden_layers or (architecture.embedding_dim,),
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
        self.positive_attention = nn.MultiheadAttention(
            embed_dim=architecture.embedding_dim,
            num_heads=architecture.num_attention_heads,
            dropout=architecture.dropout,
            batch_first=True,
        )
        self.negative_attention = nn.MultiheadAttention(
            embed_dim=architecture.embedding_dim,
            num_heads=architecture.num_attention_heads,
            dropout=architecture.dropout,
            batch_first=True,
        )

        taste_hidden_layers = architecture.taste_hidden_layers or (architecture.embedding_dim,)
        self.band_1_taste_fusion = _build_mlp(
            input_dim=architecture.embedding_dim * 5,
            hidden_dims=taste_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_2_3_taste_fusion = _build_mlp(
            input_dim=architecture.embedding_dim * 6,
            hidden_dims=taste_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_4_5_taste_fusion = _build_mlp(
            input_dim=architecture.embedding_dim * 6,
            hidden_dims=taste_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_6_20_taste_fusion = _build_mlp(
            input_dim=architecture.embedding_dim * 5,
            hidden_dims=taste_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_gt_20_taste_fusion = _build_mlp(
            input_dim=architecture.embedding_dim * 5,
            hidden_dims=taste_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )

        baseline_hidden_layers = architecture.baseline_hidden_layers or (architecture.embedding_dim // 2, architecture.embedding_dim // 4)
        self.baseline_head = _build_mlp(
            input_dim=architecture.baseline_input_dim,
            hidden_dims=baseline_hidden_layers,
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )

        residual_hidden_layers = architecture.taste_hidden_layers or (architecture.embedding_dim, architecture.embedding_dim // 2)
        gate_input_dim = architecture.user_aux_input_dim + architecture.history_band_embedding_dim + 2
        self.band_1_residual_hidden = _build_mlp(
            input_dim=architecture.embedding_dim * 7 + 4,
            hidden_dims=residual_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_2_3_residual_hidden = _build_mlp(
            input_dim=(architecture.embedding_dim * 8) + architecture.user_aux_input_dim + 10,
            hidden_dims=residual_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_4_5_residual_hidden = _build_mlp(
            input_dim=(architecture.embedding_dim * 8) + architecture.user_aux_input_dim + 10,
            hidden_dims=residual_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_6_20_residual_hidden = _build_mlp(
            input_dim=architecture.embedding_dim * 7 + 4,
            hidden_dims=residual_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_gt_20_residual_hidden = _build_mlp(
            input_dim=architecture.embedding_dim * 7 + 4,
            hidden_dims=residual_hidden_layers,
            output_dim=architecture.embedding_dim,
            dropout=architecture.dropout,
            output_activation=True,
        )
        self.band_1_correction_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_2_3_correction_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_4_5_correction_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_6_20_correction_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_gt_20_correction_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_1_like_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_2_3_like_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_4_5_like_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_6_20_like_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_gt_20_like_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_1_dislike_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_2_3_dislike_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_4_5_dislike_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_6_20_dislike_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_gt_20_dislike_head = nn.Linear(architecture.embedding_dim, 1)
        self.band_1_gate_head = _build_mlp(
            input_dim=gate_input_dim,
            hidden_dims=(max(architecture.gate_hidden_dim // 2, 16),),
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )
        self.band_2_3_gate_head = _build_mlp(
            input_dim=gate_input_dim + 8,
            hidden_dims=(architecture.gate_hidden_dim,),
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )
        self.band_4_5_gate_head = _build_mlp(
            input_dim=gate_input_dim + 8,
            hidden_dims=(architecture.gate_hidden_dim,),
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )
        self.band_6_20_gate_head = _build_mlp(
            input_dim=gate_input_dim,
            hidden_dims=(architecture.gate_hidden_dim,),
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )
        self.band_gt_20_gate_head = _build_mlp(
            input_dim=gate_input_dim,
            hidden_dims=(architecture.gate_hidden_dim,),
            output_dim=1,
            dropout=architecture.dropout,
            output_activation=False,
        )
        self.register_buffer(
            "band_1_id_tensor",
            torch.tensor(list(architecture.expert_band_ids.get("band_1", ())), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "band_2_3_id_tensor",
            torch.tensor(list(architecture.expert_band_ids.get("band_2_3", ())), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "band_4_5_id_tensor",
            torch.tensor(list(architecture.expert_band_ids.get("band_4_5", ())), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "band_6_20_id_tensor",
            torch.tensor(list(architecture.expert_band_ids.get("band_6_20", ())), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "band_gt_20_id_tensor",
            torch.tensor(list(architecture.expert_band_ids.get("band_gt_20", ())), dtype=torch.long),
            persistent=False,
        )

    @staticmethod
    def _id_membership(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.numel() == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        return (values.unsqueeze(1) == candidates.unsqueeze(0)).any(dim=1)

    def _expert_masks(self, history_band_ids: torch.Tensor, history_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        band_1 = self._id_membership(history_band_ids, self.band_1_id_tensor.to(history_band_ids.device))
        band_2_5 = self._id_membership(history_band_ids, self.band_2_3_id_tensor.to(history_band_ids.device))
        band_2_3 = band_2_5 & history_lengths.le(3)
        band_4_5 = band_2_5 & history_lengths.ge(4)
        band_6_20 = self._id_membership(history_band_ids, self.band_6_20_id_tensor.to(history_band_ids.device))
        band_gt_20 = self._id_membership(history_band_ids, self.band_gt_20_id_tensor.to(history_band_ids.device))
        fallback = ~(band_1 | band_2_3 | band_4_5 | band_6_20 | band_gt_20)
        return {
            "band_1": band_1,
            "band_2_3": band_2_3,
            "band_4_5": band_4_5,
            "band_6_20": band_6_20,
            "band_gt_20": band_gt_20 | fallback,
        }

    def _run_band_experts(
        self,
        *,
        history_band_ids: torch.Tensor,
        candidate_business_vec: torch.Tensor,
        user_type_vec: torch.Tensor,
        history_context: torch.Tensor,
        positive_context: torch.Tensor,
        negative_context: torch.Tensor,
        user_aux_features: torch.Tensor,
        band_embedding: torch.Tensor,
        baseline_hat: torch.Tensor,
        incumbent_prediction_raw: torch.Tensor,
        history_event_vec: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = candidate_business_vec.shape[0]
        device = candidate_business_vec.device
        dtype = candidate_business_vec.dtype
        history_lengths = history_mask.long().sum(dim=1).clamp_min(1)
        masks = self._expert_masks(history_band_ids, history_lengths)
        user_taste_vec = torch.zeros(batch_size, self.architecture.embedding_dim, device=device, dtype=dtype)
        correction_hidden = torch.zeros(batch_size, self.architecture.embedding_dim, device=device, dtype=dtype)
        alpha = torch.zeros(batch_size, device=device, dtype=dtype)
        correction_scale = torch.zeros(batch_size, device=device, dtype=dtype)
        expert_index = torch.full((batch_size,), 4, device=device, dtype=torch.long)
        incumbent_gap = (incumbent_prediction_raw - baseline_hat).unsqueeze(-1)
        common_gate_tail = torch.cat([incumbent_gap, incumbent_prediction_raw.unsqueeze(-1)], dim=1)
        batch_indices = torch.arange(batch_size, device=device)
        last_history_vec = history_event_vec[batch_indices, history_lengths - 1]

        expert_specs = [
            (
                "band_1",
                0,
                self.band_1_taste_fusion,
                self.band_1_residual_hidden,
                self.band_1_gate_head,
                float(self.architecture.expert_correction_scales.get("band_1", 0.5)),
            ),
            (
                "band_2_3",
                1,
                self.band_2_3_taste_fusion,
                self.band_2_3_residual_hidden,
                self.band_2_3_gate_head,
                float(self.architecture.expert_correction_scales.get("band_2_3", 0.9)),
            ),
            (
                "band_4_5",
                2,
                self.band_4_5_taste_fusion,
                self.band_4_5_residual_hidden,
                self.band_4_5_gate_head,
                float(self.architecture.expert_correction_scales.get("band_4_5", 0.95)),
            ),
            (
                "band_6_20",
                3,
                self.band_6_20_taste_fusion,
                self.band_6_20_residual_hidden,
                self.band_6_20_gate_head,
                float(self.architecture.expert_correction_scales.get("band_6_20", 1.0)),
            ),
            (
                "band_gt_20",
                4,
                self.band_gt_20_taste_fusion,
                self.band_gt_20_residual_hidden,
                self.band_gt_20_gate_head,
                float(self.architecture.expert_correction_scales.get("band_gt_20", 0.95)),
            ),
        ]
        for expert_name, expert_idx, taste_module, residual_module, gate_module, scale in expert_specs:
            mask = masks[expert_name]
            if not mask.any():
                continue
            expert_index[mask] = expert_idx
            candidate_slice = candidate_business_vec[mask]
            user_type_slice = user_type_vec[mask]
            history_slice = history_context[mask]
            positive_slice = positive_context[mask]
            negative_slice = negative_context[mask]
            aux_slice = user_aux_features[mask]
            band_slice = band_embedding[mask]
            last_slice = last_history_vec[mask]
            incumbent_gap_slice = incumbent_gap[mask]
            incumbent_raw_slice = incumbent_prediction_raw[mask].unsqueeze(-1)
            if expert_name in {"band_2_3", "band_4_5"}:
                candidate_last_dot = (candidate_slice * last_slice).sum(dim=1, keepdim=True)
                candidate_positive_dot = (candidate_slice * positive_slice).sum(dim=1, keepdim=True)
                candidate_negative_dot = (candidate_slice * negative_slice).sum(dim=1, keepdim=True)
                candidate_last_cos = F.cosine_similarity(candidate_slice, last_slice, dim=1, eps=1e-8).unsqueeze(-1)
                candidate_positive_cos = F.cosine_similarity(candidate_slice, positive_slice, dim=1, eps=1e-8).unsqueeze(-1)
                candidate_negative_cos = F.cosine_similarity(candidate_slice, negative_slice, dim=1, eps=1e-8).unsqueeze(-1)
                positive_negative_cos = F.cosine_similarity(positive_slice, negative_slice, dim=1, eps=1e-8).unsqueeze(-1)
                candidate_history_cos = F.cosine_similarity(candidate_slice, history_slice, dim=1, eps=1e-8).unsqueeze(-1)
                affinity_features = torch.cat(
                    [
                        candidate_last_dot,
                        candidate_positive_dot,
                        candidate_negative_dot,
                        candidate_last_cos,
                        candidate_positive_cos,
                        candidate_negative_cos,
                        positive_negative_cos,
                        candidate_history_cos,
                    ],
                    dim=1,
                )
                taste_input = torch.cat(
                    [
                        history_slice,
                        positive_slice,
                        negative_slice,
                        candidate_slice,
                        user_type_slice,
                        last_slice,
                    ],
                    dim=1,
                )
                taste_candidate = taste_module(taste_input)
                residual_input = torch.cat(
                    [
                        candidate_slice,
                        user_type_slice,
                        taste_candidate,
                        history_slice,
                        positive_slice,
                        negative_slice,
                        last_slice,
                        torch.abs(candidate_slice - last_slice),
                        aux_slice,
                        incumbent_gap_slice,
                        incumbent_raw_slice,
                        affinity_features,
                    ],
                    dim=1,
                )
                gate_input = torch.cat([aux_slice, band_slice, common_gate_tail[mask], affinity_features], dim=1)
            else:
                taste_input = torch.cat(
                    [
                        history_slice,
                        positive_slice,
                        negative_slice,
                        candidate_slice,
                        user_type_slice,
                    ],
                    dim=1,
                )
                taste_candidate = taste_module(taste_input)
                residual_input = torch.cat(
                    [
                        candidate_slice,
                        user_type_slice,
                        taste_candidate,
                        history_slice,
                        positive_slice,
                        negative_slice,
                        incumbent_gap_slice,
                        incumbent_raw_slice,
                        torch.abs(candidate_slice - taste_candidate),
                        (candidate_slice * taste_candidate).sum(dim=1, keepdim=True),
                        F.cosine_similarity(candidate_slice, taste_candidate, dim=1, eps=1e-8).unsqueeze(-1),
                    ],
                    dim=1,
                )
                gate_input = torch.cat([aux_slice, band_slice, common_gate_tail[mask]], dim=1)
            user_taste_vec[mask] = taste_candidate
            correction_hidden[mask] = residual_module(residual_input)
            alpha[mask] = torch.sigmoid(gate_module(gate_input).squeeze(-1))
            correction_scale[mask] = scale

        correction_logits = torch.zeros(batch_size, device=device, dtype=dtype)
        like_logits = torch.zeros(batch_size, device=device, dtype=dtype)
        dislike_logits = torch.zeros(batch_size, device=device, dtype=dtype)
        for expert_name, expert_idx, correction_head, like_head, dislike_head in [
            ("band_1", 0, self.band_1_correction_head, self.band_1_like_head, self.band_1_dislike_head),
            ("band_2_3", 1, self.band_2_3_correction_head, self.band_2_3_like_head, self.band_2_3_dislike_head),
            ("band_4_5", 2, self.band_4_5_correction_head, self.band_4_5_like_head, self.band_4_5_dislike_head),
            ("band_6_20", 3, self.band_6_20_correction_head, self.band_6_20_like_head, self.band_6_20_dislike_head),
            ("band_gt_20", 4, self.band_gt_20_correction_head, self.band_gt_20_like_head, self.band_gt_20_dislike_head),
        ]:
            mask = masks[expert_name]
            if not mask.any():
                continue
            correction_logits[mask] = correction_head(correction_hidden[mask]).squeeze(-1)
            like_logits[mask] = like_head(correction_hidden[mask]).squeeze(-1)
            dislike_logits[mask] = dislike_head(correction_hidden[mask]).squeeze(-1)
        return {
            "user_taste_vec": user_taste_vec,
            "correction_logits": correction_logits,
            "like_logits": like_logits,
            "dislike_logits": dislike_logits,
            "alpha": alpha,
            "correction_scale": correction_scale,
            "expert_index": expert_index,
        }

    @classmethod
    def from_contract(
        cls,
        contract: Any,
        config: KnownUserDeepE2EConfig | None = None,
    ) -> "KnownUserDeepE2EModel":
        config = config or KnownUserDeepE2EConfig()
        architecture = build_known_user_deep_e2e_architecture(contract, config)
        return cls(architecture)

    def encode_business(self, business_features: torch.Tensor) -> torch.Tensor:
        return self.business_tower(business_features)

    def encode_user_type(
        self,
        *,
        user_numeric_features: torch.Tensor,
        user_aux_features: torch.Tensor,
        user_categorical_ids: torch.Tensor,
        history_band_ids: torch.Tensor,
    ) -> torch.Tensor:
        categorical_parts: list[torch.Tensor] = []
        for column_index, embedding in enumerate(self.user_categorical_embeddings):
            categorical_parts.append(embedding(user_categorical_ids[:, column_index].clamp_min(0)))

        band_embedding = self.history_band_embedding(history_band_ids.clamp_min(0))
        encoder_input = torch.cat(
            [
                user_numeric_features,
                user_aux_features,
                *categorical_parts,
                band_embedding,
            ],
            dim=1,
        )
        return self.user_type_encoder(encoder_input)

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

    def _encode_history_tokens(
        self,
        *,
        candidate_business_vec: torch.Tensor,
        user_type_vec: torch.Tensor,
        history_business_features: torch.Tensor,
        history_rating_features: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, history_len, business_dim = history_business_features.shape
        history_business_vec = self.business_tower(history_business_features.reshape(batch_size * history_len, business_dim))
        history_business_vec = history_business_vec.reshape(batch_size, history_len, self.architecture.embedding_dim)
        event_input = torch.cat([history_business_vec, history_rating_features], dim=-1)
        event_vec = self.event_encoder(event_input.reshape(batch_size * history_len, -1))
        event_vec = event_vec.reshape(batch_size, history_len, self.architecture.embedding_dim)

        history_float_mask = history_mask.float()
        valid_count = history_float_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        positive_mask = history_float_mask * history_rating_features[:, :, 3]
        negative_mask = history_float_mask * history_rating_features[:, :, 4]
        recency_weights = history_float_mask * history_rating_features[:, :, 8]

        mean_token = (event_vec * history_float_mask.unsqueeze(-1)).sum(dim=1) / valid_count
        positive_count = positive_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        negative_count = negative_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        positive_token = (event_vec * positive_mask.unsqueeze(-1)).sum(dim=1) / positive_count
        negative_token = (event_vec * negative_mask.unsqueeze(-1)).sum(dim=1) / negative_count
        recency_denominator = recency_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        recency_token = (event_vec * recency_weights.unsqueeze(-1)).sum(dim=1) / recency_denominator

        summary_tokens = torch.stack([mean_token, positive_token, negative_token, recency_token], dim=1)
        summary_valid = torch.stack(
            [
                history_float_mask.any(dim=1),
                positive_mask.any(dim=1),
                negative_mask.any(dim=1),
                history_float_mask.any(dim=1),
            ],
            dim=1,
        )

        all_tokens = torch.cat([event_vec, summary_tokens], dim=1)
        all_valid = torch.cat([history_mask, summary_valid], dim=1)
        positive_summary_valid = torch.stack(
            [
                torch.zeros_like(summary_valid[:, 0], dtype=torch.bool),
                summary_valid[:, 1],
                torch.zeros_like(summary_valid[:, 0], dtype=torch.bool),
                torch.zeros_like(summary_valid[:, 0], dtype=torch.bool),
            ],
            dim=1,
        )
        negative_summary_valid = torch.stack(
            [
                torch.zeros_like(summary_valid[:, 0], dtype=torch.bool),
                torch.zeros_like(summary_valid[:, 0], dtype=torch.bool),
                summary_valid[:, 2],
                torch.zeros_like(summary_valid[:, 0], dtype=torch.bool),
            ],
            dim=1,
        )
        positive_valid = torch.cat([positive_mask.bool(), positive_summary_valid], dim=1)
        negative_valid = torch.cat([negative_mask.bool(), negative_summary_valid], dim=1)

        query = self.query_projection(torch.cat([candidate_business_vec, user_type_vec], dim=1))
        history_context = self._safe_attention(
            self.history_attention,
            query=query,
            tokens=all_tokens,
            valid_mask=all_valid,
        )
        positive_context = self._safe_attention(
            self.positive_attention,
            query=query,
            tokens=all_tokens,
            valid_mask=positive_valid,
        )
        negative_context = self._safe_attention(
            self.negative_attention,
            query=query,
            tokens=all_tokens,
            valid_mask=negative_valid,
        )
        return history_context, positive_context, negative_context, event_vec

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
        candidate_business_vec = self.encode_business(candidate_business_features)
        user_type_vec = self.encode_user_type(
            user_numeric_features=user_numeric_features,
            user_aux_features=user_aux_features,
            user_categorical_ids=user_categorical_ids,
            history_band_ids=history_band_ids,
        )
        history_context, positive_context, negative_context, history_event_vec = self._encode_history_tokens(
            candidate_business_vec=candidate_business_vec,
            user_type_vec=user_type_vec,
            history_business_features=history_business_features,
            history_rating_features=history_rating_features,
            history_mask=history_mask,
        )
        band_embedding = self.history_band_embedding(history_band_ids.clamp_min(0))

        baseline_raw = self.baseline_head(baseline_features).squeeze(-1)
        baseline_hat = self.architecture.global_mean + (2.0 * torch.tanh(baseline_raw))
        expert_outputs = self._run_band_experts(
            history_band_ids=history_band_ids,
            candidate_business_vec=candidate_business_vec,
            user_type_vec=user_type_vec,
            history_context=history_context,
            positive_context=positive_context,
            negative_context=negative_context,
            user_aux_features=user_aux_features,
            band_embedding=band_embedding,
            baseline_hat=baseline_hat,
            incumbent_prediction_raw=incumbent_prediction_raw,
            history_event_vec=history_event_vec,
            history_mask=history_mask,
        )
        correction_hat = expert_outputs["correction_scale"] * torch.tanh(expert_outputs["correction_logits"])
        if self.architecture.use_direct_predictor:
            # Direct predictor: no alpha gate, no correction scale clamp.
            # The correction head learns the full residual; incumbent is just an input feature.
            correction_hat = expert_outputs["correction_logits"]
            predicted_rating = torch.clamp(incumbent_prediction_raw + correction_hat, 1.0, 5.0)
        else:
            predicted_rating = torch.clamp(incumbent_prediction_raw + (expert_outputs["alpha"] * correction_hat), 1.0, 5.0)

        return {
            "predicted_rating": predicted_rating,
            "baseline_hat": baseline_hat,
            "residual_hat": correction_hat,
            "correction_hat": correction_hat,
            "alpha": expert_outputs["alpha"],
            "user_type_vec": user_type_vec,
            "user_taste_vec": expert_outputs["user_taste_vec"],
            "candidate_business_vec": candidate_business_vec,
            "like_logits": expert_outputs["like_logits"],
            "dislike_logits": expert_outputs["dislike_logits"],
            "incumbent_prediction_raw": incumbent_prediction_raw,
            "expert_index": expert_outputs["expert_index"],
        }


def compute_known_user_deep_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_rating: torch.Tensor,
    like_target: torch.Tensor,
    dislike_target: torch.Tensor,
    incumbent_prediction_raw: torch.Tensor,
    history_band_ids: torch.Tensor,
    config: KnownUserDeepE2EConfig,
) -> torch.Tensor:
    mask = torch.isfinite(target_rating)
    rating_target = torch.nan_to_num(target_rating, nan=0.0)
    rating_pred = outputs["predicted_rating"]
    correction_pred = outputs.get("correction_hat", outputs["residual_hat"])
    correction_target = torch.clamp(target_rating - incumbent_prediction_raw, min=-1.5, max=1.5)

    if mask.any():
        main_loss = F.l1_loss(rating_pred[mask], rating_target[mask])
        baseline_loss = F.l1_loss(outputs["baseline_hat"][mask], rating_target[mask])
        correction_loss = F.l1_loss(correction_pred[mask], correction_target[mask])
    else:
        main_loss = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)
        baseline_loss = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)
        correction_loss = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)

    distill_weights = torch.ones_like(incumbent_prediction_raw, dtype=rating_pred.dtype)
    distill_config = config.band_distillation_weights or {"1": 0.20, "2-3": 0.09, "4-5": 0.07, "6-20": 0.05, ">20": 0.05}
    expert_index = outputs.get("expert_index")
    if expert_index is None:
        band_weight_map = {
            1: 0.0,
            2: float(distill_config.get("1", 0.20)),
            3: float(distill_config.get("2-3", distill_config.get("2-5", 0.09))),
            4: float(distill_config.get("6-20", 0.05)),
            5: float(distill_config.get(">20", 0.05)),
            6: float(distill_config.get("__unknown__", 0.05)),
        }
        for band_id, weight in band_weight_map.items():
            distill_weights = torch.where(history_band_ids == band_id, torch.full_like(distill_weights, weight), distill_weights)
    else:
        expert_weight_map = {
            0: float(distill_config.get("1", 0.20)),
            1: float(distill_config.get("2-3", distill_config.get("2-5", 0.09))),
            2: float(distill_config.get("4-5", distill_config.get("2-5", 0.07))),
            3: float(distill_config.get("6-20", 0.05)),
            4: float(distill_config.get(">20", 0.05)),
        }
        for idx, weight in expert_weight_map.items():
            distill_weights = torch.where(expert_index == idx, torch.full_like(distill_weights, weight), distill_weights)
    if mask.any():
        distill_loss = ((rating_pred[mask] - incumbent_prediction_raw[mask]).abs() * distill_weights[mask]).mean()
        alpha_regularization = outputs["alpha"][mask].mean()
    else:
        distill_loss = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)
        alpha_regularization = torch.zeros((), device=rating_pred.device, dtype=rating_pred.dtype)

    like_loss = F.binary_cross_entropy_with_logits(outputs["like_logits"], like_target.float())
    dislike_loss = F.binary_cross_entropy_with_logits(outputs["dislike_logits"], dislike_target.float())
    # When direct_predictor is on the gate head is bypassed, so alpha reg is meaningless
    effective_alpha_reg = 0.0 if config.use_direct_predictor else config.alpha_regularization_weight
    return (
        main_loss
        + (config.correction_loss_weight * correction_loss)
        + (config.baseline_loss_weight * baseline_loss)
        + distill_loss
        + (config.aux_like_weight * like_loss)
        + (config.aux_dislike_weight * dislike_loss)
        + (effective_alpha_reg * alpha_regularization)
    )


@dataclass(slots=True)
class KnownUserDeepE2ECheckpoint:
    config: KnownUserDeepE2EConfig
    architecture: KnownUserDeepE2EArchitecture
    feature_contract: dict[str, Any]
    training_summary: dict[str, Any]
    model_state_dict: dict[str, Any]

    def to_model(self) -> KnownUserDeepE2EModel:
        model = KnownUserDeepE2EModel(self.architecture)
        model.load_state_dict(self.model_state_dict)
        return model

    @classmethod
    def from_model(
        cls,
        *,
        model: KnownUserDeepE2EModel,
        config: KnownUserDeepE2EConfig,
        feature_contract: Any,
        training_summary: dict[str, Any] | None = None,
    ) -> "KnownUserDeepE2ECheckpoint":
        architecture = model.architecture
        return cls(
            config=config,
            architecture=architecture,
            feature_contract=_jsonable(feature_contract),
            training_summary=training_summary or {},
            model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        )

    def save(self, save_dir: str | Path) -> dict[str, str]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / "known_user_deep_checkpoint.pt"
        config_path = save_dir / "known_user_deep_config.json"
        summary_path = save_dir / "known_user_deep_training_summary.json"

        torch.save(
            {
                "config": _jsonable(self.config),
                "architecture": _jsonable(self.architecture),
                "feature_contract": self.feature_contract,
                "training_summary": self.training_summary,
                "model_state_dict": self.model_state_dict,
            },
            checkpoint_path,
        )
        config_path.write_text(json.dumps(_jsonable(self.config), indent=2, ensure_ascii=True), encoding="utf-8")
        summary_path.write_text(json.dumps(_jsonable(self.training_summary), indent=2, ensure_ascii=True), encoding="utf-8")

        return {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "training_summary": str(summary_path),
        }

    @classmethod
    def load(cls, save_dir: str | Path, *, map_location: str | torch.device = "cpu") -> "KnownUserDeepE2ECheckpoint":
        save_dir = Path(save_dir)
        payload = torch.load(save_dir / "known_user_deep_checkpoint.pt", map_location=map_location)
        config = KnownUserDeepE2EConfig(**payload["config"])
        architecture = KnownUserDeepE2EArchitecture(**payload["architecture"])
        return cls(
            config=config,
            architecture=architecture,
            feature_contract=dict(payload["feature_contract"]),
            training_summary=dict(payload["training_summary"]),
            model_state_dict=dict(payload["model_state_dict"]),
        )
