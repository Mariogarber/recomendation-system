from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from model.known_user_deep_e2e import (
    KnownUserDeepE2EArchitecture,
    KnownUserDeepE2EConfig,
    KnownUserDeepE2EModel,
    build_known_user_deep_e2e_architecture,
    compute_known_user_deep_loss,
)

from .io import canonicalize_reviews
from .lgbm_raw_features import RAW_CORE_FEATURE_SET, build_raw_feature_frame, fit_raw_feature_spec, history_band_from_count
from .lgbm_raw_router_features import build_router_feature_frame, fit_router_feature_spec


SAFE_BUSINESS_VIEW = "content"
HISTORY_BAND_LEVELS = ["0", "1", "2-5", "6-20", ">20", "__unknown__"]
KNOWN_USER_CATEGORICAL_COLUMNS = ["user_archetype_id", "user_activity_bucket", "user_reputation_bucket", "user_tenure_bucket"]
KNOWN_USER_BASELINE_FEATURE_COLUMNS = [
    "user_average_stars",
    "business_stars",
    "user_minus_global_mean",
    "business_minus_global_mean",
    "user_business_metadata_gap",
    "user_review_count_log1p",
    "business_review_count_log1p",
    "user_review_count_x_business_review_count",
    "review_total_votes",
    "review_useful",
    "review_funny",
    "review_cool",
    "review_days_since_train_start",
    "review_days_since_train_end",
    "business_rating_per_review",
    "business_attributes_count",
    "business_attribute_true_count",
    "business_attribute_false_count",
    "business_attribute_string_count",
    "business_weekly_open_minutes",
    "business_open_days_count",
    "business_weekend_days_open",
    "business_late_night_days",
    "business_latitude",
    "business_longitude",
    "business_geo_abs",
    "history_rating_range",
    "history_count_is_2",
    "history_count_is_3",
    "history_count_is_4",
    "history_count_is_5",
    "history_count_is_2_3",
    "history_count_is_4_5",
    "business_review_count_log1p_x_history_count",
    "business_rating_per_review_x_history_count",
]
KNOWN_USER_NUMERIC_FEATURE_COLUMNS = [
    "user_average_stars",
    "user_review_count",
    "user_review_count_log1p",
    "user_total_votes",
    "user_total_votes_log1p",
    "user_engagement_log1p",
    "user_friends_count",
    "user_friends_log1p",
    "user_fans",
    "user_tenure_days",
    "user_tenure_years",
    "user_elite_years_count",
    "user_elite_any",
    "user_compliment_total",
    "user_compliment_log1p_total",
    "user_compliment_nonzero_count",
    "user_compliment_hot",
    "user_compliment_more",
    "user_compliment_profile",
    "user_compliment_cute",
    "user_compliment_list",
    "user_compliment_note",
    "user_compliment_plain",
    "user_compliment_cool",
    "user_compliment_funny",
    "user_compliment_writer",
    "user_compliment_photos",
    "user_metadata_completeness",
    "user_metadata_sparse_flag",
    "history_count",
    "history_count_log1p",
]
KNOWN_USER_AUX_FEATURE_COLUMNS = [
    "history_count",
    "history_count_log1p",
    "history_count_is_2",
    "history_count_is_3",
    "history_count_is_4",
    "history_count_is_5",
    "history_count_is_2_3",
    "history_count_is_4_5",
    "history_rating_mean",
    "history_rating_std",
    "history_rating_range",
    "history_rating_min",
    "history_rating_max",
    "history_last_rating",
    "history_positive_share",
    "history_negative_share",
    "history_recency_days_mean",
    "history_rating_std_x_count",
    "business_review_count_log1p_x_history_count",
    "user_metadata_completeness",
    "user_metadata_sparse_flag",
]
EVENT_SCALAR_FEATURE_NAMES = [
    "rating",
    "rating_centered_user",
    "rating_centered_global",
    "liked_flag",
    "disliked_flag",
    "rating_abs_dev_user",
    "days_since_interaction",
    "log1p_days_since_interaction",
    "exp_decay_days_since_interaction",
]


@dataclass(slots=True)
class KnownUserDeepDataConfig:
    business_repr_root: str
    max_history_len: int = 20
    n_user_archetypes: int = 64
    max_top_cities: int = 100
    max_top_categories: int = 32
    random_seed: int = 42
    recency_half_life_days: float = 180.0


@dataclass(slots=True)
class KnownUserDeepTrainingConfig:
    embedding_dim: int = 128
    event_hidden_dim: int = 128
    user_type_hidden_dim: int = 128
    scorer_hidden_dim: int = 256
    business_hidden_layers: tuple[int, ...] = (512, 384, 256)
    scorer_hidden_layers: tuple[int, ...] = (256, 128)
    num_attention_heads: int = 4
    dropout: float = 0.15
    batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    max_epochs: int = 20
    early_stopping_patience: int = 4
    auxiliary_loss_weight: float = 0.15
    device: str = "auto"
    random_seed: int = 42
    band_sample_weights: dict[str, float] | None = None
    band_correction_scales: dict[str, float] | None = None
    band_distillation_weights: dict[str, float] | None = None
    alpha_regularization_weight: float = 0.0
    recency_weight_scale: float = 1.0
    selective_replace_alpha_thresholds: dict[str, float] | None = None
    selective_replace_abs_correction_thresholds: dict[str, float] | None = None
    use_direct_predictor: bool = False

    def to_model_config(self, *, max_history_len: int, history_summary_tokens: int) -> KnownUserDeepE2EConfig:
        return KnownUserDeepE2EConfig(
            max_history_len=max_history_len,
            history_summary_tokens=history_summary_tokens,
            embedding_dim=self.embedding_dim,
            business_hidden_layers=tuple(self.business_hidden_layers),
            event_hidden_layers=(self.event_hidden_dim,),
            user_hidden_layers=(self.user_type_hidden_dim,),
            taste_hidden_layers=tuple(self.scorer_hidden_layers) if self.scorer_hidden_layers else (self.scorer_hidden_dim, max(self.scorer_hidden_dim // 2, 32)),
            baseline_hidden_layers=(128, 64),
            gate_hidden_dim=max(self.user_type_hidden_dim // 2, 32),
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            aux_like_weight=self.auxiliary_loss_weight,
            aux_dislike_weight=self.auxiliary_loss_weight,
            band_correction_scales=dict(self.band_correction_scales) if self.band_correction_scales else None,
            band_distillation_weights=dict(self.band_distillation_weights) if self.band_distillation_weights else None,
            alpha_regularization_weight=self.alpha_regularization_weight,
            use_direct_predictor=self.use_direct_predictor,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            random_seed=self.random_seed,
            device=self.device,
        )


@dataclass(slots=True)
class KnownUserDeepFeatureContract:
    business_repr_root: str
    business_view: str
    business_feature_names: list[str]
    baseline_feature_names: list[str]
    user_numeric_feature_names: list[str]
    user_aux_feature_names: list[str]
    user_categorical_feature_names: list[str]
    user_categorical_levels: dict[str, list[str]]
    history_band_levels: list[str]
    event_scalar_feature_names: list[str]
    baseline_feature_means: list[float]
    baseline_feature_stds: list[float]
    user_numeric_feature_means: list[float]
    user_numeric_feature_stds: list[float]
    user_aux_feature_means: list[float]
    user_aux_feature_stds: list[float]
    event_scalar_feature_means: list[float]
    event_scalar_feature_stds: list[float]
    max_history_len: int
    history_summary_tokens: int
    global_mean: float


@dataclass(slots=True)
class KnownUserDeepContext:
    data_config: KnownUserDeepDataConfig
    feature_contract: KnownUserDeepFeatureContract
    raw_spec: Any
    router_spec: Any
    business_ids: pd.Series
    business_matrix: np.ndarray
    business_index: pd.Series


@dataclass(slots=True)
class KnownUserDeepPreparedDataset:
    frame: pd.DataFrame
    history_item_idx: np.ndarray
    history_rating_features: np.ndarray
    candidate_item_idx: np.ndarray
    user_numeric_features: np.ndarray
    user_aux_features: np.ndarray
    user_categorical_ids: np.ndarray
    history_band_ids: np.ndarray
    baseline_features: np.ndarray
    incumbent_prediction_raw: np.ndarray
    targets: np.ndarray


@dataclass(slots=True)
class KnownUserDeepTrainingResult:
    model_config: KnownUserDeepE2EConfig
    architecture: KnownUserDeepE2EArchitecture
    model_state_dict: dict[str, Any]
    learning_curves: pd.DataFrame
    best_epoch: int
    best_val_mae: float
    best_val_rmse: float
    train_size: int
    val_size: int


class _KnownUserTensorDataset(Dataset):
    def __init__(self, prepared: KnownUserDeepPreparedDataset) -> None:
        self.prepared = prepared

    def __len__(self) -> int:
        return int(len(self.prepared.targets))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "history_item_idx": torch.from_numpy(self.prepared.history_item_idx[index]),
            "history_rating_features": torch.from_numpy(self.prepared.history_rating_features[index]),
            "candidate_item_idx": torch.tensor(self.prepared.candidate_item_idx[index], dtype=torch.long),
            "user_numeric_features": torch.from_numpy(self.prepared.user_numeric_features[index]),
            "user_aux_features": torch.from_numpy(self.prepared.user_aux_features[index]),
            "user_categorical_ids": torch.from_numpy(self.prepared.user_categorical_ids[index]),
            "history_band_ids": torch.tensor(self.prepared.history_band_ids[index], dtype=torch.long),
            "baseline_features": torch.from_numpy(self.prepared.baseline_features[index]),
            "incumbent_prediction_raw": torch.tensor(self.prepared.incumbent_prediction_raw[index], dtype=torch.float32),
            "target_rating": torch.tensor(self.prepared.targets[index], dtype=torch.float32),
        }


def load_safe_business_feature_block(root: str | Path) -> tuple[pd.Series, np.ndarray, list[str]]:
    root_path = Path(root)
    business_ids = pd.read_csv(root_path / "business_ids.csv")["business_id"].astype(str)
    content_matrix = sparse.load_npz(root_path / "business_content_features.npz").astype(np.float32)
    feature_names = json.loads((root_path / "business_feature_names.json").read_text(encoding="utf-8"))["content_features"]
    return business_ids, content_matrix.toarray().astype(np.float32, copy=False), list(feature_names)


def prepare_known_user_context(
    *,
    context_reviews: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    data_config: KnownUserDeepDataConfig,
) -> KnownUserDeepContext:
    raw_spec = fit_raw_feature_spec(
        context_reviews,
        users_df,
        businesses_df,
        feature_set=RAW_CORE_FEATURE_SET,
    )
    router_spec = fit_router_feature_spec(
        context_reviews,
        users_df,
        businesses_df,
        n_user_archetypes=data_config.n_user_archetypes,
        max_top_cities=data_config.max_top_cities,
        max_top_categories=data_config.max_top_categories,
        random_seed=data_config.random_seed,
    )
    business_ids, business_matrix, business_feature_names = load_safe_business_feature_block(data_config.business_repr_root)
    business_index = pd.Series(np.arange(len(business_ids), dtype=np.int32), index=business_ids.to_numpy())
    normalization_stats = _compute_feature_normalization_stats(
        context_reviews=context_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
        raw_spec=raw_spec,
        router_spec=router_spec,
        business_index=business_index,
        max_history_len=int(data_config.max_history_len),
        global_mean=float(raw_spec.global_mean),
        half_life_days=float(data_config.recency_half_life_days),
    )
    feature_contract = KnownUserDeepFeatureContract(
        business_repr_root=str(data_config.business_repr_root),
        business_view=SAFE_BUSINESS_VIEW,
        business_feature_names=business_feature_names,
        baseline_feature_names=KNOWN_USER_BASELINE_FEATURE_COLUMNS.copy(),
        user_numeric_feature_names=KNOWN_USER_NUMERIC_FEATURE_COLUMNS.copy(),
        user_aux_feature_names=KNOWN_USER_AUX_FEATURE_COLUMNS.copy(),
        user_categorical_feature_names=KNOWN_USER_CATEGORICAL_COLUMNS.copy(),
        user_categorical_levels={column: _normalize_levels(router_spec.categorical_levels.get(column, [])) for column in KNOWN_USER_CATEGORICAL_COLUMNS},
        history_band_levels=HISTORY_BAND_LEVELS.copy(),
        event_scalar_feature_names=EVENT_SCALAR_FEATURE_NAMES.copy(),
        baseline_feature_means=normalization_stats["baseline_means"],
        baseline_feature_stds=normalization_stats["baseline_stds"],
        user_numeric_feature_means=normalization_stats["user_numeric_means"],
        user_numeric_feature_stds=normalization_stats["user_numeric_stds"],
        user_aux_feature_means=normalization_stats["user_aux_means"],
        user_aux_feature_stds=normalization_stats["user_aux_stds"],
        event_scalar_feature_means=normalization_stats["event_scalar_means"],
        event_scalar_feature_stds=normalization_stats["event_scalar_stds"],
        max_history_len=int(data_config.max_history_len),
        history_summary_tokens=4,
        global_mean=float(raw_spec.global_mean),
    )
    return KnownUserDeepContext(
        data_config=data_config,
        feature_contract=feature_contract,
        raw_spec=raw_spec,
        router_spec=router_spec,
        business_ids=business_ids,
        business_matrix=business_matrix,
        business_index=business_index,
    )


def build_known_user_train_dataset(
    train_reviews: pd.DataFrame,
    *,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    context: KnownUserDeepContext,
    training_config: KnownUserDeepTrainingConfig | None = None,
    incumbent_frame: pd.DataFrame | None = None,
) -> KnownUserDeepPreparedDataset:
    training_config = training_config or KnownUserDeepTrainingConfig()
    prepared_reviews = _prepare_review_frame(train_reviews, business_index=context.business_index)
    ordered_reviews = _order_reviews(prepared_reviews)
    raw_frame = _align_feature_frame(build_raw_feature_frame(train_reviews, users_df, businesses_df, context.raw_spec), ordered_reviews["review_id"])
    router_frame = _align_feature_frame(build_router_feature_frame(train_reviews, users_df, businesses_df, context.router_spec), ordered_reviews["review_id"])
    arrays = _build_prefix_arrays_with_recency(ordered_reviews, context.data_config.max_history_len)
    return _materialize_known_user_dataset(
        raw_frame=raw_frame,
        router_frame=router_frame,
        target_frame=ordered_reviews,
        history_item_idx=arrays["history_item_idx"],
        history_ratings=arrays["history_ratings"],
        history_days=arrays["history_days"],
        exact_history_count=arrays["exact_history_count"],
        incumbent_frame=incumbent_frame,
        context=context,
        training_config=training_config,
    )


def build_known_user_eval_dataset(
    target_reviews: pd.DataFrame,
    context_reviews: pd.DataFrame,
    *,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    context: KnownUserDeepContext,
    training_config: KnownUserDeepTrainingConfig | None = None,
    incumbent_frame: pd.DataFrame | None = None,
) -> KnownUserDeepPreparedDataset:
    training_config = training_config or KnownUserDeepTrainingConfig()
    prepared_target = _prepare_review_frame(target_reviews, business_index=context.business_index)
    ordered_target = _order_reviews(prepared_target)
    prepared_context = _prepare_review_frame(context_reviews, business_index=context.business_index)
    raw_frame = _align_feature_frame(build_raw_feature_frame(target_reviews, users_df, businesses_df, context.raw_spec), ordered_target["review_id"])
    router_frame = _align_feature_frame(build_router_feature_frame(target_reviews, users_df, businesses_df, context.router_spec), ordered_target["review_id"])
    arrays = _build_fixed_context_arrays_with_recency(
        target_interactions=ordered_target,
        context_interactions=prepared_context,
        max_history_len=context.data_config.max_history_len,
    )
    return _materialize_known_user_dataset(
        raw_frame=raw_frame,
        router_frame=router_frame,
        target_frame=ordered_target,
        history_item_idx=arrays["history_item_idx"],
        history_ratings=arrays["history_ratings"],
        history_days=arrays["history_days"],
        exact_history_count=arrays["exact_history_count"],
        incumbent_frame=incumbent_frame,
        context=context,
        training_config=training_config,
    )


def train_known_user_deep_model(
    *,
    train_data: KnownUserDeepPreparedDataset,
    val_data: KnownUserDeepPreparedDataset,
    context: KnownUserDeepContext,
    training_config: KnownUserDeepTrainingConfig,
) -> KnownUserDeepTrainingResult:
    torch.manual_seed(training_config.random_seed)
    np.random.seed(training_config.random_seed)
    model_config = training_config.to_model_config(
        max_history_len=context.feature_contract.max_history_len,
        history_summary_tokens=context.feature_contract.history_summary_tokens,
    )
    architecture = build_known_user_deep_e2e_architecture(context.feature_contract, model_config)
    device = _resolve_device(model_config.device)
    print(f"Validation Training model with architecture: {asdict(architecture)} on device: {device}")
    model = KnownUserDeepE2EModel(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    train_loader = _build_loader(train_data, training_config, shuffle=True)
    business_tensor = torch.tensor(context.business_matrix, dtype=torch.float32, device=device)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 1
    best_val_mae = float("inf")
    best_val_rmse = float("inf")
    patience_left = int(model_config.early_stopping_patience)
    learning_rows: list[dict[str, Any]] = []

    for epoch in range(1, model_config.max_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            optimizer.zero_grad()
            batch_tensors = _prepare_batch_tensors(batch=batch, business_tensor=business_tensor, device=device)
            outputs = model(
                candidate_business_features=batch_tensors["candidate_business_features"],
                history_business_features=batch_tensors["history_business_features"],
                history_rating_features=batch_tensors["history_rating_features"],
                history_mask=batch_tensors["history_mask"],
                user_numeric_features=batch_tensors["user_numeric_features"],
                user_aux_features=batch_tensors["user_aux_features"],
                user_categorical_ids=batch_tensors["user_categorical_ids"],
                history_band_ids=batch_tensors["history_band_ids"],
                baseline_features=batch_tensors["baseline_features"],
                incumbent_prediction_raw=batch_tensors["incumbent_prediction_raw"],
            )
            targets = batch_tensors["target_rating"]
            loss = compute_known_user_deep_loss(
                outputs,
                target_rating=targets,
                like_target=targets.ge(4.0),
                dislike_target=targets.le(2.0),
                incumbent_prediction_raw=batch_tensors["incumbent_prediction_raw"],
                history_band_ids=batch_tensors["history_band_ids"],
                config=model_config,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_predictions = predict_known_user_dataset(
            model=model,
            prepared=val_data,
            context=context,
            batch_size=model_config.batch_size,
            device=device,
            business_tensor=business_tensor,
        )
        val_pred_raw = val_predictions["predicted_rating"].to_numpy(dtype=np.float32)
        val_pred_rounded = np.floor(val_pred_raw + 0.5).clip(1, 5)
        val_diff = (
            val_predictions["rating"].to_numpy(dtype=np.float32)
            - val_pred_rounded
        )
        val_mae = float(np.mean(np.abs(val_diff)))
        val_rmse = float(np.sqrt(np.mean(val_diff ** 2)))
        learning_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_epoch = int(epoch)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(model_config.early_stopping_patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return KnownUserDeepTrainingResult(
        model_config=model_config,
        architecture=architecture,
        model_state_dict=best_state,
        learning_curves=pd.DataFrame(learning_rows),
        best_epoch=best_epoch,
        best_val_mae=best_val_mae,
        best_val_rmse=best_val_rmse,
        train_size=int(len(train_data.targets)),
        val_size=int(len(val_data.targets)),
    )


def fit_known_user_deep_final_model(
    *,
    train_data: KnownUserDeepPreparedDataset,
    context: KnownUserDeepContext,
    training_config: KnownUserDeepTrainingConfig,
    architecture: KnownUserDeepE2EArchitecture,
    final_epochs: int,
) -> dict[str, Any]:
    model_config = training_config.to_model_config(
        max_history_len=context.feature_contract.max_history_len,
        history_summary_tokens=context.feature_contract.history_summary_tokens,
    )
    device = _resolve_device(model_config.device)
    print(f"Training final model for {final_epochs} epochs on device: {device}")
    model = KnownUserDeepE2EModel(architecture).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the model: {total_params}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
    )
    loader = _build_loader(train_data, training_config, shuffle=True)
    business_tensor = torch.tensor(context.business_matrix, dtype=torch.float32, device=device)

    for _ in range(max(int(final_epochs), 1)):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            batch_tensors = _prepare_batch_tensors(batch=batch, business_tensor=business_tensor, device=device)
            outputs = model(
                candidate_business_features=batch_tensors["candidate_business_features"],
                history_business_features=batch_tensors["history_business_features"],
                history_rating_features=batch_tensors["history_rating_features"],
                history_mask=batch_tensors["history_mask"],
                user_numeric_features=batch_tensors["user_numeric_features"],
                user_aux_features=batch_tensors["user_aux_features"],
                user_categorical_ids=batch_tensors["user_categorical_ids"],
                history_band_ids=batch_tensors["history_band_ids"],
                baseline_features=batch_tensors["baseline_features"],
                incumbent_prediction_raw=batch_tensors["incumbent_prediction_raw"],
            )
            targets = batch_tensors["target_rating"]
            loss = compute_known_user_deep_loss(
                outputs,
                target_rating=targets,
                like_target=targets.ge(4.0),
                dislike_target=targets.le(2.0),
                incumbent_prediction_raw=batch_tensors["incumbent_prediction_raw"],
                history_band_ids=batch_tensors["history_band_ids"],
                config=model_config,
            )
            loss.backward()
            optimizer.step()

    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def predict_known_user_dataset(
    *,
    model: KnownUserDeepE2EModel,
    prepared: KnownUserDeepPreparedDataset,
    context: KnownUserDeepContext,
    batch_size: int,
    device: str | torch.device,
    business_tensor: torch.Tensor | None = None,
) -> pd.DataFrame:
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    if business_tensor is None:
        business_tensor = torch.tensor(context.business_matrix, dtype=torch.float32, device=device_obj)
    loader = DataLoader(
        _KnownUserTensorDataset(prepared),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    predictions: list[np.ndarray] = []
    alphas: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch_tensors = _prepare_batch_tensors(batch=batch, business_tensor=business_tensor, device=device_obj)
            outputs = model(
                candidate_business_features=batch_tensors["candidate_business_features"],
                history_business_features=batch_tensors["history_business_features"],
                history_rating_features=batch_tensors["history_rating_features"],
                history_mask=batch_tensors["history_mask"],
                user_numeric_features=batch_tensors["user_numeric_features"],
                user_aux_features=batch_tensors["user_aux_features"],
                user_categorical_ids=batch_tensors["user_categorical_ids"],
                history_band_ids=batch_tensors["history_band_ids"],
                baseline_features=batch_tensors["baseline_features"],
                incumbent_prediction_raw=batch_tensors["incumbent_prediction_raw"],
            )
            predictions.append(outputs["predicted_rating"].cpu().numpy().astype(np.float32))
            alphas.append(outputs["alpha"].cpu().numpy().astype(np.float32))
            baselines.append(outputs["baseline_hat"].cpu().numpy().astype(np.float32))
            residuals.append(outputs["correction_hat"].cpu().numpy().astype(np.float32))
    frame = prepared.frame.copy()
    frame["predicted_rating"] = np.concatenate(predictions) if predictions else np.array([], dtype=np.float32)
    frame["alpha"] = np.concatenate(alphas) if alphas else np.array([], dtype=np.float32)
    frame["baseline_hat"] = np.concatenate(baselines) if baselines else np.array([], dtype=np.float32)
    frame["correction_hat"] = np.concatenate(residuals) if residuals else np.array([], dtype=np.float32)
    frame["residual_hat"] = frame["correction_hat"].astype(np.float32)
    return frame


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[KnownUserDeepE2EModel, dict[str, Any]]:
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    architecture = KnownUserDeepE2EArchitecture(**payload["architecture"])
    model = KnownUserDeepE2EModel(architecture)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def save_known_user_checkpoint(
    *,
    path: str | Path,
    model_state_dict: dict[str, Any],
    architecture: KnownUserDeepE2EArchitecture,
    feature_contract: KnownUserDeepFeatureContract,
    data_config: KnownUserDeepDataConfig,
    training_config: KnownUserDeepTrainingConfig,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    model_config = training_config.to_model_config(
        max_history_len=feature_contract.max_history_len,
        history_summary_tokens=feature_contract.history_summary_tokens,
    )
    payload = {
        "config": asdict(model_config),
        "architecture": asdict(architecture),
        "feature_contract": asdict(feature_contract),
        "data_config": asdict(data_config),
        "training_config": asdict(training_config),
        "model_state_dict": model_state_dict,
        "summary": extra_summary or {},
    }
    torch.save(payload, Path(path))


def _build_loader(
    prepared: KnownUserDeepPreparedDataset,
    training_config: KnownUserDeepTrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = _KnownUserTensorDataset(prepared)
    band_weights = training_config.band_sample_weights or {}
    if band_weights:
        weights = np.array(
            [float(band_weights.get(str(value), 1.0)) for value in prepared.frame["history_band"].astype(str)],
            dtype=np.float64,
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=training_config.batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=training_config.batch_size, shuffle=shuffle, num_workers=0)


def _resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(raw_device)


def _prepare_batch_tensors(
    *,
    batch: dict[str, torch.Tensor],
    business_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    history_item_idx = batch["history_item_idx"].to(device=device, dtype=torch.long)
    history_mask = history_item_idx.ge(0)
    candidate_item_idx = batch["candidate_item_idx"].to(device=device, dtype=torch.long)
    history_business_features = _gather_business_features(business_tensor, history_item_idx, history_mask)
    return {
        "history_business_features": history_business_features,
        "history_rating_features": batch["history_rating_features"].to(device=device, dtype=torch.float32),
        "history_mask": history_mask,
        "candidate_business_features": _gather_candidate_features(business_tensor, candidate_item_idx),
        "user_numeric_features": batch["user_numeric_features"].to(device=device, dtype=torch.float32),
        "user_aux_features": batch["user_aux_features"].to(device=device, dtype=torch.float32),
        "user_categorical_ids": batch["user_categorical_ids"].to(device=device, dtype=torch.long),
        "history_band_ids": batch["history_band_ids"].to(device=device, dtype=torch.long),
        "baseline_features": batch["baseline_features"].to(device=device, dtype=torch.float32),
        "incumbent_prediction_raw": batch["incumbent_prediction_raw"].to(device=device, dtype=torch.float32),
        "target_rating": batch["target_rating"].to(device=device, dtype=torch.float32),
    }


def _materialize_known_user_dataset(
    *,
    raw_frame: pd.DataFrame,
    router_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    history_item_idx: np.ndarray,
    history_ratings: np.ndarray,
    history_days: np.ndarray,
    exact_history_count: np.ndarray,
    incumbent_frame: pd.DataFrame | None,
    context: KnownUserDeepContext,
    training_config: KnownUserDeepTrainingConfig,
) -> KnownUserDeepPreparedDataset:
    feature_frame, history_item_idx, history_ratings, history_days = _assemble_known_user_feature_frame(
        raw_frame=raw_frame,
        router_frame=router_frame,
        target_frame=target_frame,
        history_item_idx=history_item_idx,
        history_ratings=history_ratings,
        history_days=history_days,
        exact_history_count=exact_history_count,
    )
    baseline_features = _normalize_dense_block(
        feature_frame[context.feature_contract.baseline_feature_names].fillna(0.0).to_numpy(dtype=np.float32),
        context.feature_contract.baseline_feature_means,
        context.feature_contract.baseline_feature_stds,
    )
    user_numeric_features = _normalize_dense_block(
        feature_frame[context.feature_contract.user_numeric_feature_names].fillna(0.0).to_numpy(dtype=np.float32),
        context.feature_contract.user_numeric_feature_means,
        context.feature_contract.user_numeric_feature_stds,
    )
    user_aux_features = _normalize_dense_block(
        feature_frame[context.feature_contract.user_aux_feature_names].fillna(0.0).to_numpy(dtype=np.float32),
        context.feature_contract.user_aux_feature_means,
        context.feature_contract.user_aux_feature_stds,
    )
    user_categorical_ids = np.column_stack(
        [
            _encode_categorical_values(feature_frame[column].astype(str).tolist(), context.feature_contract.user_categorical_levels[column])
            for column in context.feature_contract.user_categorical_feature_names
        ]
    ).astype(np.int64, copy=False)
    history_band_ids = _encode_categorical_values(
        feature_frame["history_band"].astype(str).tolist(),
        context.feature_contract.history_band_levels,
    )
    history_rating_features = _build_history_rating_features(
        history_ratings=history_ratings,
        history_days=history_days,
        user_average_stars=feature_frame["user_average_stars"].fillna(context.feature_contract.global_mean).to_numpy(dtype=np.float32),
        exact_history_count=feature_frame["history_count"].to_numpy(dtype=np.int32),
        global_mean=float(context.feature_contract.global_mean),
        half_life_days=float(context.data_config.recency_half_life_days),
        recency_weight_scale=float(training_config.recency_weight_scale),
    )
    history_rating_features = _normalize_event_feature_block(
        history_rating_features,
        context.feature_contract.event_scalar_feature_means,
        context.feature_contract.event_scalar_feature_stds,
    )
    if incumbent_frame is None:
        incumbent_raw = np.full(len(feature_frame), float(context.feature_contract.global_mean), dtype=np.float32)
        incumbent_branch = np.full(len(feature_frame), "unknown_model", dtype=object)
    else:
        incumbent_lookup = incumbent_frame.copy()
        incumbent_lookup["review_id"] = incumbent_lookup["review_id"].astype(str)
        merged_incumbent = (
            feature_frame[["review_id"]]
            .assign(review_id=lambda df: df["review_id"].astype(str))
            .merge(
                incumbent_lookup[["review_id", "incumbent_prediction_raw", "incumbent_branch"]],
                on="review_id",
                how="left",
            )
        )
        incumbent_raw = merged_incumbent["incumbent_prediction_raw"].fillna(float(context.feature_contract.global_mean)).to_numpy(dtype=np.float32)
        incumbent_branch = merged_incumbent["incumbent_branch"].fillna("unknown_model").astype(str).to_numpy(dtype=object)
    frame = feature_frame[
        [
            "review_id",
            "user",
            "item",
            "rating",
            "history_band",
            "history_count",
            "history_rating_std",
            "history_positive_share",
            "history_negative_share",
        ]
    ].copy()
    frame["incumbent_prediction_raw"] = incumbent_raw
    frame["incumbent_branch"] = incumbent_branch
    return KnownUserDeepPreparedDataset(
        frame=frame,
        history_item_idx=history_item_idx.astype(np.int32, copy=False),
        history_rating_features=history_rating_features,
        candidate_item_idx=feature_frame["item_idx"].fillna(-1).to_numpy(dtype=np.int32),
        user_numeric_features=user_numeric_features,
        user_aux_features=user_aux_features,
        user_categorical_ids=user_categorical_ids,
        history_band_ids=history_band_ids.astype(np.int64, copy=False),
        baseline_features=baseline_features,
        incumbent_prediction_raw=incumbent_raw,
        targets=feature_frame["rating"].to_numpy(dtype=np.float32),
    )


def _assemble_known_user_feature_frame(
    *,
    raw_frame: pd.DataFrame,
    router_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    history_item_idx: np.ndarray,
    history_ratings: np.ndarray,
    history_days: np.ndarray,
    exact_history_count: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    feature_frame = target_frame[["review_id", "user", "item", "rating", "timestamp", "item_idx"]].copy()
    feature_frame = feature_frame.merge(raw_frame, on="review_id", how="left", suffixes=("", "_raw"))
    feature_frame = feature_frame.merge(
        router_frame[
            [
                "review_id",
                "user_archetype_id",
                "user_metadata_completeness",
                "user_metadata_sparse_flag",
                "user_activity_bucket",
                "user_reputation_bucket",
                "user_tenure_bucket",
            ]
        ],
        on="review_id",
        how="left",
    )
    feature_frame["history_count"] = exact_history_count.astype(np.float32)
    feature_frame["history_count_log1p"] = np.log1p(feature_frame["history_count"]).astype(np.float32)
    feature_frame["history_band"] = pd.Series(
        [history_band_from_count(int(value)) for value in exact_history_count],
        index=feature_frame.index,
        dtype="string",
    )
    stats = _compute_history_scalar_stats(history_ratings, history_days, exact_history_count)
    for column, values in stats.items():
        feature_frame[column] = values.astype(np.float32)
    feature_frame["history_rating_range"] = (
        feature_frame["history_rating_max"].to_numpy(dtype=np.float32)
        - feature_frame["history_rating_min"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    feature_frame["history_count_is_2"] = (feature_frame["history_count"] == 2.0).astype(np.float32)
    feature_frame["history_count_is_3"] = (feature_frame["history_count"] == 3.0).astype(np.float32)
    feature_frame["history_count_is_4"] = (feature_frame["history_count"] == 4.0).astype(np.float32)
    feature_frame["history_count_is_5"] = (feature_frame["history_count"] == 5.0).astype(np.float32)
    feature_frame["history_count_is_2_3"] = feature_frame["history_count"].between(2.0, 3.0, inclusive="both").astype(np.float32)
    feature_frame["history_count_is_4_5"] = feature_frame["history_count"].between(4.0, 5.0, inclusive="both").astype(np.float32)
    feature_frame["history_rating_std_x_count"] = (
        feature_frame["history_rating_std"].to_numpy(dtype=np.float32)
        * feature_frame["history_count"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    feature_frame["business_review_count_log1p_x_history_count"] = (
        feature_frame["business_review_count_log1p"].fillna(0.0).to_numpy(dtype=np.float32)
        * feature_frame["history_count"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    feature_frame["business_rating_per_review_x_history_count"] = (
        feature_frame["business_rating_per_review"].fillna(0.0).to_numpy(dtype=np.float32)
        * feature_frame["history_count"].to_numpy(dtype=np.float32)
    ).astype(np.float32)

    known_mask = feature_frame["history_count"].to_numpy(dtype=np.float32) > 0.0
    feature_frame = feature_frame.loc[known_mask].reset_index(drop=True)
    return (
        feature_frame,
        history_item_idx[known_mask],
        history_ratings[known_mask],
        history_days[known_mask],
    )


def _compute_feature_normalization_stats(
    *,
    context_reviews: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    raw_spec: Any,
    router_spec: Any,
    business_index: pd.Series,
    max_history_len: int,
    global_mean: float,
    half_life_days: float,
) -> dict[str, list[float]]:
    prepared_reviews = _prepare_review_frame(context_reviews, business_index=business_index)
    ordered_reviews = _order_reviews(prepared_reviews)
    raw_frame = _align_feature_frame(build_raw_feature_frame(context_reviews, users_df, businesses_df, raw_spec), ordered_reviews["review_id"])
    router_frame = _align_feature_frame(build_router_feature_frame(context_reviews, users_df, businesses_df, router_spec), ordered_reviews["review_id"])
    arrays = _build_prefix_arrays_with_recency(ordered_reviews, max_history_len)
    feature_frame, history_item_idx, history_ratings, history_days = _assemble_known_user_feature_frame(
        raw_frame=raw_frame,
        router_frame=router_frame,
        target_frame=ordered_reviews,
        history_item_idx=arrays["history_item_idx"],
        history_ratings=arrays["history_ratings"],
        history_days=arrays["history_days"],
        exact_history_count=arrays["exact_history_count"],
    )
    event_features = _build_history_rating_features(
        history_ratings=history_ratings,
        history_days=history_days,
        user_average_stars=feature_frame["user_average_stars"].fillna(global_mean).to_numpy(dtype=np.float32),
        exact_history_count=feature_frame["history_count"].to_numpy(dtype=np.int32),
        global_mean=float(global_mean),
        half_life_days=float(half_life_days),
        recency_weight_scale=1.0,
    )
    return {
        "baseline_means": _feature_means(feature_frame, KNOWN_USER_BASELINE_FEATURE_COLUMNS),
        "baseline_stds": _feature_stds(feature_frame, KNOWN_USER_BASELINE_FEATURE_COLUMNS),
        "user_numeric_means": _feature_means(feature_frame, KNOWN_USER_NUMERIC_FEATURE_COLUMNS),
        "user_numeric_stds": _feature_stds(feature_frame, KNOWN_USER_NUMERIC_FEATURE_COLUMNS),
        "user_aux_means": _feature_means(feature_frame, KNOWN_USER_AUX_FEATURE_COLUMNS),
        "user_aux_stds": _feature_stds(feature_frame, KNOWN_USER_AUX_FEATURE_COLUMNS),
        "event_scalar_means": _event_feature_means(event_features),
        "event_scalar_stds": _event_feature_stds(event_features),
    }


def _feature_means(frame: pd.DataFrame, columns: list[str]) -> list[float]:
    values = frame[columns].fillna(0.0).to_numpy(dtype=np.float32)
    return values.mean(axis=0, dtype=np.float64).astype(np.float32).tolist()


def _feature_stds(frame: pd.DataFrame, columns: list[str]) -> list[float]:
    values = frame[columns].fillna(0.0).to_numpy(dtype=np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(np.isfinite(std) & (std >= 1e-3), std, 1.0).astype(np.float32)
    return std.tolist()


def _event_feature_means(event_features: np.ndarray) -> list[float]:
    flattened = event_features.reshape(-1, event_features.shape[-1]).astype(np.float32, copy=False)
    return flattened.mean(axis=0, dtype=np.float64).astype(np.float32).tolist()


def _event_feature_stds(event_features: np.ndarray) -> list[float]:
    flattened = event_features.reshape(-1, event_features.shape[-1]).astype(np.float32, copy=False)
    std = flattened.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(np.isfinite(std) & (std >= 1e-3), std, 1.0).astype(np.float32)
    return std.tolist()


def _normalize_dense_block(values: np.ndarray, means: list[float], stds: list[float]) -> np.ndarray:
    means_arr = np.asarray(means, dtype=np.float32)
    stds_arr = np.asarray(stds, dtype=np.float32)
    return ((values - means_arr[None, :]) / stds_arr[None, :]).astype(np.float32, copy=False)


def _normalize_event_feature_block(values: np.ndarray, means: list[float], stds: list[float]) -> np.ndarray:
    means_arr = np.asarray(means, dtype=np.float32)
    stds_arr = np.asarray(stds, dtype=np.float32)
    return ((values - means_arr[None, None, :]) / stds_arr[None, None, :]).astype(np.float32, copy=False)


def _prepare_review_frame(
    reviews_df: pd.DataFrame,
    *,
    business_index: pd.Series,
) -> pd.DataFrame:
    frame = canonicalize_reviews(reviews_df)
    if "review_id" not in frame.columns:
        frame["review_id"] = np.arange(len(frame), dtype=np.int64)
    if "rating" not in frame.columns:
        frame["rating"] = np.nan
    if "timestamp" not in frame.columns:
        raise ValueError("Review table is missing timestamp/date.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["item_idx"] = frame["item"].map(business_index).fillna(-1).astype(np.int32)
    return frame[["review_id", "user", "item", "rating", "timestamp", "item_idx"]].copy()


def _order_reviews(reviews_df: pd.DataFrame) -> pd.DataFrame:
    return reviews_df.sort_values(["user", "timestamp", "item"], kind="stable").reset_index(drop=True)


def _align_feature_frame(frame: pd.DataFrame, ordered_review_ids: pd.Series) -> pd.DataFrame:
    lookup = frame.copy()
    lookup["review_id"] = lookup["review_id"].astype(str)
    ordered = pd.DataFrame({"review_id": ordered_review_ids.astype(str)})
    return ordered.merge(lookup, on="review_id", how="left")


def _build_prefix_arrays_with_recency(
    interactions: pd.DataFrame,
    max_history_len: int,
) -> dict[str, np.ndarray]:
    ordered = _order_reviews(interactions)
    n_rows = len(ordered)
    history_item_idx = np.full((n_rows, max_history_len), -1, dtype=np.int32)
    history_ratings = np.zeros((n_rows, max_history_len), dtype=np.float32)
    history_days = np.zeros((n_rows, max_history_len), dtype=np.float32)
    exact_history_count = np.zeros(n_rows, dtype=np.int32)

    for _, group in ordered.groupby("user", sort=False):
        group_indices = group.index.to_numpy(dtype=np.int32)
        item_values = group["item_idx"].to_numpy(dtype=np.int32)
        rating_values = group["rating"].to_numpy(dtype=np.float32)
        timestamps = group["timestamp"].astype("int64", copy=False).to_numpy(dtype=np.int64)
        for local_idx, global_idx in enumerate(group_indices):
            exact_history_count[global_idx] = int(local_idx)
            start = max(0, local_idx - max_history_len)
            history_slice_items = item_values[start:local_idx]
            history_slice_ratings = rating_values[start:local_idx]
            history_slice_times = timestamps[start:local_idx]
            if len(history_slice_items) == 0:
                continue
            history_item_idx[global_idx, : len(history_slice_items)] = history_slice_items
            history_ratings[global_idx, : len(history_slice_ratings)] = history_slice_ratings
            delta_days = (timestamps[local_idx] - history_slice_times).astype(np.float64) / 86_400_000_000_000.0
            history_days[global_idx, : len(delta_days)] = np.clip(delta_days, a_min=0.0, a_max=None).astype(np.float32)

    return {
        "history_item_idx": history_item_idx,
        "history_ratings": history_ratings,
        "history_days": history_days,
        "exact_history_count": exact_history_count,
    }


def _build_fixed_context_arrays_with_recency(
    *,
    target_interactions: pd.DataFrame,
    context_interactions: pd.DataFrame,
    max_history_len: int,
) -> dict[str, np.ndarray]:
    ordered_targets = _order_reviews(target_interactions)
    n_rows = len(ordered_targets)
    history_item_idx = np.full((n_rows, max_history_len), -1, dtype=np.int32)
    history_ratings = np.zeros((n_rows, max_history_len), dtype=np.float32)
    history_days = np.zeros((n_rows, max_history_len), dtype=np.float32)
    exact_history_count = np.zeros(n_rows, dtype=np.int32)

    history_lookup = _build_history_lookup(context_interactions, max_history_len=max_history_len)
    target_timestamps = ordered_targets["timestamp"].astype("int64", copy=False).to_numpy(dtype=np.int64)
    for row_idx, row in ordered_targets.reset_index(drop=True).iterrows():
        items, ratings, timestamps, total_count = history_lookup.get(
            row["user"],
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64), 0),
        )
        exact_history_count[row_idx] = int(total_count)
        if len(items) == 0:
            continue
        history_item_idx[row_idx, : len(items)] = items
        history_ratings[row_idx, : len(ratings)] = ratings
        delta_days = (target_timestamps[row_idx] - timestamps).astype(np.float64) / 86_400_000_000_000.0
        history_days[row_idx, : len(items)] = np.clip(delta_days, a_min=0.0, a_max=None).astype(np.float32)

    return {
        "history_item_idx": history_item_idx,
        "history_ratings": history_ratings,
        "history_days": history_days,
        "exact_history_count": exact_history_count,
    }


def _build_history_lookup(
    interactions: pd.DataFrame,
    *,
    max_history_len: int,
) -> dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    ordered = _order_reviews(interactions)
    lookup: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for user, group in ordered.groupby("user", sort=False):
        items = group["item_idx"].to_numpy(dtype=np.int32)
        ratings = group["rating"].to_numpy(dtype=np.float32)
        timestamps = group["timestamp"].astype("int64", copy=False).to_numpy(dtype=np.int64)
        lookup[user] = (
            items[-max_history_len:].copy(),
            ratings[-max_history_len:].copy(),
            timestamps[-max_history_len:].copy(),
            int(len(items)),
        )
    return lookup


def _compute_history_scalar_stats(
    history_ratings: np.ndarray,
    history_days: np.ndarray,
    exact_history_count: np.ndarray,
) -> dict[str, np.ndarray]:
    max_history_len = history_ratings.shape[1]
    positions = np.arange(max_history_len, dtype=np.int32)
    clipped_count = np.minimum(exact_history_count, max_history_len)
    mask = positions[None, :] < clipped_count[:, None]
    count = np.clip(clipped_count.astype(np.float32), 1.0, None)
    rating_sum = (history_ratings * mask).sum(axis=1, dtype=np.float32)
    rating_mean = rating_sum / count
    centered = (history_ratings - rating_mean[:, None]) * mask
    rating_std = np.sqrt((centered ** 2).sum(axis=1, dtype=np.float32) / count).astype(np.float32)
    min_rating = np.where(mask, history_ratings, np.inf).min(axis=1)
    max_rating = np.where(mask, history_ratings, -np.inf).max(axis=1)
    safe_last_positions = np.clip(clipped_count - 1, 0, max_history_len - 1)
    last_rating = history_ratings[np.arange(len(history_ratings)), safe_last_positions]
    last_rating = np.where(exact_history_count > 0, last_rating, 0.0).astype(np.float32)
    has_history = clipped_count > 0
    min_rating = np.where(has_history, min_rating, 0.0).astype(np.float32)
    max_rating = np.where(has_history, max_rating, 0.0).astype(np.float32)
    positive_share = ((history_ratings >= 4.0) & mask).sum(axis=1, dtype=np.float32) / count
    negative_share = ((history_ratings <= 2.0) & mask).sum(axis=1, dtype=np.float32) / count
    recency_days_mean = (history_days * mask).sum(axis=1, dtype=np.float32) / count
    return {
        "history_rating_mean": np.where(has_history, rating_mean, 0.0).astype(np.float32),
        "history_rating_std": np.where(has_history, rating_std, 0.0).astype(np.float32),
        "history_rating_min": min_rating,
        "history_rating_max": max_rating,
        "history_last_rating": last_rating,
        "history_positive_share": np.where(has_history, positive_share, 0.0).astype(np.float32),
        "history_negative_share": np.where(has_history, negative_share, 0.0).astype(np.float32),
        "history_recency_days_mean": np.where(has_history, recency_days_mean, 0.0).astype(np.float32),
    }


def _build_history_rating_features(
    *,
    history_ratings: np.ndarray,
    history_days: np.ndarray,
    user_average_stars: np.ndarray,
    exact_history_count: np.ndarray,
    global_mean: float,
    half_life_days: float,
    recency_weight_scale: float,
) -> np.ndarray:
    max_history_len = history_ratings.shape[1]
    positions = np.arange(max_history_len, dtype=np.int32)
    clipped_count = np.minimum(exact_history_count, max_history_len)
    mask = positions[None, :] < clipped_count[:, None]
    ratings = np.where(mask, history_ratings, 0.0).astype(np.float32)
    days = np.where(mask, history_days, 0.0).astype(np.float32)
    user_centered = (ratings - user_average_stars[:, None]).astype(np.float32)
    global_centered = (ratings - np.float32(global_mean)).astype(np.float32)
    liked = ((ratings >= 4.0) & mask).astype(np.float32)
    disliked = ((ratings <= 2.0) & mask).astype(np.float32)
    abs_dev_user = np.abs(user_centered).astype(np.float32)
    log_days = np.log1p(days).astype(np.float32)
    decay = np.exp(-np.log(2.0) * (days / max(half_life_days, 1e-6))).astype(np.float32)
    decay = (decay * np.float32(recency_weight_scale)).astype(np.float32)
    features = np.stack([ratings, user_centered, global_centered, liked, disliked, abs_dev_user, days, log_days, decay], axis=-1).astype(np.float32, copy=False)
    features *= mask[..., None].astype(np.float32)
    return features


def _gather_business_features(business_tensor: torch.Tensor, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch_size, history_len = indices.shape
    feature_dim = business_tensor.shape[1]
    gathered = business_tensor[indices.clamp_min(0).reshape(-1)].reshape(batch_size, history_len, feature_dim)
    return gathered * mask.unsqueeze(-1).to(gathered.dtype)


def _gather_candidate_features(business_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    feature_dim = business_tensor.shape[1]
    gathered = business_tensor[indices.clamp_min(0)]
    valid_mask = indices.ge(0).unsqueeze(-1)
    zeros = torch.zeros(indices.shape[0], feature_dim, device=business_tensor.device, dtype=business_tensor.dtype)
    return torch.where(valid_mask, gathered, zeros)


def _normalize_levels(levels: list[str]) -> list[str]:
    seen: list[str] = []
    for value in list(levels) + ["__unknown__"]:
        text = str(value)
        if text not in seen:
            seen.append(text)
    return seen


def _encode_categorical_values(values: list[str], levels: list[str]) -> np.ndarray:
    mapping = {level: index for index, level in enumerate(levels, start=1)}
    default_idx = mapping.get("__unknown__", 0)
    return np.array([mapping.get(str(value), default_idx) for value in values], dtype=np.int64)
