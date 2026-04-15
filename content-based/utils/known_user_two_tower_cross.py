from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from model.known_user_two_tower_cross import (
    KnownUserTwoTowerArchitecture,
    KnownUserTwoTowerConfig,
    KnownUserTwoTowerCrossModel,
    build_known_user_two_tower_architecture,
    compute_known_user_two_tower_loss,
)
from .business_representation import BusinessRepresentationBuilder, BusinessRepresentationConfig
from .known_user_deep_e2e import (
    EVENT_SCALAR_FEATURE_NAMES,
    HISTORY_BAND_LEVELS,
    KNOWN_USER_AUX_FEATURE_COLUMNS,
    KNOWN_USER_BASELINE_FEATURE_COLUMNS,
    KNOWN_USER_CATEGORICAL_COLUMNS,
    KNOWN_USER_NUMERIC_FEATURE_COLUMNS,
    _align_feature_frame,
    _build_fixed_context_arrays_with_recency,
    _build_loader,
    _build_prefix_arrays_with_recency,
    _compute_feature_normalization_stats,
    _materialize_known_user_dataset,
    _normalize_levels,
    _order_reviews,
    _prepare_batch_tensors,
    _prepare_review_frame,
    _resolve_device,
    load_safe_business_feature_block,
)
from .lgbm_raw_features import RAW_CORE_FEATURE_SET, build_raw_feature_frame, fit_raw_feature_spec
from .lgbm_raw_router_features import build_router_feature_frame, fit_router_feature_spec


@dataclass(slots=True)
class KnownUserTwoTowerDataConfig:
    business_source: str = "structured_from_scratch"
    business_repr_root: str | None = None
    business_repr_view: str = "content"
    max_history_len: int = 20
    n_user_archetypes: int = 64
    max_top_cities: int = 100
    max_top_categories: int = 32
    random_seed: int = 42
    recency_half_life_days: float = 180.0
    structured_min_city_freq: int = 20
    structured_max_city_ohe: int = 200
    structured_city_hash_dim: int = 64
    structured_min_category_freq: int = 20
    structured_min_attribute_value_freq: int = 30
    structured_max_attribute_values_per_key: int = 12


@dataclass(slots=True)
class KnownUserTwoTowerTrainingConfig:
    embedding_dim: int = 128
    event_hidden_dim: int = 128
    user_type_hidden_dim: int = 128
    business_hidden_layers: tuple[int, ...] = (512, 256)
    fusion_hidden_layers: tuple[int, ...] = (256, 128)
    baseline_hidden_layers: tuple[int, ...] = (128, 64)
    cross_hidden_layers: tuple[int, ...] = (512, 256)
    cross_depth: int = 3
    num_attention_heads: int = 4
    dropout: float = 0.15
    batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 2e-5
    max_epochs: int = 20
    early_stopping_patience: int = 4
    device: str = "auto"
    random_seed: int = 42
    band_sample_weights: dict[str, float] | None = None
    recency_weight_scale: float = 1.0
    band_correction_scales: dict[str, float] | None = None
    band_distillation_weights: dict[str, float] | None = None

    def to_model_config(self, *, max_history_len: int) -> KnownUserTwoTowerConfig:
        return KnownUserTwoTowerConfig(
            max_history_len=max_history_len,
            embedding_dim=self.embedding_dim,
            business_hidden_layers=tuple(self.business_hidden_layers),
            event_hidden_layers=(self.event_hidden_dim,),
            user_hidden_layers=(self.user_type_hidden_dim,),
            fusion_hidden_layers=tuple(self.fusion_hidden_layers),
            baseline_hidden_layers=tuple(self.baseline_hidden_layers),
            cross_hidden_layers=tuple(self.cross_hidden_layers),
            cross_depth=int(self.cross_depth),
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            random_seed=self.random_seed,
            device=self.device,
            band_correction_scales=self.band_correction_scales,
            band_distillation_weights=self.band_distillation_weights,
        )


@dataclass(slots=True)
class KnownUserTwoTowerFeatureContract:
    business_source: str
    business_repr_root: str | None
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
    global_mean: float


@dataclass(slots=True)
class KnownUserTwoTowerContext:
    data_config: KnownUserTwoTowerDataConfig
    feature_contract: KnownUserTwoTowerFeatureContract
    raw_spec: Any
    router_spec: Any
    business_ids: pd.Series
    business_matrix: np.ndarray
    business_index: pd.Series
    business_source_summary: dict[str, Any]


@dataclass(slots=True)
class KnownUserTwoTowerTrainingResult:
    model_config: KnownUserTwoTowerConfig
    architecture: KnownUserTwoTowerArchitecture
    model_state_dict: dict[str, Any]
    learning_curves: pd.DataFrame
    best_epoch: int
    best_val_mae: float
    best_val_rmse: float
    train_size: int
    val_size: int


def _load_business_features(
    *,
    context_reviews: pd.DataFrame,
    businesses_df: pd.DataFrame,
    data_config: KnownUserTwoTowerDataConfig,
) -> tuple[pd.Series, np.ndarray, list[str], dict[str, Any]]:
    business_source = str(data_config.business_source)
    if business_source == "structured_from_scratch":
        representation_reviews = context_reviews.copy()
        rename_back: dict[str, str] = {}
        if "item" in representation_reviews.columns and "business_id" not in representation_reviews.columns:
            rename_back["item"] = "business_id"
        if "rating" in representation_reviews.columns and "stars" not in representation_reviews.columns:
            rename_back["rating"] = "stars"
        if "timestamp" in representation_reviews.columns and "date" not in representation_reviews.columns:
            rename_back["timestamp"] = "date"
        if rename_back:
            representation_reviews = representation_reviews.rename(columns=rename_back)
        builder = BusinessRepresentationBuilder(
            BusinessRepresentationConfig(
                min_city_freq=data_config.structured_min_city_freq,
                max_city_ohe=data_config.structured_max_city_ohe,
                city_hash_dim=data_config.structured_city_hash_dim,
                min_category_freq=data_config.structured_min_category_freq,
                min_attribute_value_freq=data_config.structured_min_attribute_value_freq,
                max_attribute_values_per_key=data_config.structured_max_attribute_values_per_key,
                include_geo_clusters=False,
                include_priors=False,
            )
        )
        bundle = builder.fit_transform(businesses_df, representation_reviews)
        matrix = bundle.get_matrix(view=data_config.business_repr_view).toarray().astype(np.float32, copy=False)
        feature_names = bundle.content_feature_names if data_config.business_repr_view == "content" else bundle.full_feature_names
        summary = {
            "business_source": business_source,
            "business_repr_view": data_config.business_repr_view,
            "representation_summary": bundle.representation_summary,
        }
        return bundle.business_ids.astype(str), matrix, list(feature_names), summary
    if business_source in {"bundle", "bundle_iter03", "bundle_iter04"}:
        if not data_config.business_repr_root:
            raise ValueError("business_repr_root is required when business_source uses a bundle.")
        business_ids, business_matrix, feature_names = load_safe_business_feature_block(data_config.business_repr_root)
        summary = {
            "business_source": business_source,
            "business_repr_root": str(data_config.business_repr_root),
            "business_repr_view": "content",
        }
        return business_ids.astype(str), business_matrix, list(feature_names), summary
    raise ValueError(f"Unsupported business_source: {business_source}")


def prepare_known_user_two_tower_context(
    *,
    context_reviews: pd.DataFrame,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    data_config: KnownUserTwoTowerDataConfig,
) -> KnownUserTwoTowerContext:
    raw_spec = fit_raw_feature_spec(context_reviews, users_df, businesses_df, feature_set=RAW_CORE_FEATURE_SET)
    router_spec = fit_router_feature_spec(
        context_reviews,
        users_df,
        businesses_df,
        n_user_archetypes=data_config.n_user_archetypes,
        max_top_cities=data_config.max_top_cities,
        max_top_categories=data_config.max_top_categories,
        random_seed=data_config.random_seed,
    )
    business_ids, business_matrix, business_feature_names, business_source_summary = _load_business_features(
        context_reviews=context_reviews,
        businesses_df=businesses_df,
        data_config=data_config,
    )
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
    feature_contract = KnownUserTwoTowerFeatureContract(
        business_source=str(data_config.business_source),
        business_repr_root=str(data_config.business_repr_root) if data_config.business_repr_root else None,
        business_view=str(data_config.business_repr_view),
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
        global_mean=float(raw_spec.global_mean),
    )
    return KnownUserTwoTowerContext(
        data_config=data_config,
        feature_contract=feature_contract,
        raw_spec=raw_spec,
        router_spec=router_spec,
        business_ids=business_ids,
        business_matrix=business_matrix,
        business_index=business_index,
        business_source_summary=business_source_summary,
    )


def build_known_user_two_tower_train_dataset(
    train_reviews: pd.DataFrame,
    *,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    context: KnownUserTwoTowerContext,
    training_config: KnownUserTwoTowerTrainingConfig | None = None,
    incumbent_frame: pd.DataFrame | None = None,
) -> Any:
    training_config = training_config or KnownUserTwoTowerTrainingConfig()
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


def build_known_user_two_tower_eval_dataset(
    target_reviews: pd.DataFrame,
    context_reviews: pd.DataFrame,
    *,
    users_df: pd.DataFrame,
    businesses_df: pd.DataFrame,
    context: KnownUserTwoTowerContext,
    training_config: KnownUserTwoTowerTrainingConfig | None = None,
    incumbent_frame: pd.DataFrame | None = None,
) -> Any:
    training_config = training_config or KnownUserTwoTowerTrainingConfig()
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


def train_known_user_two_tower_model(
    *,
    train_data: Any,
    val_data: Any,
    context: KnownUserTwoTowerContext,
    training_config: KnownUserTwoTowerTrainingConfig,
) -> KnownUserTwoTowerTrainingResult:
    torch.manual_seed(training_config.random_seed)
    np.random.seed(training_config.random_seed)
    model_config = training_config.to_model_config(max_history_len=context.feature_contract.max_history_len)
    architecture = build_known_user_two_tower_architecture(context.feature_contract, model_config)
    device = _resolve_device(model_config.device)
    model = KnownUserTwoTowerCrossModel(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
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
            loss = compute_known_user_two_tower_loss(
                outputs,
                target_rating=batch_tensors["target_rating"],
                incumbent_prediction_raw=batch_tensors["incumbent_prediction_raw"],
                history_band_ids=batch_tensors["history_band_ids"],
                config=model_config,
            )
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        val_predictions = predict_known_user_two_tower_dataset(
            model=model,
            prepared=val_data,
            context=context,
            batch_size=model_config.batch_size,
            device=device,
            business_tensor=business_tensor,
        )
        val_diff = val_predictions["rating"].to_numpy(dtype=np.float32) - val_predictions["predicted_rating"].to_numpy(dtype=np.float32)
        val_mae = float(np.mean(np.abs(val_diff)))
        val_rmse = float(np.sqrt(np.mean(val_diff ** 2)))
        learning_rows.append({"epoch": int(epoch), "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"), "val_mae": val_mae, "val_rmse": val_rmse})
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

    return KnownUserTwoTowerTrainingResult(
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


def fit_known_user_two_tower_final_model(
    *,
    train_data: Any,
    context: KnownUserTwoTowerContext,
    training_config: KnownUserTwoTowerTrainingConfig,
    architecture: KnownUserTwoTowerArchitecture,
    final_epochs: int,
) -> dict[str, Any]:
    model_config = training_config.to_model_config(max_history_len=context.feature_contract.max_history_len)
    device = _resolve_device(model_config.device)
    model = KnownUserTwoTowerCrossModel(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=model_config.weight_decay)
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
            loss = compute_known_user_two_tower_loss(
                outputs,
                target_rating=batch_tensors["target_rating"],
                incumbent_prediction_raw=batch_tensors["incumbent_prediction_raw"],
                history_band_ids=batch_tensors["history_band_ids"],
                config=model_config,
            )
            loss.backward()
            optimizer.step()
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def predict_known_user_two_tower_dataset(
    *,
    model: KnownUserTwoTowerCrossModel,
    prepared: Any,
    context: KnownUserTwoTowerContext,
    batch_size: int,
    device: str | torch.device,
    business_tensor: torch.Tensor | None = None,
) -> pd.DataFrame:
    from .known_user_deep_e2e import _KnownUserTensorDataset

    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    if business_tensor is None:
        business_tensor = torch.tensor(context.business_matrix, dtype=torch.float32, device=device_obj)
    loader = torch.utils.data.DataLoader(_KnownUserTensorDataset(prepared), batch_size=batch_size, shuffle=False, num_workers=0)
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


def save_known_user_two_tower_checkpoint(
    *,
    path: str | Path,
    model_state_dict: dict[str, Any],
    architecture: KnownUserTwoTowerArchitecture,
    feature_contract: KnownUserTwoTowerFeatureContract,
    data_config: KnownUserTwoTowerDataConfig,
    training_config: KnownUserTwoTowerTrainingConfig,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    model_config = training_config.to_model_config(max_history_len=feature_contract.max_history_len)
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
