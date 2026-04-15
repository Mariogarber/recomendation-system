import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from pipelines.deep.train_known_user_deep import _history_band_lookup_from_context, _run_summary
from model.known_user_deep_e2e import KnownUserDeepE2EConfig, KnownUserDeepE2EModel, build_known_user_deep_e2e_architecture
from utils.known_user_deep_e2e import (
    KnownUserDeepDataConfig,
    KnownUserDeepTrainingConfig,
    _prepare_batch_tensors,
    build_known_user_eval_dataset,
    build_known_user_train_dataset,
    build_model_from_checkpoint,
    load_safe_business_feature_block,
    prepare_known_user_context,
    save_known_user_checkpoint,
)


def _make_users() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "review_count": [3, 1],
            "yelping_since": ["2020-01-01", "2021-01-01"],
            "useful": [10, 2],
            "funny": [1, 0],
            "cool": [5, 1],
            "elite": ["2020,2021", ""],
            "friends": ["a,b,c", ""],
            "fans": [2, 0],
            "average_stars": [4.2, 3.0],
            "compliment_hot": [1, 0],
            "compliment_more": [0, 0],
            "compliment_profile": [0, 0],
            "compliment_cute": [0, 0],
            "compliment_list": [0, 0],
            "compliment_note": [1, 0],
            "compliment_plain": [0, 0],
            "compliment_cool": [1, 0],
            "compliment_funny": [0, 0],
            "compliment_writer": [1, 0],
            "compliment_photos": [0, 0],
        }
    )


def _make_businesses() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "business_id": ["b1", "b2", "b3"],
            "city": ["A", "A", "B"],
            "state": ["X", "X", "Y"],
            "postal_code": ["1", "1", "2"],
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [1.0, 2.0, 3.0],
            "stars": [4.0, 3.5, 4.5],
            "review_count": [10, 20, 5],
            "is_open": [1, 1, 0],
            "attributes": ["{}", "{}", "{}"],
            "categories": ["Food, Cafe", "Food", "Bars"],
            "hours": ["{}", "{}", "{}"],
        }
    )


def _make_train_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "user_id": ["u1", "u1", "u1", "u2"],
            "business_id": ["b1", "b2", "b3", "b1"],
            "stars": [5.0, 1.0, 4.0, 3.0],
            "date": [
                "2023-01-01 10:00:00",
                "2023-01-02 10:00:00",
                "2023-01-03 10:00:00",
                "2023-01-04 10:00:00",
            ],
            "useful": [0, 1, 0, 0],
            "funny": [0, 0, 0, 0],
            "cool": [0, 0, 1, 0],
        }
    )


def _write_business_repr(tmp_path) -> str:
    root = tmp_path / "business_repr"
    root.mkdir()
    pd.DataFrame({"business_id": ["b1", "b2", "b3"]}).to_csv(root / "business_ids.csv", index=False)
    sparse.save_npz(
        root / "business_content_features.npz",
        sparse.csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            )
        ),
    )
    (root / "business_feature_names.json").write_text(
        json.dumps({"content_features": ["f1", "f2", "f3", "bias"]}),
        encoding="utf-8",
    )
    return str(root)


def _build_context(tmp_path):
    data_config = KnownUserDeepDataConfig(business_repr_root=_write_business_repr(tmp_path), max_history_len=5, n_user_archetypes=4)
    return prepare_known_user_context(
        context_reviews=_make_train_reviews(),
        users_df=_make_users(),
        businesses_df=_make_businesses(),
        data_config=data_config,
    )


def test_train_dataset_is_prefix_safe(tmp_path):
    context = _build_context(tmp_path)
    dataset = build_known_user_train_dataset(
        _make_train_reviews(),
        users_df=_make_users(),
        businesses_df=_make_businesses(),
        context=context,
    )
    rows = dataset.frame.set_index("review_id")
    assert rows.loc["r2", "history_count"] == 1.0
    assert rows.loc["r3", "history_count"] == 2.0
    assert dataset.history_item_idx[0, 0] == 0
    assert dataset.history_item_idx[1, 0] == 0
    assert dataset.history_item_idx[1, 1] == 1


def test_eval_dataset_uses_context_only(tmp_path):
    context = _build_context(tmp_path)
    train_context = _make_train_reviews().iloc[:2].copy()
    val_target = pd.DataFrame(
        {
            "review_id": ["v1"],
            "user_id": ["u1"],
            "business_id": ["b3"],
            "stars": [4.0],
            "date": ["2023-01-03 10:00:00"],
        }
    )
    dataset = build_known_user_eval_dataset(
        val_target,
        train_context,
        users_df=_make_users(),
        businesses_df=_make_businesses(),
        context=context,
    )
    assert dataset.frame.iloc[0]["history_count"] == 2.0
    assert dataset.history_item_idx.shape == (1, 5)
    assert dataset.history_item_idx[0, 0] == 0
    assert dataset.history_item_idx[0, 1] == 1


def test_model_forward_and_checkpoint_roundtrip(tmp_path):
    context = _build_context(tmp_path)
    dataset = build_known_user_train_dataset(
        _make_train_reviews(),
        users_df=_make_users(),
        businesses_df=_make_businesses(),
        context=context,
    )
    model_config = KnownUserDeepE2EConfig(
        max_history_len=context.feature_contract.max_history_len,
        history_summary_tokens=context.feature_contract.history_summary_tokens,
        embedding_dim=32,
        business_hidden_layers=(32,),
        event_hidden_layers=(32,),
        user_hidden_layers=(32,),
        taste_hidden_layers=(32,),
        baseline_hidden_layers=(16,),
        gate_hidden_dim=16,
        categorical_embedding_dim=4,
        history_band_embedding_dim=4,
        num_attention_heads=4,
        dropout=0.0,
    )
    architecture = build_known_user_deep_e2e_architecture(context.feature_contract, model_config)
    model = KnownUserDeepE2EModel(architecture)
    batch = {
        "history_item_idx": torch.from_numpy(dataset.history_item_idx[:2]),
        "history_rating_features": torch.from_numpy(dataset.history_rating_features[:2]),
        "candidate_item_idx": torch.from_numpy(dataset.candidate_item_idx[:2]),
        "user_numeric_features": torch.from_numpy(dataset.user_numeric_features[:2]),
        "user_aux_features": torch.from_numpy(dataset.user_aux_features[:2]),
        "user_categorical_ids": torch.from_numpy(dataset.user_categorical_ids[:2]),
        "history_band_ids": torch.from_numpy(dataset.history_band_ids[:2]),
        "baseline_features": torch.from_numpy(dataset.baseline_features[:2]),
        "incumbent_prediction_raw": torch.from_numpy(dataset.incumbent_prediction_raw[:2]),
        "target_rating": torch.from_numpy(dataset.targets[:2]),
    }
    prepared = _prepare_batch_tensors(
        batch=batch,
        business_tensor=torch.tensor(context.business_matrix, dtype=torch.float32),
        device=torch.device("cpu"),
    )
    outputs = model(
        candidate_business_features=prepared["candidate_business_features"],
        history_business_features=prepared["history_business_features"],
        history_rating_features=prepared["history_rating_features"],
        history_mask=prepared["history_mask"],
        user_numeric_features=prepared["user_numeric_features"],
        user_aux_features=prepared["user_aux_features"],
        user_categorical_ids=prepared["user_categorical_ids"],
        history_band_ids=prepared["history_band_ids"],
        baseline_features=prepared["baseline_features"],
        incumbent_prediction_raw=prepared["incumbent_prediction_raw"],
    )
    assert outputs["predicted_rating"].shape == (2,)
    assert outputs["alpha"].shape == (2,)
    assert outputs["expert_index"].shape == (2,)
    assert torch.all(outputs["correction_hat"].abs() <= 1.01)

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_known_user_checkpoint(
        path=checkpoint_path,
        model_state_dict=model.state_dict(),
        architecture=architecture,
        feature_contract=context.feature_contract,
        data_config=context.data_config,
        training_config=KnownUserDeepTrainingConfig(
            embedding_dim=32,
            event_hidden_dim=32,
            user_type_hidden_dim=32,
            scorer_hidden_dim=64,
            business_hidden_layers=(32,),
            scorer_hidden_layers=(32,),
            num_attention_heads=4,
            dropout=0.0,
        ),
    )
    loaded_model, payload = build_model_from_checkpoint(checkpoint_path)
    assert isinstance(loaded_model, KnownUserDeepE2EModel)
    assert payload["architecture"]["embedding_dim"] == 32
    assert load_safe_business_feature_block(context.data_config.business_repr_root)[0].tolist() == ["b1", "b2", "b3"]


def test_dataset_carries_incumbent_prediction_default(tmp_path):
    context = _build_context(tmp_path)
    dataset = build_known_user_train_dataset(
        _make_train_reviews(),
        users_df=_make_users(),
        businesses_df=_make_businesses(),
        context=context,
    )
    assert "incumbent_prediction_raw" in dataset.frame.columns
    assert np.allclose(dataset.frame["incumbent_prediction_raw"].to_numpy(dtype=np.float32), dataset.incumbent_prediction_raw)


def test_feature_injected_short_history_columns_exist(tmp_path):
    context = _build_context(tmp_path)
    dataset = build_known_user_train_dataset(
        _make_train_reviews(),
        users_df=_make_users(),
        businesses_df=_make_businesses(),
        context=context,
    )
    assert "history_count_is_2" in context.feature_contract.user_aux_feature_names
    assert "history_rating_range" in context.feature_contract.baseline_feature_names
    assert dataset.user_aux_features.shape[1] == len(context.feature_contract.user_aux_feature_names)
    assert dataset.baseline_features.shape[1] == len(context.feature_contract.baseline_feature_names)


def test_expert_dispatch_uses_band_groups():
    tmp_dir = tempfile.mkdtemp(dir=Path.cwd())
    context = _build_context(Path(tmp_dir))
    dataset = build_known_user_train_dataset(
        _make_train_reviews(),
        users_df=_make_users(),
        businesses_df=_make_businesses(),
        context=context,
    )
    model_config = KnownUserDeepE2EConfig(
        max_history_len=context.feature_contract.max_history_len,
        history_summary_tokens=context.feature_contract.history_summary_tokens,
        embedding_dim=16,
        business_hidden_layers=(16,),
        event_hidden_layers=(16,),
        user_hidden_layers=(16,),
        taste_hidden_layers=(16,),
        baseline_hidden_layers=(16,),
        gate_hidden_dim=8,
        categorical_embedding_dim=4,
        history_band_embedding_dim=4,
        num_attention_heads=4,
        dropout=0.0,
    )
    architecture = build_known_user_deep_e2e_architecture(context.feature_contract, model_config)
    model = KnownUserDeepE2EModel(architecture)
    history_band_ids = torch.tensor([2, 3, 3, 4, 5], dtype=torch.long)
    history_lengths = torch.tensor([1, 2, 5, 8, 25], dtype=torch.long)
    masks = model._expert_masks(history_band_ids, history_lengths)
    assert masks["band_1"].tolist() == [True, False, False, False, False]
    assert masks["band_2_3"].tolist() == [False, True, False, False, False]
    assert masks["band_4_5"].tolist() == [False, False, True, False, False]
    assert masks["band_6_20"].tolist() == [False, False, False, True, False]
    assert masks["band_gt_20"].tolist() == [False, False, False, False, True]


class _DummyTrainingResult:
    best_epoch = 2
    best_val_mae = 0.7
    best_val_rmse = 1.0
    train_size = 10
    val_size = 4


def _make_summary_training_config() -> KnownUserDeepTrainingConfig:
    return KnownUserDeepTrainingConfig(
        embedding_dim=32,
        event_hidden_dim=32,
        user_type_hidden_dim=32,
        scorer_hidden_dim=64,
        business_hidden_layers=(32,),
        scorer_hidden_layers=(32,),
        num_attention_heads=4,
        dropout=0.0,
    )


def test_run_summary_compares_only_rows_with_deep_coverage():
    incumbent_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "rating": [1.0, 5.0],
            "history_band": ["6-20", "6-20"],
            "incumbent_prediction_raw": [1.0, 1.0],
            "incumbent_prediction": [1.0, 1.0],
            "incumbent_branch": ["known_prefix_deep_model", "known_prefix_deep_model"],
        }
    )
    deep_val = pd.DataFrame(
        {
            "review_id": ["r2"],
            "rating": [5.0],
            "history_band": ["6-20"],
            "deep_prediction_raw": [1.0],
            "deep_prediction": [1.0],
            "alpha": [0.5],
        }
    )

    summary = _run_summary(
        run_name="dummy",
        training_config=_make_summary_training_config(),
        incumbent_val=incumbent_val,
        deep_val=deep_val,
        training_result=_DummyTrainingResult(),
        enable_margin=0.001,
    )

    comparison = {row["history_band"]: row for row in summary["band_comparison"]}
    assert comparison["6-20"]["deep_available_rows"] == 1
    assert np.isclose(comparison["6-20"]["incumbent_mae"], 4.0)
    assert np.isclose(comparison["6-20"]["deep_mae"], 4.0)
    assert comparison["6-20"]["enabled_for_router"] is False
    assert summary["router_replacement_eval"]["replaced_rows"] == 0


def test_run_summary_enables_band_only_when_replacement_improves_same_slice():
    incumbent_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "rating": [3.0, 4.0, 5.0],
            "history_band": ["2-5", "2-5", "0"],
            "incumbent_prediction_raw": [3.0, 3.0, 5.0],
            "incumbent_prediction": [3.0, 3.0, 5.0],
            "incumbent_branch": ["known_model", "known_model", "cold_model"],
        }
    )
    deep_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "rating": [3.0, 4.0],
            "history_band": ["2-5", "2-5"],
            "deep_prediction_raw": [3.0, 4.0],
            "deep_prediction": [3.0, 4.0],
            "alpha": [0.5, 0.5],
        }
    )

    summary = _run_summary(
        run_name="dummy",
        training_config=_make_summary_training_config(),
        incumbent_val=incumbent_val,
        deep_val=deep_val,
        training_result=_DummyTrainingResult(),
        enable_margin=0.0,
    )

    comparison = {row["history_band"]: row for row in summary["band_comparison"]}
    assert comparison["2-5"]["enabled_for_router"] is True
    assert summary["enabled_bands"] == ["2-5"]
    assert summary["router_replacement_eval"]["replaced_rows"] == 2
    final_band_metrics = {row["history_band"]: row for row in summary["final_band_metrics"]}
    assert np.isclose(final_band_metrics["2-5"]["mae"], 0.0)


def test_run_summary_can_enable_band_1_with_non_negative_margin():
    incumbent_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "rating": [1.0, 4.0],
            "history_band": ["1", "1"],
            "incumbent_prediction_raw": [2.0, 5.0],
            "incumbent_prediction": [2.0, 5.0],
            "incumbent_branch": ["known_model", "known_model"],
        }
    )
    deep_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "rating": [1.0, 4.0],
            "history_band": ["1", "1"],
            "deep_prediction_raw": [1.0, 4.0],
            "deep_prediction": [1.0, 4.0],
            "alpha": [0.4, 0.3],
        }
    )

    summary = _run_summary(
        run_name="dummy",
        training_config=_make_summary_training_config(),
        incumbent_val=incumbent_val,
        deep_val=deep_val,
        training_result=_DummyTrainingResult(),
        enable_margin=0.01,
    )

    comparison = {row["history_band"]: row for row in summary["band_comparison"]}
    assert comparison["1"]["enabled_for_router"] is True
    assert "1" in summary["enabled_bands"]
    assert summary["router_replacement_eval"]["replaced_rows"] == 2


def test_run_summary_supports_selective_replacement_for_band_2_5():
    incumbent_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "rating": [3.0, 5.0, 1.0],
            "history_band": ["2-5", "2-5", "6-20"],
            "history_count": [2, 4, 8],
            "incumbent_prediction_raw": [3.0, 3.0, 1.0],
            "incumbent_prediction": [3.0, 3.0, 1.0],
            "incumbent_branch": ["known_model", "known_model", "known_prefix_deep_model"],
        }
    )
    deep_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "rating": [3.0, 5.0, 1.0],
            "history_band": ["2-5", "2-5", "6-20"],
            "deep_prediction_raw": [3.0, 5.0, 1.0],
            "deep_prediction": [3.0, 5.0, 1.0],
            "alpha": [0.40, 0.80, 0.5],
            "correction_hat": [0.05, 0.40, 0.0],
        }
    )

    config = _make_summary_training_config()
    config.selective_replace_alpha_thresholds = {"2-5": 0.55}
    config.selective_replace_abs_correction_thresholds = {"2-5": 0.2}

    summary = _run_summary(
        run_name="dummy",
        training_config=config,
        incumbent_val=incumbent_val,
        deep_val=deep_val,
        training_result=_DummyTrainingResult(),
        enable_margin=0.001,
    )

    comparison = {row["history_band"]: row for row in summary["band_comparison"]}
    assert comparison["2-5"]["replacement_candidate_rows"] == 1
    assert comparison["2-5"]["enabled_for_router"] is True
    assert summary["router_replacement_eval"]["replaced_rows"] == 1
    assert summary["router_replacement_eval"]["replace_policy"]["band_thresholds_applied"]["2-5"]["selected_rows"] == 1
    assert summary["router_replacement_eval"]["short_history_diagnostics"]["alpha_threshold_eval"]
    assert summary["short_history_metrics"]["deep"]


def test_run_summary_persists_short_history_segments():
    incumbent_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "rating": [2.0, 3.0, 4.0, 5.0],
            "history_band": ["2-5", "2-5", "2-5", "2-5"],
            "history_count": [2, 3, 4, 5],
            "incumbent_prediction_raw": [3.0, 3.0, 3.0, 3.0],
            "incumbent_prediction": [3.0, 3.0, 3.0, 3.0],
            "incumbent_branch": ["known_model", "known_model", "known_model", "known_model"],
        }
    )
    deep_val = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4"],
            "rating": [2.0, 3.0, 4.0, 5.0],
            "history_band": ["2-5", "2-5", "2-5", "2-5"],
            "history_count": [2, 3, 4, 5],
            "deep_prediction_raw": [2.0, 3.0, 4.0, 5.0],
            "deep_prediction": [2.0, 3.0, 4.0, 5.0],
            "alpha": [0.5, 0.5, 0.5, 0.5],
            "correction_hat": [0.5, 0.0, 1.0, 2.0],
        }
    )

    summary = _run_summary(
        run_name="dummy",
        training_config=_make_summary_training_config(),
        incumbent_val=incumbent_val,
        deep_val=deep_val,
        training_result=_DummyTrainingResult(),
        enable_margin=0.0,
    )

    deep_segments = {row["history_count_segment"] for row in summary["short_history_metrics"]["deep"]}
    final_segments = {row["history_count_segment"] for row in summary["router_replacement_eval"]["final_short_history_metrics"]}
    diagnostic_segments = {row["history_count_bin"] for row in summary["router_replacement_eval"]["short_history_diagnostics"]["history_count_bins"]}
    assert {"2", "3", "4", "5", "2-3", "4-5"}.issubset(deep_segments)
    assert {"2", "3", "4", "5", "2-3", "4-5"}.issubset(final_segments)
    assert {"2", "3", "4", "5", "2-3", "4-5"}.issubset(diagnostic_segments)


def test_history_band_lookup_from_context_uses_context_counts():
    context_reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
            "user_id": ["u1", "u2", "u2", "u3", "u3", "u3", "u3"],
        }
    )

    lookup = _history_band_lookup_from_context(context_reviews)

    assert lookup["u1"] == "1"
    assert lookup["u2"] == "2-5"
    assert lookup["u3"] == "2-5"
