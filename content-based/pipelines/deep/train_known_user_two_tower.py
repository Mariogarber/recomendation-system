from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any

import numpy as np
import torch

from model.known_user_two_tower_cross import KnownUserTwoTowerCrossModel
from train_known_user_deep_router import (
    _alpha_correction_stats_by_band,
    _attach_deep_predictions,
    _coverage_by_band,
    _deep_model_band_eval,
    _load_incumbent_stack,
    _predict_incumbent_router,
    _round_half_up,
    _save_json,
)
from utils.io import canonicalize_reviews, get_default_data_dir, load_businesses, load_test_reviews, load_train_reviews, load_users
from utils.known_user_two_tower_cross import (
    KnownUserTwoTowerDataConfig,
    KnownUserTwoTowerTrainingConfig,
    build_known_user_two_tower_eval_dataset,
    build_known_user_two_tower_train_dataset,
    fit_known_user_two_tower_final_model,
    predict_known_user_two_tower_dataset,
    prepare_known_user_two_tower_context,
    save_known_user_two_tower_checkpoint,
    train_known_user_two_tower_model,
)
from utils.split import temporal_train_validation_split


def _load_run_configs(max_runs: int, *, max_epochs: int, early_stopping_patience: int) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = [
        {
            "run_name": "run01_structured_base",
            "training": KnownUserTwoTowerTrainingConfig(
                embedding_dim=128,
                event_hidden_dim=128,
                user_type_hidden_dim=128,
                business_hidden_layers=(512, 256),
                fusion_hidden_layers=(256, 128),
                cross_hidden_layers=(512, 256),
                cross_depth=3,
                num_attention_heads=4,
                dropout=0.15,
                batch_size=512,
                learning_rate=8e-4,
                weight_decay=2e-5,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                band_sample_weights={"1": 1.4, "2-5": 1.8, "6-20": 1.0, ">20": 1.0},
            ),
        },
        {
            "run_name": "run02_structured_stable",
            "training": KnownUserTwoTowerTrainingConfig(
                embedding_dim=128,
                event_hidden_dim=128,
                user_type_hidden_dim=128,
                business_hidden_layers=(512, 256),
                fusion_hidden_layers=(256, 128),
                cross_hidden_layers=(512, 256),
                cross_depth=3,
                num_attention_heads=4,
                dropout=0.20,
                batch_size=512,
                learning_rate=4e-4,
                weight_decay=1e-4,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                band_sample_weights={"1": 1.4, "2-5": 1.8, "6-20": 1.0, ">20": 1.0},
            ),
        },
        {
            "run_name": "run03_structured_capacity",
            "training": KnownUserTwoTowerTrainingConfig(
                embedding_dim=160,
                event_hidden_dim=160,
                user_type_hidden_dim=160,
                business_hidden_layers=(512, 384, 256),
                fusion_hidden_layers=(320, 160),
                cross_hidden_layers=(640, 320),
                cross_depth=4,
                num_attention_heads=8,
                dropout=0.15,
                batch_size=512,
                learning_rate=8e-4,
                weight_decay=2e-5,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                band_sample_weights={"1": 1.5, "2-5": 2.0, "6-20": 1.0, ">20": 1.0},
            ),
        },
    ]
    return configs[: max(1, min(max_runs, len(configs)))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-tower + cross + prefix-memory known-user residual router.")
    parser.add_argument("--data-dir", type=Path, default=get_default_data_dir())
    parser.add_argument("--save-root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts" / "known_user_two_tower_router_v1")
    parser.add_argument("--incumbent-root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts" / "lgbm_raw_router_prefix_deep_v1")
    parser.add_argument("--business-source", type=str, default="structured_from_scratch")
    parser.add_argument("--business-repr-root", type=Path, default=None)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--known-enable-margin", type=float, default=0.002)
    parser.add_argument("--max-history-len", type=int, default=20)
    parser.add_argument("--n-user-archetypes", type=int, default=64)
    parser.add_argument("--max-top-cities", type=int, default=100)
    parser.add_argument("--max-top-categories", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = args.save_root
    save_root.mkdir(parents=True, exist_ok=True)

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = canonicalize_reviews(load_train_reviews(args.data_dir))
    test_reviews = canonicalize_reviews(load_test_reviews(args.data_dir))
    train_split, val_split = temporal_train_validation_split(train_reviews, val_size=float(args.validation_size))

    incumbent_stack = _load_incumbent_stack(args.incumbent_root)
    train_incumbent = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=train_split,
        context_reviews=train_split,
        users_df=users_df,
        businesses_df=businesses_df,
    )
    val_incumbent = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=val_split,
        context_reviews=train_split,
        users_df=users_df,
        businesses_df=businesses_df,
    )

    data_config = KnownUserTwoTowerDataConfig(
        business_source=str(args.business_source),
        business_repr_root=str(args.business_repr_root) if args.business_repr_root is not None else None,
        max_history_len=int(args.max_history_len),
        n_user_archetypes=int(args.n_user_archetypes),
        max_top_cities=int(args.max_top_cities),
        max_top_categories=int(args.max_top_categories),
        random_seed=int(args.seed),
    )
    context = prepare_known_user_two_tower_context(
        context_reviews=train_split,
        users_df=users_df,
        businesses_df=businesses_df,
        data_config=data_config,
    )

    best_record: dict[str, Any] | None = None
    run_summaries: list[dict[str, Any]] = []
    for run_cfg in _load_run_configs(
        args.max_runs,
        max_epochs=max(int(args.max_epochs), 1),
        early_stopping_patience=max(int(args.early_stopping_patience), 1),
    ):
        run_name = str(run_cfg["run_name"])
        run_dir = save_root / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        training_cfg: KnownUserTwoTowerTrainingConfig = run_cfg["training"]
        train_data = build_known_user_two_tower_train_dataset(
            train_split,
            users_df=users_df,
            businesses_df=businesses_df,
            context=context,
            training_config=training_cfg,
            incumbent_frame=train_incumbent,
        )
        val_data = build_known_user_two_tower_eval_dataset(
            val_split,
            train_split,
            users_df=users_df,
            businesses_df=businesses_df,
            context=context,
            training_config=training_cfg,
            incumbent_frame=val_incumbent,
        )
        training_result = train_known_user_two_tower_model(
            train_data=train_data,
            val_data=val_data,
            context=context,
            training_config=training_cfg,
        )
        device = torch.device("cuda" if torch.cuda.is_available() and training_result.model_config.device == "auto" else "cpu")
        model = KnownUserTwoTowerCrossModel(training_result.architecture)
        model.load_state_dict(training_result.model_state_dict)
        model.to(device)
        val_predictions = predict_known_user_two_tower_dataset(
            model=model,
            prepared=val_data,
            context=context,
            batch_size=training_result.model_config.batch_size,
            device=device,
        )
        val_predictions["deep_prediction_raw"] = val_predictions["predicted_rating"].astype(np.float32)
        val_predictions = val_predictions.rename(columns={"predicted_rating": "deep_prediction", "incumbent_prediction_raw": "incumbent_prediction"})
        eval_frame = _attach_deep_predictions(val_incumbent.rename(columns={"prediction": "incumbent_prediction"}), val_predictions)
        band_eval = _deep_model_band_eval(eval_frame)
        enabled_bands = [row["history_band"] for row in band_eval if float(row["delta_mae"]) <= -float(args.known_enable_margin)]
        summary = {
            "run_name": run_name,
            "success": True,
            "business_source_summary": context.business_source_summary,
            "best_epoch": int(training_result.best_epoch),
            "best_val_mae": float(training_result.best_val_mae),
            "best_val_rmse": float(training_result.best_val_rmse),
            "enabled_bands": enabled_bands,
            "band_eval": band_eval,
            "alpha_stats_by_band": _alpha_correction_stats_by_band(eval_frame),
            "coverage_by_band": _coverage_by_band(eval_frame, branch_col="incumbent_branch"),
        }
        training_result.learning_curves.to_csv(run_dir / "learning_curves.csv", index=False)
        save_known_user_two_tower_checkpoint(
            path=run_dir / "known_user_two_tower_checkpoint.pt",
            model_state_dict=training_result.model_state_dict,
            architecture=training_result.architecture,
            feature_contract=context.feature_contract,
            data_config=context.data_config,
            training_config=training_cfg,
            extra_summary=summary,
        )
        _save_json(run_dir / "validation_summary.json", summary)
        run_summaries.append(summary)
        if best_record is None or float(summary["best_val_mae"]) < float(best_record["best_val_mae"]):
            best_record = summary | {"training_cfg": training_cfg, "architecture": training_result.architecture}

    if best_record is None:
        raise RuntimeError("No training run completed.")

    full_incumbent_train = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=train_reviews,
        context_reviews=train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
    )
    full_context = prepare_known_user_two_tower_context(
        context_reviews=train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
        data_config=data_config,
    )
    full_train_data = build_known_user_two_tower_train_dataset(
        train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
        context=full_context,
        training_config=best_record["training_cfg"],
        incumbent_frame=full_incumbent_train,
    )
    final_state = fit_known_user_two_tower_final_model(
        train_data=full_train_data,
        context=full_context,
        training_config=best_record["training_cfg"],
        architecture=best_record["architecture"],
        final_epochs=int(best_record["best_epoch"]),
    )
    save_known_user_two_tower_checkpoint(
        path=save_root / "known_user_two_tower_checkpoint.pt",
        model_state_dict=final_state,
        architecture=best_record["architecture"],
        feature_contract=full_context.feature_contract,
        data_config=full_context.data_config,
        training_config=best_record["training_cfg"],
        extra_summary={"best_run_name": best_record["run_name"], "run_summaries": run_summaries},
    )

    test_incumbent = _predict_incumbent_router(
        spec=incumbent_stack["spec"],
        known_booster=incumbent_stack["known_booster"],
        known_prefix_booster=incumbent_stack["known_prefix_booster"],
        cold_booster=incumbent_stack["cold_booster"],
        target_reviews=test_reviews,
        context_reviews=train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
    )
    test_data = build_known_user_two_tower_eval_dataset(
        test_reviews,
        train_reviews,
        users_df=users_df,
        businesses_df=businesses_df,
        context=full_context,
        training_config=best_record["training_cfg"],
        incumbent_frame=test_incumbent,
    )
    device = torch.device("cuda" if torch.cuda.is_available() and best_record["training_cfg"].device == "auto" else "cpu")
    final_model = KnownUserTwoTowerCrossModel(best_record["architecture"])
    final_model.load_state_dict(final_state)
    final_model.to(device)
    deep_test_predictions = predict_known_user_two_tower_dataset(
        model=final_model,
        prepared=test_data,
        context=full_context,
        batch_size=best_record["training_cfg"].batch_size,
        device=device,
    )
    enabled_bands = set(best_record["enabled_bands"])
    test_lookup = deep_test_predictions[deep_test_predictions["history_band"].astype(str).isin(enabled_bands)][["review_id", "predicted_rating"]].copy()
    test_lookup["review_id"] = test_lookup["review_id"].astype(str)
    submission = test_incumbent[["review_id", "incumbent_prediction_raw"]].copy()
    submission["review_id"] = submission["review_id"].astype(str)
    submission = submission.merge(test_lookup, on="review_id", how="left")
    submission["final_prediction"] = np.where(
        np.isfinite(submission["predicted_rating"]),
        submission["predicted_rating"],
        submission["incumbent_prediction_raw"],
    ).astype(np.float32)
    submission["stars"] = _round_half_up(np.clip(submission["final_prediction"].to_numpy(dtype=np.float32), 1.0, 5.0))
    submission[["review_id", "stars"]].to_csv(save_root / "submission.csv", index=False)

    validation_summary = {
        "best_run_name": best_record["run_name"],
        "business_source_summary": full_context.business_source_summary,
        "best_val_mae": float(best_record["best_val_mae"]),
        "best_val_rmse": float(best_record["best_val_rmse"]),
        "enabled_known_two_tower_bands": list(best_record["enabled_bands"]),
        "run_summaries": run_summaries,
    }
    _save_json(save_root / "validation_summary.json", validation_summary)
    _save_json(
        save_root / "submission_summary.json",
        {
            "rows": int(len(submission)),
            "deep_served_rows": int(np.isfinite(submission["predicted_rating"]).sum()),
            "enabled_known_two_tower_bands": list(best_record["enabled_bands"]),
            "business_source_summary": full_context.business_source_summary,
        },
    )


if __name__ == "__main__":
    main()
