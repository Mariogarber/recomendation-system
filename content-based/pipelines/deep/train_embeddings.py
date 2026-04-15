from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.business_representation import (
    BusinessRepresentationBuilder,
    BusinessRepresentationConfig,
)
from utils.deep_user_embeddings import (
    DeepUserEmbeddingBuilder,
    DeepUserEmbeddingConfig,
    build_target_user_ids,
)
from utils.io import (
    get_default_data_dir,
    load_businesses,
    load_test_reviews,
    load_train_reviews,
    load_users,
)
from utils.user_representation import (
    UserRepresentationBuilder,
    UserRepresentationConfig,
)


def _parse_hidden_layers(raw_value: str) -> tuple[int, ...]:
    text = raw_value.strip()
    if not text or text.lower() in {"default", "none", "auto"}:
        return ()
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            dim = int(token)
        except ValueError as exc:  # pragma: no cover - argparse validation path
            raise argparse.ArgumentTypeError(
                f"Invalid hidden layer specification '{raw_value}'. Expected comma-separated integers."
            ) from exc
        if dim <= 0:
            raise argparse.ArgumentTypeError(
                f"Invalid hidden layer specification '{raw_value}'. Hidden layer sizes must be positive."
            )
        values.append(dim)
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build business, manual-user, and deep-user embeddings for competition.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_default_data_dir(),
        help="Directory containing usuarios.csv, negocios.csv, train_reviews.csv, and test_reviews.csv",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "competition_embeddings_v1",
        help="Root directory where all embedding artifacts will be saved.",
    )

    parser.add_argument("--min-city-freq", type=int, default=20)
    parser.add_argument("--max-city-ohe", type=int, default=200)
    parser.add_argument("--city-hash-dim", type=int, default=64)
    parser.add_argument("--min-category-freq", type=int, default=20)
    parser.add_argument("--min-attribute-value-freq", type=int, default=30)
    parser.add_argument("--max-attribute-values-per-key", type=int, default=12)
    parser.add_argument("--include-geo-clusters", action="store_true")
    parser.add_argument("--geo-cluster-count", type=int, default=32)
    parser.add_argument("--no-business-priors", action="store_true")

    parser.add_argument("--manual-aggregation-mode", choices=["mean", "rating", "centered", "recency"], default="centered")
    parser.add_argument("--manual-business-view", choices=["content", "prior", "full"], default="full")
    parser.add_argument("--manual-no-metadata", action="store_true")
    parser.add_argument("--manual-recency-half-life-days", type=float, default=180.0)

    parser.add_argument("--deep-business-view", choices=["content", "prior", "full"], default="full")
    parser.add_argument("--deep-max-history-len", type=int, default=20)
    parser.add_argument("--deep-batch-size", type=int, default=768)
    parser.add_argument("--deep-learning-rate", type=float, default=8e-4)
    parser.add_argument("--deep-weight-decay", type=float, default=2e-5)
    parser.add_argument("--deep-max-epochs", type=int, default=20)
    parser.add_argument("--deep-early-stopping-patience", type=int, default=4)
    parser.add_argument("--deep-embedding-dim", type=int, default=128)
    parser.add_argument("--deep-business-hidden-dim", type=int, default=384)
    parser.add_argument("--deep-rating-hidden-dim", type=int, default=32)
    parser.add_argument("--deep-metadata-hidden-dim", type=int, default=64)
    parser.add_argument("--deep-scorer-hidden-dim", type=int, default=256)
    parser.add_argument("--deep-business-hidden-layers", type=_parse_hidden_layers, default=(512, 384, 256))
    parser.add_argument("--deep-rating-hidden-layers", type=_parse_hidden_layers, default=(64, 32))
    parser.add_argument("--deep-metadata-hidden-layers", type=_parse_hidden_layers, default=(128, 64))
    parser.add_argument("--deep-scorer-hidden-layers", type=_parse_hidden_layers, default=(256, 128))
    parser.add_argument("--deep-dropout", type=float, default=0.15)
    parser.add_argument("--deep-history-shrinkage-temperature", type=float, default=3.0)
    parser.add_argument("--deep-rating-modulation-scale", type=float, default=0.35)
    parser.add_argument("--deep-val-size", type=float, default=0.2)
    parser.add_argument("--deep-device", type=str, default="auto")
    args = parser.parse_args()

    data_dir = args.data_dir
    businesses_df = load_businesses(data_dir)
    users_df = load_users(data_dir)
    train_reviews = load_train_reviews(data_dir)
    test_reviews = load_test_reviews(data_dir)
    target_user_ids = build_target_user_ids(
        users_df=users_df,
        train_reviews=train_reviews,
        test_reviews=test_reviews,
    )

    business_config = BusinessRepresentationConfig(
        min_city_freq=args.min_city_freq,
        max_city_ohe=args.max_city_ohe,
        city_hash_dim=args.city_hash_dim,
        min_category_freq=args.min_category_freq,
        min_attribute_value_freq=args.min_attribute_value_freq,
        max_attribute_values_per_key=args.max_attribute_values_per_key,
        include_geo_clusters=args.include_geo_clusters,
        geo_cluster_count=args.geo_cluster_count,
        include_priors=not args.no_business_priors,
    )
    business_bundle = BusinessRepresentationBuilder(business_config).fit_transform(
        businesses_df,
        train_reviews,
    )

    manual_config = UserRepresentationConfig(
        aggregation_mode=args.manual_aggregation_mode,
        business_view=args.manual_business_view,
        include_metadata=not args.manual_no_metadata,
        recency_half_life_days=args.manual_recency_half_life_days,
    )
    manual_bundle = UserRepresentationBuilder(manual_config).fit_transform(
        train_reviews=train_reviews,
        business_bundle=business_bundle,
        users_df=users_df,
        target_user_ids=target_user_ids,
    )

    deep_config = DeepUserEmbeddingConfig(
        business_view=args.deep_business_view,
        max_history_len=args.deep_max_history_len,
        batch_size=args.deep_batch_size,
        learning_rate=args.deep_learning_rate,
        weight_decay=args.deep_weight_decay,
        max_epochs=args.deep_max_epochs,
        early_stopping_patience=args.deep_early_stopping_patience,
        embedding_dim=args.deep_embedding_dim,
        business_hidden_dim=args.deep_business_hidden_dim,
        rating_hidden_dim=args.deep_rating_hidden_dim,
        metadata_hidden_dim=args.deep_metadata_hidden_dim,
        scorer_hidden_dim=args.deep_scorer_hidden_dim,
        business_hidden_layers=args.deep_business_hidden_layers,
        rating_hidden_layers=args.deep_rating_hidden_layers,
        metadata_hidden_layers=args.deep_metadata_hidden_layers,
        scorer_hidden_layers=args.deep_scorer_hidden_layers,
        dropout=args.deep_dropout,
        history_shrinkage_temperature=args.deep_history_shrinkage_temperature,
        rating_modulation_scale=args.deep_rating_modulation_scale,
        temporal_val_size=args.deep_val_size,
        device=args.deep_device,
    )
    deep_bundle = DeepUserEmbeddingBuilder(deep_config).fit_transform(
        train_reviews=train_reviews,
        business_bundle=business_bundle,
        users_df=users_df,
        test_reviews=test_reviews,
        target_user_ids=target_user_ids,
    )

    save_root = args.save_root
    business_paths = business_bundle.save(save_root / "business_repr")
    manual_paths = manual_bundle.save(save_root / "user_manual_repr")
    deep_paths = deep_bundle.save(save_root / "user_deep_repr")

    print("=== BUSINESS REPRESENTATION SUMMARY ===")
    print(json.dumps(business_bundle.representation_summary, indent=2))
    print("\n=== MANUAL USER REPRESENTATION SUMMARY ===")
    print(json.dumps(manual_bundle.profile_summary, indent=2))
    print("\n=== DEEP USER REPRESENTATION SUMMARY ===")
    print(json.dumps(deep_bundle.training_summary, indent=2))
    print("\n=== EFFECTIVE PIPELINE SOURCES ===")
    print(json.dumps(
        {
            "manual_profile_source": manual_bundle.profile_summary.get("profile_source"),
            "manual_business_view": manual_bundle.profile_summary.get("business_view"),
            "deep_business_view": deep_bundle.training_summary.get("business_view"),
            "deep_business_feature_source": deep_bundle.training_summary.get("business_feature_source"),
        },
        indent=2,
    ))
    print("\n=== TARGET USER COVERAGE ===")
    print(json.dumps({"n_target_users": int(len(target_user_ids))}, indent=2))
    print("\n=== SAVED ARTIFACTS ===")
    print(json.dumps(
        {
            "business": business_paths,
            "user_manual": manual_paths,
            "user_deep": deep_paths,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
