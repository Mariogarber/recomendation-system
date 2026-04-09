from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.business_representation import (
    BusinessRepresentationBuilder,
    BusinessRepresentationConfig,
)
from utils.io import (
    get_default_data_dir,
    load_businesses,
    load_train_reviews,
    load_users,
)
from utils.user_representation import (
    UserRepresentationBuilder,
    UserRepresentationConfig,
)


def _parse_business_blocks(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None
    blocks = [token.strip() for token in raw_value.split(",") if token.strip()]
    return blocks or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build User Representation V1 artifacts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_default_data_dir(),
        help="Directory containing usuarios.csv, negocios.csv, and train_reviews.csv",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, the script only builds and prints summaries.",
    )
    parser.add_argument(
        "--aggregation-mode",
        choices=["mean", "rating", "centered", "recency"],
        default="centered",
        help="Aggregation used to build the user content profile from rated businesses.",
    )
    parser.add_argument(
        "--business-view",
        choices=["content", "prior", "full"],
        default="content",
        help="View of the business representation used to build user profiles.",
    )
    parser.add_argument(
        "--business-blocks",
        type=str,
        default=None,
        help="Optional comma-separated business blocks for ablations, e.g. categories,attributes",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable the safe user metadata block.",
    )
    parser.add_argument(
        "--recency-half-life-days",
        type=float,
        default=180.0,
        help="Half-life in days for recency weighting.",
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
    args = parser.parse_args()

    data_dir = args.data_dir
    businesses_df = load_businesses(data_dir)
    users_df = load_users(data_dir)
    train_reviews = load_train_reviews(data_dir)

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

    user_config = UserRepresentationConfig(
        aggregation_mode=args.aggregation_mode,
        business_view=args.business_view,
        business_blocks=_parse_business_blocks(args.business_blocks),
        include_metadata=not args.no_metadata,
        recency_half_life_days=args.recency_half_life_days,
    )
    user_bundle = UserRepresentationBuilder(user_config).fit_transform(
        train_reviews=train_reviews,
        business_bundle=business_bundle,
        users_df=users_df,
    )

    print("=== USER REPRESENTATION SUMMARY ===")
    print(json.dumps(user_bundle.profile_summary, indent=2))
    print("\n=== USER METADATA AUDIT SUMMARY ===")
    print(json.dumps(user_bundle.user_metadata_audit_summary, indent=2))

    if args.save_dir is not None:
        saved_paths = user_bundle.save(args.save_dir)
        print("\n=== SAVED ARTIFACTS ===")
        print(json.dumps(saved_paths, indent=2))


if __name__ == "__main__":
    main()
