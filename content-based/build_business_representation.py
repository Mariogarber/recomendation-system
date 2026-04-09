from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.business_representation import (
    BusinessRepresentationBuilder,
    BusinessRepresentationConfig,
)
from utils.io import get_default_data_dir, load_businesses, load_train_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Business Representation V1 artifacts.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_default_data_dir(),
        help="Directory containing negocios.csv and train_reviews.csv",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, the script only builds and prints summaries.",
    )
    parser.add_argument("--min-city-freq", type=int, default=20)
    parser.add_argument("--max-city-ohe", type=int, default=200)
    parser.add_argument("--city-hash-dim", type=int, default=64)
    parser.add_argument("--min-category-freq", type=int, default=20)
    parser.add_argument("--min-attribute-value-freq", type=int, default=30)
    parser.add_argument("--max-attribute-values-per-key", type=int, default=12)
    parser.add_argument("--include-geo-clusters", action="store_true")
    parser.add_argument("--geo-cluster-count", type=int, default=32)
    parser.add_argument("--no-priors", action="store_true")
    args = parser.parse_args()

    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)

    config = BusinessRepresentationConfig(
        min_city_freq=args.min_city_freq,
        max_city_ohe=args.max_city_ohe,
        city_hash_dim=args.city_hash_dim,
        min_category_freq=args.min_category_freq,
        min_attribute_value_freq=args.min_attribute_value_freq,
        max_attribute_values_per_key=args.max_attribute_values_per_key,
        include_geo_clusters=args.include_geo_clusters,
        geo_cluster_count=args.geo_cluster_count,
        include_priors=not args.no_priors,
    )

    builder = BusinessRepresentationBuilder(config)
    bundle = builder.fit_transform(businesses_df, train_reviews)

    print("=== BUSINESS REPRESENTATION SUMMARY ===")
    print(json.dumps(bundle.representation_summary, indent=2))
    print("\n=== BLOCK SUMMARY ===")
    print(bundle.block_summary.to_string(index=False))

    if args.save_dir is not None:
        saved_paths = bundle.save(args.save_dir)
        print("\n=== SAVED ARTIFACTS ===")
        print(json.dumps(saved_paths, indent=2))


if __name__ == "__main__":
    main()
