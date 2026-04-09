from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.audit import (
    compare_business_metadata_with_train,
    compare_user_metadata_with_train,
    summarize_business_comparison,
    summarize_user_comparison,
)
from utils.business_features import extract_hours_features, summarize_attribute_keys
from utils.io import (
    canonicalize_reviews,
    get_default_data_dir,
    load_businesses,
    load_test_reviews,
    load_train_reviews,
    load_users,
)
from utils.split import cold_start_breakdown


def build_dataset_summary(train_reviews: pd.DataFrame, test_reviews: pd.DataFrame) -> dict[str, float | int]:
    train_canonical = canonicalize_reviews(train_reviews)
    test_canonical = canonicalize_reviews(test_reviews)

    unique_users = int(train_canonical["user"].nunique())
    unique_items = int(train_canonical["item"].nunique())
    train_rows = int(len(train_canonical))

    return {
        "train_rows": train_rows,
        "test_rows": int(len(test_canonical)),
        "train_unique_users": unique_users,
        "train_unique_items": unique_items,
        "train_density": float(train_rows / (unique_users * unique_items)),
        "rating_mean": float(train_canonical["rating"].mean()),
        "rating_std": float(train_canonical["rating"].std(ddof=1)),
        "rating_min": float(train_canonical["rating"].min()),
        "rating_max": float(train_canonical["rating"].max()),
    }


def build_business_parser_summary(businesses_df: pd.DataFrame) -> dict[str, float | int]:
    total = len(businesses_df)
    attribute_counts = summarize_attribute_keys(businesses_df)
    hours_features = businesses_df["hours"].apply(extract_hours_features)
    hours_df = pd.DataFrame(hours_features.tolist())

    return {
        "business_rows": int(total),
        "categories_present_rate": float(businesses_df["categories"].notna().mean()),
        "attributes_present_rate": float(businesses_df["attributes"].notna().mean()),
        "hours_present_rate": float(businesses_df["hours"].notna().mean()),
        "unique_states": int(businesses_df["state"].nunique(dropna=True)),
        "unique_cities": int(businesses_df["city"].nunique(dropna=True)),
        "unique_attribute_keys": int(len(attribute_counts)),
        "mean_open_days_count": float(hours_df["open_days_count"].mean()),
        "mean_weekly_open_minutes": float(hours_df["weekly_open_minutes"].mean()),
    }


def save_json(payload: dict, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 audit for the content-based pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=get_default_data_dir(),
        help="Directory containing usuarios.csv, negocios.csv, train_reviews.csv and test_reviews.csv",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory where JSON/CSV reports will be written.",
    )
    args = parser.parse_args()

    users_df = load_users(args.data_dir)
    businesses_df = load_businesses(args.data_dir)
    train_reviews = load_train_reviews(args.data_dir)
    test_reviews = load_test_reviews(args.data_dir)

    dataset_summary = build_dataset_summary(train_reviews, test_reviews)
    cold_start_summary = cold_start_breakdown(
        canonicalize_reviews(train_reviews),
        canonicalize_reviews(test_reviews),
    )

    user_comparison = compare_user_metadata_with_train(users_df, train_reviews)
    business_comparison = compare_business_metadata_with_train(businesses_df, train_reviews)
    user_leakage_summary = summarize_user_comparison(user_comparison)
    business_leakage_summary = summarize_business_comparison(business_comparison)
    business_parser_summary = build_business_parser_summary(businesses_df)
    attribute_key_counts = summarize_attribute_keys(businesses_df)

    print("=== DATASET SUMMARY ===")
    print(json.dumps(dataset_summary, indent=2))
    print("\n=== COLD START SUMMARY ===")
    print(json.dumps(cold_start_summary, indent=2))
    print("\n=== USER LEAKAGE SUMMARY ===")
    print(json.dumps(user_leakage_summary, indent=2))
    print("\n=== BUSINESS LEAKAGE SUMMARY ===")
    print(json.dumps(business_leakage_summary, indent=2))
    print("\n=== BUSINESS PARSER SUMMARY ===")
    print(json.dumps(business_parser_summary, indent=2))

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        save_json(dataset_summary, args.save_dir / "dataset_summary.json")
        save_json(cold_start_summary, args.save_dir / "cold_start_summary.json")
        save_json(user_leakage_summary, args.save_dir / "user_leakage_summary.json")
        save_json(business_leakage_summary, args.save_dir / "business_leakage_summary.json")
        save_json(business_parser_summary, args.save_dir / "business_parser_summary.json")
        user_comparison.to_csv(args.save_dir / "user_leakage_details.csv", index=False)
        business_comparison.to_csv(args.save_dir / "business_leakage_details.csv", index=False)
        attribute_key_counts.to_csv(args.save_dir / "attribute_key_counts.csv", index=False)


if __name__ == "__main__":
    main()
