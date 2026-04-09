from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import MultiLabelBinarizer

from .audit import (
    build_business_train_aggregates,
    compare_business_metadata_with_train,
    summarize_business_comparison,
)
from .business_features import extract_hours_features, parse_attributes, parse_categories


@dataclass(slots=True)
class BusinessRepresentationConfig:
    min_city_freq: int = 20
    max_city_ohe: int = 200
    city_hash_dim: int = 64
    min_category_freq: int = 20
    min_attribute_value_freq: int = 30
    max_attribute_values_per_key: int = 12
    include_geo_clusters: bool = False
    geo_cluster_count: int = 32
    include_priors: bool = True


@dataclass(slots=True)
class BusinessRepresentationBundle:
    business_ids: pd.Series
    clean_business_table: pd.DataFrame
    content_matrix: sparse.csr_matrix
    prior_matrix: sparse.csr_matrix
    full_matrix: sparse.csr_matrix
    block_summary: pd.DataFrame
    feature_metadata: pd.DataFrame
    business_prior_audit_summary: dict[str, Any]
    business_prior_audit_details: pd.DataFrame
    representation_summary: dict[str, Any]
    content_feature_names: list[str]
    prior_feature_names: list[str]
    full_feature_names: list[str]

    def get_matrix(self, view: str = "full", blocks: list[str] | None = None) -> sparse.csr_matrix:
        if blocks is not None:
            mask = self.feature_metadata["block_name"].isin(blocks)
            indices = self.feature_metadata.loc[mask, "feature_index"].to_numpy(dtype=int)
            return self.full_matrix[:, indices]

        if view == "content":
            return self.content_matrix
        if view == "prior":
            return self.prior_matrix
        if view == "full":
            return self.full_matrix
        raise ValueError("view must be one of: 'content', 'prior', 'full'")

    def save(self, save_dir: str | Path) -> dict[str, str]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, str] = {}

        business_ids_path = save_dir / "business_ids.csv"
        self.business_ids.to_frame(name="business_id").to_csv(business_ids_path, index=False)
        saved_paths["business_ids"] = str(business_ids_path)

        sparse.save_npz(save_dir / "business_content_features.npz", self.content_matrix)
        sparse.save_npz(save_dir / "business_prior_features.npz", self.prior_matrix)
        sparse.save_npz(save_dir / "business_full_features.npz", self.full_matrix)
        saved_paths["content_matrix"] = str(save_dir / "business_content_features.npz")
        saved_paths["prior_matrix"] = str(save_dir / "business_prior_features.npz")
        saved_paths["full_matrix"] = str(save_dir / "business_full_features.npz")

        feature_names_path = save_dir / "business_feature_names.json"
        feature_names_path.write_text(
            json.dumps(
                {
                    "content_features": self.content_feature_names,
                    "prior_features": self.prior_feature_names,
                    "full_features": self.full_feature_names,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        saved_paths["feature_names"] = str(feature_names_path)

        block_summary_path = save_dir / "business_block_summary.csv"
        self.block_summary.to_csv(block_summary_path, index=False)
        saved_paths["block_summary"] = str(block_summary_path)

        feature_metadata_path = save_dir / "feature_metadata.csv"
        self.feature_metadata.to_csv(feature_metadata_path, index=False)
        saved_paths["feature_metadata"] = str(feature_metadata_path)

        prior_summary_path = save_dir / "business_prior_leakage_summary.json"
        prior_summary_path.write_text(
            json.dumps(self.business_prior_audit_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        saved_paths["prior_leakage_summary"] = str(prior_summary_path)

        prior_details_path = save_dir / "business_prior_leakage_details.csv"
        self.business_prior_audit_details.to_csv(prior_details_path, index=False)
        saved_paths["prior_leakage_details"] = str(prior_details_path)

        summary_path = save_dir / "business_representation_summary.json"
        summary_path.write_text(
            json.dumps(self.representation_summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        saved_paths["representation_summary"] = str(summary_path)

        clean_table_path = _save_clean_table(self.clean_business_table, save_dir / "clean_business_table")
        saved_paths["clean_business_table"] = str(clean_table_path)

        return saved_paths


class BusinessRepresentationBuilder:
    def __init__(self, config: BusinessRepresentationConfig | None = None) -> None:
        self.config = config or BusinessRepresentationConfig()

    def fit_transform(
        self,
        businesses_df: pd.DataFrame,
        train_reviews: pd.DataFrame,
    ) -> BusinessRepresentationBundle:
        businesses = businesses_df.copy()
        if "business_id" not in businesses.columns:
            raise ValueError("businesses_df must contain 'business_id'")
        if businesses["business_id"].duplicated().any():
            raise ValueError("business_id must be unique in businesses_df")

        prior_audit_details = compare_business_metadata_with_train(businesses, train_reviews)
        prior_audit_summary = summarize_business_comparison(prior_audit_details)
        business_priors = build_business_train_aggregates(train_reviews).rename(
            columns={
                "train_review_count": "prior_train_review_count",
                "train_average_stars": "prior_train_average_stars",
                "train_rating_std": "prior_train_rating_std",
                "train_first_review_date": "prior_train_first_review_date",
                "train_last_review_date": "prior_train_last_review_date",
            }
        )

        prepared = self._prepare_business_table(businesses, business_priors)

        geo_block = self._build_geo_block(prepared)
        category_block = self._build_category_block(prepared)
        attribute_block = self._build_attribute_block(prepared)
        hours_block = self._build_hours_block(prepared)
        prior_block = self._build_prior_block(prepared)

        content_blocks = [geo_block, category_block, attribute_block, hours_block]
        content_matrix = sparse.hstack([block["matrix"] for block in content_blocks], format="csr")
        prior_matrix = prior_block["matrix"]
        full_matrix = sparse.hstack([content_matrix, prior_matrix], format="csr")

        block_summary = self._build_block_summary(content_blocks, prior_block)
        feature_metadata = self._build_feature_metadata(content_blocks, prior_block)
        feature_names = feature_metadata["feature_name"].tolist()
        content_feature_names = [name for block in content_blocks for name in block["feature_names"]]
        prior_feature_names = prior_block["feature_names"]

        clean_business_table = self._build_clean_business_table(prepared)
        validation_summary = self._build_validation_summary(
            prepared,
            content_matrix,
            prior_matrix,
            full_matrix,
            feature_metadata,
        )

        representation_summary = {
            "n_businesses": int(len(prepared)),
            "content_shape": [int(content_matrix.shape[0]), int(content_matrix.shape[1])],
            "prior_shape": [int(prior_matrix.shape[0]), int(prior_matrix.shape[1])],
            "full_shape": [int(full_matrix.shape[0]), int(full_matrix.shape[1])],
            "config": asdict(self.config),
            "city_strategy_used": geo_block["extra"]["city_strategy"],
            "kept_category_count": category_block["extra"]["kept_token_count"],
            "kept_attribute_feature_count": attribute_block["extra"]["kept_token_count"],
            "prior_audit_summary": prior_audit_summary,
            "validation_summary": validation_summary,
        }

        self._run_validation_checks(prepared, feature_metadata, full_matrix, validation_summary)

        return BusinessRepresentationBundle(
            business_ids=prepared["business_id"].copy(),
            clean_business_table=clean_business_table,
            content_matrix=content_matrix,
            prior_matrix=prior_matrix,
            full_matrix=full_matrix,
            block_summary=block_summary,
            feature_metadata=feature_metadata,
            business_prior_audit_summary=prior_audit_summary,
            business_prior_audit_details=prior_audit_details,
            representation_summary=representation_summary,
            content_feature_names=content_feature_names,
            prior_feature_names=prior_feature_names,
            full_feature_names=feature_names,
        )

    def _prepare_business_table(
        self,
        businesses: pd.DataFrame,
        business_priors: pd.DataFrame,
    ) -> pd.DataFrame:
        prepared = businesses.merge(business_priors, on="business_id", how="left")
        prepared["state_clean"] = prepared.get("state", pd.Series(index=prepared.index)).apply(_normalize_token)
        prepared["city_clean"] = prepared.get("city", pd.Series(index=prepared.index)).apply(_normalize_token)
        prepared["state_clean"] = prepared["state_clean"].replace("", "__unknown__").fillna("__unknown__")
        prepared["city_clean"] = prepared["city_clean"].replace("", "__unknown__").fillna("__unknown__")

        prepared["latitude_filled"] = pd.to_numeric(prepared.get("latitude"), errors="coerce")
        prepared["longitude_filled"] = pd.to_numeric(prepared.get("longitude"), errors="coerce")
        prepared["latitude_filled"] = prepared["latitude_filled"].fillna(prepared["latitude_filled"].median())
        prepared["longitude_filled"] = prepared["longitude_filled"].fillna(prepared["longitude_filled"].median())

        prepared["is_open_clean"] = pd.to_numeric(prepared.get("is_open"), errors="coerce").fillna(0.0).astype(float)

        prepared["parsed_categories"] = prepared.get("categories", pd.Series(index=prepared.index)).apply(parse_categories)
        prepared["parsed_attributes"] = prepared.get("attributes", pd.Series(index=prepared.index)).apply(parse_attributes)
        hours_features = prepared.get("hours", pd.Series(index=prepared.index)).apply(extract_hours_features)
        hours_df = pd.DataFrame(hours_features.tolist(), index=prepared.index)
        for column in hours_df.columns:
            prepared[column] = pd.to_numeric(hours_df[column], errors="coerce").fillna(0.0)

        prepared["prior_seen_in_train"] = prepared["prior_train_review_count"].notna().astype(float)
        prepared["prior_train_review_count"] = prepared["prior_train_review_count"].fillna(0.0)
        prepared["prior_train_average_stars"] = prepared["prior_train_average_stars"].fillna(0.0)
        prepared["prior_train_rating_std"] = prepared["prior_train_rating_std"].fillna(0.0)
        prepared["prior_train_review_count_log1p"] = np.log1p(prepared["prior_train_review_count"])
        max_reviews = float(prepared["prior_train_review_count"].max())
        if max_reviews > 0:
            prepared["prior_train_support_percentile"] = prepared["prior_train_review_count"] / max_reviews
        else:
            prepared["prior_train_support_percentile"] = 0.0

        return prepared

    def _build_geo_block(self, prepared: pd.DataFrame) -> dict[str, Any]:
        n_rows = len(prepared)

        state_dummies = pd.get_dummies(prepared["state_clean"], prefix="geo__state", dtype=np.float32)
        state_feature_names = state_dummies.columns.tolist()
        state_matrix = sparse.csr_matrix(state_dummies.to_numpy(dtype=np.float32))

        city_counts = prepared["city_clean"].value_counts()
        kept_cities = city_counts[city_counts >= self.config.min_city_freq].index.tolist()
        prepared["city_bucket"] = prepared["city_clean"].where(prepared["city_clean"].isin(kept_cities), "__other__")
        city_unique = prepared["city_bucket"].nunique(dropna=False)

        if city_unique <= self.config.max_city_ohe:
            city_strategy = "one_hot"
            city_dummies = pd.get_dummies(prepared["city_bucket"], prefix="geo__city", dtype=np.float32)
            city_feature_names = city_dummies.columns.tolist()
            city_matrix = sparse.csr_matrix(city_dummies.to_numpy(dtype=np.float32))
        else:
            city_strategy = "hashing"
            hasher = FeatureHasher(
                n_features=self.config.city_hash_dim,
                input_type="string",
                alternate_sign=False,
            )
            city_values = prepared["city_bucket"].astype(str).apply(lambda value: [value]).tolist()
            city_matrix = hasher.transform(city_values).tocsr().astype(np.float32)
            city_feature_names = [
                f"geo__city_hash__{index:03d}"
                for index in range(city_matrix.shape[1])
            ]

        lat_values = prepared["latitude_filled"].to_numpy(dtype=np.float32)
        lon_values = prepared["longitude_filled"].to_numpy(dtype=np.float32)
        lat_z = _safe_zscore(lat_values)
        lon_z = _safe_zscore(lon_values)
        numeric_matrix = sparse.csr_matrix(
            np.column_stack(
                [
                    lat_z,
                    lon_z,
                    prepared["is_open_clean"].to_numpy(dtype=np.float32),
                ]
            ).astype(np.float32)
        )
        numeric_feature_names = [
            "geo__latitude_z",
            "geo__longitude_z",
            "geo__is_open",
        ]

        matrices = [state_matrix, city_matrix, numeric_matrix]
        feature_names = state_feature_names + city_feature_names + numeric_feature_names

        if self.config.include_geo_clusters:
            cluster_count = min(self.config.geo_cluster_count, n_rows)
            kmeans = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(
                np.column_stack([prepared["latitude_filled"], prepared["longitude_filled"]])
            )
            cluster_dummies = pd.get_dummies(cluster_labels, prefix="geo__cluster", dtype=np.float32)
            cluster_matrix = sparse.csr_matrix(cluster_dummies.to_numpy(dtype=np.float32))
            matrices.append(cluster_matrix)
            feature_names.extend(cluster_dummies.columns.tolist())

        matrix = sparse.hstack(matrices, format="csr").astype(np.float32)
        return {
            "block_name": "geo",
            "matrix": matrix,
            "feature_names": feature_names,
            "coverage": float(
                prepared[["state_clean", "city_clean", "latitude_filled", "longitude_filled"]].notna().all(axis=1).mean()
            ),
            "pruning_rule": f"city_min_freq>={self.config.min_city_freq}; city_strategy={city_strategy}",
            "source": "business_metadata",
            "requires_audit": False,
            "default_rule": "missing state/city->unknown; missing lat/lon->median; missing is_open->0",
            "extra": {
                "city_strategy": city_strategy,
                "kept_city_count": int(len(kept_cities)),
            },
        }

    def _build_category_block(self, prepared: pd.DataFrame) -> dict[str, Any]:
        row_tokens = [_unique_preserving_order(tokens) for tokens in prepared["parsed_categories"]]
        token_counts: Counter[str] = Counter()
        for tokens in row_tokens:
            token_counts.update(tokens)

        kept_tokens = sorted(
            token
            for token, count in token_counts.items()
            if count >= self.config.min_category_freq
        )
        kept_set = set(kept_tokens)
        filtered_tokens = [[token for token in tokens if token in kept_set] for tokens in row_tokens]

        mlb = MultiLabelBinarizer(classes=kept_tokens, sparse_output=True)
        matrix = mlb.fit_transform(filtered_tokens).tocsr().astype(np.float32)
        feature_names = [f"category__{_safe_feature_name(token)}" for token in kept_tokens]

        prepared["kept_category_tokens"] = filtered_tokens
        prepared["kept_category_count"] = [len(tokens) for tokens in filtered_tokens]

        return {
            "block_name": "categories",
            "matrix": matrix,
            "feature_names": feature_names,
            "coverage": float(np.mean([len(tokens) > 0 for tokens in filtered_tokens])) if len(filtered_tokens) else 0.0,
            "pruning_rule": f"category_freq>={self.config.min_category_freq}",
            "source": "business_metadata.categories",
            "requires_audit": False,
            "default_rule": "missing categories->empty set",
            "extra": {
                "kept_token_count": int(len(kept_tokens)),
                "raw_unique_token_count": int(len(token_counts)),
            },
        }

    def _build_attribute_block(self, prepared: pd.DataFrame) -> dict[str, Any]:
        parsed_dicts = prepared["parsed_attributes"].tolist()
        key_value_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in parsed_dicts:
            for key, value in row.items():
                key_value_counts[key][_normalize_attribute_value(value)] += 1

        eligible_keys = {
            key
            for key, counts in key_value_counts.items()
            if len(counts) <= self.config.max_attribute_values_per_key
        }

        kept_pairs: set[str] = set()
        other_keys: set[str] = set()
        for key in eligible_keys:
            for value, count in key_value_counts[key].items():
                token = f"{key}={value}"
                if count >= self.config.min_attribute_value_freq:
                    kept_pairs.add(token)
                else:
                    other_keys.add(key)

        row_tokens: list[list[str]] = []
        for row in parsed_dicts:
            tokens_for_row: set[str] = set()
            for key, value in row.items():
                if key not in eligible_keys:
                    continue
                token = f"{key}={_normalize_attribute_value(value)}"
                if token in kept_pairs:
                    tokens_for_row.add(token)
                elif key in other_keys:
                    tokens_for_row.add(f"{key}=other")
            row_tokens.append(sorted(tokens_for_row))

        all_tokens = sorted({token for tokens in row_tokens for token in tokens})
        mlb = MultiLabelBinarizer(classes=all_tokens, sparse_output=True)
        matrix = mlb.fit_transform(row_tokens).tocsr().astype(np.float32)
        feature_names = [f"attribute__{_safe_feature_name(token)}" for token in all_tokens]

        prepared["kept_attribute_tokens"] = row_tokens
        prepared["kept_attribute_count"] = [len(tokens) for tokens in row_tokens]

        return {
            "block_name": "attributes",
            "matrix": matrix,
            "feature_names": feature_names,
            "coverage": float(np.mean([len(tokens) > 0 for tokens in row_tokens])) if len(row_tokens) else 0.0,
            "pruning_rule": (
                f"attribute_value_freq>={self.config.min_attribute_value_freq}; "
                f"max_values_per_key<={self.config.max_attribute_values_per_key}"
            ),
            "source": "business_metadata.attributes",
            "requires_audit": False,
            "default_rule": "missing or unparsable attributes->empty set",
            "extra": {
                "kept_token_count": int(len(all_tokens)),
                "eligible_key_count": int(len(eligible_keys)),
                "raw_key_count": int(len(key_value_counts)),
            },
        }

    def _build_hours_block(self, prepared: pd.DataFrame) -> dict[str, Any]:
        columns = [
            "open_days_count",
            "weekly_open_minutes",
            "weekend_days_open",
            "late_night_days",
        ]
        numeric = prepared[columns].astype(np.float32).copy()
        for column in columns:
            numeric[column] = _safe_zscore(numeric[column].to_numpy(dtype=np.float32))

        matrix = sparse.csr_matrix(numeric.to_numpy(dtype=np.float32))
        feature_names = [f"hours__{column}_z" for column in columns]

        raw_hours_present = prepared.get("hours", pd.Series(index=prepared.index)).notna().mean()
        return {
            "block_name": "hours",
            "matrix": matrix,
            "feature_names": feature_names,
            "coverage": float(raw_hours_present),
            "pruning_rule": "no pruning; compact numeric hours features",
            "source": "business_metadata.hours",
            "requires_audit": False,
            "default_rule": "missing or invalid hours->0 then z-score",
            "extra": {
                "kept_token_count": int(len(feature_names)),
            },
        }

    def _build_prior_block(self, prepared: pd.DataFrame) -> dict[str, Any]:
        if not self.config.include_priors:
            matrix = sparse.csr_matrix((len(prepared), 0), dtype=np.float32)
            return {
                "block_name": "priors",
                "matrix": matrix,
                "feature_names": [],
                "coverage": 0.0,
                "pruning_rule": "priors disabled",
                "source": "train_reviews aggregates",
                "requires_audit": True,
                "default_rule": "not included",
                "extra": {
                    "kept_token_count": 0,
                },
            }

        columns = [
            "prior_seen_in_train",
            "prior_train_review_count_log1p",
            "prior_train_support_percentile",
            "prior_train_average_stars",
            "prior_train_rating_std",
        ]
        numeric = prepared[columns].astype(np.float32).copy()
        matrix = sparse.csr_matrix(numeric.to_numpy(dtype=np.float32))
        feature_names = [
            "prior__seen_in_train",
            "prior__train_review_count_log1p",
            "prior__train_support_percentile",
            "prior__train_average_stars",
            "prior__train_rating_std",
        ]

        return {
            "block_name": "priors",
            "matrix": matrix,
            "feature_names": feature_names,
            "coverage": float(prepared["prior_seen_in_train"].mean()),
            "pruning_rule": "derived only from train_reviews aggregates",
            "source": "train_reviews aggregates",
            "requires_audit": True,
            "default_rule": "missing train aggregate->0 plus seen flag",
            "extra": {
                "kept_token_count": int(len(feature_names)),
            },
        }

    def _build_block_summary(
        self,
        content_blocks: list[dict[str, Any]],
        prior_block: dict[str, Any],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        start_index = 0
        for block in [*content_blocks, prior_block]:
            width = int(block["matrix"].shape[1])
            rows.append(
                {
                    "block_name": block["block_name"],
                    "n_columns": width,
                    "coverage": float(block["coverage"]),
                    "pruning_rule": block["pruning_rule"],
                    "source": block["source"],
                    "requires_audit": bool(block["requires_audit"]),
                    "default_rule": block["default_rule"],
                    "start_index": start_index,
                    "end_index_exclusive": start_index + width,
                }
            )
            start_index += width
        return pd.DataFrame(rows)

    def _build_feature_metadata(
        self,
        content_blocks: list[dict[str, Any]],
        prior_block: dict[str, Any],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        feature_index = 0
        for block in [*content_blocks, prior_block]:
            for feature_name in block["feature_names"]:
                rows.append(
                    {
                        "feature_index": feature_index,
                        "feature_name": feature_name,
                        "block_name": block["block_name"],
                        "source": block["source"],
                        "requires_audit": bool(block["requires_audit"]),
                        "default_rule": block["default_rule"],
                    }
                )
                feature_index += 1
        return pd.DataFrame(rows)

    def _build_clean_business_table(self, prepared: pd.DataFrame) -> pd.DataFrame:
        table = pd.DataFrame(
            {
                "business_id": prepared["business_id"],
                "state_clean": prepared["state_clean"],
                "city_clean": prepared["city_clean"],
                "city_bucket": prepared["city_bucket"],
                "latitude_filled": prepared["latitude_filled"],
                "longitude_filled": prepared["longitude_filled"],
                "is_open_clean": prepared["is_open_clean"],
                "parsed_categories": prepared["parsed_categories"].apply(lambda values: "|".join(values)),
                "kept_category_tokens": prepared["kept_category_tokens"].apply(lambda values: "|".join(values)),
                "kept_category_count": prepared["kept_category_count"],
                "kept_attribute_tokens": prepared["kept_attribute_tokens"].apply(lambda values: "|".join(values)),
                "kept_attribute_count": prepared["kept_attribute_count"],
                "open_days_count": prepared["open_days_count"],
                "weekly_open_minutes": prepared["weekly_open_minutes"],
                "weekend_days_open": prepared["weekend_days_open"],
                "late_night_days": prepared["late_night_days"],
                "prior_seen_in_train": prepared["prior_seen_in_train"],
                "prior_train_review_count": prepared["prior_train_review_count"],
                "prior_train_review_count_log1p": prepared["prior_train_review_count_log1p"],
                "prior_train_average_stars": prepared["prior_train_average_stars"],
                "prior_train_rating_std": prepared["prior_train_rating_std"],
                "prior_train_support_percentile": prepared["prior_train_support_percentile"],
            }
        )
        return table

    def _build_validation_summary(
        self,
        prepared: pd.DataFrame,
        content_matrix: sparse.csr_matrix,
        prior_matrix: sparse.csr_matrix,
        full_matrix: sparse.csr_matrix,
        feature_metadata: pd.DataFrame,
    ) -> dict[str, Any]:
        categories_present = prepared.get("categories", pd.Series(index=prepared.index)).notna()
        attributes_present = prepared.get("attributes", pd.Series(index=prepared.index)).notna()
        hours_present = prepared.get("hours", pd.Series(index=prepared.index)).notna()

        known_nested_keys = set()
        for row in prepared["parsed_attributes"]:
            known_nested_keys.update(key for key in row if key.startswith("BusinessParking.") or key.startswith("Ambience."))

        return {
            "one_row_per_business": bool(len(prepared) == prepared["business_id"].nunique()),
            "content_plus_prior_matches_full": bool(content_matrix.shape[1] + prior_matrix.shape[1] == full_matrix.shape[1]),
            "category_parse_success_rate_on_present_rows": float(
                np.mean([len(tokens) > 0 for tokens in prepared.loc[categories_present, "parsed_categories"]])
            ) if categories_present.any() else 1.0,
            "attribute_parse_success_rate_on_present_rows": float(
                np.mean([len(values) > 0 for values in prepared.loc[attributes_present, "parsed_attributes"]])
            ) if attributes_present.any() else 1.0,
            "hours_zero_fill_consistent": bool(
                (
                    prepared.loc[~hours_present, ["open_days_count", "weekly_open_minutes", "weekend_days_open", "late_night_days"]]
                    .fillna(0.0)
                    .to_numpy()
                    == 0.0
                ).all()
            ),
            "known_nested_attribute_keys_detected": sorted(known_nested_keys)[:20],
            "raw_prior_columns_excluded": bool(
                not any(name.endswith("__stars") or name.endswith("__review_count") for name in feature_metadata["feature_name"])
            ),
        }

    def _run_validation_checks(
        self,
        prepared: pd.DataFrame,
        feature_metadata: pd.DataFrame,
        full_matrix: sparse.csr_matrix,
        validation_summary: dict[str, Any],
    ) -> None:
        if not validation_summary["one_row_per_business"]:
            raise ValueError("Business representation must keep a 1:1 relationship with business_id.")

        if not validation_summary["content_plus_prior_matches_full"]:
            raise ValueError("content_matrix + prior_matrix width does not match full_matrix width.")

        if full_matrix.shape[0] != len(prepared):
            raise ValueError("Feature matrix row count must match number of businesses.")

        if feature_metadata["feature_index"].nunique() != len(feature_metadata):
            raise ValueError("feature_index must be unique.")

        if feature_metadata["feature_index"].max() != len(feature_metadata) - 1:
            raise ValueError("feature_index must be contiguous.")


def _normalize_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "__unknown__"
    text = str(value).strip()
    return text if text else "__unknown__"


def _normalize_attribute_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.4f}".rstrip("0").rstrip(".")
    return _normalize_token(value)


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def _safe_feature_name(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "unknown"


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    if std == 0.0:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - mean) / std).astype(np.float32)


def _save_clean_table(df: pd.DataFrame, filepath_without_suffix: Path) -> Path:
    parquet_path = filepath_without_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError, ValueError):
        csv_path = filepath_without_suffix.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path
