# Content-Based Branch

This document is the living documentation for the `content-based/` module.
Its purpose is to describe what exists today, how it works, what has been validated,
which artifacts are produced, and what is still missing.

## Goal

The goal of the content-based branch is to predict the rating a user will give to a business by:

- representing each business with a structured content vector
- building user preferences from previously rated businesses
- scoring candidate businesses against those user profiles

The main metric is:

- `MAE`

## Current Development State

The branch is in an intermediate but useful state:

- Data audit is implemented
- Leakage analysis is implemented
- Business metadata parsers are implemented
- Business representation V1 is implemented
- User representation V1 is implemented
- Final content-based scoring and regression are not implemented yet

In checklist terms:

- Phase 1: done
- Phase 2: pending
- Phase 3: done
- Phase 4 core builder: done
- Phase 5+: pending

See also:

- [Plan](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/PLAN.md)
- [Technical Checklist](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/TECHNICAL_CHECKLIST.md)
- [Project Status](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/project-status.md)

## Dataset Used By This Module

The module currently assumes the files under `content-based/data/`:

- `usuarios.csv`
- `negocios.csv`
- `train_reviews.csv`
- `test_reviews.csv`

Current audited summary:

- `train_rows`: `967,784`
- `test_rows`: `414,765`
- `train_unique_users`: `541,915`
- `train_unique_items`: `30,064`
- `train_density`: `0.0000594`
- rating mean: `3.7599`
- rating std: `1.4795`

## Main Findings So Far

### 1. The dataset is extremely sparse

This makes pure interaction-based history weak for many users and increases the value of a strong business representation.

### 2. Cold start is mainly a user problem

Current test breakdown from the audit:

- `both_known`: `244,812` (`59.02%`)
- `new_user_known_item`: `169,927` (`40.97%`)
- `known_user_new_item`: `16`
- `both_new`: `10`

This means the project should prioritize:

- robust business features
- sensible new-user behavior
- explicit cold-start evaluation

### 3. Some metadata is likely leakage-prone

Business metadata compared with aggregates recomputed only from `train_reviews.csv` shows:

- `review_count_exact_match_rate`: `5.18%`
- `average_stars_mean_abs_diff`: `0.2255`

User metadata shows an even stronger mismatch:

- `review_count_exact_match_rate`: `9.23%`
- `average_stars_mean_abs_diff`: `0.7894`

As a consequence:

- `stars` and `review_count` from `negocios.csv` are not used directly as raw priors
- any aggregate prior used in the model must be recomputed from train

### 4. Business metadata has strong signal

Current parser coverage summary:

- `categories_present_rate`: `99.94%`
- `attributes_present_rate`: `90.79%`
- `hours_present_rate`: `84.41%`
- unique states: `17`
- unique cities: `776`
- unique flattened attribute keys: `87`

This supports the design choice of making business representation the core of the branch.

## What Is Implemented

### Data IO

File:

- [io.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/io.py)

Responsibilities:

- load users, businesses, train reviews, and test reviews
- normalize review tables into a common schema when needed

### Splits And Cold Start Analysis

File:

- [split.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/split.py)

Responsibilities:

- random train/validation split
- temporal train/validation split
- cold-start breakdown between train and evaluation sets

### Leakage Audit

File:

- [audit.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/audit.py)

Responsibilities:

- build train-only aggregates for users and businesses
- compare metadata against train-derived aggregates
- summarize likely leakage

### Business Metadata Parsing

File:

- [business_features.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/business_features.py)

Responsibilities:

- parse category strings into token lists
- parse nested business attributes into flattened key/value mappings
- parse opening hours into compact numerical features

Examples of supported nested keys:

- `BusinessParking.*`
- `Ambience.*`

### Business Representation V1

Files:

- [business_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/business_representation.py)
- [build_business_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)

This is currently the most advanced part of the content-based branch.

It builds separate blocks for:

- `geo`
- `categories`
- `attributes`
- `hours`
- `priors`

The representation explicitly separates:

- `content_matrix`
- `prior_matrix`
- `full_matrix`

Current V1 output summary:

- businesses: `30,069`
- content features: `871`
- prior features: `5`
- full features: `876`
- city strategy used: `hashing`
- kept category features: `558`
- kept attribute features: `225`

Current block widths:

- `geo`: `84`
- `categories`: `558`
- `attributes`: `225`
- `hours`: `4`
- `priors`: `5`

Validation checks currently passing:

- one row per business
- content + prior dimensions match full representation
- category parse success on present rows
- attribute parse success on present rows
- consistent zero-fill for missing hours
- raw `stars` and raw `review_count` excluded from direct priors

### User Representation V1

Files:

- [user_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/user_representation.py)
- [build_user_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)

This builder creates one row per train user and separates two blocks:

- `profile_matrix`
- `metadata_matrix`

The final output is:

- `full_user_matrix`

Core design choices implemented:

- the main user taste profile is computed from businesses rated in `train`
- the default aggregation is centered weighting:
  - `weight = rating - mean_user_rating_train`
- single-review users fall back to a simple positive weight
- zero-sum centered profiles fall back to a simple positive weight
- the default business source is `business_view="content"`
- business priors are not mixed into the pure taste profile by default
- safe user metadata is stored as a separate block

Current metadata features supported:

- `yelping_since` transformed to tenure
- `elite` transformed to count and binary flag
- `useful`, `funny`, `cool`, `fans`
- `compliment_*`

Explicitly excluded from V1:

- `friends`
- raw user `review_count`
- raw user `average_stars`

Supported aggregation modes:

- `mean`
- `rating`
- `centered`
- `recency`

Supported ablation controls:

- full business view selection with `content`, `prior`, or `full`
- block selection through `--business-blocks`

Current validated V1 output with the default smoke test:

- users with profile: `541,915`
- profile features: `871`
- metadata features: `18`
- full user features: `889`
- aggregation mode: `centered`
- one-review users: `397,130`
- users in `2-5`: `124,980`
- users in `6-20`: `17,564`
- users in `>20`: `2,241`
- centered fallback for single-review users: `397,130`
- centered fallback for zero-absolute-weight users: `42,895`

Current metadata coverage note:

- `usuarios.csv` is missing exactly one train user: `ufZfni7nb_KdJC6DXNfVHQ`
- the builder still creates that user profile and fills metadata defaults when needed

### Minimal BaseModel

File:

- [base.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/base.py)

Purpose:

- provide a minimal common interface for future content-based models

## Available Scripts

### Phase 1 Audit

Command:

```powershell
python .\content-based\phase1_audit.py
```

Purpose:

- print and optionally persist dataset summary, cold-start summary, leakage summaries, and parser summary

### Build Business Representation

Command:

```powershell
python .\content-based\build_business_representation.py --save-dir .\content-based\artifacts\business_repr_v1
```

Main configurable parameters:

- `--min-city-freq`
- `--max-city-ohe`
- `--city-hash-dim`
- `--min-category-freq`
- `--min-attribute-value-freq`
- `--max-attribute-values-per-key`
- `--include-geo-clusters`
- `--geo-cluster-count`
- `--no-priors`

### Build User Representation

Command:

```powershell
python .\content-based\build_user_representation.py --save-dir .\content-based\artifacts\user_repr_v1
```

Main configurable parameters:

- `--aggregation-mode`
- `--business-view`
- `--business-blocks`
- `--no-metadata`
- `--recency-half-life-days`
- business representation knobs inherited by the script:
  - `--min-city-freq`
  - `--max-city-ohe`
  - `--city-hash-dim`
  - `--min-category-freq`
  - `--min-attribute-value-freq`
  - `--max-attribute-values-per-key`
  - `--include-geo-clusters`
  - `--geo-cluster-count`
  - `--no-business-priors`

## Artifacts Produced Today

Example artifact directory already generated:

- [business_repr_v1_smoke](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/business_repr_v1_smoke)
- [user_repr_v1_smoke](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/user_repr_v1_smoke)

Files produced by the business representation builder:

- `business_ids.csv`
- `business_content_features.npz`
- `business_prior_features.npz`
- `business_full_features.npz`
- `business_feature_names.json`
- `business_block_summary.csv`
- `feature_metadata.csv`
- `business_prior_leakage_summary.json`
- `business_prior_leakage_details.csv`
- `business_representation_summary.json`
- `clean_business_table.parquet`

Files produced by the user representation builder:

- `user_ids.csv`
- `user_profile_features.npz`
- `user_metadata_features.npz`
- `user_full_features.npz`
- `user_feature_names.json`
- `user_feature_metadata.csv`
- `user_profile_summary.json`
- `user_metadata_audit_summary.json`
- `user_metadata_audit_details.csv`
- `clean_user_table.parquet`

## What Is Not Implemented Yet

The following parts are still missing:

- leakage-safe baseline models for the content-based branch
- business-only ablation runner
- similarity scoring between user profile and candidate business
- supervised regression on top of user profile + business features
- segmented MAE evaluation for content-based models
- explicit cold-start policy in prediction code
- empirical comparison of user-profile aggregation variants

## Documentation Maintenance

This file should be updated whenever one of these happens:

- a new parser or builder is added
- a feature block changes shape or semantics
- a major dataset finding changes modeling assumptions
- a new artifact type is introduced
- a phase from the checklist is completed

When updating this document, try to keep four sections in sync:

- `Current Development State`
- `Main Findings So Far`
- `What Is Implemented`
- `What Is Not Implemented Yet`
