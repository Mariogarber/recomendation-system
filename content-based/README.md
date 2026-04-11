# Content-Based

> Estado documental: este archivo pasa a ser un hub corto y legacy-friendly.
>
> La documentacion canónica del modulo vive en:
> - [docs/architecture/content-based-current.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/architecture/content-based-current.md)
> - [docs/flows/content-based-pipeline.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/flows/content-based-pipeline.md)
> - [docs/training/content-based-deep-user.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-deep-user.md)
> - [docs/training/content-based-frozen-regressor.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-frozen-regressor.md)
> - [docs/training/content-based-gbm-blend.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-gbm-blend.md)
> - [docs/training/content-based-lgbm-raw-features.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-lgbm-raw-features.md)
> - [docs/training/content-based-lgbm-deep-embeddings.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-lgbm-deep-embeddings.md)
> - [docs/training/content-based-lgbm-raw-router.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-lgbm-raw-router.md)

This document is the living documentation for the `content-based/` module. It reflects what is implemented today, how the manual and deep embedding families are built, and what the current report can and cannot prove.

## Goal

The goal of the content-based branch is to predict the rating a user will give to a business by:

- representing each business with a structured content vector
- building user preferences from previously rated businesses
- scoring candidate businesses against those user profiles

The main metric remains:

- `MAE`

## Current Development State

The branch is now beyond the first manual-only prototype:

- Data audit is implemented
- Leakage analysis is implemented
- Business metadata parsers are implemented
- Business representation V1 is implemented
- User representation V1 is implemented
- The competition deep-user pipeline is implemented
- The embedding quality report is implemented as an offline diagnostic
- A competition-oriented downstream scorer over frozen deep embeddings is implemented
- A hybrid competition submission path over deep predictions plus a GBM fallback is implemented
- Two standalone LightGBM competition baselines are implemented:
  - raw tabular features only
  - deep embeddings plus scalar priors
- A routed LightGBM stack over `raw_core` plus metadata-driven user archetypes is implemented

In checklist terms:

- Phase 1: done
- Phase 3: done
- Phase 4 manual profile: done
- Phase 4 deep competition embeddings: done
- Phase 2, Phase 5, and Phase 6: still pending as formal, leak-safe prediction work

See also:

- [Plan](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/PLAN.md)
- [Technical Checklist](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/TECHNICAL_CHECKLIST.md)
- [Project Status](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/project-status.md)
- [Feature Guide](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-features-guide.md)
- [Embedding Quality Report Guide](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/embedding_quality_report_guide.md)
- [Deep User Embeddings RFC](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-embeddings-rfc.md)
- [Deep User Embeddings Dataflow](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-embeddings-dataflow.md)
- [Deep User Embeddings Experiments](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-embeddings-experiments.md)
- [Deep User Model Flow](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-model-flow.md)
- [Frozen Embedding Regressor](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-frozen-embedding-regressor.md)
- [GBM Blend Training Note](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/training/content-based-gbm-blend.md)

## Current Competition Submission Path

The active competition export path is now hybrid:

- deep submission: `content-based/artifacts/frozen_embedding_submission_v1/submission.csv`
- GBM submission: `content-based/artifacts/gbm_submission_v1/submission.csv`
- final blended submission: `content-based/artifacts/blended_submission_v1/submission.csv`

Prediction rule:

- known train users: average the already rounded deep and GBM stars, then round half-up
- new users: use the GBM star directly

Current deliverable columns:

- `review_id`
- `stars`

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

This makes pure interaction history weak for many users and increases the value of a strong business representation.

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

### 3. Some metadata is leakage-prone if used raw

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

This is the main business-side representation used by both the manual and deep user pipelines.

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

Current validated smoke-test output:

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
- the builder default is `business_view="content"`
- the competition pipeline overrides that and uses `business_view="full"` for the manual bundle
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

### Deep Competition Embeddings

Files:

- [deep_user_embeddings.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/deep_user_embeddings.py)
- [build_competition_embeddings.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_competition_embeddings.py)
- [analyze_embeddings_report.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/analyze_embeddings_report.py)

This pipeline is implemented and exports a second family of user embeddings:

- `user_deep_features`
- `business_deep_features`

Important defaults in the competition script:

- manual user bundle uses `business_view="full"`
- deep user encoder uses `business_view="full"`
- both pipelines keep `include_metadata=True` unless disabled explicitly

The deep pipeline:

- trains a business tower on the existing business representation
- uses temporal validation during training
- learns a dense user embedding from history + ratings + safe metadata
- exports `history`, `metadata_only`, and `default_only` coverage in `user_deep_summary.json`

Latest iterative search on the current codebase:

- recommended exported embedding bundle: [competition_embeddings_v3_iter03](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter03)
- recommended training-head reference bundle: [competition_embeddings_v3_iter04](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04)
- detailed loop notes: [content-based-deep-user-embeddings-experiments.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-embeddings-experiments.md)
- the report now separates `utility_honest_validation.csv` from `utility_post_export_diagnostics.csv`

### Downstream Frozen Regressor

Files:

- [train_frozen_embedding_regressor.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_frozen_embedding_regressor.py)
- [frozen_embedding_regressor.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/model/frozen_embedding_regressor.py)
- [frozen_embedding_regression.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/frozen_embedding_regression.py)

This downstream pipeline is now implemented.

It trains a rating regressor on top of:

- frozen `user_deep_features`
- frozen `business_deep_features`
- inference-safe review-context features:
  - `useful`
  - `funny`
  - `cool`
  - `date`

Current references:

- diagnostic run only: [frozen_embedding_regressor_v1](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_v1)
- honest held-out run: [frozen_embedding_regressor_honest_v1](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_regressor_honest_v1)

Current status:

- the downstream scorer is implemented and reproducible
- a full-train competition submission flow is implemented on top of the best leaky downstream run
- the honest run does not yet beat the Ridge baseline on `MAE`
- the honest baseline and the best trainable run are documented in [Frozen Embedding Regressor](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-frozen-embedding-regressor.md)
- the current competition submission artifact is [frozen_embedding_submission_v1](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_submission_v1)
- the final rounded CSV is [submission.csv](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_submission_v1/submission.csv)

### Competition Submission Flow

Files:

- [train_frozen_embedding_submission_model.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_frozen_embedding_submission_model.py)
- [predict_frozen_embedding_submission.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/predict_frozen_embedding_submission.py)

This flow is intentionally competition-oriented and uses the original full embedding exports.

Current default:

- source run: `iter04_with_review`
- embedding bundle: [competition_embeddings_v3_iter04](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v3_iter04)
- final full-train epochs: `18`
- review-context enabled: `true`

The inference script writes a competition-ready CSV with exactly:

- `ids`
- `prediction`

Predictions are clipped to `[1, 5]` and rounded before export.

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

### Build Competition Embeddings

Command:

```powershell
python .\content-based\build_competition_embeddings.py --save-root .\content-based\artifacts\competition_embeddings_v1
```

Purpose:

- build the business bundle
- build the manual user bundle
- train and export the deep user bundle
- save a single competition-ready artifact tree

### Train Final Frozen Submission Model

Command:

```powershell
python .\content-based\train_frozen_embedding_submission_model.py --device cuda
```

Purpose:

- load the winning leaky downstream configuration from `iter04_with_review`
- train the final model on all original `train_reviews.csv`
- save the checkpoint, full-train summary, and review-context transforms

### Generate Competition Submission

Command:

```powershell
python .\content-based\predict_frozen_embedding_submission.py --device cuda
```

Purpose:

- load the final frozen submission checkpoint
- score all rows from the original `test_reviews.csv`
- save a rounded competition CSV in `ids,prediction` format

## Artifacts Produced Today

Example artifact directories already generated:

- [business_repr_v1_smoke](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/business_repr_v1_smoke)
- [user_repr_v1_smoke](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/user_repr_v1_smoke)
- [competition_embeddings_v1](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/competition_embeddings_v1)
- [frozen_embedding_submission_v1](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/frozen_embedding_submission_v1)

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
- `clean_business_table.parquet` or `clean_business_table.csv` when parquet is unavailable

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
- `clean_user_table.parquet` or `clean_user_table.csv` when parquet is unavailable

Files produced by the competition deep pipeline:

- `business_repr/`
- `user_manual_repr/`
- `user_deep_repr/`

The deep bundle exports:

- `user_deep_ids.csv`
- `business_deep_ids.csv`
- `user_deep_features.npz`
- `business_deep_features.npz`
- `user_deep_feature_names.json`
- `user_deep_feature_metadata.csv`
- `business_deep_feature_metadata.csv`
- `user_deep_summary.json`
- `user_deep_clean_table.parquet` or `user_deep_clean_table.csv`
- `business_deep_clean_table.parquet` or `business_deep_clean_table.csv`
- `deep_user_encoder_checkpoint.pt`

Files produced by the competition submission flow:

- `checkpoint.pt`
- `config.json`
- `train_summary.json`
- `review_context_summary.json`
- `submission_summary.json`
- `submission.csv`

## What Is Not Implemented Yet

The following parts are still missing as formal prediction work:

- leakage-safe baseline models for the content-based branch
- business-only ablation runner
- similarity scoring between user profile and candidate business as a standalone baseline
- segmented MAE evaluation for content-based models outside the diagnostic report
- explicit cold-start policy in prediction code
- empirical comparison of user-profile aggregation variants as a tracked experiment suite

## Documentation Maintenance

This file should be updated whenever one of these happens:

- a new parser or builder is added
- a feature block changes shape or semantics
- a major dataset finding changes modeling assumptions
- a new artifact type is introduced
- a phase from the checklist is completed

When updating this document, keep these sections in sync:

- `Current Development State`
- `Main Findings So Far`
- `What Is Implemented`
- `What Is Not Implemented Yet`
