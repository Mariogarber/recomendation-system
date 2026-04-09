# Project Status

This document is the living status report for the whole repository.
It should describe the current state of the project as it actually exists in code,
not only the intended design.

## Project Goal

The repository is focused on rating prediction for a reduced Yelp dataset.
The current working task is:

- predict the rating a user will give to a business

The main evaluation metric for the current content-based dataset is:

- `MAE`

## Repository Overview

The repository currently has two main development branches:

### 1. Collaborative Filtering

Directory:

- [colaborative-filtering](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/colaborative-filtering)

This branch is the most mature one.
It already contains multiple implemented models, utilities, metrics, and ensemble methods.

Main documented modules:

- [Models](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/models.md)
- [Metrics](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/metrics.md)
- [Ensembles](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/ensemble.md)
- [Utils](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/utils.md)

Implemented capabilities include:

- baseline predictors
- KNN variants
- matrix factorization
- Bayesian factorization variants
- ensemble methods
- evaluation utilities and profiling

### 2. Content-Based

Directory:

- [content-based](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based)

This branch is under active development and has recently moved from rough planning to reusable code.

Main documents:

- [Content-Based README](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/README.md)
- [Content-Based Plan](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/PLAN.md)
- [Technical Checklist](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/TECHNICAL_CHECKLIST.md)
- [Content-Based Feature Guide](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-features-guide.md)
- [Deep User Embeddings RFC](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-embeddings-rfc.md)
- [Deep User Embeddings Dataflow](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-embeddings-dataflow.md)
- [Deep User Embeddings Experiments](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/content-based-deep-user-embeddings-experiments.md)

## Current Content-Based Status

### Completed

- dataset loading utilities
- random and temporal split helpers
- cold-start breakdown utility
- leakage audit utilities for users and businesses
- business metadata parsers for `categories`, `attributes`, and `hours`
- minimal base model abstraction
- executable Phase 1 audit script
- business representation V1 builder
- user representation V1 builder
- persistence of business representation artifacts
- persistence of user representation artifacts

### Not Yet Implemented

- leakage-safe baselines for the content-based branch
- business-to-user similarity scoring
- supervised content-based regressor
- segmented evaluation pipeline for content-based models
- explicit cold-start prediction policy

There is now also a pre-implementation deep-user-embedding proposal documented as an RFC with dedicated dataflow and experiment annexes. This proposal does not replace the current builder yet; it defines a future parallel learned user-embedding family.

## Dataset Facts Already Established

These facts are already supported by the implemented audit scripts and should be treated as current repo knowledge unless new data replaces them.

### Dataset Scale

- `train_reviews`: `967,784`
- `test_reviews`: `414,765`
- unique train users: `541,915`
- unique train businesses: `30,064`
- train density: `0.0000594`

### Cold Start

The main cold-start problem is on the user side:

- `both_known`: `244,812`
- `new_user_known_item`: `169,927`
- `known_user_new_item`: `16`
- `both_new`: `10`

This means roughly `40.97%` of test rows involve a new user and a known business.

### Leakage Risk

Some metadata cannot be trusted as direct model input without audit:

Business metadata compared with train-derived aggregates:

- `review_count_exact_match_rate`: `5.18%`
- `average_stars_mean_abs_diff`: `0.2255`

User metadata compared with train-derived aggregates:

- `review_count_exact_match_rate`: `9.23%`
- `average_stars_mean_abs_diff`: `0.7894`

Current modeling consequence:

- direct raw aggregates from metadata are considered risky
- business priors should be recomputed from `train_reviews.csv`

### Business Metadata Coverage

- `categories_present_rate`: `99.94%`
- `attributes_present_rate`: `90.79%`
- `hours_present_rate`: `84.41%`
- unique states: `17`
- unique cities: `776`
- unique flattened attribute keys: `87`

This is why the business representation is currently the center of the content-based branch.

## Current Business Representation V1

The current V1 builder is implemented in:

- [business_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/business_representation.py)
- [build_business_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)

### Representation Structure

The business representation is currently split into these blocks:

- `geo`
- `categories`
- `attributes`
- `hours`
- `priors`

And exposed in three matrices:

- `content_matrix`
- `prior_matrix`
- `full_matrix`

### Latest Validated Output

Latest smoke-test output:

- businesses: `30,069`
- content features: `871`
- prior features: `5`
- total features: `876`
- city strategy used: `hashing`
- kept category features: `558`
- kept attribute features: `225`

Current block widths:

- `geo`: `84`
- `categories`: `558`
- `attributes`: `225`
- `hours`: `4`
- `priors`: `5`

Validation checks already passing:

- one row per business
- content and prior widths match full representation
- category parse success on rows where categories are present
- attribute parse success on rows where attributes are present
- missing hours converted consistently
- raw metadata priors excluded

## Current User Representation V1

The current V1 builder is implemented in:

- [user_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/user_representation.py)
- [build_user_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)

### Representation Structure

The user representation is currently split into these blocks:

- `profile`
- `metadata`

And exposed in these matrices:

- `profile_matrix`
- `metadata_matrix`
- `full_user_matrix`

### Current Behavior

- profile vectors are aggregated from train interactions and business vectors
- default aggregation is centered weighting
- single-review users fall back to a simple positive profile
- zero-sum centered profiles fall back to a simple positive profile
- default business source is the business `content` view
- safe user metadata is added as a separate calibration block

Current metadata block candidates included in code:

- tenure derived from `yelping_since`
- `elite` count and binary indicator
- log-scaled and normalized `useful`, `funny`, `cool`, `fans`
- log-scaled and normalized `compliment_*`

Current exclusions:

- `friends`
- raw `review_count`
- raw `average_stars`

### Latest Validated Output

Latest smoke-test output:

- users with profile: `541,915`
- profile features: `871`
- metadata features: `18`
- total user features: `889`
- aggregation mode: `centered`
- one-review users: `397,130`
- `2-5` reviews: `124,980`
- `6-20` reviews: `17,564`
- `>20` reviews: `2,241`
- centered single-review fallback count: `397,130`
- centered zero-absolute-weight fallback count: `42,895`

Metadata coverage note:

- exactly one train user is absent from `usuarios.csv`: `ufZfni7nb_KdJC6DXNfVHQ`
- the current builder keeps that user in the profile matrix and fills metadata defaults

## Available Content-Based Scripts

### Phase 1 Audit

Script:

- [phase1_audit.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/phase1_audit.py)

Purpose:

- dataset summary
- cold-start summary
- leakage summary
- parser summary

Example:

```powershell
python .\content-based\phase1_audit.py
```

### Business Representation Builder

Script:

- [build_business_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_business_representation.py)

Purpose:

- build business content/prior/full matrices
- generate feature metadata and block summary
- persist artifacts for future experiments

Example:

```powershell
python .\content-based\build_business_representation.py --save-dir .\content-based\artifacts\business_repr_v1
```

### User Representation Builder

Script:

- [build_user_representation.py](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/build_user_representation.py)

Purpose:

- build user profile, metadata, and full matrices
- generate user feature metadata and profile summary
- persist user representation artifacts for downstream experiments

Example:

```powershell
python .\content-based\build_user_representation.py --save-dir .\content-based\artifacts\user_repr_v1
```

## Existing Artifacts

An example artifact directory already exists:

- [business_repr_v1_smoke](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/business_repr_v1_smoke)
- [user_repr_v1_smoke](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/artifacts/user_repr_v1_smoke)

Artifacts currently produced by the builder:

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

## Recommended Next Steps

From the current state of the repository, the next logical tasks are:

1. Implement leakage-safe baselines for the content-based branch
2. Build user profiles from train interactions and business content vectors
3. Add simple similarity-based scoring
4. Add supervised regression using user profile + business features
5. Add segmented MAE reporting, especially for new users

## Documentation Maintenance

This file should be updated whenever any of these change:

- branch status
- dataset findings
- implemented scripts
- artifact formats
- business representation dimensions
- phase completion in the content-based checklist

Recommended documentation update points:

- `docs/project-status.md`: global snapshot
- `content-based/README.md`: module-specific technical state
- `content-based/PLAN.md`: intended future design
- `content-based/TECHNICAL_CHECKLIST.md`: execution progress
