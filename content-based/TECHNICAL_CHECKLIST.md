# Technical Checklist

This checklist turns the content-based plan into concrete implementation work.
It is ordered so we can execute Phase 1 first and then move into modeling.

## Phase 1: Audit and Dataset Contracts

- [x] Create reusable data loaders in `content-based/utils/io.py`
- [x] Create random and temporal validation helpers in `content-based/utils/split.py`
- [x] Create leakage-audit utilities in `content-based/utils/audit.py`
- [x] Create business parsers for `categories`, `attributes`, and `hours` in `content-based/utils/business_features.py`
- [x] Create a minimal model interface in `content-based/model/base.py`
- [x] Create an executable Phase 1 audit script in `content-based/phase1_audit.py`

Outputs expected from Phase 1:

- Train/test interaction summary
- Cold-start breakdown for validation or test
- User metadata vs train aggregate comparison
- Business metadata vs train aggregate comparison
- Coverage report for `categories`, `attributes`, and `hours`
- Parsed attribute-key frequency summary

## Phase 1 execution steps

Run from repo root:

```powershell
python .\content-based\phase1_audit.py
```

Optionally save reports:

```powershell
python .\content-based\phase1_audit.py --save-dir .\content-based\reports\phase1
```

Files produced when `--save-dir` is used:

- `dataset_summary.json`
- `cold_start_summary.json`
- `user_leakage_summary.json`
- `business_leakage_summary.json`
- `user_leakage_details.csv`
- `business_leakage_details.csv`
- `business_parser_summary.json`
- `attribute_key_counts.csv`

## Phase 2: Baselines and Validation

- [ ] Implement global mean baseline
- [ ] Implement business-mean baseline from train only
- [ ] Implement user-mean baseline from train only
- [ ] Implement bias baseline: `global + user_bias + business_bias`
- [ ] Evaluate baselines on random split
- [ ] Evaluate baselines on temporal split
- [ ] Compare MAE globally and by cold-start segment

Recommended output files:

- `baseline_results.csv`
- `baseline_segment_results.csv`

## Phase 3: Business Representation

- [x] Build a clean business table after leakage-safe filtering
- [x] Encode categories as multi-hot or TF-IDF
- [x] Flatten and encode parsed attributes
- [x] Convert hours into compact numeric features
- [x] Normalize geographic features
- [x] Persist the business feature matrix and feature names

Recommended output files:

- `clean_business_table.parquet`
- `business_content_features.npz`
- `business_prior_features.npz`
- `business_full_features.npz`
- `business_feature_names.json`
- `business_block_summary.csv`
- `feature_metadata.csv`

Run from repo root:

```powershell
python .\content-based\build_business_representation.py --save-dir .\content-based\artifacts\business_repr_v1
```

## Phase 4: User Profiles

- [x] Build item-content vectors for every business
- [x] Build user profiles from businesses rated in train
- [ ] Test aggregation variants:
  - mean
  - rating-weighted mean
  - centered weighting
  - recency weighting
- [x] Store user profile coverage and history-length diagnostics
- [x] Add a safe user metadata block separated from the profile block

Recommended output files:

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

Run from repo root:

```powershell
python .\content-based\build_user_representation.py --save-dir .\content-based\artifacts\user_repr_v1
```

Core behavior implemented in V1:

- primary profile built from rated business vectors
- default aggregation: centered weighting
- support for `mean`, `rating`, `centered`, and `recency`
- support for business-view ablations through `--business-view` and `--business-blocks`
- optional safe user metadata block through `--no-metadata`

## Phase 5: Content-Based Prediction

- [ ] Implement similarity scoring between user profile and candidate business
- [ ] Build a supervised regressor using:
  - user profile vector
  - candidate business vector
  - similarity features
  - safe metadata features
  - optional review-context features
- [ ] Clip predictions to the valid rating range
- [ ] Compare Ridge vs boosting models

Recommended output files:

- `content_model_results.csv`
- `content_model_segment_results.csv`

## Phase 6: Cold Start Policies

- [ ] Define separate behavior for:
  - known user, known business
  - new user, known business
  - known user, new business
  - new user, new business
- [ ] Compare a single unified model against segmented policies
- [ ] Measure MAE specifically for new-user rows

## Phase 7: Optional Extensions

- [ ] Add friend-summary features
- [x] Add user metadata calibration block
- [ ] Add lightweight hybridization with collaborative predictions
- [ ] Run feature-family ablations

## Acceptance Criteria

- [ ] No feature enters the model without leakage classification
- [ ] Baselines are reproducible
- [ ] Every experiment reports global MAE and segmented MAE
- [ ] Cold-start behavior is explicitly documented
- [ ] The final model beats naive global and business-mean baselines
