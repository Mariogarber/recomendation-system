# Technical Checklist

> Documento legacy. La fuente canónica actual es:
> - [docs/status/current-state.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/status/current-state.md)

This checklist turns the content-based plan into concrete implementation work. It now reflects both the manual builders and the implemented deep competition pipeline.

## Phase 1: Audit and Dataset Contracts

- [x] Create reusable data loaders in `content-based/utils/io.py`
- [x] Create random and temporal validation helpers in `content-based/utils/split.py`
- [x] Create leakage-audit utilities in `content-based/utils/audit.py`
- [x] Create business parsers for `categories`, `attributes`, and `hours` in `content-based/utils/business_features.py`
- [x] Create a minimal model interface in `content-based/model/base.py`
- [x] Create an executable Phase 1 audit script in `content-based/phase1_audit.py`

Outputs covered by Phase 1:

- train/test interaction summary
- cold-start breakdown for validation or test
- user metadata vs train aggregate comparison
- business metadata vs train aggregate comparison
- coverage report for `categories`, `attributes`, and `hours`
- parsed attribute-key frequency summary

## Phase 2: Baselines and Validation

- [ ] Implement global mean baseline
- [ ] Implement business-mean baseline from train only
- [ ] Implement user-mean baseline from train only
- [ ] Implement bias baseline: `global + user_bias + business_bias`
- [ ] Evaluate baselines on random split
- [ ] Evaluate baselines on temporal split
- [ ] Compare MAE globally and by cold-start segment

Notes:

- The report already contains diagnostic post-export scorers, but those are not a substitute for a proper baseline suite.
- Any new baseline must be leakage-safe at the level of user history construction.

Recommended output files:

- `baseline_results.csv`
- `baseline_segment_results.csv`

## Phase 3: Business Representation

- [x] Build a clean business table after leakage-safe filtering
- [x] Encode categories as multi-hot features
- [x] Flatten and encode parsed attributes
- [x] Convert hours into compact numeric features
- [x] Normalize geographic features
- [x] Persist the business feature matrix and feature names
- [x] Recompute business priors from `train_reviews.csv`

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

## Phase 4: User Profiles and Deep Embeddings

### 4A. Manual user profiles

- [x] Build item-content vectors for every business
- [x] Build user profiles from businesses rated in train
- [x] Support `mean`, `rating`, `centered`, and `recency`
- [x] Support business-view ablations through `--business-view` and `--business-blocks`
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

### 4B. Deep competition embeddings

- [x] Build a business tower on top of the business representation
- [x] Train a rating-aware deep user encoder with temporal validation
- [x] Export `user_deep_features` and `business_deep_features`
- [x] Export `history`, `metadata_only`, and `default_only` coverage in the deep summary
- [x] Generate the competition embedding quality report and report CSVs

Recommended output files:

- `business_repr/`
- `user_manual_repr/`
- `user_deep_repr/`
- `embedding_quality_report.html`
- `embedding_quality_report_summary.json`

Run from repo root:

```powershell
python .\content-based\build_competition_embeddings.py --save-root .\content-based\artifacts\competition_embeddings_v1
```

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

- [ ] Add friend-summary features as a model feature block
- [x] Add user metadata calibration block
- [ ] Add lightweight hybridization with collaborative predictions
- [ ] Run feature-family ablations as a tracked experiment suite

## Acceptance Criteria

- [ ] No feature enters the model without leakage classification
- [ ] Baselines are reproducible
- [ ] Every experiment reports global MAE and segmented MAE
- [ ] Cold-start behavior is explicitly documented
- [ ] The final model beats naive global and business-mean baselines
