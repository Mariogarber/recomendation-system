# Hybrid Meta-LightGBM Stack: CF + Content-Based

- Date: 2026-04-13
- Status: approved
- Goal: break through the 0.65 leaderboard ceiling by stacking collaborative filtering and content-based predictions under a LightGBM meta-learner

## Context

Both the simple LightGBM router and the deep content-based router (`known_user_deep_router_v2_eval_v3`) achieve the same ~0.65 MAE on the leaderboard despite the deep model reaching ~0.5999 on validation. This gap indicates the content-based branch has hit a generalization ceiling caused by temporal distribution shift between the validation split and the test set. The CF branch is mature (BPMF, NNBPMF, KNN, PMF, ensembles) but has never been submitted or blended with content-based. The target is to reach 0.60 MAE on the leaderboard.

## Architecture

Two-layer stack:

### Layer 1 — Base Models (unchanged)

- **Content-based:** `known_user_deep_router_v2_eval_v3` — existing router, no changes
- **CF:** best available CF model checkpoint (`.pt` format in `artifacts/`), loaded and used for inference only

Neither base model is retrained.

### Layer 2 — Meta-Learner

A LightGBM model trained on base model predictions over the validation set.

**Input features (7 total):**

| Feature | Source |
|---|---|
| `cf_prediction` | CF model inference |
| `cb_prediction` | Content-based router output |
| `history_band` | Dataset field |
| `user_review_count` | User metadata |
| `user_mean_rating` | User metadata |
| `business_prior_mean` | Business metadata |
| `cf_cb_diff` | `cf_prediction - cb_prediction` |

**LightGBM config (conservative to avoid overfitting on small validation set):**
- `num_leaves = 16`
- `min_data_in_leaf = 50`
- `learning_rate = 0.05`
- `n_estimators = 200`
- early stopping on an inner split of the validation set

### Cold Start (`history_band = 0`)

CF models fail for users with no history. For `history_band = 0` rows, bypass the meta-model entirely and use the existing cold model output from the content-based router directly.

## Leakage Prevention

The meta-model trains only on validation set predictions — rows that neither base model saw during their own training (both were trained on the training set only). This is safe because:

- CF checkpoint was trained on training data only
- Content-based router was trained on training data only
- Meta-LightGBM trains on validation predictions with validation ground-truth ratings as target

## Data Flow

```
Training phase:
  training set -> [CF model training] -> CF checkpoint (already done)
  training set -> [CB router training] -> CB checkpoint (already done)
  validation set -> CF checkpoint -> cf_val_predictions.csv
  validation set -> CB router (re-run inference, not from artifacts summary) -> cb_val_predictions.csv
  [cf_val, cb_val, meta_features_val, val_ratings] -> train meta-LightGBM

Submission phase:
  test set -> CF checkpoint -> cf_test_predictions.csv
  test set -> CB router -> cb_test_predictions.csv
  [cf_test, cb_test, meta_features_test] -> meta-LightGBM -> submission.csv
```

## New Scripts

Two new scripts, no changes to existing code:

- `content-based/predict_cf_for_meta.py`
  - loads CF checkpoint from `artifacts/`
  - runs inference on validation rows and test rows
  - outputs `cf_val_predictions.csv` and `cf_test_predictions.csv`

- `content-based/train_meta_lgbm.py`
  - loads `cf_val_predictions.csv`, `cb_val_predictions.csv`, meta-features, validation ratings
  - trains meta-LightGBM with early stopping
  - applies to test set
  - outputs `submission.csv` to a new artifact folder `meta_lgbm_hybrid_v1`

## Success Criteria

- Meta-LightGBM validation MAE lower than 0.5999 (current best CB-only)
- Leaderboard MAE at or below 0.60
- No leakage: meta-model trained only on validation set predictions

## Out of Scope

- Retraining CF or content-based base models
- Changes to the cold start routing logic
- Adding more than 7 meta-features in the first iteration
