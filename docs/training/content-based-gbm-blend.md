# Content-Based GBM Blend

- Date: `2026-04-10`
- Status: `implemented`
- Scope: final competition-oriented submission pipeline over the leaky deep model plus a LightGBM fallback

## What Was Built

The content-based branch now has a hybrid submission path with these components:

- `train_gbm_regressor.py`: trains and validates a LightGBM regressor over frozen embeddings, scalar priors, and review context
- `train_gbm_submission_model.py`: retrains the GBM on all `train_reviews`
- `predict_gbm_submission.py`: generates a rounded GBM submission with columns `review_id,stars`
- `blend_deep_gbm_submission.py`: blends the already-trained deep submission with the GBM submission

## Blend Rule

The final submission uses this routing policy:

```text
if user_id is in train_reviews:
    final_star = round_half_up((deep_star + gbm_star) / 2)
else:
    final_star = gbm_star
```

This means:

- known users use the mean of the already rounded deep and GBM predictions
- new users fall back to the GBM only

## Cold-Start Handling

The GBM feature builder treats cold start against train-derived priors, not bundle membership.
This is required because the exported deep bundle may already include metadata-only user embeddings.

The GBM training path also adds synthetic cold-start rows by masking user-side features on a random subset of train rows.
That gives the tree model real examples of `history_count = 0` and `is_new_user = 1` during fitting.

## Selected Validation Run

Selected validation artifact:

- `content-based/artifacts/gbm_regressor_v1_cs30`

Key metrics from `validation_summary.json`:

- `val_mae_gbm_raw`: `1.0054`
- `val_mae_gbm_rounded`: `1.0091`
- `blend_validation.blend_mae`: `0.9457`
- `synthetic_cold_start_fraction`: `0.3`
- `best_iteration`: `113`

Cold-start-heavy validation bands for the blend:

- `history_band 0`: `1.0588`
- `history_band 1`: `0.7945`

## Final Submission Artifacts

Artifacts generated from the selected run:

- GBM full-train model: `content-based/artifacts/gbm_submission_v1`
- Final hybrid submission: `content-based/artifacts/blended_submission_v1/submission.csv`

Final submission summary:

- rows: `414,765`
- known-user blended rows: `244,828`
- GBM fallback rows: `169,937`
- columns: `review_id,stars`

## Commands Used

Validation:

```bash
cd content-based
python train_gbm_regressor.py --n-estimators 300 --early-stopping-rounds 30 --synthetic-cold-start-fraction 0.3 --save-root artifacts/gbm_regressor_v1_cs30
```

Full-train GBM:

```bash
cd content-based
python train_gbm_submission_model.py --source-run artifacts/gbm_regressor_v1_cs30 --save-root artifacts/gbm_submission_v1
```

GBM submission:

```bash
cd content-based
python predict_gbm_submission.py --artifact-root artifacts/gbm_submission_v1 --save-path artifacts/gbm_submission_v1/submission.csv
```

Hybrid submission:

```bash
cd content-based
python blend_deep_gbm_submission.py --deep-submission artifacts/frozen_embedding_submission_v1/submission.csv --gbm-submission artifacts/gbm_submission_v1/submission.csv --save-root artifacts/blended_submission_v1
```
