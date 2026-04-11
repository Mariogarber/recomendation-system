# Content-Based LightGBM Raw Features

- Date: `2026-04-10`
- Status: `implemented`
- Scope: standalone raw tabular LightGBM over `usuarios.csv`, `negocios.csv`, `train_reviews.csv`, and `test_reviews.csv`

## Goal

Train a LightGBM model that predicts review stars using only raw tabular signals from:

- user metadata from `usuarios.csv`
- business metadata from `negocios.csv`
- review context from `train_reviews.csv` / `test_reviews.csv`

This pipeline is intentionally separate from the deep-embedding branch and does not depend on frozen embeddings.

## Scripts

- training + validation + full-train export:
  - [`content-based/train_lgbm_raw_features.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/train_lgbm_raw_features.py)
- submission inference:
  - [`content-based/predict_lgbm_raw_features_submission.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/predict_lgbm_raw_features_submission.py)
- feature builder:
  - [`content-based/utils/lgbm_raw_features.py`](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/content-based/utils/lgbm_raw_features.py)

## Feature Sets

Two ablations are supported:

- `raw_core`
  - direct user metadata
  - direct business metadata
  - review context features
  - business category / attribute / hours summaries
  - location categoricals
- `raw_priors`
  - everything in `raw_core`
  - train-derived user priors
  - train-derived business priors
  - train-derived city / state / postal priors

In the current run, `raw_core` is the stronger variant. `raw_priors` is kept as a diagnostic ablation because it overfit on the temporal validation split.

## Outputs

Each run writes an artifact directory with:

- `validation_summary.json`
- `validation_predictions.csv`
- `validation_model.txt`
- `validation_spec.joblib`
- `feature_importance.csv`
- `training_summary.json`
- `submission_model.txt`
- `submission_spec.joblib`
- `submission_summary.json` after inference

The final submission CSV uses exactly:

- `review_id`
- `stars`

## Commands

Run the `raw_core` ablation:

```bash
cd content-based
python train_lgbm_raw_features.py --feature-set raw_core --save-root artifacts/lgbm_raw_core_v1
python predict_lgbm_raw_features_submission.py --artifact-root artifacts/lgbm_raw_core_v1 --save-path artifacts/lgbm_raw_core_v1/submission.csv
```

Run the `raw_priors` ablation:

```bash
cd content-based
python train_lgbm_raw_features.py --feature-set raw_priors --save-root artifacts/lgbm_raw_priors_v1
python predict_lgbm_raw_features_submission.py --artifact-root artifacts/lgbm_raw_priors_v1 --save-path artifacts/lgbm_raw_priors_v1/submission.csv
```

## Validation Notes

Validation is temporal:

- `train_reviews.csv` is sorted by date
- the tail `validation-size` fraction is held out
- metrics are reported on rounded predictions because the competition submission is integer-valued

The validation summary also includes:

- cold-start breakdown
- MAE and RMSE by user history band
- feature count by family

## Validation Snapshot

Current run results:

- `raw_core`
  - `validation_mae_rounded = 0.6204`
  - `best_iteration = 1115`
  - `new_user_known_item_pct = 0.5874`
- `raw_priors`
  - `validation_mae_rounded = 1.2346`
  - `best_iteration = 226`
  - overfit signal is strongest on cold-start-heavy bands

Implementation note:

- review vote log features clamp negative sentinels before `log1p`, because `train_reviews.csv` contains rows with `useful = -1`

If you need one submission to trust first, use `raw_core`.

## Practical Guidance

- Start with `raw_core` to verify the pipeline is healthy end-to-end.
- Use `raw_core` for the submission you actually plan to send unless a future retune makes `raw_priors` competitive.
- If you change `--validation-size` or the LightGBM hyperparameters, retrain before generating the final submission so the saved `submission_spec.joblib` and `submission_model.txt` stay aligned.
