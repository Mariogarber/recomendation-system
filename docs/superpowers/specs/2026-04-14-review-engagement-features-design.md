# Design: Review Engagement Features — LightGBM Experiment

- Date: 2026-04-14
- Status: approved
- Author: brainstorming session

## Context

All current submissions score ~0.653 on the leaderboard regardless of model architecture.
Local best (MoE v3) shows 0.5999 — a gap of ~0.053 that is immune to architectural changes.

The test_reviews.csv file contains `useful`, `funny`, `cool`, and `date` columns for every
review being predicted. These review-level engagement features are completely unused in all
current models. They are non-zero in the test set and represent valid, non-leaky prediction
signals (votes were cast before competition data collection).

Known Yelp signal properties:
- `funny` anti-correlates with high ratings (negative/snarky reviews get funny votes)
- `useful` skews toward 3–4 star reviews
- `cool` correlates with 4–5 star reviews

## Goal

Determine whether adding review engagement features to the LightGBM router improves
leaderboard MAE from ~0.653 toward 0.630. One experiment, one submission.

## Scope

Single experiment: add review features to the LightGBM router only.
No changes to the deep model (known_user_deep_e2e) in this iteration.

## Architecture

The review being predicted contributes 6 new tabular features:

| Feature | Derivation |
|---|---|
| `review_useful_log1p` | log1p(useful) |
| `review_funny_log1p` | log1p(funny) |
| `review_cool_log1p` | log1p(cool) |
| `review_dow` | day of week from date (0=Mon … 6=Sun) |
| `review_month` | month from date (1–12) |
| `review_hour` | hour of day from datetime (0–23) |

These features are appended to the existing tabular feature set in `utils/gbm_features.py`.
All 6 are derived purely from the review row itself (no external references) and are
available in both train_reviews and test_reviews with no train/test asymmetry.

Note: `review_days_since_train_end` was considered and rejected — it would be 0 for
all training rows (all predate the cutoff) and positive only in test, creating asymmetry.

## Component Changes

### `utils/gbm_features.py`
- Add `build_review_context_features(reviews_df, train_end_date)` function
- Computes the 6 features above from a reviews DataFrame
- Modify existing `build_gbm_features()` (or equivalent entry point) to call this and
  merge result by row index / review_id

### `train_lgbm_raw_router.py`
- Pass `train_end_date` (derived from max date in train_reviews) into the dataset builder
- No other structural change

## Artifact Naming

- LightGBM artifact: `lgbm_review_features_v1`
- Submission file: `lgbm_review_features_v1/submission.csv`

## Evaluation

- Primary: leaderboard MAE vs current 0.653 baseline
- Local validation: compare against `lgbm_hybrid_conservative_v1` (0.6265) and
  `known_user_deep_router_v2_eval_v3` (0.5999) on local validation
- Feature importance from LightGBM to confirm the new features are used

## Docs Update

After the experiment:
- Add artifact to `docs/experiments/registry.md`
- Update `docs/status/current-state.md` with result and conclusion
- Update `docs/proposals/content-based-next-ideas.md` (remove or promote the review
  features idea based on leaderboard result)

## Next Steps (conditional)

- If leaderboard improves: inject review features into the deep model
  (`known_user_deep_e2e`) as a `review_context_encoder` block — separate experiment
- If leaderboard does not improve: review features are not discriminative enough at
  this stage; focus shifts to temporal validation realignment (Option B)
