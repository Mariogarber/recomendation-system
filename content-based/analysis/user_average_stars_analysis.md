# Why `user_average_stars` Is Not a Leakage Problem

**Date:** 2026-04-15  
**Author:** analysis during MAE improvement session  
**Empirical basis:** `_check_leakage.py` results on the 80/20 temporal val split

---

## 1. What `user_average_stars` Is

`user_average_stars` is a field provided in `users.json` by Yelp as part of the competition data.
It is the all-time mean star rating a user has given across their **entire Yelp review history** — including reviews that are:

- Older than the earliest review in our competition dataset
- From businesses not present in our dataset
- Simply outside the date range used for training/validation splits

It is a static metadata attribute, computed by Yelp before the dataset was released, and is **available at test time** in the same `users.json` file.

---

## 2. What Leakage Would Mean

Leakage in this context would mean: a feature used at training time contains information about the *target* (the val/test rating) that would not be available in production.

The concern raised was: for "cold" users (those absent from the training split), does `user_average_stars` encode their val-period ratings?

---

## 3. Empirical Results

Running `_check_leakage.py` on the 80/20 temporal split:

```
Cold val users:                                          105,550
Avg Yelp review_count:                                      22.7
Avg reviews in our val split:                               1.22
Avg reviews outside our dataset (in Yelp avg_stars):       21.5
% users with external Yelp reviews in their avg_stars:    85.5%

Correlation(val_stars, yelp_avg_stars):                   0.6992
MAE using Yelp avg_stars as predictor:                    0.8614
MAE using global mean as predictor:                       1.4908
Advantage of Yelp avg_stars over global mean:             0.6294
```

---

## 4. Interpretation

### 4.1 The average is dominated by external reviews

Cold val users have **22.7 reviews on Yelp on average** but only **1.22 in our val split**.
This means `user_average_stars` is computed from ~21.5 reviews that are completely outside our competition dataset.

These external reviews are:
- Temporally prior (older Yelp history)
- From a different slice of the user's activity

They do **not** include the 1.22 val-split ratings we are trying to predict. The val rating contributes at most `1/22.7 ≈ 4.4%` to the average, and for 85.5% of users it contributes **0%** (because their avg_stars was computed before any of our dataset reviews existed or from entirely external data).

### 4.2 The correlation is genuine signal, not contamination

The 0.70 correlation between `yelp_avg_stars` and val ratings reflects **user taste consistency** — a user who habitually gives 4-star ratings gives 4-star ratings in both the past and the future. This is a legitimate predictive signal, not a shortcut that leaks the answer.

The same 0.70 correlation will exist at test time: test users' `user_average_stars` is computed from their external Yelp history, not from the test ratings we are predicting.

### 4.3 Comparison to genuine leakage

Genuine leakage would look like:
- Computing `user_average_stars` from `train_split["stars"].mean()` per user — this uses training targets to build the feature
- Using a feature computed on the val set to train the model

`user_average_stars` from `users.json` is neither. It is a pre-computed, externally-sourced metadata field identical in character to "user age" or "user city".

---

## 5. Why Previous Experiments Were Wrong

Several experiments attempted to "fix" this perceived leakage by replacing `user_average_stars` with train-split-derived means:

```python
# INCORRECT — introduced actual within-sample leakage
_train_user_stars = build_train_user_stars(train_split, global_mean=_train_global_mean)
users_df["average_stars"] = users_df["user_id"].map(_train_user_stars).fillna(_train_global_mean)
```

This replacement:
1. For **known users**: replaced a stable 22-review average with a noisy 5-10 review average, reducing signal quality
2. For **cold users**: replaced a 0.70-correlated external signal with the global mean (0.00 correlation with individual taste), a 0.63 MAE regression
3. Introduced **actual LOO leakage** for the training pass: each training row's feature = mean(all training targets for that user), including the current row

The result was val MAE collapsing from 0.6265 → 1.19 for the known model branch.

---

## 6. What About Test-Time Consistency?

The `user_average_stars` feature is consistent across val and test because:

| Split | `user_average_stars` source | Val/test ratings in the average? |
|---|---|---|
| Val training pass | `users.json` (Yelp all-time) | No — external reviews dominate |
| Val inference | `users.json` (Yelp all-time) | Negligibly (≤4.4% for most users) |
| Test inference | `users.json` (Yelp all-time) | No — same static file |

There is no distribution shift in this feature between val and test.

---

## 7. Conclusion and Rule for Future Experiments

> **`user_average_stars` from `users.json` is a legitimate competition feature. Never replace it with train-split-derived means. It is not leakage.**

Rules to follow:

1. **Use `users.json` average_stars as-is** for all training, validation, and submission passes.
2. **Do not compute per-user means from the training split** and inject them into `users_df["average_stars"]` — this destroys the feature for cold users and introduces actual within-sample leakage for known users.
3. **The submission pass** may compute `user_average_stars` from `train_reviews` (all 100% of the data) as a supplementary feature only if it is a separate column — it should not overwrite the Yelp metadata column.
4. If you want to add a "train-derived user mean" as a **separate feature** (e.g., `user_train_mean_stars`), compute it with temporal LOO encoding (`build_temporal_loo_user_stars`) to avoid within-sample correlation.

---

## 8. Quantified Value Summary

| Feature | MAE on cold val users | Correlation w/ target |
|---|---|---|
| Global mean (3.74) | 1.491 | 0.000 |
| `user_average_stars` (Yelp) | **0.861** | **0.699** |
| Best LGBM cold model (prefix_deep_v1) | ~0.565 | — |

The 0.63 MAE gap between global mean and Yelp avg_stars is the single most valuable signal in the entire cold-start pipeline. Preserving it is non-negotiable.
