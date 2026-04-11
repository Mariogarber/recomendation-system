# Content-Based Recommendation Plan

> Documento legacy. Para estado vigente y siguientes ideas usar:
> - [docs/status/current-state.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/status/current-state.md)
> - [docs/proposals/content-based-next-ideas.md](/C:/Users/mario/OneDrive/Documentos/UPM/Master_Data/Sistemas_recomendacion/recomendation-system/docs/proposals/content-based-next-ideas.md)

This document defines a concrete plan for the content-based branch of the competition.
The objective is to predict the rating that a user will give to a business while being
careful with leakage, cold start, and evaluation quality.

Status note:

- the manual business and user builders are already implemented
- the deep competition embeddings pipeline is already implemented
- the remaining work is centered on leak-safe baselines, final scoring, and cold-start policy

## 1. Main principles

- The first version must be a true content-based model.
- User preferences must be built from the content of previously rated businesses.
- Metadata that may contain information beyond `train_reviews.csv` must not be used
  directly without a leakage audit.
- The plan must include explicit handling for cold-start users, because they are a
  large fraction of the test set.
- Every new feature family must be validated through ablation, not only intuition.

## 2. Data view

We can describe the dataset as a graph:
- **User nodes**
- **Business nodes**
- **User -> Business** review edges
- **User -> User** friend edges

For the content-based model, the core information flow will be:
- Build a robust business content vector.
- Build each user profile from the businesses already rated by that user in train.
- Score a candidate business against the user profile.

Friend relations will be treated as an optional secondary feature block, not as part of
the first baseline.

## 3. Leakage audit

Before training any model, classify each feature into one of these groups:
- **Safe direct feature**: can be used as it appears in the metadata file.
- **Safe only if recomputed from train**: may use future information if taken directly.
- **Unsafe for this competition setting**: should not be used in the first version.

### 3.1 Features to audit first

User-level candidates:
- `review_count`
- `average_stars`
- `yelping_since`
- `useful`, `funny`, `cool`
- `fans`
- `elite`
- `compliment_*`
- `friends`

Business-level candidates:
- `stars`
- `review_count`
- `is_open`
- `city`
- `state`
- `postal_code`
- `latitude`, `longitude`
- `categories`
- `attributes`
- `hours`

Review-level candidates:
- `date`
- `useful`, `funny`, `cool`

### 3.2 Leakage tasks

- Compare user metadata aggregates against aggregates recomputed only from `train_reviews.csv`.
- Compare business metadata aggregates against aggregates recomputed only from `train_reviews.csv`.
- Decide which aggregate features must be removed or recomputed from train only.
- Document the final whitelist of allowed features before any model comparison.

## 4. Exploratory analysis tasks

### 4.1 Dataset structure

- Measure train/test sizes, unique users, unique businesses, and interaction density.
- Quantify cold start:
  - new user + known business
  - known user + new business
  - new user + new business
- Compute review-count distributions for users and businesses.
- Check target distribution of stars and business/user skew.

### 4.2 Missingness and cardinality

- Measure missing rates for `attributes`, `categories`, and `hours`.
- Measure cardinality for `city`, `state`, postal zones, and category vocabulary.
- Count how many distinct business attributes exist after parsing.
- Detect rare categories and rare attributes to decide pruning thresholds.

### 4.3 Temporal analysis

- Inspect train and test date ranges.
- Compare random split validation against time-based validation.
- Evaluate whether recency weighting helps when building user profiles.

### 4.4 User-side analysis

- Study how user history length affects predictability.
- Segment users by number of train reviews: `1`, `2-5`, `6-20`, `>20`.
- Measure whether `fans`, `compliments`, and engagement counts correlate with rating bias.
- Leave friend features for a later ablation after the core content model is stable.

### 4.5 Business-side analysis

- Parse `categories` as multi-label text.
- Parse `attributes` as structured key/value features.
- Convert `hours` into usable schedule features:
  - open days count
  - total weekly opening hours
  - weekend availability
  - late-night availability if possible
- Check whether geographic information should be raw, normalized, clustered, or both.

## 5. Business representation

The first strong business vector should combine the following blocks:

### 5.1 Structured business features

- `state` as one-hot or embedding
- `city` as one-hot, hashing, or embedding depending on cardinality
- normalized `latitude` and `longitude`
- `is_open`
- leakage-safe business popularity features only if approved by the audit

### 5.2 Category features

- Parse category strings into tokens.
- Build a multi-hot or TF-IDF representation.
- Remove overly rare categories if they create too much sparsity.

### 5.3 Attribute features

- Parse the `attributes` dictionary.
- Flatten boolean and low-cardinality values into indicator variables.
- Group rare values into `other` when necessary.
- Keep a parser report with coverage and parsing errors.

### 5.4 Hours features

- Parse the `hours` dictionary.
- Build compact numerical features instead of keeping raw strings.

## 6. User representation

The user profile must be created from previously rated businesses, not only from user metadata.

### 6.1 Core user profile

For each user:
- collect all businesses rated in train
- map them to business content vectors
- aggregate them into a user profile vector

Aggregation variants to test:
- simple mean of rated business vectors
- rating-weighted mean
- centered weighting using `(rating - user_mean_train)`
- recency-weighted aggregation

### 6.2 User metadata block

User metadata can be added as a secondary block only after the leakage audit.
Candidate metadata:
- `yelping_since`
- engagement counts
- `fans`
- `elite`
- `compliment_*`

These should be treated as calibration features, not as the main source of preference.

### 6.3 Friend block

Friend information is optional and should be postponed to a later iteration.
If tested, the first approach should be:
- compute friend aggregate statistics only for friends seen in train
- add friend-summary features, not a full graph model
- compare against the no-friends baseline through ablation

## 7. Prediction strategy

The model design will be staged.

### 7.1 Baselines

Implement and evaluate:
- global mean
- business mean from train
- user mean from train
- bias baseline: `global + user_bias + business_bias`
- simple content score using user-profile vs business-vector cosine similarity

### 7.2 First content-based regressor

Build a supervised model using:
- user profile vector
- candidate business vector
- similarity features between user and business
- safe user metadata
- safe business metadata
- optional review-context features available in test: `date`, `useful`, `funny`, `cool`

Candidate regressors:
- Ridge regression
- Gradient boosting
- XGBoost or LightGBM if infrastructure is available

### 7.3 Cold-start policy

Because most cold-start cases are new users with known businesses, define explicit rules:

- **Known user, known business**:
  use the full content-based regressor
- **New user, known business**:
  use business features + safe review-context features + global/business priors
- **Known user, new business**:
  use user profile + business content if available from metadata
- **New user, new business**:
  use the safest fallback baseline

## 8. Validation protocol

The competition metric is MAE, so the analysis must optimize for MAE first.

### 8.1 Core evaluation

- Use MAE as the main metric
- Track RMSE as a secondary metric
- Clip predictions to the valid rating range

### 8.2 Validation splits

- random validation split for fast iteration
- time-aware validation split for robustness

### 8.3 Mandatory segmented evaluation

Report MAE by:
- cold-start users vs seen users
- users with short vs long history
- businesses with low vs high support
- time slices if temporal drift is relevant

### 8.4 Ablation table

Run an ablation table for:
- categories only
- categories + attributes
- categories + attributes + hours
- adding user metadata
- adding review-context features
- adding friend features

## 9. Ordered implementation phases

### Phase 1: Data audit and parsers

- Build leakage report
- Build `categories` parser
- Build `attributes` parser
- Build `hours` parser
- Produce a clean feature-ready business table

### Phase 2: Baselines and validation

- Implement leakage-safe baselines
- Create random and temporal validation splits
- Establish benchmark MAE values

### Phase 3: Business and user profiles

- Build business vectors
- Build user profile aggregation pipeline
- Evaluate simple similarity-based recommenders

### Phase 4: Supervised content model

- Train regressors on profile, business, and similarity features
- Add safe metadata blocks
- Tune the best model on validation

### Phase 5: Cold-start specialization

- Implement explicit cold-start branches
- Compare a unified model against segmented policies

### Phase 6: Optional extensions

- Friend-summary features
- richer temporal weighting
- lightweight hybridization with collaborative predictions if needed

## 10. Deliverables

Each phase should end with:
- a short experiment note
- MAE results on validation
- segmented error analysis
- decision on what stays, what is removed, and why

## 11. Definition of success

The content-based branch will be considered successful if it:
- beats the naive global and business-mean baselines
- remains leakage-safe
- behaves reasonably on cold-start users
- has a clear ablation story explaining which feature families really help
