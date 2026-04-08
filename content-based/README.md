# Yelp Recommendation System Challenge

## Overview

Welcome to the Yelp Recommendation System Challenge. The goal of this competition is to build a recommendation system capable of predicting the rating that a user will assign to a business.

The dataset is a reduced version (25%) of the original Yelp dataset and includes detailed information about users, businesses, and their interactions through reviews.

Participants are encouraged to explore content-based, collaborative filtering, and hybrid recommendation approaches.

---

## Objective

The main task is:

Predict the rating (stars) that a user will give to a business.

This is a regression problem. Typical evaluation metrics include:

- Mean Absolute Error (MAE)

---

## Dataset Description

The dataset is composed of three main entities:

- Users
- Businesses
- Reviews (user-business interactions)

This structure enables the use of multiple recommendation strategies, including content-based filtering, collaborative filtering, and hybrid models.

---

## Files

- `usuarios.csv`: Information about users
- `negocios.csv`: Information about businesses (items)
- `train_reviews.csv`: Training data with known ratings
- `test_reviews.csv`: Test data where ratings must be predicted

---

## Data Details

### Users (usuarios.csv)

This file contains metadata about users, including:

- `user_id`: Unique user identifier
- `name`: User name
- `review_count`: Number of reviews written
- `yelping_since`: Date when the user joined
- `friends`: List of user connections
- `useful`, `funny`, `cool`: Interaction metrics
- `fans`: Number of followers
- `average_stars`: Average rating given by the user
- Compliments: Various engagement indicators (e.g., `compliment_hot`, `compliment_funny`, etc.)

These features can be used to model user behavior and preferences.

---

### Businesses (negocios.csv)

This file contains metadata about businesses:

- `business_id`: Unique business identifier
- `name`, `address`, `city`, `state`, `postal_code`
- `latitude`, `longitude`
- `stars`: Average rating of the business
- `review_count`: Number of reviews
- `is_open`: Whether the business is open
- `attributes`: Business characteristics (e.g., parking, takeout)
- `categories`: List of categories
- `hours`: Opening hours

These features are essential for content-based recommendation systems.

---

### Reviews (train_reviews.csv, test_reviews.csv)

This file contains user-business interactions:

- `review_id`: Unique review identifier
- `user_id`: User identifier
- `business_id`: Business identifier
- `stars`: Rating (target variable in training set)
- `date`: Review date
- `useful`, `funny`, `cool`: Engagement metrics

This is the core dataset for collaborative filtering methods.

---

## Recommended Approaches

### Content-Based Filtering

- Encode business features such as categories and attributes
- Represent users based on previously interacted items
- Use similarity measures such as cosine similarity

---

### Collaborative Filtering

- Matrix factorization techniques (e.g., SVD, ALS)
- Neighborhood-based approaches (user-based or item-based)

---

### Hybrid Models

- Combine user features, business features, and interaction data
- Use machine learning models such as:
  - Gradient Boosting (XGBoost, LightGBM)
  - Neural Networks

---

## Baseline Example

```python
import pandas as pd

# Load data
reviews = pd.read_csv("train_reviews.csv")

# Compute average rating per business
business_mean = reviews.groupby("business_id")["stars"].mean()

global_mean = reviews["stars"].mean()

def predict(user_id, business_id):
    return business_mean.get(business_id, global_mean)

# Example
print(predict("user_1", "business_1"))
```

This baseline uses the average rating of each business. While simple, it does not capture personalization or contextual information.

---

## Advanced Ideas

Participants are encouraged to explore more advanced techniques, such as:

- Feature engineering on user and business metadata
- Temporal dynamics (e.g., time-aware recommendations)
- Natural Language Processing on reviews (if available)
- Embeddings for users and items
- Graph-based approaches using user-user or item-item relationships
- Deep learning architectures for recommendation systems

## Submission Format

The submission file must contain predictions for each row in test_reviews.csv.

Example:

```python
review_id,stars
abc123,4.2
def456,3.8
```

## Evaluation

Submissions are evaluated based on prediction error between true and predicted ratings. Specifically, MAE will be evaluate.