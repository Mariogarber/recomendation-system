import pandas as pd, numpy as np
from utils.io import load_users, load_train_reviews
from utils.split import temporal_train_validation_split

users = load_users()
train_reviews = load_train_reviews()
train_split, val_split = temporal_train_validation_split(train_reviews, val_size=0.2, timestamp_col="date")

train_user_ids = set(train_split["user_id"].unique())
val_user_ids = set(val_split["user_id"].unique())
cold_val_users = val_user_ids - train_user_ids

cold_users_df = users[users["user_id"].isin(cold_val_users)][["user_id","average_stars","review_count"]]
reviews_in_dataset = val_split[val_split["user_id"].isin(cold_val_users)].groupby("user_id").size().rename("reviews_in_dataset")
merged = cold_users_df.merge(reviews_in_dataset, on="user_id", how="left").fillna(0)
merged["reviews_outside"] = merged["review_count"] - merged["reviews_in_dataset"]

print(f"Cold val users: {len(cold_val_users):,}")
print(f"Avg Yelp review_count: {merged['review_count'].mean():.1f}")
print(f"Avg reviews in our val split: {merged['reviews_in_dataset'].mean():.2f}")
print(f"Avg reviews outside our dataset (leaked into avg_stars): {merged['reviews_outside'].mean():.1f}")
print(f"% users where avg_stars includes reviews NOT in our dataset: {(merged['reviews_outside'] > 0).mean()*100:.1f}%")

# Also: correlation between Yelp average_stars and actual val rating for cold users
cold_val_with_stars = val_split[val_split["user_id"].isin(cold_val_users)].merge(
    cold_users_df[["user_id","average_stars"]], on="user_id", how="left"
)
corr = cold_val_with_stars[["stars","average_stars"]].corr().iloc[0,1]
print(f"\nCorrelation(val_stars, yelp_avg_stars) for cold users: {corr:.4f}")

# Compare: MAE if predicting with Yelp avg_stars vs global mean
global_mean = float(train_split["stars"].mean())
mae_yelp = (cold_val_with_stars["stars"] - cold_val_with_stars["average_stars"]).abs().mean()
mae_global = (cold_val_with_stars["stars"] - global_mean).abs().mean()
print(f"MAE using Yelp avg_stars: {mae_yelp:.4f}")
print(f"MAE using global mean: {mae_global:.4f}")
print(f"Yelp avg_stars advantage: {mae_global - mae_yelp:.4f}")
