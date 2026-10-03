import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

import preprocessing

# Load and preprocess the data, filter price outliers
df = preprocessing.get_preprocessed_data()
df = df.dropna(subset=["price"])
df = df[(df["price"] >= 10) & (df["price"] <= df["price"].quantile(0.995))]

# Exclude non-features, free text, URLs, and target-leakage columns
cols_to_drop = [
    "price", "id", "listing_url", "scrape_id", "last_scraped", "source",
    "picture_url", "host_id", "host_url", "host_profile_id",
    "host_profile_url", "host_thumbnail_url", "host_picture_url",
    "name", "description", "neighborhood_overview",
    "host_about", "amenities", "license", "first_review", "last_review",
    "bathrooms_text", "bathrooms", "host_name", "host_location",
    "price_quote_checkin_date", "price_quote_checkout_date", "price_quote_raw",
    "price_quote_total_price", "price_quote_price_per_night",
    "estimated_revenue_l365d", "calendar_last_scraped", "neighbourhood",
]

X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
y = df["price"].values

# Identify categorical and numerical features
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# Building imputers for missing values
numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the data
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, max_depth=10,random_state=42, n_jobs=-1)),
])

print(f"Training baseline model on {len(X_train)} samples with {X.shape[1]} features...")
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)
print("\n--- Baseline Results ---")
print(f"Test MAE  : ${mae:.2f}")
print(f"Test RMSE : ${rmse:.2f}")