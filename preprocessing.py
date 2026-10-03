import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd

# Default city filenames mapping
DEFAULT_CITY_FILES = {
    "Toronto": "data/toronto_listings.csv",
    "Ottawa": "data/ottawa_listings.csv",
    "Vancouver": "data/vancouver_listings.csv",
    "Victoria": "data/victoria_listings.csv",
    "Montreal": "data/montreal_listings.csv",
    "Winnipeg": "data/winnipeg_listings.csv",
    "Quebec": "data/quebec_listings.csv",
    "New Brunswick": "data/new_brunswick_listings.csv",
}

HIGH_VALUE_AMENITIES = {
    "has_free_parking": ["free parking", "free driveway parking", "free residential parking"],
    "has_pool": ["pool"],
    "has_hot_tub": ["hot tub"],
    "has_air_con": ["air conditioning"],
    "has_gym": ["gym"],
    "has_balcony_patio": ["balcony", "patio"],
    "has_dedicated_workspace": ["dedicated workspace"],
    "has_view": ["view", "waterfront", "skyline view"],
    "has_ev_charger": ["ev charger"],
    "has_bbq_grill": ["bbq grill", "barbecue"],
}

REVIEW_SCORE_WEIGHTS = {
    "review_scores_rating": 0.35,
    "review_scores_cleanliness": 0.25,
    "review_scores_location": 0.25,
    "review_scores_value": 0.15,
}


def load_raw_listings(city_files: Optional[dict] = None) -> pd.DataFrame:
    """Load and concatenate listings from all specified city CSV files."""
    if city_files is None:
        city_files = DEFAULT_CITY_FILES

    dfs = []
    for city, filepath in city_files.items():
        if os.path.exists(filepath):
            dfs.append(pd.read_csv(filepath).assign(city=city))

    if not dfs:
        raise FileNotFoundError("No listing CSV files found.")

    return pd.concat(dfs, ignore_index=True)


def parse_amenities(val) -> List[str]:
    """Parse JSON string of amenities into a list of strings."""
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        return json.loads(val)
    except Exception:
        return []


def engineer_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Extract amenity count and binary flags for high-value amenities."""
    df = df.copy()
    if "amenities" not in df.columns:
        return df

    amenities_series = df["amenities"].apply(parse_amenities)
    df["num_amenities"] = amenities_series.apply(len)

    amenities_text = amenities_series.apply(lambda lst: " ".join(item.lower() for item in lst))

    for col_name, keywords in HIGH_VALUE_AMENITIES.items():
        df[col_name] = amenities_text.apply(lambda text: int(any(kw in text for kw in keywords)))

    df = df.drop(columns=["amenities"])
    return df


def engineer_license(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer binary flags for presence and exemption of listing license."""
    df = df.copy()
    if "license" not in df.columns:
        return df

    df["has_license"] = df["license"].notna().astype(int)
    df["is_license_exempt"] = (
        df["license"].fillna("").astype(str).str.strip().str.lower().str.contains("exempt").astype(int)
    )
    df = df.drop(columns=["license"])
    return df


def engineer_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Extract bathroom count and shared bathroom indicator from bathrooms_text."""
    df = df.copy()
    if "bathrooms_text" in df.columns:
        bath_text = df["bathrooms_text"].fillna("").astype(str).str.lower()
        df["is_shared_bath"] = bath_text.str.contains("shared").astype(int)

        bath_num = bath_text.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
        is_half_bath = bath_text.str.contains("half-bath")
        df["bathrooms_num"] = bath_num.fillna(is_half_bath.map({True: 0.5, False: np.nan}))

    df = df.drop(columns=["bathrooms_text", "bathrooms"], errors="ignore")
    return df


def engineer_reviews_and_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Extract review activity flags, recency, and longevity metrics."""
    df = df.copy()

    if "number_of_reviews" in df.columns:
        df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)

    last_review_date = (
        pd.to_datetime(df["last_review"], errors="coerce")
        if "last_review" in df.columns
        else pd.Series(index=df.index, dtype="datetime64[ns]")
    )
    reference_date = last_review_date.max()

    if "last_review" in df.columns:
        df["days_since_last_review"] = (reference_date - last_review_date).dt.days

    if "first_review" in df.columns:
        first_review_date = pd.to_datetime(df["first_review"], errors="coerce")
        df["days_since_first_review"] = (reference_date - first_review_date).dt.days

    df = df.drop(columns=["first_review", "last_review"], errors="ignore")
    return df


def engineer_quality_and_host(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer composite review score and host experience tiers."""
    df = df.copy()

    # 1. Composite Review Score
    # Check which review sub-score columns exist
    available_weights = {
        col: w for col, w in REVIEW_SCORE_WEIGHTS.items() if col in df.columns
    }
    if available_weights:
        total_weight = sum(available_weights.values())
        # Compute weighted sum as a pd.Series
        weighted_sum = pd.Series(0.0, index=df.index)
        for col, weight in available_weights.items():
            median_val = df[col].median()
            filled_col = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            weighted_sum = weighted_sum + (filled_col * (weight / total_weight))

        max_score = 5.0
        # Explicitly pd.Series.clip to avoid IDE type inference warning
        df["review_score_composite"] = (weighted_sum / max_score).clip(lower=0.0, upper=1.0)

    # 2. Host Experience Tier
    if "hosts_time_as_host_years" in df.columns:
        df["host_experience_tier"] = pd.cut(
            df["hosts_time_as_host_years"].fillna(0),
            bins=[-float("inf"), 1, 3, 5, float("inf")],
            labels=[0, 1, 2, 3],
            right=False,
        ).astype(int)

    return df


def clean_target_price(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the price column by stripping currency symbols and casting to float."""
    df = df.copy()
    if "price" in df.columns:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .replace("nan", np.nan)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering and preprocessing pipeline to listings."""
    df = clean_target_price(df)
    df = engineer_amenities(df)
    df = engineer_license(df)
    df = engineer_bathrooms(df)
    df = engineer_reviews_and_dates(df)
    df = engineer_quality_and_host(df)
    return df


# For backwards compatibility with scripts importing preprocessing.df
def get_preprocessed_data() -> pd.DataFrame:
    raw_df = load_raw_listings()
    return preprocess_pipeline(raw_df)


# Alias df for scripts that import preprocessing.df directly
def __getattr__(name):
    if name == "df":
        global _cached_df
        if "_cached_df" not in globals():
            _cached_df = get_preprocessed_data()
        return _cached_df
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    print("Loading raw listings and running preprocessing pipeline...")
    processed_df = get_preprocessed_data()
    print(f"Pipeline complete! Shape: {processed_df.shape}")
    print("\nSummary of engineered features:")
    check_cols = [
        "num_amenities", "has_free_parking", "has_license", "is_shared_bath",
        "bathrooms_num", "review_score_composite", "host_experience_tier"
    ]
    print(processed_df[[c for c in check_cols if c in processed_df.columns]].describe().round(3))