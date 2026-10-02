import json

import pandas as pd

# Read multiple cities listings
df_tor = pd.read_csv("data/toronto_listings.csv").assign(city="Toronto")
df_ott = pd.read_csv("data/ottawa_listings.csv").assign(city="Ottawa")
df_van = pd.read_csv("data/vancouver_listings.csv").assign(city="Vancouver")
df_vic = pd.read_csv("data/victoria_listings.csv").assign(city="Victoria")
df_mon = pd.read_csv("data/montreal_listings.csv").assign(city="Montreal")
df_win = pd.read_csv("data/winnipeg_listings.csv").assign(city="Winnipeg")
df_que = pd.read_csv("data/quebec_listings.csv").assign(city="Quebec")
df_new = pd.read_csv("data/new_brunswick_listings.csv").assign(city="New Brunswick")

# Combine into a single dataframe
df = pd.concat([df_tor, df_ott, df_van, df_vic, df_mon, df_win, df_que, df_new], ignore_index=True)

# Strip '$' sign and convert value to float
df["price"] = df["price"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)

# FEATURE ENGINEERING
# ----------------------- AMENITIES ------------------------------
def parse_amenities(val):
    if pd.isna(val) or not isinstance(val, str):
        return []
    try:
        return json.loads(val)
    except Exception:
        return []

# Apply transformation on 'amenities'
amenities_series = df["amenities"].apply(parse_amenities)

# New feat - number of amenities
df["num_amenities"] = amenities_series.apply(len)

# Lowercased text representation for fast keyword matching
amenities_text = amenities_series.apply(lambda lst: " ".join(item.lower() for item in lst))
print(amenities_text.head(5))

# Create a mapping dictionary for new key/premium amenities and their keywords
high_value_amenities = {
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

# Create the new features from premium amenities
for col_name, keywords in high_value_amenities.items():
    df[col_name] = amenities_text.apply(lambda text: int(any(kw in text for kw in keywords)))
    
df = df.drop(columns=["amenities"])

# ----------------------- LICENSE ------------------------------
# Has any license or registration
df["has_license"] = df["license"].notna().astype(int)

# License is exempted
df["is_license_exempt"] = (df["license"].fillna("").str.strip().str.lower().str.contains("exempt").astype(int))

df = df.drop(columns=["license"])

# ----------------------- BATHROOMS ------------------------------
# Check if the bathroom is shared, 1 means shared while 0 means private
df["is_shared_bath"] = (
    df["bathrooms_text"]
    .fillna("")
    .str.lower()
    .str.contains("shared")
    .astype(int)
)

# Extract the numeric value of bathroom count
bath_num = (
    df["bathrooms_text"]
    .fillna("")
    .str.lower()
    .str.extract(r"(\d+(?:\.\d+)?)")[0]
    .astype(float)
)
is_half_bath = df["bathrooms_text"].fillna("").str.lower().str.contains("half-bath")
df["bathrooms_num"] = bath_num.fillna(is_half_bath.map({True: 0.5, False: None}))

# Drop text column and redundant null-heavy bathrooms column
df = df.drop(columns=["bathrooms_text", "bathrooms"], errors="ignore")

# ----------------------- REVIEWS & DATES ---------------------------
df["has_reviews"] = (df["number_of_reviews"] > 0).astype(int)

last_review_date = pd.to_datetime(df["last_review"], errors="coerce")
reference_date = last_review_date.max()
df["days_since_last_review"] = (reference_date - last_review_date).dt.days

first_review_date = pd.to_datetime(df["first_review"], errors="coerce")
df["days_since_first_review"] = (reference_date - first_review_date).dt.days

df = df.drop(columns=["first_review", "last_review"], errors="ignore")

# ----------------------- REVIEW SCORE COMPOSITE -----------------------
# Weighted average of the four main review sub-scores, normalized to [0, 1].
# Weights reflect how much each dimension typically influences booking decisions:
#   location & cleanliness tend to drive price more than check-in logistics.
_review_weights = {
    "review_scores_rating":       0.35,
    "review_scores_cleanliness":  0.25,
    "review_scores_location":     0.25,
    "review_scores_value":        0.15,
}
_max_score = 5.0  # Airbnb sub-scores are on a 1–5 scale

_weighted_sum = sum(
    df[col].fillna(df[col].median()) * w
    for col, w in _review_weights.items()
)
df["review_score_composite"] = (_weighted_sum / _max_score).clip(0, 1)

# ----------------------- HOST EXPERIENCE TIER -------------------------
# Ordinal bucket derived from hosts_time_as_host_years.
#   0 → new      (< 1 year)
#   1 → growing  (1–3 years)
#   2 → experienced (3–5 years)
#   3 → veteran  (5+ years)
# More-experienced hosts tend to price listings higher and get better reviews.
df["host_experience_tier"] = pd.cut(
    df["hosts_time_as_host_years"].fillna(0),
    bins=[-float("inf"), 1, 3, 5, float("inf")],
    labels=[0, 1, 2, 3],
    right=False,
).astype(int)