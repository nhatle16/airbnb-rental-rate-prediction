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