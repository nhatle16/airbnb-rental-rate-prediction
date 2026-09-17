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
