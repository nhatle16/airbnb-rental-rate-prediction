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

