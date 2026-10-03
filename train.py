import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import preprocessing

df = preprocessing.get_preprocessed_data()
df = df.dropna(subset=["price"])
df = df[(df["price"] >= 10) & (df["price"] <= df["price"].quantile(0.995))]