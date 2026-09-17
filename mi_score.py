
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif

def make_mi_score(X, y):
    """Calculate the mutual information score for each feature in X with respect to the target variable y."""
    X = X.copy()
    
    # Drop columns that are completely null
    all_null_cols = X.columns[X.isna().all()].tolist()
    if all_null_cols:
        X = X.drop(columns=all_null_cols)
    
    # Identify the columns by type - numerical or categorical
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    
    # Preprocessing numerical data
    if numerical_cols:
        numerical_imputer = SimpleImputer(strategy='mean')
        X[numerical_cols] = numerical_imputer.fit_transform(X[numerical_cols])
        
    # Preprocessing categorical data
    if categorical_cols:
        categorical_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
        X[categorical_cols] = categorical_imputer.fit_transform(X[categorical_cols])
        
        # Encode categorical features
        encoder = OrdinalEncoder()
        X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
        
    # Check for discrete features (encoded categorical features or integer features)
    discrete_features = [col in categorical_cols or X[col].dtype == int for col in X.columns]
    
    # Calculate mutual information scores
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    
    mi_scores = pd.Series(mi_scores, name="MI Score", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores

def plot_mi_scores(mi_scores):
    """Plot the mutual information scores."""
    mi_scores = mi_scores.sort_values(ascending=False)
    positions = np.arange(len(mi_scores))
    ticks = list(mi_scores.index)
    plt.barh(positions, mi_scores)
    plt.yticks(positions, ticks)
    plt.title("Mutual Information Scores")
    
def clear_airbnb_data(df):
    """Clean Airbnb data by converting percentage and currency strings to float."""
    df = df.copy()
    
    for col in df.select_dtypes(include='object').columns:
        # Remove all None and NaN values from the column
        valid_series = df[col].dropna().astype(str).str.strip()
        
        # Filter out empty strings
        valid_series = valid_series[valid_series != '']
        
        if valid_series.empty:
            continue
        
        # Strip percentage signs at the endand convert to float
        if valid_series.str.match(r'^\d+(\.\d+)?%$').mean() > 0.8:
            df[col] = df[col].astype(str).str.rstrip('%').replace('nan', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100.0
        
        # Strip dollar signs at the beginning and convert to float
        if valid_series.str.match(r'^\$[\d,]+(\.\d+)?$').mean() > 0.8:
            df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).replace('nan', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/toronto_listings.csv")
    
    df = df.dropna(subset=["price"])  # Drop rows where price is NaN
    
    X = df.drop(columns=["price"])
    y = df["price"]
    
    mi_scores = make_mi_score(X, y)
    print(f"Num. of feature with 0 score: {(mi_scores <= 0).sum()} / {len(mi_scores)}")
    
    plt.figure(dpi=100, figsize=(10, 14))
    plot_mi_scores(mi_scores)
    plt.show()
