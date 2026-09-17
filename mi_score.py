
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif

def make_mi_score(X, y):
    """Calculate the mutual information score for each feature in X with respect to the target variable y."""
    X = X.copy()
    
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