"""
src/preprocess.py
-------------------
Handles data cleaning and feature encoding for the customer churn dataset.

This logic is kept in a separate module so that both the training script
and the Streamlit app use the exact same cleaning and encoding steps,
ensuring predictions are generated using the same transformations that
were applied during training.
"""

import pandas as pd
import numpy as np


# Columns that are simple Yes/No flags -> mapped straight to 1/0
BINARY_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

# Columns with more than two categories -> handled with one-hot encoding
CATEGORICAL_COLUMNS = [
    "gender",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Continuous numeric columns -> get scaled before modeling
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data(path: str) -> pd.DataFrame:
    """Reads the raw CSV into a DataFrame."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fixes missing values and removes columns with no predictive value.

    Specifically:
    1. Fills the missing TotalCharges values with the median, a simple
       and defensible choice for a right-skewed numeric column.
    2. Drops the customerID column since it carries no predictive signal.
    3. Ensures SeniorCitizen is treated as a category (0/1), not a count.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        median_value = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_value)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def encode_features(df: pd.DataFrame, feature_columns: list = None):
    """Converts cleaned, human-readable categories into numeric features a
    model can use.

    Parameters
    ----------
    df : DataFrame that has already been through clean_data().
    feature_columns : if provided (at prediction time), the resulting
        DataFrame is re-indexed to exactly match this column list -- this is
        what makes single-row predictions safe even if that one row doesn't
        happen to contain every category value seen during training.

    Returns
    -------
    X : encoded feature DataFrame
    y : target Series (None if "Churn" isn't in df, e.g. at prediction time)
    columns : the final list of feature column names (only returned when
        feature_columns was not supplied, i.e. during training)
    """
    df = df.copy()

    y = None
    if "Churn" in df.columns:
        y = df["Churn"].map({"Yes": 1, "No": 0})
        df = df.drop(columns=["Churn"])

    # Simple Yes/No -> 1/0 mapping
    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encode the remaining categorical columns
    df = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLUMNS if c in df.columns])

    if feature_columns is not None:
        # Align prediction-time data to the exact training columns.
        # Any category the model has never seen becomes all-zero (i.e. "not
        # this category"), and any missing expected column is added as 0.
        df = df.reindex(columns=feature_columns, fill_value=0)
        return df, y

    return df, y, list(df.columns)


def split_features_target(df: pd.DataFrame):
    """Convenience wrapper: clean + encode in one call, returning X, y, and
    the final list of feature columns (used during training)."""
    cleaned = clean_data(df)
    X, y, columns = encode_features(cleaned)
    return X, y, columns