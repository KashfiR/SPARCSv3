"""Loading and preprocessing for the curated CuMiDa microarray datasets."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_cumida(path):
    df = pd.read_csv(path)
    return df.drop(columns=['samples']) if 'samples' in df.columns else df


def preprocess_data(df, verbose=True):
    """Drop duplicate probes and zero-variance columns, impute, encode labels."""
    duplicated = df.columns.duplicated()
    if duplicated.any():
        if verbose:
            print(f"dropping {int(duplicated.sum())} duplicate probe columns")
        df = df.loc[:, ~duplicated]

    X = df.drop(columns=['type'])
    y = df['type']

    if verbose:
        print(f"shape {X.shape}, {len(y)} samples")
        print(f"class distribution:\n{y.value_counts()}")

    missing = X.isnull().sum().sum()
    if missing:
        if verbose:
            print(f"imputing {int(missing)} missing values with column medians")
        X = X.fillna(X.median())

    constant = X.columns[X.std() == 0]
    if len(constant):
        if verbose:
            print(f"removing {len(constant)} zero-variance probes, {X.shape[1]-len(constant)} remain")
        X = X.drop(columns=constant)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return X, y, y_encoded, encoder, list(X.columns)


def split_and_normalize(X, y_encoded, test_size=0.3, random_state=42, verbose=True):
    """Stratified split followed by standardization fitted on the training half."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                  columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test),
                                 columns=X_train.columns, index=X_test.index)

    if verbose:
        print(f"train {X_train_scaled.shape[0]} samples, test {X_test_scaled.shape[0]} samples")

    return X_train_scaled, X_test_scaled, np.asarray(y_train), np.asarray(y_test), scaler
