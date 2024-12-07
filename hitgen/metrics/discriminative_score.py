import pandas as pd
import numpy as np
from tsfeatures import tsfeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def split_train_test(data, train_ratio=0.8, max_train_series=50, max_test_series=10):
    """Splits data indices into train and test indices with limits on the number of time series."""

    unique_ids = data["unique_id"].unique()
    n_series = len(unique_ids)

    # limit the number of time series for training and testing
    n_series_train = min(int(train_ratio * n_series), max_train_series)
    n_series_test = min(n_series - n_series_train, max_test_series)

    train_indices = np.random.choice(unique_ids, size=n_series_train, replace=False)
    test_indices = np.setdiff1d(unique_ids, train_indices)[:n_series_test]

    return train_indices, test_indices


def filter_data_by_indices(data, indices, label_value):
    """Filters data by indices and assigns labels."""
    filtered_data = data[data["unique_id"].isin(indices)]
    labels = pd.DataFrame([label_value] * len(indices))
    return filtered_data, labels


def generate_features(data, freq):
    """Generates time series features using tsfeatures."""
    return tsfeatures(data, freq=freq)


def compute_discriminative_score(original_data, synthetic_data, freq):
    train_idx, test_idx = split_train_test(original_data)

    FREQS = {"H": 24, "D": 1, "M": 12, "Q": 4, "W": 1, "Y": 1}
    freq = FREQS[freq]

    # original data
    original_data_train, original_data_train_y = filter_data_by_indices(
        original_data, train_idx, label_value=0
    )
    original_data_test, original_data_test_y = filter_data_by_indices(
        original_data, test_idx, label_value=0
    )

    original_features_train = generate_features(original_data_train, freq=freq)
    original_features_test = generate_features(original_data_test, freq=freq)

    # synthetic data
    synthetic_data_train, synthetic_data_train_y = filter_data_by_indices(
        synthetic_data, train_idx, label_value=1
    )
    synthetic_data_test, synthetic_data_test_y = filter_data_by_indices(
        synthetic_data, test_idx, label_value=1
    )

    synthetic_features_train = generate_features(synthetic_data_train, freq=freq)
    synthetic_features_test = generate_features(synthetic_data_test, freq=freq)

    # Classifier
    X_train = pd.concat(
        (original_features_train, synthetic_features_train), ignore_index=True
    ).drop(columns=["unique_id"], errors="ignore")
    y_train = pd.concat(
        (original_data_train_y, synthetic_data_train_y), ignore_index=True
    )

    X_test = pd.concat(
        (original_features_test, synthetic_features_test), ignore_index=True
    ).drop(columns=["unique_id"], errors="ignore")
    y_test = pd.concat((original_data_test_y, synthetic_data_test_y), ignore_index=True)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return f1_score(y_test, y_pred)
