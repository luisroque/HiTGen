from typing import Tuple, List
import pandas as pd
import numpy as np
import json
import os
import pickle
import hashlib
import warnings
from tsfeatures import tsfeatures
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from datetime import datetime

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS


def split_train_test(
    unique_ids,
    key,
    split_dir,
    train_ratio=0.8,
    max_train_series=100,
    max_test_series=20,
):
    """
    Splits data indices into train and test indices, stores them in a JSON file,
    and retrieves them by a numeric key.
    """
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, "splits.json")

    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            splits = json.load(f)
    else:
        splits = {}

    if str(key) in splits:
        print(f"         Key {key} already exists. Skipping split creation.")
        return splits[str(key)]["train_indices"], splits[str(key)]["test_indices"]

    n_series = len(unique_ids)

    n_series_train = min(int(train_ratio * n_series), max_train_series)
    n_series_test = min(n_series - n_series_train, max_test_series)

    train_indices = np.random.choice(unique_ids, size=n_series_train, replace=False)
    test_indices = np.setdiff1d(unique_ids, train_indices)[:n_series_test]

    splits[str(key)] = {
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }

    with open(split_file, "w") as f:
        json.dump(splits, f, indent=4)

    return splits[str(key)]["train_indices"], splits[str(key)]["test_indices"]


def filter_data_by_indices(
    data: pd.DataFrame,
    indices: List[str],
    label_value: int,
    downstream_forecast: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filters data by indices and assigns labels."""
    filtered_data = data[data["unique_id"].isin(indices)].copy()
    unique_ids_n = filtered_data["unique_id"].nunique()
    labels = pd.DataFrame([label_value] * unique_ids_n)
    if downstream_forecast:
        filtered_data["unique_id"] = filtered_data["unique_id"] + "_synth"

    return filtered_data, labels


def safe_generate_features(
    data,
    freq,
    dataset_name,
    dataset_group,
    data_cat,
    split,
    method,
    train_idx,
    test_idx,
    store_features=True,
):
    """
    Safely generates time series features using tsfeatures
    """
    features_dir = "assets/features/"
    os.makedirs(features_dir, exist_ok=True)

    if not isinstance(data_cat, str) or data_cat not in {"real", "synthetic"}:
        raise ValueError(
            f"Invalid data_cat value: {data_cat}. Expected 'real' or 'synthetic'."
        )

    if not isinstance(split, str) or split not in {
        "train",
        "test",
        "test_train",
        "test_test",
        "hypertuning_train",
        "hypertuning_test",
    }:
        raise ValueError(f"Invalid split value: {split}. Expected 'train' or 'test'.")

    if not isinstance(train_idx, list) or not isinstance(test_idx, list):
        raise ValueError("train_idx and test_idx must be lists.")

    # create a unique hash key for this dataset/method/data_cat/split/split_idx combination
    key = (
        f"{dataset_name}_{dataset_group}_{method}_{data_cat}_{split}_"
        f"{hashlib.md5(str(train_idx).encode()).hexdigest()}_"
        f"{hashlib.md5(str(test_idx).encode()).hexdigest()}"
    )
    feature_file = os.path.join(features_dir, f"{key}.pkl")

    if os.path.exists(feature_file):
        print(f"             Loading cached features from {feature_file}")
        with open(feature_file, "rb") as f:
            return pickle.load(f)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:

            features = tsfeatures(data, freq=freq)
            print("             Features created successfully.")

            if store_features:
                with open(feature_file, "wb") as f:
                    pickle.dump(features, f)
                print(f"            Features saved to {feature_file}")

            for warning in w:
                if "divide by zero" in str(warning.message):
                    print("             Detected problematic data. Skipping.")
                    return None

            return features
        except Exception as e:
            print(f"                Error generating features: {e}")
            return None


def plot_feature_importance(
    feature_names, feature_importances, score, loss, dataset_name, dataset_group
):
    """
    Plots and saves feature importance.
    """
    sorted_idx = feature_importances.argsort()
    sorted_features = feature_names[sorted_idx]
    sorted_importances = feature_importances[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importances, color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.tight_layout()

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = (
        f"assets/plots/{current_datetime}_feature_importance_vae_generated_vs_original_"
        f"{dataset_name}_{dataset_group}_{round(score, 2)}_{round(loss, 2)}.pdf"
    )
    plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved to {filename}")


def compute_discriminative_score(
    unique_ids,
    original_data,
    synthetic_data,
    method,
    freq,
    dataset_name,
    dataset_group,
    loss,
    generate_feature_plot=False,
    samples=1,
    store_score=True,
    store_features_synth=True,
    split="train",
):
    score_file = f"assets/results/{dataset_name}_{dataset_group}_{method}_discriminative_score.json"
    os.makedirs("assets/results", exist_ok=True)

    if os.path.exists(score_file) and store_score:
        print(f"Score file '{score_file}' exists. Loading score...")
        with open(score_file, "r") as f:
            final_score = json.load(f).get("final_score", None)
        print(f"Loaded final score: {final_score:.4f}")
        return final_score

    scores = []
    FREQS = {"H": 24, "D": 1, "M": 12, "Q": 4, "W": 1, "Y": 1}
    freq = FREQS[freq]
    for sample in range(samples):
        print(f"    Sample {sample} of {samples}")
        train_idx, test_idx = split_train_test(
            unique_ids,
            sample,
            split_dir=f"assets/model_weights/{dataset_name}_{dataset_group}_{split}_data_split_discriminator",
        )

        # original data
        original_data_train, original_data_train_y = filter_data_by_indices(
            original_data, train_idx, label_value=0
        )
        original_data_test, original_data_test_y = filter_data_by_indices(
            original_data, test_idx, label_value=0
        )

        original_features_train = safe_generate_features(
            original_data_train,
            freq=freq,
            dataset_name=dataset_name,
            dataset_group=dataset_group,
            data_cat="real",
            split=f"{split}_train",
            method=method,
            train_idx=train_idx,
            test_idx=test_idx,
        )
        original_features_test = safe_generate_features(
            original_data_test,
            freq=freq,
            dataset_name=dataset_name,
            dataset_group=dataset_group,
            data_cat="real",
            split=f"{split}_test",
            method=method,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        if original_features_train is None or original_features_test is None:
            print("Feature generation failed for original data. Skipping iteration.")
            continue

        # synthetic data
        synthetic_data_train, synthetic_data_train_y = filter_data_by_indices(
            synthetic_data, train_idx, label_value=1
        )
        synthetic_data_test, synthetic_data_test_y = filter_data_by_indices(
            synthetic_data, test_idx, label_value=1
        )

        synthetic_features_train = safe_generate_features(
            synthetic_data_train,
            freq=freq,
            dataset_name=dataset_name,
            dataset_group=dataset_group,
            data_cat="synthetic",
            split="train",
            method=method,
            train_idx=train_idx,
            test_idx=test_idx,
            store_features=store_features_synth,
        )
        synthetic_features_test = safe_generate_features(
            synthetic_data_test,
            freq=freq,
            dataset_name=dataset_name,
            dataset_group=dataset_group,
            data_cat="synthetic",
            split="test",
            method=method,
            train_idx=train_idx,
            test_idx=test_idx,
            store_features=store_features_synth,
        )

        if synthetic_features_train is None or synthetic_features_test is None:
            print("Feature generation failed for synthetic data. Skipping iteration.")
            continue

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
        y_test = pd.concat(
            (original_data_test_y, synthetic_data_test_y), ignore_index=True
        )

        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        X_test, y_test = shuffle(X_test, y_test, random_state=42)

        classifier = XGBClassifier()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # feature importance
        feature_importances = classifier.feature_importances_
        score = f1_score(y_test, y_pred)
        print("F1 score:", score)
        if generate_feature_plot:
            plot_feature_importance(
                X_train.columns,
                feature_importances,
                score,
                loss,
                dataset_name,
                dataset_group,
            )
        scores.append(score)

    if scores:
        final_score = np.average(scores)
        print(f"\n\n### -> Final score: {final_score:.4f}")
    else:
        print("No valid iterations completed. Final score is undefined.")
        final_score = None

    if store_score:
        with open(score_file, "w") as f:
            json.dump({"final_score": final_score}, f)
            print(f"Final score saved to '{score_file}'")

    return final_score


def smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_value = 100 * np.mean(
        np.where(denominator == 0, 0, 2 * np.abs(y_true - y_pred) / denominator)
    )
    return smape_value


def tstr(
    unique_ids,
    original_data,
    synthetic_data,
    method,
    freq,
    dataset_name,
    dataset_group,
    horizon,
    samples=3,
    split="test",
):
    """
    Train two models:
        1) TRTR (Train on Real, Test on Real)
        2) TSTR (Train on Synthetic, Test on Real)

    Compare their performance on a hold-out test set across multiple splits.
    The final metric reported is the SMAPE.
    """
    results_file = (
        f"assets/results/{dataset_name}_{dataset_group}_{method}_TSTR_results.json"
    )
    os.makedirs("assets/results", exist_ok=True)

    if os.path.exists(results_file):
        print(f"Results file '{results_file}' exists. Loading results...")
        with open(results_file, "r") as f:
            final_results = json.load(f)
        return final_results

    results_trtr = []
    results_tstr = []

    synthetic_data = synthetic_data.drop(columns=["method"], errors="ignore")

    for sample_idx in range(samples):
        print(f"\n--- Sample {sample_idx+1} of {samples} ---")

        trtr_cache_file = (
            f"assets/results/{dataset_name}_{dataset_group}_{method}_{split}"
            f"TRTR_sample{sample_idx}.json"
        )

        train_idx, test_idx = split_train_test(
            unique_ids,
            sample_idx,
            split_dir=f"assets/model_weights/{dataset_name}_{dataset_group}_{split}_data_split_forecast",
        )

        df_test_real, _ = filter_data_by_indices(original_data, test_idx, label_value=0)
        df_test_synth, _ = filter_data_by_indices(
            synthetic_data, test_idx, label_value=1, downstream_forecast=True
        )

        input_size = 50

        model_trtr = NHITS(
            h=horizon,
            max_steps=500,
            input_size=input_size,
            start_padding_enabled=True,
            scaler_type="standard",
        )
        model_tstr = NHITS(
            h=horizon,
            max_steps=500,
            input_size=input_size,
            start_padding_enabled=True,
            scaler_type="standard",
        )

        if os.path.exists(trtr_cache_file):
            print(f"    [TRTR] Cache file found for sample={sample_idx}, loading ...")
            with open(trtr_cache_file, "r") as f:
                cached_data = json.load(f)
            smape_trtr = cached_data["smape_trtr"]
            print(f"    [TRTR] Loaded SMAPE={smape_trtr:.4f} from cache.")
        else:
            print("    [TRTR] Training on Real, Testing on Real ...")
            nf_trtr = NeuralForecast(models=[model_trtr], freq=freq)
            cv_model_trtr = nf_trtr.cross_validation(
                df=df_test_real, test_size=horizon, n_windows=None
            )
            fcst_trtr = cv_model_trtr.reset_index()

            smape_trtr = smape(fcst_trtr["y"], fcst_trtr["NHITS"])
            print(f"    [TRTR] Computed SMAPE={smape_trtr:.4f}")

            with open(trtr_cache_file, "w") as f:
                json.dump(
                    {"smape_trtr": smape_trtr},
                    f,
                )
            print(f"    [TRTR] Results saved to '{trtr_cache_file}'")

        print("    [TRTR] Training on Real, Testing on Real ...")
        nf_trtr = NeuralForecast(models=[model_trtr], freq=freq)
        cv_model_trtr = nf_trtr.cross_validation(
            df=df_test_real, test_size=horizon, n_windows=None
        )
        fcst_trtr = cv_model_trtr.reset_index()

        cutoff_value = fcst_trtr["cutoff"].iloc[0]

        df_test_synth_concat = df_test_synth.loc[
            df_test_synth["ds"] <= cutoff_value
        ].copy()

        df_test_synth_concat["unique_id"] = df_test_synth_concat[
            "unique_id"
        ].str.replace("_synth$", "", regex=True)

        df_test_real_concat = df_test_real.loc[df_test_real["ds"] > cutoff_value].copy()

        df_tstr = pd.concat(
            [df_test_synth_concat, df_test_real_concat], axis=0, ignore_index=True
        ).sort_values(by=["unique_id", "ds"])

        print("    [TSTR] Training on Synthetic, Testing on Real ...")
        nf_tstr = NeuralForecast(models=[model_tstr], freq=freq)
        cv_model_tstr = nf_tstr.cross_validation(
            df=df_tstr, test_size=horizon, n_windows=None
        )
        fcst_tstr = cv_model_tstr.reset_index()

        smape_trtr = smape(fcst_trtr["y"], fcst_trtr["NHITS"])
        smape_tstr = smape(fcst_tstr["y"], fcst_tstr["NHITS"])

        print(f"    SMAPE (TRTR - Real->Real):  {smape_trtr:.4f}")
        print(f"    SMAPE (TSTR - Synth->Real): {smape_tstr:.4f}")

        results_trtr.append(smape_trtr)
        results_tstr.append(smape_tstr)

    if results_trtr and results_tstr:
        avg_smape_trtr = np.mean(results_trtr)
        avg_smape_tstr = np.mean(results_tstr)
        print("\n\n### Final Results across samples ###")
        print(f"Avg SMAPE (TRTR):  {avg_smape_trtr:.4f}  (Train on Real, Test on Real)")
        print(
            f"Avg SMAPE (TSTR):  {avg_smape_tstr:.4f}  (Train on Synth, Test on Real)"
        )
    else:
        avg_smape_trtr = None
        avg_smape_tstr = None
        print("No valid iterations completed. Final results are undefined.")

    final_results = {
        "avg_smape_trtr": avg_smape_trtr,
        "avg_smape_tstr": avg_smape_tstr,
        "results_trtr_samples": results_trtr,
        "results_tstr_samples": results_tstr,
    }

    with open(results_file, "w") as f:
        json.dump(final_results, f)
        print(f"Results saved to '{results_file}'")

    return final_results


def compute_downstream_forecast(
    unique_ids,
    original_data,
    synthetic_data,
    method,
    freq,
    dataset_name,
    dataset_group,
    horizon,
    samples=3,
    split="test",
):
    """
    Train two NHITS models:
        1) On original_data only.
        2) On original_data + synthetic_data (concatenated).
    Compare their performance on a hold-out test set.
    """
    results_file = f"assets/results/{dataset_name}_{dataset_group}_{method}_downstream_task_results.json"
    os.makedirs("assets/results", exist_ok=True)

    if os.path.exists(results_file):
        print(f"Results file '{results_file}' exists. Loading results...")
        with open(results_file, "r") as f:
            final_results = json.load(f)
        return final_results

    results_original = []
    results_concatenated = []

    synthetic_data = synthetic_data.drop(columns=["method"], errors="ignore")

    for sample_idx in range(samples):
        print(f"\n--- Sample {sample_idx+1} of {samples} ---")

        original_downstream_forecast_cache_file = (
            f"assets/results/{dataset_name}_{dataset_group}_{method}_{split}"
            f"downstream_forecast_original_sample{sample_idx}.json"
        )

        train_idx, test_idx = split_train_test(
            unique_ids,
            sample_idx,
            split_dir=f"assets/model_weights/{dataset_name}_{dataset_group}_{split}_data_split_forecast",
        )

        df_train_original, _ = filter_data_by_indices(
            original_data, train_idx, label_value=0
        )
        df_test_original, _ = filter_data_by_indices(
            original_data, test_idx, label_value=0
        )

        df_train_synthetic, _ = filter_data_by_indices(
            synthetic_data, train_idx, label_value=1, downstream_forecast=True
        )
        df_test_synthetic, _ = filter_data_by_indices(
            synthetic_data, test_idx, label_value=1, downstream_forecast=True
        )

        df_train_concat = pd.concat(
            [df_train_original, df_train_synthetic], ignore_index=True
        )
        df_test_concat = pd.concat(
            [df_test_original, df_test_synthetic], ignore_index=True
        )

        input_size = 50

        if os.path.exists(original_downstream_forecast_cache_file):
            print(
                f"    [ORIGINAL] Cache file found for sample={sample_idx}, loading..."
            )
            with open(original_downstream_forecast_cache_file, "r") as f:
                cached_data = json.load(f)
            smape_original = cached_data["smape_original"]
            print(f"    [ORIGINAL] Loaded SMAPE={smape_original:.4f} from cache.")
        else:
            print("    Training NHITS on original data...")
            model_original = NHITS(
                h=horizon,
                max_steps=500,
                input_size=input_size,
                start_padding_enabled=True,
                scaler_type="standard",
            )
            nf_orig = NeuralForecast(models=[model_original], freq=freq)
            cv_model_orig = nf_orig.cross_validation(
                df=df_test_original, test_size=horizon, n_windows=None
            )
            fcst_orig = cv_model_orig.reset_index()
            smape_original = smape(fcst_orig["y"], fcst_orig["NHITS"])

            with open(original_downstream_forecast_cache_file, "w") as f:
                json.dump({"smape_original": smape_original}, f)
            print(
                f"    [ORIGINAL] Computed SMAPE={smape_original:.4f} "
                f"and saved to {original_downstream_forecast_cache_file}"
            )

        print("    Training NHITS on original + synthetic data...")
        model_concat = NHITS(
            h=horizon,
            max_steps=500,
            input_size=input_size,
            start_padding_enabled=True,
            scaler_type="standard",
        )

        nf_concat = NeuralForecast(models=[model_concat], freq=freq)
        cv_model_concat = nf_concat.cross_validation(
            df=df_test_concat, test_size=horizon, n_windows=None
        )
        fcst_concat = cv_model_concat.reset_index()

        fcst_concat = fcst_concat[
            ~fcst_concat["unique_id"].str.contains("_synth", na=False)
        ]

        smape_concat = smape(fcst_concat["y"], fcst_concat["NHITS"])

        print(f"    SMAPE (original-only): {smape_original:.4f}")
        print(f"    SMAPE (concat):        {smape_concat:.4f}")

        results_original.append(smape_original)
        results_concatenated.append(smape_concat)

    if results_original and results_concatenated:
        avg_smape_original = np.mean(results_original)
        avg_smape_concat = np.mean(results_concatenated)
        std_smape_original = np.std(results_original)
        std_smape_concat = np.std(results_concatenated)

        print("\n\n### Final Results across samples ###")
        print(
            f"Avg SMAPE (original-only): {avg_smape_original:.4f} ± {std_smape_original:.4f}"
        )
        print(
            f"Avg SMAPE (concat):        {avg_smape_concat:.4f} ± {std_smape_concat:.4f}"
        )
    else:
        avg_smape_original = None
        avg_smape_concat = None
        std_smape_original = None
        std_smape_concat = None
        print("No valid iterations completed. Final results are undefined.")

    final_results = {
        "avg_smape_original": avg_smape_original,
        "std_smape_original": std_smape_original,
        "avg_smape_concat": avg_smape_concat,
        "std_smape_concat": std_smape_concat,
        "results_original_samples": results_original,
        "results_concatenated_samples": results_concatenated,
    }

    if split == "hypertuning":
        return avg_smape_concat
    else:
        with open(results_file, "w") as f:
            json.dump(final_results, f)
            print(f"Results saved to '{results_file}'")
        return final_results
