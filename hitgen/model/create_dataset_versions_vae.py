import os
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
from tensorflow import keras
import tensorflow as tf
from tensorflow import data as tfdata
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow import (
    float32,
)
from hitgen.model.models import (
    CVAE,
    KLAnnealingAndNoiseScalingCallback,
    TemporalizeGenerator,
)
from hitgen.feature_engineering.feature_transformations import (
    detemporalize,
)
from hitgen.preprocessing.pre_processing_datasets import (
    PreprocessDatasets as ppc,
)
from hitgen.model.models import get_CVAE
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
)
from hitgen.load_data.config import DATASETS, DATASETS_FREQ
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)


class InvalidFrequencyError(Exception):
    pass


class CreateTransformedVersionsCVAE:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).

    This class contains several methods to preprocess data, fit a CVAE, generate new time series, and
    save transformed versions of the dataset. It's designed to be used with time-series data.

    The class follows the Singleton design pattern ensuring that only one instance can exist.

    Args:
        dataset_name: Name of the dataset.
        freq: Frequency of the time series data.
        input_dir: Directory where the input data is located. Defaults to "./".
        transf_data: Type of transformation applied to the data. Defaults to "whole".
        top: Number of top series to select. Defaults to None.
        window_size: Window size for the sliding window. Defaults to 10.
        weekly_m5: If True, use the M5 competition's weekly grouping. Defaults to True.
        test_size: Size of the test set. If None, the size is determined automatically. Defaults to None.

        Below are parameters for the synthetic data creation:
            num_base_series_time_points: Number of base time points in the series. Defaults to 100.
            num_latent_dim: Dimension of the latent space. Defaults to 3.
            num_variants: Number of variants for the transformation. Defaults to 20.
            noise_scale: Scale of the Gaussian noise. Defaults to 0.1.
            amplitude: Amplitude of the time series data. Defaults to 1.0.
    """

    _instance = None

    def __new__(cls, *args, **kwargs) -> "CreateTransformedVersionsCVAE":
        """
        Override the __new__ method to implement the Singleton design pattern.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        dataset_name: str,
        dataset_group: str,
        freq: str,
        batch_size: int = 8,
        shuffle: bool = True,
        input_dir: str = "./assets/",
        transf_data: str = "whole",
        top: int = None,
        window_size: int = 12,
        weekly_m5: bool = True,
        test_size: int = None,
        num_base_series_time_points: int = 100,
        num_latent_dim: int = 3,
        num_variants: int = 20,
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
        stride_temporalize: int = 2,
        bi_rnn: bool = True,
        forecasting: bool = True,
        annealing: bool = True,
        kl_weight_init: float = None,
        noise_scale_init: float = None,
        n_blocks_encoder: int = 3,
        n_blocks_decoder: int = 3,
        n_hidden: int = 16,
        n_layers: int = 3,
        kernel_size: int = 2,
        pooling_mode: str = "average",
        patience: int = 30,
    ):
        self.dataset_name = dataset_name
        self.dataset_group = dataset_group
        self.input_dir = input_dir
        self.transf_data = transf_data
        self.freq = freq
        self.top = top
        self.test_size = test_size
        self.weekly_m5 = weekly_m5
        self.num_base_series_time_points = num_base_series_time_points
        self.num_latent_dim = num_latent_dim
        self.num_variants = num_variants
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        self.stride_temporalize = stride_temporalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bi_rnn = bi_rnn
        self.forecasting = forecasting
        self.annealing = annealing
        self.kl_weight_init = kl_weight_init
        self.noise_scale_init = noise_scale_init
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.patience = patience
        self.kernel_size = kernel_size
        self.pooling_mode = pooling_mode
        (self.data, self.s, self.freq) = self.load_data(
            self.dataset_name, self.dataset_group
        )
        self.s_train = None
        if window_size:
            self.window_size = window_size
        self.y = self.data
        self.n = self.data.shape[0]
        self.df = pd.DataFrame(self.data)
        self.df.asfreq(self.freq)

        num_series = self.df["unique_id"].nunique()
        avg_time_points = self.df.groupby("unique_id").size().mean()

        print(f"Dataset Summary for {dataset_name} ({dataset_group}):")
        print(f"   - Total number of time series: {num_series}")
        print(f"   - Average number of time points per series: {avg_time_points:.2f}")

        self.features_input = (None, None, None)
        self._create_directories()
        self.long_properties = {}
        self.split_path = f"assets/model_weights/data_split/{dataset_name}_{dataset_group}_data_split.json"
        self.unique_ids = self.df["unique_id"].unique()

    @staticmethod
    def load_data(dataset_name, group):
        data_cls = DATASETS[dataset_name]
        print(dataset_name, group)

        try:
            ds = data_cls.load_data(group)
        except FileNotFoundError as e:
            print(f"Error loading data for {dataset_name} - {group}: {e}")

        h = data_cls.horizons_map[group]
        n_lags = data_cls.context_length[group]
        freq = data_cls.frequency_pd[group]
        season_len = data_cls.frequency_map[group]
        n_series = ds.nunique()["unique_id"]
        return ds, n_series, freq

    def create_dataset_long_form(self, data, unique_ids=None) -> pd.DataFrame:
        df = pd.DataFrame(data)

        if unique_ids is None:
            df.columns = self.long_properties["unique_id"]
        else:
            df.columns = unique_ids
        df["ds"] = pd.date_range(
            self.long_properties["ds"][0],
            periods=data.shape[0],
            freq=self.freq,
        )

        data_long = df.melt(id_vars=["ds"], var_name="unique_id", value_name="y")

        return data_long

    def _get_dataset(self):
        """
        Get dataset and apply preprocessing
        """
        ppc_args = {
            "dataset": self.dataset_name,
            "freq": self.freq,
            "input_dir": self.input_dir,
            "top": self.top,
            "test_size": self.test_size,
            "weekly_m5": self.weekly_m5,
            "num_base_series_time_points": self.num_base_series_time_points,
            "num_latent_dim": self.num_latent_dim,
            "num_variants": self.num_variants,
            "noise_scale": self.noise_scale,
            "amplitude": self.amplitude,
        }

        dataset = ppc(**ppc_args).apply_preprocess()

        return dataset

    def _create_directories(self):
        """
        Create dynamically the directories to store the data if they don't exist
        """
        # Create directory to store transformed datasets if does not exist
        Path(f"{self.input_dir}data").mkdir(parents=True, exist_ok=True)

    def _load_or_create_split(
        self,
        train_test_split: float,
        train_test_absolute: int,
    ) -> (np.ndarray, np.ndarray):
        """Load split from file if it exists, otherwise create and save a new split."""
        if os.path.exists(self.split_path):
            with open(self.split_path, "r") as f:
                split_data = json.load(f)
                return np.array(split_data["train_ids"]), np.array(
                    split_data["test_ids"]
                )

        np.random.shuffle(self.unique_ids)
        train_size = int(len(self.unique_ids) * train_test_split)
        if train_test_absolute:
            train_ids = self.unique_ids[:train_test_absolute]
        else:
            train_ids = self.unique_ids[:train_size]

        test_ids = self.unique_ids[train_size:]

        os.makedirs(os.path.dirname(self.split_path), exist_ok=True)
        with open(self.split_path, "w") as f:
            json.dump(
                {"train_ids": train_ids.tolist(), "test_ids": test_ids.tolist()}, f
            )

        return train_ids, test_ids

    @staticmethod
    def _transform_log_returns(x):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        x_log = np.log(x + 1)
        x_diff = x_log.diff()
        return x_diff

    @staticmethod
    def _transform_diff(x):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        x_diff = x.diff()
        return x_diff

    @staticmethod
    def _transform_diff_minmax(x):
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        x_diff = x.diff()
        return x_diff

    @staticmethod
    def _backtransform_log_returns(x_diff: pd.DataFrame, initial_value: pd.DataFrame):
        """
        Back-transform log returns.
        """
        x_diff["ds"] = pd.to_datetime(x_diff["ds"])
        initial_value["ds"] = pd.to_datetime(initial_value["ds"])

        # filter x_diff to exclude any value before the first true value
        x_diff = x_diff.merge(
            initial_value[["unique_id", "ds"]],
            on="unique_id",
            suffixes=("", "_initial"),
        )
        x_diff = x_diff[x_diff["ds"] > x_diff["ds_initial"]]
        x_diff = x_diff.drop(columns=["ds_initial"])

        # compute log-transformed initial values
        initial_value = initial_value.set_index("unique_id")
        initial_value["y"] = np.log(initial_value["y"] + 1)

        x_diff = pd.concat(
            [x_diff.reset_index(drop=True), initial_value.reset_index()]
        ).sort_values(by=["unique_id", "ds"])

        # set the index for x_diff for alignment and compute the cumulative sum
        x_diff["y"] = x_diff.groupby("unique_id")["y"].cumsum()

        x_diff["y"] = np.exp(x_diff["y"]) - 1

        return x_diff

    @staticmethod
    def _backtransform_diff(x_diff: pd.DataFrame, initial_value: pd.DataFrame):
        """
        Back-transform log returns.
        """
        x_diff["ds"] = pd.to_datetime(x_diff["ds"])
        initial_value["ds"] = pd.to_datetime(initial_value["ds"])

        # filter x_diff to exclude any value before the first true value
        x_diff = x_diff.merge(
            initial_value[["unique_id", "ds"]],
            on="unique_id",
            suffixes=("", "_initial"),
        )
        x_diff = x_diff[x_diff["ds"] > x_diff["ds_initial"]]
        x_diff = x_diff.drop(columns=["ds_initial"])

        # compute log-transformed initial values
        initial_value = initial_value.set_index("unique_id")

        x_diff = pd.concat(
            [x_diff.reset_index(drop=True), initial_value.reset_index()]
        ).sort_values(by=["unique_id", "ds"])

        # set the index for x_diff for alignment and compute the cumulative sum
        x_diff["y"] = x_diff.groupby("unique_id")["y"].cumsum()

        return x_diff

    def _preprocess_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Sort and preprocess the data for feature engineering."""
        df = df.sort_values(by=["unique_id", "ds"])
        x_wide = df.pivot(index="ds", columns="unique_id", values="y")

        self.long_properties["ds"] = x_wide.reset_index()["ds"].values
        self.long_properties["unique_id"] = x_wide.columns.values

        # create mask before padding
        mask = (~x_wide.isna()).astype(int)

        # x_wide_log_returns = self._transform_log_returns(x_wide)
        # x_wide_diff = self._transform_diff(x_wide)

        # padding
        x_wide_filled = x_wide.fillna(0.0)
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(x_wide_filled)

        return scaled_data, mask, x_wide

    def plot_sample_time_series_long_format(self, df, data_cat, state):
        import matplotlib.pyplot as plt

        selected_ids = df["unique_id"].unique()[:8]
        df_selected = df[df["unique_id"].isin(selected_ids)]

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 10), sharex=True)

        axes = axes.flatten()

        for i, uid in enumerate(selected_ids):
            subset = df_selected[df_selected["unique_id"] == uid]
            ax = axes[i]
            ax.plot(subset["ds"], subset["y"], marker="o", linestyle="-")
            ax.set_title(f"Series: {uid}")
            ax.grid(True)

        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_dir = "assets/plots/"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"{self.dataset_name}_{self.dataset_group}_{data_cat}_{state}_time_series.png",
        )
        plt.savefig(output_path, dpi=300)

    def _feature_engineering(
        self, train_test_split=0.7, train_size_absolute=None
    ) -> Dict[str, pd.DataFrame]:
        """Apply preprocessing to raw time series and split into training and testing."""
        x_wide_transf, mask_wide, x_wide = self._preprocess_data(self.df)
        x_long = self.create_dataset_long_form(x_wide)
        x_long_transf = self.create_dataset_long_form(x_wide_transf)
        # self.plot_sample_time_series_long_format(
        #     x_long_transf, "original", "preprocess"
        # )

        mask_long = self.create_dataset_long_form(mask_wide)

        train_ids, test_ids = self._load_or_create_split(
            train_test_split, train_size_absolute
        )

        self.s_train = len(train_ids)

        train_data = x_long_transf[x_long_transf["unique_id"].isin(train_ids)]
        train_data_no_transf = x_long[x_long["unique_id"].isin(train_ids)]
        mask_train = mask_long[mask_long["unique_id"].isin(train_ids)]
        test_data = x_long_transf[x_long_transf["unique_id"].isin(test_ids)]
        test_data_no_transf = x_long[x_long["unique_id"].isin(test_ids)]
        mask_test = mask_long[mask_long["unique_id"].isin(test_ids)]
        original_data = x_long_transf
        original_data_no_transf = x_long
        original_mask = mask_long

        x_train_wide = train_data.pivot(index="ds", columns="unique_id", values="y")
        x_test_wide = test_data.pivot(index="ds", columns="unique_id", values="y")

        # extract only training min/max params from the full scaler
        train_columns = x_train_wide.columns
        scaler_train = MinMaxScaler()

        # filter the fitted params from self.scaler for only training series
        scaler_train.min_ = self.scaler.min_[
            np.isin(self.long_properties["unique_id"], train_columns)
        ]
        scaler_train.scale_ = self.scaler.scale_[
            np.isin(self.long_properties["unique_id"], train_columns)
        ]
        scaler_train.data_min_ = self.scaler.data_min_[
            np.isin(self.long_properties["unique_id"], train_columns)
        ]
        scaler_train.data_max_ = self.scaler.data_max_[
            np.isin(self.long_properties["unique_id"], train_columns)
        ]
        scaler_train.data_range_ = self.scaler.data_range_[
            np.isin(self.long_properties["unique_id"], train_columns)
        ]
        scaler_train.feature_names_in = self.scaler.feature_names_in_[
            np.isin(self.long_properties["unique_id"], train_columns)
        ]

        self.scaler_train = scaler_train

        x_train_no_transf_wide = train_data_no_transf.pivot(
            index="ds", columns="unique_id", values="y"
        )
        x_test_no_transf_wide = test_data_no_transf.pivot(
            index="ds", columns="unique_id", values="y"
        )
        x_original_wide = original_data.pivot(
            index="ds", columns="unique_id", values="y"
        )
        x_original_no_transf_wide = original_data_no_transf.pivot(
            index="ds", columns="unique_id", values="y"
        )
        mask_train_wide = mask_train.pivot(index="ds", columns="unique_id", values="y")
        self.mask_train_tf = tf.convert_to_tensor(
            mask_train_wide.values, dtype=tf.float32
        )
        mask_test_wide = mask_test.pivot(index="ds", columns="unique_id", values="y")
        self.mask_test_tf = tf.convert_to_tensor(
            mask_test_wide.values, dtype=tf.float32
        )
        mask_original_wide = original_mask.pivot(
            index="ds", columns="unique_id", values="y"
        )
        self.mask_original_tf = tf.convert_to_tensor(
            mask_original_wide.values, dtype=tf.float32
        )

        original_data_train_long = x_train_wide.reset_index().melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )
        original_data_train_no_transf_long = x_train_no_transf_wide.reset_index().melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )
        original_data_test_long = x_test_wide.reset_index().melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )
        original_data_test_no_transf_long = x_test_no_transf_wide.reset_index().melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )
        original_data_long = x_original_wide.reset_index().melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )
        original_data_no_transf_long = x_original_no_transf_wide.reset_index().melt(
            id_vars=["ds"], var_name="unique_id", value_name="y"
        )

        self.X_train_raw = x_train_wide.reset_index(drop=True)
        self.X_test_raw = x_test_wide.reset_index(drop=True)
        self.X_orig_raw = x_original_wide.reset_index(drop=True)

        def compute_fourier_features(dates, n):
            """Compute Fourier terms for a given frequency."""
            t = dates.astype(np.int64) / 10**9  # convert datetime to seconds
            freq_to_period = {
                "D": 365.25,
                "W": 52.18,
                "MS": 12,
                "M": 12,
                "QS": 4,
                "Q": 4,
                "Y": 1,
                "YS": 1,
            }
            period = freq_to_period.get(self.freq, 1)
            features = {}
            for k in range(1, n + 1):
                features[f"sin_{self.freq}_{k}"] = np.sin(2 * np.pi * k * t / period)
                features[f"cos_{self.freq}_{k}"] = np.cos(2 * np.pi * k * t / period)
            return pd.DataFrame(features)

        fourier_features_train = compute_fourier_features(x_train_wide.index, 3)
        fourier_features_test = compute_fourier_features(x_test_wide.index, 3)
        fourier_features_original = compute_fourier_features(x_original_wide.index, 3)

        return {
            "x_train_wide": x_train_wide,
            "x_test_wide": x_test_wide,
            "x_original_wide": x_original_wide,
            "original_data_train_long": original_data_train_long,
            "original_data_test_long": original_data_test_long,
            "original_data_long": original_data_long,
            "mask_train_wide": mask_train_wide,
            "mask_test_wide": mask_test_wide,
            "mask_original_wide": mask_original_wide,
            "original_data_no_transf_long": original_data_no_transf_long,
            "original_data_train_no_transf_long": original_data_train_no_transf_long,
            "original_data_test_no_transf_long": original_data_test_no_transf_long,
            "fourier_features_train": fourier_features_train,
            "fourier_features_test": fourier_features_test,
            "fourier_features_original": fourier_features_original,
        }

    @staticmethod
    def _generate_noise(self, n_batches, window_size):
        while True:
            yield np.random.uniform(low=0, high=1, size=(n_batches, window_size))

    def get_batch_noise(
        self,
        batch_size,
        size=None,
    ):
        return iter(
            tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
            .batch(batch_size if size is None else size)
            .repeat()
        )

    def fit(
        self,
        epochs: int = 750,
        patience: int = 30,
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        hyper_tuning: bool = False,
        load_weights: bool = True,
    ) -> tuple[CVAE, dict, EarlyStopping]:
        """Training our CVAE"""
        feature_dict = self._feature_engineering()

        original_data = feature_dict["x_original_wide"]
        original_mask = feature_dict["mask_original_wide"]
        original_features = feature_dict["fourier_features_original"]

        data_mask_temporalized = TemporalizeGenerator(
            original_data,
            original_mask,
            original_features,
            window_size=self.window_size,
            stride=self.stride_temporalize,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        encoder, decoder = get_CVAE(
            window_size=self.window_size,
            n_series=self.s,
            latent_dim=latent_dim,
            bi_rnn=self.bi_rnn,
            noise_scale_init=self.noise_scale_init,
            n_blocks_encoder=self.n_blocks_encoder,
            n_blocks_decoder=self.n_blocks_decoder,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            pooling_mode=self.pooling_mode,
            forecasting=self.forecasting,
        )

        cvae = CVAE(
            encoder,
            decoder,
            kl_weight_initial=self.kl_weight_init,
            forecasting=self.forecasting,
        )
        cvae.compile(
            optimizer=keras.optimizers.legacy.Adam(
                learning_rate=learning_rate, clipnorm=1.0, clipvalue=1.0
            ),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        es = EarlyStopping(
            patience=patience,
            verbose=1,
            monitor="loss",
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.2, patience=10, min_lr=1e-6, cooldown=3, verbose=1
        )

        weights_folder = "assets/model_weights"
        os.makedirs(weights_folder, exist_ok=True)

        weights_file = os.path.join(
            weights_folder, f"{self.dataset_name}_{self.dataset_group}__vae.weights.h5"
        )
        history_file = os.path.join(
            weights_folder,
            f"{self.dataset_name}_{self.dataset_group}_training_history.json",
        )
        history = None

        if os.path.exists(weights_file) and not hyper_tuning and load_weights:
            print("Loading existing weights...")
            cvae.load_weights(weights_file)

            if os.path.exists(history_file):
                print("Loading training history...")
                with open(history_file, "r") as f:
                    history = json.load(f)
            else:
                print("No history file found. Skipping history loading.")
        else:

            mc = ModelCheckpoint(
                weights_file,
                save_best_only=True,
                save_weights_only=True,
                monitor="loss",
                mode="auto",
                verbose=1,
            )

            history = cvae.fit(
                x=data_mask_temporalized,
                epochs=epochs,
                batch_size=self.batch_size,
                shuffle=False,
                callbacks=[es, mc, reduce_lr],
            )

            if history is not None:
                history = history.history
                history_dict = {
                    key: [float(val) for val in values]
                    for key, values in history.items()
                }
                with open(history_file, "w") as f:
                    json.dump(history_dict, f)

        return cvae, history, es

    def update_best_scores(
        self,
        original_data,
        synthetic_data,
        score,
        latent_dim,
        window_size,
        patience,
        kl_weight,
        n_blocks_encoder,
        n_blocks_decoder,
        n_hidden,
        n_layers,
        kernel_size,
        pooling_mode,
        batch_size,
        epochs,
        learning_rate,
        bi_rnn,
        shuffle,
        forecasting,
        noise_scale_init,
        loss,
    ):
        scores_path = f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_best_hyperparameters.jsonl"

        if os.path.exists(scores_path):
            with open(scores_path, "r") as f:
                scores_data = [json.loads(line) for line in f.readlines()]
        else:
            scores_data = []

        new_score = {
            "latent_dim": latent_dim,
            "window_size": window_size,
            "patience": patience,
            "kl_weight": kl_weight,
            "n_blocks_encoder": n_blocks_encoder,
            "n_blocks_decoder": n_blocks_decoder,
            "n_hidden": n_hidden,
            "n_layers": n_layers,
            "kernel_size": kernel_size,
            "pooling_mode": pooling_mode,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "bi_rnn": bi_rnn,
            "shuffle": shuffle,
            "forecasting": forecasting,
            "noise_scale_init": noise_scale_init,
            "loss": loss,
            "score": score,
        }

        added_score = False

        # always add the score if there are fewer than 20 entries
        if len(scores_data) < 20:
            scores_data.append(new_score)
            added_score = True

        # if the list is full, add only if the score is better than the worst
        elif score < max([entry["score"] for entry in scores_data]):
            scores_data.append(new_score)
            added_score = True

        if added_score:
            plot_generated_vs_original(
                synth_data=synthetic_data,
                original_test_data=original_data,
                score=score,
                loss=loss,
                dataset_name=self.dataset_name,
                dataset_group=self.dataset_group,
                n_series=8,
            )

        scores_data.sort(key=lambda x: x["score"])
        scores_data = scores_data[:20]

        os.makedirs(os.path.dirname(scores_path), exist_ok=True)
        with open(scores_path, "w") as f:
            for score_entry in scores_data:
                f.write(json.dumps(score_entry) + "\n")

        print(f"Best scores updated and saved to {scores_path}")

    @staticmethod
    def compute_mean_discriminative_score(
        unique_ids,
        original_data,
        synthetic_data,
        method,
        freq,
        dataset_name,
        dataset_group,
        loss,
        store_score,
        store_features_synth,
        split,
        num_iterations=1,
        generate_feature_plot=False,
    ):
        scores = []
        for i in range(num_iterations):
            score = compute_discriminative_score(
                unique_ids=unique_ids,
                original_data=original_data,
                synthetic_data=synthetic_data,
                method=method,
                freq=freq,
                dataset_name=dataset_name,
                dataset_group=dataset_group,
                loss=loss,
                samples=3,
                generate_feature_plot=generate_feature_plot,
                store_score=store_score,
                store_features_synth=store_features_synth,
                split=split,
            )
            scores.append(score)

        mean_score = np.mean(scores)
        return mean_score

    def objective(self, trial):
        """
        Objective function for Optuna to tune the CVAE hyperparameters.
        """
        # try:
        latent_dim = trial.suggest_int("latent_dim", 8, 300, step=8)
        # window_size = trial.suggest_int("window_size", 6, 24)
        patience = trial.suggest_int("patience", 20, 40, step=5)
        kl_weight = trial.suggest_float("kl_weight", 0.05, 0.5)
        n_blocks_encoder = trial.suggest_int("n_blocks_encoder", 1, 5)
        n_blocks_decoder = trial.suggest_int("n_blocks_decoder", 1, 5)
        n_hidden = trial.suggest_int("n_hidden", 16, 128, step=16)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        pooling_mode = trial.suggest_categorical("pooling_mode", ["max", "average"])
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
        epochs = trial.suggest_int("epochs", 200, 2000, step=100)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        # bi_rnn = trial.suggest_categorical("bi_rnn", [True, False])
        forecasting = trial.suggest_categorical("forecasting", [True, False])
        # shuffle = trial.suggest_categorical("shuffle", [True, False])
        noise_scale_init = trial.suggest_float("noise_scale_init", 0.01, 0.5)

        bi_rnn = False
        shuffle = False

        feature_dict = self._feature_engineering()

        train_data = feature_dict["x_train_wide"]
        train_mask = feature_dict["mask_train_wide"]
        original_data_train_no_transf_long = feature_dict[
            "original_data_train_no_transf_long"
        ]
        train_dyn_features = feature_dict["fourier_features_train"]

        data_mask_temporalized = TemporalizeGenerator(
            train_data,
            train_mask,
            train_dyn_features,
            window_size=self.window_size,
            stride=self.stride_temporalize,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        encoder, decoder = get_CVAE(
            window_size=self.window_size,
            n_series=self.s_train,
            latent_dim=latent_dim,
            bi_rnn=bi_rnn,
            noise_scale_init=noise_scale_init,
            n_blocks_encoder=n_blocks_encoder,
            n_blocks_decoder=n_blocks_decoder,
            n_hidden=n_hidden,
            n_layers=n_layers,
            kernel_size=kernel_size,
            pooling_mode=pooling_mode,
            forecasting=forecasting,
        )

        cvae = CVAE(
            encoder, decoder, kl_weight_initial=kl_weight, forecasting=forecasting
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        es = EarlyStopping(
            patience=self.patience,
            verbose=1,
            monitor="loss",
            mode="auto",
            restore_best_weights=True,
        )

        history = cvae.fit(
            x=data_mask_temporalized,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
        )

        loss = min(history.history["loss"])

        _, synthetic_data_long_no_transf = self.predict_train(
            cvae,
            data_mask_temporalized=data_mask_temporalized,
            samples=data_mask_temporalized.indices.shape[0],
            window_size=self.window_size,
            latent_dim=latent_dim,
        )

        # compute the discriminative score x times to account for variability

        score = self.compute_mean_discriminative_score(
            unique_ids=original_data_train_no_transf_long["unique_id"].unique(),
            original_data=original_data_train_no_transf_long,
            synthetic_data=synthetic_data_long_no_transf,
            method="hitgen",
            freq="M",
            dataset_name=self.dataset_name,
            dataset_group=self.dataset_group,
            loss=loss,
            generate_feature_plot=False,
            store_score=False,
            store_features_synth=False,
            split="hypertuning",
        )

        if score is None:
            print("No valid scores computed. Pruning this trial.")
            raise optuna.exceptions.TrialPruned()

        self.update_best_scores(
            original_data_train_no_transf_long,
            synthetic_data_long_no_transf,
            score,
            latent_dim,
            self.window_size,
            patience,
            kl_weight,
            n_blocks_encoder,
            n_blocks_decoder,
            n_hidden,
            n_layers,
            kernel_size,
            pooling_mode,
            batch_size,
            epochs,
            learning_rate,
            bi_rnn,
            shuffle,
            forecasting,
            noise_scale_init,
            loss,
        )

        # clean GPU memory between trials
        del cvae, encoder, decoder
        tf.keras.backend.clear_session()
        gc.collect()
        os.system("nvidia-smi --gpu-reset")

        return score

        # except Exception as e:
        #     print(f"Error in trial: {e}")
        #     raise optuna.exceptions.TrialPruned()

    def hyper_tune_and_train(self, n_trials=30):
        """
        Run Optuna hyperparameter tuning for the CVAE and train the best model.
        """
        study_name = "opt_vae"

        storage = "sqlite:///optuna_study.db"

        feature_dict = self._feature_engineering()

        data_train = feature_dict["x_train_wide"]
        mask_train = feature_dict["mask_train_wide"]
        dyn_features_train = feature_dict["fourier_features_train"]

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )

        if len(study.trials) == 0:
            print("No trials have been completed yet. Running hyperparameter tuning...")
            study.optimize(self.objective, n_trials=n_trials)

        if len(study.trials) == 0:
            raise RuntimeError(
                "Optuna tuning did not complete successfully. No trials were found."
            )

        # retrieve the best trial
        best_trial = study.best_trial
        self.best_params = best_trial.params

        with open(
            f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_best_params.json",
            "w",
        ) as f:
            json.dump(self.best_params, f)

        self.best_params["bi_rnn"] = False
        self.best_params["shuffle"] = True

        print(f"Best Hyperparameters: {self.best_params}")

        data_mask_temporalized = TemporalizeGenerator(
            data_train,
            mask_train,
            dyn_features_train,
            window_size=self.best_params["latent_dim"],
            stride=self.stride_temporalize,
            batch_size=self.best_params["batch_size"],
            shuffle=self.best_params["shuffle"],
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params["latent_dim"],
            n_series=self.s_train,
            latent_dim=self.best_params["latent_dim"],
            bi_rnn=self.best_params["bi_rnn"],
            noise_scale_init=self.best_params["noise_scale_init"],
            n_blocks_encoder=self.best_params["n_blocks_encoder"],
            n_blocks_decoder=self.best_params["n_blocks_decoder"],
            n_hidden=self.best_params["n_hidden"],
            n_layers=self.best_params["n_layers"],
            kernel_size=self.best_params["kernel_size"],
            pooling_mode=self.best_params["pooling_mode"],
            forecasting=self.best_params["forecasting"],
        )

        cvae = CVAE(
            encoder,
            decoder,
            kl_weight_initial=self.best_params["kl_weight"],
            forecasting=self.best_params["forecasting"],
        )
        cvae.compile(
            optimizer=keras.optimizers.legacy.Adam(
                learning_rate=self.best_params["learning_rate"]
            ),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        # final training with best parameters
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.best_params["patience"],
            restore_best_weights=True,
        )
        history = cvae.fit(
            x=data_mask_temporalized,
            epochs=self.best_params["epochs"],
            batch_size=self.best_params["batch_size"],
            callbacks=[early_stopping],
        )

        # Save training history
        with open(
            f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_training_history.json",
            "w",
        ) as f:
            json.dump(history.history, f)

        print("Training completed with the best hyperparameters.")

    def predict(
        self,
        cvae: CVAE,
        data_mask_temporalized,
        samples,
        window_size,
        latent_dim,
        train_test_split=0.7,
        train_size_absolute=None,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Predict original time series using VAE"""
        # new_latent_samples = np.random.normal(size=(samples, window_size, latent_dim))
        # mask_temporalized = self.temporalize(self.mask_original_tf, window_size)

        z_mean, z_log_var, z = cvae.encoder.predict(
            [
                data_mask_temporalized.temporalized_data,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )
        alpha = 3  # x times bigger variance
        epsilon = np.random.normal(size=z_mean.shape) * 0.1
        z_augmented = z_mean + np.exp(0.5 * z_log_var) * alpha * epsilon
        generated_data, predictions = cvae.decoder.predict(
            [
                z_augmented,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )

        X_hat = detemporalize(generated_data)
        X_hat_no_transf = self.scaler.inverse_transform(X_hat)

        train_ids, test_ids = self._load_or_create_split(
            train_test_split, train_size_absolute
        )

        x_hat_long = self.create_dataset_long_form(X_hat)
        x_hat_long_no_transf = self.create_dataset_long_form(X_hat_no_transf)

        X_hat_train_long = x_hat_long[x_hat_long["unique_id"].isin(train_ids)]
        X_hat_test_long = x_hat_long[x_hat_long["unique_id"].isin(test_ids)]
        X_hat_all_long = x_hat_long

        X_hat_train_long_no_transf = x_hat_long_no_transf[
            x_hat_long_no_transf["unique_id"].isin(train_ids)
        ]
        X_hat_test_long_no_transf = x_hat_long_no_transf[
            x_hat_long_no_transf["unique_id"].isin(test_ids)
        ]
        X_hat_all_long_no_transf = x_hat_long_no_transf

        return (
            X_hat_train_long,
            X_hat_test_long,
            X_hat_all_long,
            X_hat_train_long_no_transf,
            X_hat_test_long_no_transf,
            X_hat_all_long_no_transf,
        )

    def predict_train(
        self,
        cvae: CVAE,
        data_mask_temporalized,
        samples,
        window_size,
        latent_dim,
        train_test_split=0.7,
        train_size_absolute=None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Predict original time series using VAE"""
        z_mean, z_log_var, z = cvae.encoder.predict(
            [
                data_mask_temporalized.temporalized_data,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )
        alpha = 3  # x times bigger variance
        epsilon = np.random.normal(size=z_mean.shape) * 0.1
        z_augmented = z_mean + np.exp(0.5 * z_log_var) * alpha * epsilon
        generated_data, predictions = cvae.decoder.predict(
            [
                z_augmented,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )

        train_ids, test_ids = self._load_or_create_split(
            train_test_split, train_size_absolute
        )

        X_hat = detemporalize(generated_data)
        X_hat_no_transf = self.scaler_train.inverse_transform(X_hat)

        x_hat_long = self.create_dataset_long_form(X_hat, train_ids)
        x_hat_long_no_transf = self.create_dataset_long_form(X_hat_no_transf, train_ids)

        X_hat_train_long = x_hat_long[x_hat_long["unique_id"].isin(train_ids)]
        X_hat_train_long_no_transf = x_hat_long_no_transf[
            x_hat_long_no_transf["unique_id"].isin(train_ids)
        ]

        return X_hat_train_long, X_hat_train_long_no_transf

    @staticmethod
    def temporalize(tensor_2d, window_size):
        shape = tf.shape(tensor_2d)
        output = []

        for idx in range(shape[0] - window_size + 1):
            window = tensor_2d[idx : idx + window_size, :]
            output.append(window)

        output = tf.stack(output)

        return output

    @staticmethod
    def inverse_transform(data, scaler):
        if not scaler:
            return data
        # Reshape from (samples, timesteps, features) to (samples*timesteps, features)
        original_shape = data.shape
        data_reshaped = data.reshape(-1, original_shape[-1])
        data_inverse = scaler.inverse_transform(data_reshaped)
        return data_inverse.reshape(original_shape)

    def generate_new_datasets(
        self,
        cvae: CVAE,
        z_mean: np.ndarray,
        z_log_var: np.ndarray,
        transformation: Optional[str] = None,
        transf_param: List[float] = None,
        n_versions: int = 6,
        n_samples: int = 10,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate new datasets using the CVAE trained model and different samples from its latent space.

        Args:
            cvae: A trained Conditional Variational Autoencoder (CVAE) model.
            z_mean: Mean parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            z_log_var: Log variance parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            transformation: Transformation to apply to the data, if any.
            transf_param: Parameter for the transformation.
            n_versions: Number of versions of the dataset to create.
            n_samples: Number of samples of the dataset to create.
            save: If True, the generated datasets are stored locally.

        Returns:
            An array containing the new generated datasets.
        """
        if transf_param is None:
            transf_param = [0.5, 2, 4, 10, 20, 50]
        y_new = np.zeros((n_versions, n_samples, self.n, self.s))
        s = 0
        for v in range(1, n_versions + 1):
            for s in range(1, n_samples + 1):
                y_new[v - 1, s - 1] = self.generate_transformed_time_series(
                    cvae=cvae,
                    z_mean=z_mean,
                    z_log_var=z_log_var,
                    transformation=transformation,
                    transf_param=transf_param[v - 1],
                )
            if save:
                self._save_version_file(y_new[v - 1], v, s, "vae")
        return y_new
