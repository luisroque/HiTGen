import os
import gc
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import optuna
from tensorflow import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from hitgen.model.models import (
    CVAE,
    TemporalizeGenerator,
)
from hitgen.feature_engineering.feature_transformations import (
    detemporalize,
)
from hitgen.model.models import get_CVAE
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
)
from hitgen.load_data.config import DATASETS
from hitgen.visualization.model_visualization import (
    plot_generated_vs_original,
)


class InvalidFrequencyError(Exception):
    pass


class CreateTransformedVersionsCVAE:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).
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
        shuffle: bool = False,
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
        bi_rnn: bool = False,
        forecasting: bool = True,
        conv1d_blocks_backcast=2,
        filters_backcast=64,
        kernel_size_backcast=3,
        conv1d_blocks_forecast=2,
        filters_forecast=64,
        kernel_size_forecast=3,
        kl_weight_init: float = 0.1,
        noise_scale_init: float = 0.1,
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
        self.freq = freq
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bi_rnn = bi_rnn
        self.forecasting = forecasting
        self.kl_weight_init = kl_weight_init
        self.noise_scale_init = noise_scale_init
        self.n_blocks_encoder = n_blocks_encoder
        self.n_blocks_decoder = n_blocks_decoder
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.patience = patience
        self.kernel_size = kernel_size
        self.pooling_mode = pooling_mode
        self.conv1d_blocks_backcast = conv1d_blocks_backcast
        self.filters_backcast = filters_backcast
        self.kernel_size_backcast = kernel_size_backcast
        self.conv1d_blocks_forecast = conv1d_blocks_forecast
        self.filters_forecast = filters_forecast
        self.kernel_size_forecast = kernel_size_forecast
        (self.data, self.s, self.freq) = self.load_data(
            self.dataset_name, self.dataset_group
        )
        self.s_train = None
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
        self.long_properties = {}
        self.split_path = f"assets/model_weights/data_split/{dataset_name}_{dataset_group}_data_split.json"
        self.unique_ids = self.df["unique_id"].unique()

        feature_dict = self._feature_engineering()

        self.original_wide = feature_dict["original_wide"]
        self.original_train_wide = feature_dict["train_wide"]
        self.original_test_wide = feature_dict["test_wide"]

        self.original_long = feature_dict["original_long"]
        self.original_train_long = feature_dict["train_long"]
        self.original_test_long = feature_dict["test_long"]

        self.mask_wide = feature_dict["mask_wide"]
        self.mask_train_wide = feature_dict["mask_train_wide"]
        self.mask_test_wide = feature_dict["mask_test_wide"]

        self.original_long_transf = feature_dict["original_long_transf"]
        self.original_train_long_transf = feature_dict["original_train_long_transf"]
        self.original_test_long_transf = feature_dict["original_test_long_transf"]

        self.original_wide_transf = feature_dict["original_wide_transf"]
        self.original_train_wide_transf = feature_dict["original_train_wide_transf"]
        self.original_test_wide_transf = feature_dict["original_test_wide_transf"]

        self.original_dyn_features = feature_dict["fourier_features_original"]
        self.train_dyn_features = feature_dict["fourier_features_train"]
        self.test_dyn_features = feature_dict["fourier_features_test"]

    @staticmethod
    def load_data(dataset_name: str, group: str) -> Tuple[pd.DataFrame, int, str]:
        data_cls = DATASETS[dataset_name]
        print(dataset_name, group)

        try:
            ds = data_cls.load_data(group)

            # for testing purposes only
            ds = ds[ds["unique_id"].isin(ds["unique_id"].unique()[:20])]
        except FileNotFoundError as e:
            print(f"Error loading data for {dataset_name} - {group}: {e}")

        freq = data_cls.frequency_pd[group]
        n_series = int(ds.nunique()["unique_id"])
        return ds, n_series, freq

    def create_dataset_long_form(self, data, original, unique_ids=None) -> pd.DataFrame:
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
        data_long = data_long.sort_values(by=["unique_id", "ds"])

        # keep only values that exist in the original
        data_long = data_long.merge(
            original[["unique_id", "ds"]], on=["unique_id", "ds"], how="inner"
        )

        return data_long

    @staticmethod
    def _create_dataset_wide_form(
        data_long: pd.DataFrame, ids: List[str], fill_nans: bool = True
    ) -> pd.DataFrame:
        """
        Transforms a long-form dataset back to wide form ensuring the
        right order of the columns
        """
        if not {"ds", "unique_id", "y"}.issubset(data_long.columns):
            raise ValueError(
                "Input DataFrame must contain 'ds', 'unique_id', and 'y' columns."
            )

        data_long = data_long.sort_values(by=["unique_id", "ds"])
        data_wide = data_long.pivot(index="ds", columns="unique_id", values="y")
        data_wide = data_wide.reindex(columns=ids)
        if fill_nans:
            data_wide = data_wide.fillna(0)

        return data_wide

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
        self, df: pd.DataFrame, ids: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Sort and preprocess the data for feature engineering."""
        x_wide = self._create_dataset_wide_form(df, ids, fill_nans=False)

        self.long_properties["ds"] = x_wide.reset_index()["ds"].values
        self.long_properties["unique_id"] = x_wide.columns.values

        # create mask before padding
        mask = (~x_wide.isna()).astype(int)

        # padding
        x_wide_filled = x_wide.fillna(0.0)
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(x_wide_filled)

        return scaled_data, mask, x_wide_filled

    def _set_scaler_train(self, df: pd.DataFrame):
        """Sort and preprocess the data for feature engineering."""
        # padding
        x_wide_filled = df.fillna(0.0)
        self.scaler_train = StandardScaler()
        scaled_data_train = self.scaler_train.fit_transform(x_wide_filled)

    def _feature_engineering(
        self, train_test_split=0.7, train_size_absolute=None
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply preprocessing to raw time series, split into training and testing,
        compute periodic Fourier features, and return all relevant DataFrames.
        """
        self.ids = self.df["unique_id"].unique().sort()
        original_wide_transf, mask_original_wide, original_wide = self._preprocess_data(
            self.df, self.ids
        )

        original_long = self.create_dataset_long_form(original_wide, self.df)
        original_long_transf = self.create_dataset_long_form(
            original_wide_transf, self.df
        )
        mask_original_long = self.create_dataset_long_form(mask_original_wide, self.df)

        self.train_ids, self.test_ids = self._load_or_create_split(
            train_test_split, train_size_absolute
        )
        self.train_ids.sort()
        self.test_ids.sort()
        self.s_train = len(self.train_ids)

        # create training dfs
        train_long = original_long[original_long["unique_id"].isin(self.train_ids)]
        train_long_transf = original_long_transf[
            original_long_transf["unique_id"].isin(self.train_ids)
        ]
        mask_train_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.train_ids)
        ]

        # convert training long -> wide
        train_wide = self._create_dataset_wide_form(train_long, self.train_ids)
        train_wide_transf = self._create_dataset_wide_form(
            train_long_transf, self.train_ids
        )
        mask_train_wide = self._create_dataset_wide_form(
            mask_train_long, self.train_ids
        )

        # set training scaler
        self._set_scaler_train(train_wide)

        test_long = original_long[original_long["unique_id"].isin(self.test_ids)]
        test_long_transf = original_long_transf[
            original_long_transf["unique_id"].isin(self.test_ids)
        ]
        mask_test_long = mask_original_long[
            mask_original_long["unique_id"].isin(self.test_ids)
        ]

        # convert testing long -> wide
        test_wide = self._create_dataset_wide_form(test_long, self.train_ids)
        test_wide_transf = self._create_dataset_wide_form(
            test_long_transf, self.train_ids
        )
        mask_test_wide = self._create_dataset_wide_form(mask_test_long, self.train_ids)

        # convert mask to tensors
        self.mask_train_tf = tf.convert_to_tensor(
            mask_train_wide.values, dtype=tf.float32
        )
        self.mask_test_tf = tf.convert_to_tensor(
            mask_test_wide.values, dtype=tf.float32
        )
        self.mask_original_tf = tf.convert_to_tensor(
            mask_original_wide.values, dtype=tf.float32
        )

        self.X_train_raw = train_wide.reset_index(drop=True)
        self.X_test_raw = test_wide.reset_index(drop=True)
        self.X_orig_raw = original_wide.reset_index(drop=True)

        def compute_fourier_features(dates, n):
            """
            Compute Fourier terms for a given frequency.
            `self.freq` must be an attribute like 'D', 'W', 'M', etc.
            """
            t = dates.astype(np.int64) / 10**9
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
            return pd.DataFrame(features, index=dates)

        fourier_features_train = compute_fourier_features(train_wide.index, 3)
        fourier_features_test = compute_fourier_features(test_wide.index, 3)
        fourier_features_original = compute_fourier_features(original_wide.index, 3)

        return {
            # wide
            "original_wide": original_wide,
            "train_wide": train_wide,
            "test_wide": test_wide,
            # long
            "original_long": original_long,
            "train_long": train_long,
            "test_long": test_long,
            # mask wide
            "mask_train_wide": mask_train_wide,
            "mask_test_wide": mask_test_wide,
            "mask_wide": mask_original_wide,
            # transformed long
            "original_long_transf": original_long_transf,
            "original_train_long_transf": train_long_transf,
            "original_test_long_transf": test_long_transf,
            # wide long
            "original_wide_transf": original_wide_transf,
            "original_train_wide_transf": train_wide_transf,
            "original_test_wide_transf": test_wide_transf,
            # fourier features
            "fourier_features_train": fourier_features_train,
            "fourier_features_test": fourier_features_test,
            "fourier_features_original": fourier_features_original,
        }

    def fit(
        self,
        window_size: int,
        epochs: int = 750,
        patience: int = 30,
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        hyper_tuning: bool = False,
        load_weights: bool = True,
    ) -> tuple[CVAE, dict, EarlyStopping]:
        """Training our CVAE"""

        data_mask_temporalized = TemporalizeGenerator(
            self.original_wide_transf,
            self.mask_wide,
            self.original_dyn_features,
            window_size=window_size,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

        encoder, decoder = get_CVAE(
            window_size=window_size,
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
            conv1d_blocks_backcast=self.conv1d_blocks_backcast,
            filters_backcast=self.filters_backcast,
            kernel_size_backcast=self.kernel_size_backcast,
            conv1d_blocks_forecast=self.conv1d_blocks_forecast,
            filters_forecast=self.filters_forecast,
            kernel_size_forecast=self.kernel_size_forecast,
        )

        cvae = CVAE(
            encoder,
            decoder,
            kl_weight_initial=self.kl_weight_init,
            forecasting=self.forecasting,
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
                # clipnorm=1.0,
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

        dummy_input = (
            tf.random.normal((1, window_size, self.s)),  # batch_data
            tf.ones((1, window_size, self.s)),  # batch_mask
            tf.random.normal((1, window_size, 6)),  # batch_dyn_features
        )
        _ = cvae(dummy_input)

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
        conv1d_blocks_backcast,
        filters_backcast,
        kernel_size_backcast,
        conv1d_blocks_forecast,
        filters_forecast,
        kernel_size_forecast,
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
            "conv1d_blocks_backcast": conv1d_blocks_backcast,
            "filters_backcast": filters_backcast,
            "kernel_size_backcast": kernel_size_backcast,
            "conv1d_blocks_forecast": conv1d_blocks_forecast,
            "filters_forecast": filters_forecast,
            "kernel_size_forecast": kernel_size_forecast,
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
        if self.freq == "M" or self.freq == "MS":
            window_size = trial.suggest_int("window_size", 3, 24, step=3)
        elif self.freq == "Q" or self.freq == "QS":
            window_size = trial.suggest_int("window_size", 4, 12, step=2)
        elif self.freq == "Y" or self.freq == "YS":
            window_size = trial.suggest_int("window_size", 2, 6, step=1)
        else:
            window_size = trial.suggest_int("window_size", 4, 24, step=1)
        patience = trial.suggest_int("patience", 20, 40, step=5)
        kl_weight = trial.suggest_float("kl_weight", 0.05, 0.5)
        n_blocks_encoder = trial.suggest_int("n_blocks_encoder", 1, 5)
        n_blocks_decoder = trial.suggest_int("n_blocks_decoder", 1, 5)
        n_hidden = trial.suggest_int("n_hidden", 16, 128, step=16)
        n_layers = trial.suggest_int("n_layers", 1, 5)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        pooling_mode = trial.suggest_categorical("pooling_mode", ["max", "average"])
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
        epochs = trial.suggest_int("epochs", 5, 10, step=5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        # bi_rnn = trial.suggest_categorical("bi_rnn", [True, False])
        # forecasting = trial.suggest_categorical("forecasting", [True, False])
        # shuffle = trial.suggest_categorical("shuffle", [True, False])
        noise_scale_init = trial.suggest_float("noise_scale_init", 0.01, 0.5)
        conv1d_blocks_backcast = trial.suggest_int(
            "conv1d_blocks_backcast", 1, 5, step=1
        )
        filters_backcast = trial.suggest_int("filters_backcast", 16, 256, step=16)
        kernel_size_backcast = trial.suggest_int("kernel_size_backcast", 2, 4, step=1)
        conv1d_blocks_forecast = trial.suggest_int(
            "conv1d_blocks_forecast", 1, 5, step=1
        )
        filters_forecast = trial.suggest_int("filters_forecast", 16, 256, step=16)
        kernel_size_forecast = trial.suggest_int("kernel_size_forecast", 2, 4, step=1)

        bi_rnn = False
        shuffle = False
        forecasting = True

        data_mask_temporalized = TemporalizeGenerator(
            self.original_train_wide_transf,
            self.mask_train_wide,
            self.train_dyn_features,
            window_size=window_size,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        encoder, decoder = get_CVAE(
            window_size=window_size,
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
            conv1d_blocks_backcast=conv1d_blocks_backcast,
            filters_backcast=filters_backcast,
            kernel_size_backcast=kernel_size_backcast,
            conv1d_blocks_forecast=conv1d_blocks_forecast,
            filters_forecast=filters_forecast,
            kernel_size_forecast=kernel_size_forecast,
        )

        cvae = CVAE(
            encoder, decoder, kl_weight_initial=kl_weight, forecasting=forecasting
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=learning_rate,
            ),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        es = EarlyStopping(
            patience=self.patience,
            verbose=1,
            monitor="loss",
            mode="auto",
            restore_best_weights=True,
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.2, patience=10, min_lr=1e-6, cooldown=3, verbose=1
        )

        history = cvae.fit(
            x=data_mask_temporalized,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[es, reduce_lr],
        )

        loss = min(history.history["loss"])

        synthetic_long = self.predict_train(
            cvae,
            data_mask_temporalized=data_mask_temporalized,
        )

        # compute the discriminative score x times to account for variability

        score = self.compute_mean_discriminative_score(
            unique_ids=self.original_long["unique_id"].unique(),
            original_data=self.original_long,
            synthetic_data=synthetic_long,
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
            original_data=self.original_long,
            synthetic_data=synthetic_long,
            score=score,
            latent_dim=latent_dim,
            window_size=window_size,
            patience=patience,
            kl_weight=kl_weight,
            n_blocks_encoder=n_blocks_encoder,
            n_blocks_decoder=n_blocks_decoder,
            n_hidden=n_hidden,
            n_layers=n_layers,
            kernel_size=kernel_size,
            pooling_mode=pooling_mode,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            bi_rnn=bi_rnn,
            shuffle=shuffle,
            forecasting=forecasting,
            conv1d_blocks_backcast=conv1d_blocks_backcast,
            filters_backcast=filters_backcast,
            kernel_size_backcast=kernel_size_backcast,
            conv1d_blocks_forecast=conv1d_blocks_forecast,
            filters_forecast=filters_forecast,
            kernel_size_forecast=kernel_size_forecast,
            noise_scale_init=noise_scale_init,
            loss=loss,
        )

        # clean GPU memory between trials
        del cvae, encoder, decoder
        tf.keras.backend.clear_session()
        gc.collect()

        return score

        # except Exception as e:
        #     print(f"Error in trial: {e}")
        #     raise optuna.exceptions.TrialPruned()

    def hyper_tune_and_train(self, n_trials=30):
        """
        Run Optuna hyperparameter tuning for the CVAE and train the best model.
        """
        study_name = f"{self.dataset_name}_{self.dataset_group}_opt_vae"

        storage = "sqlite:///optuna_study.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,
        )

        if len(study.trials) == 0:
            print("No trials have been completed yet. Running hyperparameter tuning...")
            study.optimize(self.objective, n_trials=n_trials)

        try:
            best_trial = study.best_trial
        except ValueError:
            print(
                "No best trial found, likely due to no successful trials. Rerunning optimization..."
            )
            study.optimize(self.objective, n_trials=n_trials)
            best_trial = study.best_trial

        self.best_params = best_trial.params

        self.best_params["bi_rnn"] = False
        self.best_params["shuffle"] = False
        self.best_params["forecasting"] = True

        with open(
            f"assets/model_weights/{self.dataset_name}_{self.dataset_group}_best_params.json",
            "w",
        ) as f:
            json.dump(self.best_params, f)

        print(f"Best Hyperparameters: {self.best_params}")

        data_mask_temporalized = TemporalizeGenerator(
            self.original_wide_transf,
            self.mask_wide,
            self.original_dyn_features,
            window_size=self.best_params["window_size"],
            batch_size=self.best_params["batch_size"],
            shuffle=self.best_params["shuffle"],
        )

        encoder, decoder = get_CVAE(
            window_size=self.best_params["window_size"],
            n_series=self.s,
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
            conv1d_blocks_backcast=self.best_params["conv1d_blocks_backcast"],
            filters_backcast=self.best_params["filters_backcast"],
            kernel_size_backcast=self.best_params["kernel_size_backcast"],
            conv1d_blocks_forecast=self.best_params["conv1d_blocks_forecast"],
            filters_forecast=self.best_params["filters_forecast"],
            kernel_size_forecast=self.best_params["kernel_size_forecast"],
        )

        cvae = CVAE(
            encoder,
            decoder,
            kl_weight_initial=self.best_params["kl_weight"],
            forecasting=self.best_params["forecasting"],
        )
        cvae.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.best_params["learning_rate"]
            ),
            metrics=[cvae.reconstruction_loss_tracker, cvae.kl_loss_tracker],
        )

        # final training with best parameters
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=self.best_params["patience"],
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

        dummy_input = (
            tf.random.normal(
                (1, self.best_params["window_size"], self.s)
            ),  # batch_data
            tf.ones((1, self.best_params["window_size"], self.s)),  # batch_mask
            tf.random.normal(
                (1, self.best_params["window_size"], 6)
            ),  # batch_dyn_features
        )
        _ = cvae(dummy_input)

        if os.path.exists(weights_file):
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
                epochs=self.best_params["epochs"],
                batch_size=self.best_params["batch_size"],
                shuffle=False,
                callbacks=[early_stopping, mc, reduce_lr],
            )

            if history is not None:
                history = history.history
                history_dict = {
                    key: [float(val) for val in values]
                    for key, values in history.items()
                }
                with open(history_file, "w") as f:
                    json.dump(history_dict, f)

        print("Training completed with the best hyperparameters.")
        return cvae

    def predict(
        self,
        cvae: CVAE,
        data_mask_temporalized,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Predict original time series using VAE"""

        z_mean, z_log_var, z = cvae.encoder.predict(
            [
                data_mask_temporalized.temporalized_data,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )
        alpha = 0  # x times bigger variance
        epsilon = np.random.normal(size=z_mean.shape) * 0.1
        z_augmented = z_mean + np.exp(0.5 * z_log_var) * alpha * epsilon
        generated_data, predictions = cvae.decoder.predict(
            [
                z_augmented,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )

        X_hat_wide_transf = detemporalize(generated_data)
        X_hat_wide = self.scaler.inverse_transform(X_hat_wide_transf)

        X_hat_long = self.create_dataset_long_form(X_hat_wide, self.original_long)

        X_hat_train_long = X_hat_long[X_hat_long["unique_id"].isin(self.train_ids)]
        X_hat_test_long = X_hat_long[X_hat_long["unique_id"].isin(self.test_ids)]

        return (
            X_hat_long,
            X_hat_train_long,
            X_hat_test_long,
        )

    def predict_train(
        self,
        cvae: CVAE,
        data_mask_temporalized,
    ) -> pd.DataFrame:
        """Predict original time series using VAE"""

        z_mean, z_log_var, z = cvae.encoder.predict(
            [
                data_mask_temporalized.temporalized_data,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )
        alpha = 0  # x times bigger variance
        epsilon = np.random.normal(size=z_mean.shape) * 0.1
        z_augmented = z_mean + np.exp(0.5 * z_log_var) * alpha * epsilon
        generated_data, predictions = cvae.decoder.predict(
            [
                z_augmented,
                data_mask_temporalized.temporalized_mask,
                data_mask_temporalized.temporalized_dyn_features,
            ]
        )

        X_hat_transf = detemporalize(generated_data)
        X_hat = self.scaler_train.inverse_transform(X_hat_transf)

        x_hat_train_long = self.create_dataset_long_form(
            X_hat, self.original_train_long, self.train_ids
        )

        return x_hat_train_long
