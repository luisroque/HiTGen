import pandas as pd
import pickle
from .utils import generate_groups_data_flat, generate_groups_data_matrix
from urllib import request
from pathlib import Path
import os
import zipfile
import numpy as np
import datetime
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import hf_hub_download


class PreprocessDatasets:
    """
    A class used to preprocess datasets

    ...

    Attributes
    ----------
    dataset : str
        the dataset to download and preprocess
    freq : str
        frequency of the time series (e.g., 'D' for daily, 'W' for weekly, etc.)
    input_dir : str
        input directory where the dataset is located
    top : int
        number of top series to filter from the dataset
    test_size : int
        size of the test set
    sample_perc : float
        percentage of samples to use in the dataset
    weekly_m5 : bool
        whether to convert the M5 dataset to weekly data
    num_base_series_time_points : int
        number of time points in the base series for synthetic data
    num_latent_dim : int
        number of dimensions in the latent space for synthetic data
    num_variants : int
        number of variants to generate for synthetic data
    noise_scale : float
        scale of the noise for synthetic data
    amplitude : float
        amplitude of the seasonality component for synthetic data
    """

    def __init__(
        self,
        dataset: str,
        freq: str,
        input_dir: str = "./assets/",
        top: int = 500,
        test_size: int = None,
        sample_perc: float = None,
        weekly_m5: bool = True,
        num_base_series_time_points: int = 100,
        num_latent_dim: int = 3,
        num_variants: int = 20,
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
    ) -> None:
        self.weekly = weekly_m5
        if dataset == "m5":
            dataset = dataset.capitalize()
        self.dataset = dataset
        self.freq = freq
        self.input_dir = input_dir
        self.num_base_series_time_points = num_base_series_time_points
        self.num_latent_dim = num_latent_dim
        self.num_variants = num_variants
        self.top = top
        self.test_size = test_size
        self.sample_perc = sample_perc
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        if self.sample_perc is not None and self.sample_perc > 1:
            raise ValueError("sample_perc must be between 0 and 1")
        elif self.sample_perc:
            self.sample_perc_int = int(self.sample_perc * 100)
        else:
            self.sample_perc_int = ""
        self._create_directories()
        self.n = {"prison": 48, "tourism": 228, "m5": 275, "police": 334}

        self.pickle_path = f"{self.input_dir}data/original_datasets/{self.dataset}.pkl/{self.dataset}.pkl"

        self.dataset_mapping = {
            "m5": "zaai-ai/hierarchical_time_series_datasets",
            "police": "zaai-ai/hierarchical_time_series_datasets",
            "prison": "zaai-ai/hierarchical_time_series_datasets",
            "tourism": "zaai-ai/hierarchical_time_series_datasets",
        }

        self._get_dataset_path()

    def _create_directories(self):
        # Create directory to store original datasets if does not exist
        Path(f"{self.input_dir}data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.input_dir}data/original_datasets").mkdir(
            parents=True, exist_ok=True
        )

    @staticmethod
    def generate_time_series(length, num_series):
        scaler = MinMaxScaler()
        series = [
            scaler.fit_transform(np.random.randn(length, 1)) for _ in range(num_series)
        ]
        return series

    def create_base_time_series(
        self, length, num_series, seasonality_period, amplitude
    ):
        base_series = self.generate_time_series(length, num_series)
        for series in base_series:
            t = np.arange(len(series))
            seasonal_component = np.sin(2 * np.pi * t * amplitude / seasonality_period)
            base_series += seasonal_component[:, np.newaxis]
        return base_series

    @staticmethod
    def generate_variants(base_series, num_variants, noise_scale):
        variants = []
        for series in base_series:
            for _ in range(num_variants):
                noise = np.random.normal(scale=noise_scale, size=series.shape)
                variant = series + noise
                variants.append(variant)
        return variants

    @staticmethod
    def _floor(x, freq):
        offset = x[0].ceil(freq) - x[0] + datetime.timedelta(days=-1)
        return (x + offset).floor(freq) - offset

    def load_dataset_from_huggingface(self):
        dataset_id = self.dataset_mapping.get(self.dataset.lower())
        if not dataset_id:
            raise ValueError(f"Dataset {self.dataset} is not supported.")
        # Download the dataset file from Hugging Face Hub
        save_path = f"{self.input_dir}data/original_datasets/{self.dataset.lower()}.pkl"
        _ = hf_hub_download(
            repo_id=dataset_id,
            filename=f"{self.dataset.lower()}.pkl",
            repo_type="dataset",
            local_dir=save_path,
        )
        return save_path

    def _get_dataset_path(self):
        if not os.path.isfile(self.pickle_path):
            try:
                self.load_dataset_from_huggingface()
            except Exception as e:
                print(f"Failed to download the dataset. Error: {e}")
        return

    @staticmethod
    def _load_pickle_file(file_path):
        if os.path.isfile(file_path):
            with open(file_path, "rb") as handle:
                return pickle.load(handle)

    @staticmethod
    def _transform_and_group_stv(stv):
        """Transforms and groups the stv DataFrame."""
        stv = stv.melt(
            list(stv.columns[:6]),
            var_name="day",
            value_vars=list(stv.columns[6:]),
            ignore_index=True,
        )

        stv = (
            stv.groupby(["dept_id", "cat_id", "store_id", "state_id", "item_id", "day"])
            .sum("value")
            .reset_index()
        )

        return stv

    @staticmethod
    def _generate_calendar_days(stv, cal):
        """Generates a DataFrame of calendar days."""
        days_calendar = np.concatenate(
            (
                stv["day"].unique().reshape(-1, 1),
                cal["date"][:-56].unique().reshape(-1, 1),
            ),
            axis=1,
        )
        df_caldays = pd.DataFrame(days_calendar, columns=["day", "Date"])

        return df_caldays

    def _convert_to_weekly_data(self, stv, cols):
        """Converts the stv DataFrame to weekly data."""
        rule = "7D"
        f = self._floor(stv.index, rule)

        cols_group = cols.copy()
        cols_group.append(f)

        stv_weekly = stv.groupby(cols_group).sum()

        return stv_weekly

    def _filter_top_series(self, df, group_columns):
        """Filters the top series from the df DataFrame."""
        sort_column = (
            "value"
            if "value" in df.columns
            else "Count" if "Count" in df.columns else None
        )
        if sort_column is None:
            raise ValueError("Neither 'value' nor 'Count' column found in DataFrame")

        df_top = (
            df.groupby(group_columns)
            .sum()
            .reset_index()
            .sort_values(by=sort_column, ascending=False)
            .head(self.top)
            .drop(sort_column, axis=1)
        ).reset_index(drop=True)

        df_f = pd.merge(
            df_top,
            df.reset_index(drop=True),
            on=group_columns,
            how="left",
        )

        df_f.drop_duplicates(inplace=True)

        return df_f

    def _generate_groups(self, df_pivot, groups_input, seasonality, h):
        """Generates groups from the pivoted DataFrame."""

        groups = generate_groups_data_flat(
            y=df_pivot,
            dates=list(df_pivot.index),
            groups_input=groups_input,
            seasonality=seasonality,
            h=h,
            sample_perc=self.sample_perc,
        )
        groups = generate_groups_data_matrix(groups)

        with open(self.pickle_path, "wb") as handle:
            pickle.dump(groups, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return groups

    def _load_and_preprocess_data(self, path, date_column, drop_columns=None):
        """Loads and preprocesses data from the specified path."""
        if not path:
            return None
        df = pd.read_csv(path, sep=",")
        if self.test_size:
            test_size = self.test_size * self.n[self.dataset]
            df = df[:test_size]
        if drop_columns:
            df = df.drop(drop_columns, axis=1)
        df["Date"] = df[date_column].astype("datetime64[ns]")
        return df

    @staticmethod
    def _pivot_data(df, index, columns, values):
        """Pivots the specified DataFrame."""
        df_pivot = df.pivot(index=index, columns=columns, values=values)
        return df_pivot

    def _prison(self):
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

    def _tourism(self):
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

    def _m5(self):
        """Preprocess the M5 dataset."""
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

    def _police(self):
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

    def _synthetic(
        self,
    ):
        if self.freq == "D":
            seasonality, h = 365, 30
        elif self.freq == "W":
            seasonality, h = 52, 12
        elif self.freq == "M":
            seasonality, h = 12, 2
        elif self.freq == "Q":
            seasonality, h = 4, 2
        else:
            raise ValueError(f"Unsupported frequency: {self.freq}")

        base_series = self.create_base_time_series(
            self.num_base_series_time_points,
            self.num_latent_dim,
            seasonality,
            self.amplitude,
        )
        variants = self.generate_variants(
            base_series,
            self.num_variants,
            self.noise_scale,
        )

        start_date = pd.Timestamp("2023-01-01")
        end_date = start_date + timedelta(days=self.num_base_series_time_points - 1)
        dates = pd.date_range(start_date, end_date, freq=self.freq)

        df = pd.DataFrame(np.concatenate(variants, axis=1), index=dates)

        column_tuples = [
            (f"group_1", f"group_element_{i // self.num_variants + 1}")
            for i in range(self.num_latent_dim * self.num_variants)
        ]
        df.columns = pd.MultiIndex.from_tuples(
            column_tuples, names=["Group", "Element"]
        )

        groups_input = {f"group_1": [1]}

        groups = self._generate_groups(df, groups_input, seasonality, h)
        groups["base_series"] = np.array(base_series)

        with open(self.pickle_path, "wb") as handle:
            pickle.dump(groups, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return groups

    @staticmethod
    def _fill_missing_dates(df, cols):
        """
        Fill missing dates in DataFrame.
        """
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        if "Date" in cols:
            cols.remove("Date")

        df = df.groupby(cols).resample("D").sum().reset_index().sort_values(by="Date")

        df["Count"].fillna(0, inplace=True)

        return df

    def apply_preprocess(self):
        dataset_new = getattr(PreprocessDatasets, "_" + self.dataset.lower())(self)
        return dataset_new
