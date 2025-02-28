import os
import pandas as pd
import pickle
from metaforecast.synth import (
    DBA,
    Jittering,
    Scaling,
    MagnitudeWarping,
    TimeWarping,
    SeasonalMBB,
    TSMixup,
)


def workflow_metaforecast_methods(
    df: pd.DataFrame, freq: str, dataset: str, dataset_group: str
) -> pd.DataFrame:
    """
    Applies multiple synthetic/augmentation techniques to a time-series dataset.
    """
    cache_dir = "assets/model_weights/metaforecast/"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"{dataset}_{dataset_group}_synthetic_metaforecast.pkl"
    )

    if os.path.exists(cache_path):
        print(f"Loading cached synthetic dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    FREQS = {"H": 24, "D": 1, "M": 12, "Q": 4, "W": 1, "Y": 1}
    if freq not in FREQS:
        raise ValueError(
            f"Unsupported frequency '{freq}'. Valid options: {list(FREQS.keys())}"
        )
    per = FREQS[freq]

    print("Starting to generate synthetic series using multiple methods...")
    print(
        "Methods used for this generation: DBA, Jitter, Scaling, MagWarp, TimeWarp, MBB, TSMixup"
    )

    df.sort_values(by=["unique_id", "ds"], inplace=True)

    df_unique_ids = df["unique_id"].unique()

    dba_synth = DBA(max_n_uids=10)
    df_dba = dba_synth.transform(df)
    df_dba["method"] = "DBA"
    id_mapping = dict(zip(df_dba["unique_id"].unique(), df_unique_ids))
    df_dba["unique_id"] = df_dba["unique_id"].map(id_mapping)

    jitter_synth = Jittering()
    df_jitter = jitter_synth.transform(df)
    df_jitter["method"] = "Jitter"
    id_mapping = dict(zip(df_jitter["unique_id"].unique(), df_unique_ids))
    df_jitter["unique_id"] = df_jitter["unique_id"].map(id_mapping)

    scaling_synth = Scaling()
    df_scaling = scaling_synth.transform(df)
    df_scaling["method"] = "Scaling"
    df_scaling["unique_id"] = df["unique_id"]
    id_mapping = dict(zip(df_scaling["unique_id"].unique(), df_unique_ids))
    df_scaling["unique_id"] = df_scaling["unique_id"].map(id_mapping)

    magwarp_synth = MagnitudeWarping()
    df_magwarp = magwarp_synth.transform(df)
    df_magwarp["method"] = "MagWarp"
    id_mapping = dict(zip(df_magwarp["unique_id"].unique(), df_unique_ids))
    df_magwarp["unique_id"] = df_magwarp["unique_id"].map(id_mapping)

    timewarp_synth = TimeWarping()
    df_timewarp = timewarp_synth.transform(df)
    df_timewarp["method"] = "TimeWarp"
    id_mapping = dict(zip(df_timewarp["unique_id"].unique(), df_unique_ids))
    df_timewarp["unique_id"] = df_timewarp["unique_id"].map(id_mapping)

    mbb_synth = SeasonalMBB(seas_period=per)
    df_mbb = mbb_synth.transform(df)
    df_mbb["method"] = "MBB"
    id_mapping = dict(zip(df_mbb["unique_id"].unique(), df_unique_ids))
    df_mbb["unique_id"] = df_mbb["unique_id"].map(id_mapping)

    # TODO: TS_MIXUP NEEDS TO BE UPDATED TO CREATE SERIES WITH SIMILAR LENGTH AS THE ORIGINAL ONES
    # ts_mixup = TSMixup(max_n_uids=7, min_len=50, max_len=96)
    # df_mixup = ts_mixup.transform(df)
    # df_mixup["method"] = "TSMixup"
    # df_mixup["unique_id"] = df["unique_id"]

    df_synthetic = pd.concat(
        [
            df,
            df_dba,
            df_jitter,
            df_scaling,
            df_magwarp,
            df_timewarp,
            df_mbb,
            # df_mixup,
        ],
        ignore_index=True,
    )

    df_synthetic = df_synthetic.sort_values(by=["unique_id", "ds"]).reset_index(
        drop=True
    )

    print("Synthetic series generation completed.")

    with open(cache_path, "wb") as f:
        pickle.dump(df_synthetic, f)
    print(f"Synthetic dataset saved to {cache_path}")

    return df_synthetic
