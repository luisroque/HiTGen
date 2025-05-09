import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Optional


base_path_load = "assets/results_forecast_out_domain"
path_fm_moirai = "assets/results_forecast_out_domain_summary/moirai_results.csv"
results_files_out_domain = [
    f for f in os.listdir(base_path_load) if f.endswith(".json")
]

results_combined = []
for result in results_files_out_domain:
    with open(os.path.join(base_path_load, result), "r") as f:
        result_details = json.load(f)

    results_combined.append(result_details)

results_df = pd.DataFrame(results_combined).reset_index(drop=True)

results_filtered = results_df[
    (results_df["Dataset Source"] != results_df["Dataset"])
].copy()

results_filtered.rename(
    columns={
        "Forecast SMAPE MEAN (last window) Per Series_out_domain": "SMAPE Mean",
        "Forecast MASE MEAN (last window) Per Series_out_domain": "MASE Mean",
        "Forecast MAE MEAN (last window) Per Series_out_domain": "MAE Mean",
        "Forecast RMSE MEAN (last window) Per Series_out_domain": "RMSE Mean",
        "Forecast RMSSE MEAN (last window) Per Series_out_domain": "RMSSE Mean",
        "Dataset": "Dataset Target",
        "Group": "Dataset Group Target",
    },
    inplace=True,
)
results_filtered = results_filtered[
    [
        "Dataset Source",
        "Dataset Group Source",
        "Dataset Target",
        "Dataset Group Target",
        "Method",
        "SMAPE Mean",
        "MASE Mean",
        "MAE Mean",
        "RMSE Mean",
        "RMSSE Mean",
    ]
]

results_filtered_coreset = results_filtered[
    (results_filtered["Dataset Source"] == "MIXED")
].copy()

moirai_results = pd.read_csv(path_fm_moirai)

# union of results + moirai results
required_columns = results_filtered_coreset.columns

# add missing columns with NaN to Moirai df
for col in required_columns:
    if col not in moirai_results.columns:
        moirai_results[col] = pd.NA

moirai_results["Dataset Source"] = "MIXED"
moirai_results["Dataset Group Source"] = moirai_results.apply(
    lambda x: f"ALL_BUT_{x['Dataset Target']}_{x['Dataset Group Target']}", axis=1
)

# reorder columns to match the target df
moirai_df = moirai_results[required_columns]

results_filtered_coreset = pd.concat(
    [results_filtered_coreset, moirai_df], ignore_index=True
)

results_filtered_coreset["Source-Target Pair"] = (
    results_filtered_coreset["Dataset Source"]
    + " ("
    + results_filtered_coreset["Dataset Group Source"]
    + ") → "
    + results_filtered_coreset["Dataset Target"]
    + " ("
    + results_filtered_coreset["Dataset Group Target"]
    + ")"
)

results_filtered_coreset = results_filtered_coreset.loc[
    (results_filtered_coreset["Method"] != "AutoHiTGenDeepMixtureTempNormLossNorm")
]

results_filtered = results_filtered[
    results_filtered["Dataset Source"] != "MIXED"
].copy()

results_filtered["Source-Target Pair"] = (
    results_filtered["Dataset Source"]
    + " ("
    + results_filtered["Dataset Group Source"]
    + ") → "
    + results_filtered["Dataset Target"]
    + " ("
    + results_filtered["Dataset Group Target"]
    + ")"
)


def summarize_metric(
    df: pd.DataFrame,
    metric: str,
    mode: str,
    aggregate_by: List[str],
    rank_within: Optional[List[str]] = None,
    filter_same_seasonality: bool = False,
    src_seas_col: str = "Dataset Group Source",
    tgt_seas_col: str = "Dataset Group Target",
    out_path: Optional[Path] = None,
    fname: str | None = None,
    rank_method: str = "min",
    agg_func=np.nanmean,
) -> pd.DataFrame:
    """
    Generic summary / ranking utility for the forecast results grid.

    Parameters
        df  : cleaned results table
        metric : column to aggregate (e.g. "SMAPE Mean")
        mode : "rank" | "mean"
        rank_within : list of columns that define the grouping within which to rank.
                      Ignored when `mode == "mean"`.
        aggregate_by : final grouping columns for the summary table
        filter_same_seasonality : keep only rows where src & tgt seasonalities match
        out_path : directory to write csv (if None → just return df)
    """
    work = df.copy()

    if filter_same_seasonality:
        work = work[work[src_seas_col] == work[tgt_seas_col]]

    if mode == "rank":
        if not rank_within:
            raise ValueError("`rank_within` must be given when mode='rank'")
        work["Rank"] = work.groupby(rank_within)[metric].rank(method=rank_method)
        summary = (
            work.groupby(aggregate_by)["Rank"]
            .apply(agg_func)
            .reset_index()
            .rename(columns={"Rank": "Rank"})
        )
        if aggregate_by == ["Method"]:
            sort_by = ["Rank"]
        else:
            sort_by = aggregate_by + ["Rank"]
        summary.sort_values(by=sort_by, inplace=True)

    elif mode == "mean":
        summary = (
            work.groupby(aggregate_by)[metric]
            .apply(agg_func)
            .reset_index()
            .rename(columns={metric: metric})
        )
        if aggregate_by == ["Method"]:
            sort_by = [metric]
        else:
            sort_by = aggregate_by + [metric]
        summary.sort_values(by=sort_by, inplace=True)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        if fname is None:
            stem = "_".join(aggregate_by)
            fname = f"{mode}_{metric.replace(' ','_').lower()}_{stem}.csv"
        summary.to_csv(out_path / fname, index=False)

    return summary


base_path = Path("assets/results_forecast_out_domain_summary")
out_dir = base_path
metric = "MASE Mean"
metric_store = f"{metric.replace(' ','_').lower()}"

for results, suffix, filter_same_seasonality in zip(
    [results_filtered, results_filtered_coreset], ["", "_coreset"], [False, False]
):

    # rank – all seasonalities, grouped by src-dataset & method
    summarize_metric(
        results,
        metric=metric,
        mode="rank",
        rank_within=["Source-Target Pair"],
        filter_same_seasonality=filter_same_seasonality,
        aggregate_by=["Dataset Source", "Dataset Group Source", "Method"],
        out_path=out_dir,
        fname=f"results_ranks_all_seasonalities{suffix}.csv",
    )

    # mean SMAPE – same grouping
    summarize_metric(
        results,
        metric=metric,
        mode="mean",
        filter_same_seasonality=filter_same_seasonality,
        aggregate_by=["Dataset Source", "Dataset Group Source", "Method"],
        out_path=out_dir,
        fname=f"results_all_seasonalities_{metric_store}{suffix}.csv",
    )

    # mean SMAPE – every individual src–tgt pair
    summarize_metric(
        results,
        metric=metric,
        mode="mean",
        aggregate_by=[
            "Dataset Source",
            "Dataset Group Source",
            "Dataset Target",
            "Dataset Group Target",
            "Method",
        ],
        filter_same_seasonality=filter_same_seasonality,
        out_path=out_dir,
        fname=f"results_all_seasonalities_all_combinations_{metric_store}{suffix}.csv",
    )

    # rank – by Method only (all seasonalities)
    summarize_metric(
        results,
        metric=metric,
        mode="rank",
        rank_within=["Source-Target Pair"],
        aggregate_by=["Method"],
        filter_same_seasonality=filter_same_seasonality,
        out_path=out_dir,
        fname=f"results_ranks_all_seasonalities_by_method_{metric_store}{suffix}.csv",
    )

    # mean SMAPE – by Method only
    summarize_metric(
        results,
        metric=metric,
        mode="mean",
        aggregate_by=["Method"],
        filter_same_seasonality=filter_same_seasonality,
        out_path=out_dir,
        fname=f"results_all_seasonalities_by_method_{metric_store}{suffix}.csv",
    )

    # # rank & mean restricted to same seasonality transfers
    # for m in ("rank", "mean"):
    #     summarize_metric(
    #         results_filtered,
    #         metric=metric,
    #         mode=m,
    #         rank_within=None if m == "mean" else ["Source-Target Pair"],
    #         aggregate_by=(
    #             ["Dataset Source", "Dataset Group Source", "Method"]
    #             if m == "rank"
    #             else ["Method"]
    #         ),
    #         filter_same_seasonality=True,
    #         out_path=out_dir,
    #         fname=f"results_{m}_same_seasonalities_{'by_method' if m=='mean' else 'by_source'}_{metric_store}{suffix}.csv",
    #     )
