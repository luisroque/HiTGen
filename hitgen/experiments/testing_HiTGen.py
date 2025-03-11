import multiprocessing
import os
import argparse
import pandas as pd
from hitgen.model.models import (
    TemporalizeGenerator,
)
from hitgen.model.create_dataset_versions_vae import CreateTransformedVersionsCVAE
from hitgen.metrics.discriminative_score import (
    compute_discriminative_score,
    compute_downstream_forecast,
    tstr,
)
from hitgen.visualization import plot_generated_vs_original
from hitgen.benchmarks.metaforecast import workflow_metaforecast_methods

# from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

DATASETS_HYPERPARAMS_CONFIGS = {
    "Tourism": {
        "Monthly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        }
    },
    "M1": {
        "Monthly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        },
        "Quarterly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        },
    },
    "M3": {
        "Monthly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        },
        "Quarterly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        },
        "Yearly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        },
    },
    "M4": {
        "Monthly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        },
        "Quarterly": {
            # "timegan": {
            #     "gan_args": ModelParameters(
            #         batch_size=16,
            #         lr=2e-4,
            #         noise_dim=16,
            #         layers_dim=32,
            #         latent_dim=32,
            #         gamma=1.0,
            #     ),
            #     "train_args": TrainParameters(
            #         epochs=1000,
            #         sequence_length=24,
            #         number_sequences=4,
            #     ),
            # },
        },
    },
}

DATASET_GROUP_FREQ = {
    # "Tourism": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    # },
    # "M1": {
    #     "Monthly": {"FREQ": "M", "H": 24},
    #     "Quarterly": {"FREQ": "Q", "H": 8},
    # },
    "M3": {
        "Monthly": {"FREQ": "M", "H": 24},
        # "Quarterly": {"FREQ": "Q", "H": 8},
        # "Yearly": {"FREQ": "Y", "H": 4},
    },
}


METAFORECAST_METHODS = [
    "DBA",
    "Jitter",
    "Scaling",
    "MagWarp",
    "TimeWarp",
    "MBB",
    # "TSMixup",
]


def extract_frequency(dataset_group):
    """Safely extracts frequency from dataset group."""
    freq = dataset_group[1]["FREQ"]
    return freq


def extract_horizon(dataset_group):
    """Safely extracts horizon from dataset group."""
    h = dataset_group[1]["H"]
    return h


def extract_score(dataset_group):
    """Safely extracts frequency from dataset group."""
    score = dataset_group[1]["final_score"]
    return score


def has_final_score_in_tuple(tpl):
    """Check if the second element is a dictionary and contains 'final_score'"""
    return isinstance(tpl[1], dict) and "final_score" in tpl[1]


def set_device(
    use_gpu: bool,
):
    """Configures TensorFlow to use GPU or CPU."""
    if not use_gpu:
        print("Using CPU as specified by the user.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(
        description="Run synthetic data generation using HiTGen."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU if available (default: False, meaning it runs on CPU).",
    )
    parser.add_argument(
        "--opt-score",
        choices=["discriminative_score", "downstream_score"],
        default="discriminative_score",
        help="Select the score for the hyperparameter tuning optimization. Choices: "
        "'discriminative_score' or 'downstream_score' (default: 'discriminative_score').",
    )
    args = parser.parse_args()

    set_device(args.use_gpu)

    results = []

    for DATASET, SUBGROUPS in DATASET_GROUP_FREQ.items():
        for subgroup in SUBGROUPS.items():
            FREQ = extract_frequency(subgroup)
            H = extract_horizon(subgroup)
            DATASET_GROUP = subgroup[0]
            hitgen_score_disc = None
            if has_final_score_in_tuple(subgroup):
                print(
                    f"Dataset: {DATASET}, Dataset-group: {DATASET_GROUP}, Frequency: {FREQ} "
                    f"has a final score already, skipping HiTGen scores computations..."
                )
                hitgen_score_disc = extract_score(subgroup)

            print(
                f"Dataset: {DATASET}, Dataset-group: {DATASET_GROUP}, Frequency: {FREQ}"
            )
            if (
                DATASET not in DATASETS_HYPERPARAMS_CONFIGS
                or DATASET_GROUP not in DATASETS_HYPERPARAMS_CONFIGS[DATASET]
            ):
                raise ValueError(
                    f"Configuration for {DATASET} - {DATASET_GROUP} not found."
                )

            # dataset_config = DATASETS_HYPERPARAMS_CONFIGS[DATASET][DATASET_GROUP]

            # TIMEGAN Configurations
            # timegan_config = dataset_config["timegan"]
            # gan_args = timegan_config["gan_args"]
            # train_args = timegan_config["train_args"]

            SYNTHETIC_FILE_PATH_HITGEN = (
                f"assets/model_weights/{DATASET}_{DATASET_GROUP}_synthetic_hitgen.pkl"
            )
            SYNTHETIC_FILE_PATH_TIMEGAN = (
                f"assets/model_weights/{DATASET}_{DATASET_GROUP}_synthetic_timegan.pkl"
            )

            create_dataset_vae = CreateTransformedVersionsCVAE(
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                freq=FREQ,
                horizon=H,
                opt_score=args.opt_score,
            )

            # hypertuning
            model = create_dataset_vae.hyper_tune_and_train()

            data_mask_temporalized = TemporalizeGenerator(
                create_dataset_vae.original_wide_transf,
                create_dataset_vae.mask_wide,
                create_dataset_vae.original_dyn_features,
                window_size=create_dataset_vae.best_params["window_size"],
                batch_size=create_dataset_vae.best_params["batch_size"],
                shuffle=create_dataset_vae.best_params["shuffle"],
            )

            (
                _,
                _,
                synth_hitgen_test_long,
            ) = create_dataset_vae.predict(
                model,
                data_mask_temporalized=data_mask_temporalized,
            )

            # TimeGAN synthetic data generation

            # parallel timegan training and synthetic data generation
            # synth_timegan_data_all = Parallel(n_jobs=6)(
            #     delayed(train_and_generate_synthetic)(
            #         ts, original_data_long, DATASET, DATASET_GROUP, WINDOW_SIZE
            #     )
            #     for ts in original_data_long["unique_id"].unique()
            # )

            # best_params = hyper_tune_timegan(
            #     train_data_long, DATASET, DATASET_GROUP, window_size=24, n_trials=50
            # )
            # final_model = train_timegan_with_best_params(
            #     test_data_long, best_params, DATASET, DATASET_GROUP, window_size=24
            # )

            test_unique_ids = create_dataset_vae.original_test_long[
                "unique_id"
            ].unique()

            # hypertuning timegan
            # hyper_tune_timegan(
            #     data=original_data_long,
            #     dataset_name=DATASET,
            #     dataset_group=DATASET_GROUP,
            #     window_size=WINDOW_SIZE,
            #     long_properties=create_dataset_vae.long_properties,
            #     freq=FREQ,
            # )

            # synthetic_timegan_long = workflow_timegan(
            #     test_unique_ids,
            #     SYNTHETIC_FILE_PATH_TIMEGAN,
            #     test_data_long,
            #     DATASET,
            #     DATASET_GROUP,
            #     WINDOW_SIZE,
            #     create_dataset_vae.long_properties,
            #     FREQ,
            #     timegan_config,
            # )

            # metaforecast methods
            synthetic_metaforecast_long = workflow_metaforecast_methods(
                df=create_dataset_vae.original_test_long,
                freq=FREQ,
                dataset=DATASET,
                dataset_group=DATASET_GROUP,
            )

            plot_generated_vs_original(
                synth_data=synth_hitgen_test_long,
                original_test_data=create_dataset_vae.original_test_long,
                score=0.0,
                loss=0.0,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                n_series=4,
                suffix_name="hitgen",
            )

            if not hitgen_score_disc:
                print("\nComputing discriminative score for HiTGen synthetic data...")
                hitgen_score_disc = compute_discriminative_score(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synth_hitgen_test_long,
                    method="hitgen",
                    freq=FREQ,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    loss=0.0,
                    samples=5,
                    split="test",
                )

            print("\nComputing TSTR score for HiTGen synthetic data...")
            hitgen_score_tstr = tstr(
                unique_ids=test_unique_ids,
                original_data=create_dataset_vae.original_test_long,
                synthetic_data=synth_hitgen_test_long,
                method="hitgen",
                freq=FREQ,
                horizon=H,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                samples=5,
                split="test",
            )

            print(
                "\nComputing downstream task forecasting score for HiTGen synthetic data..."
            )
            hitgen_score_dtf = compute_downstream_forecast(
                unique_ids=test_unique_ids,
                original_data=create_dataset_vae.original_test_long,
                synthetic_data=synth_hitgen_test_long,
                method="hitgen",
                freq=FREQ,
                horizon=H,
                dataset_name=DATASET,
                dataset_group=DATASET_GROUP,
                samples=5,
                split="test",
            )

            print(f"\n\n{DATASET}")
            print(f"{DATASET_GROUP}")

            print(
                f"Discriminative score for HiTGen synthetic data: {hitgen_score_disc:.4f}"
            )

            print(
                f"TSTR score for HiTGen synthetic data: "
                f"TSTR {hitgen_score_tstr['avg_smape_tstr']:.4f} vs TRTR "
                f"score {hitgen_score_tstr['avg_smape_trtr']:.4f}"
            )

            print(
                f"Downstream task forecasting score for HiTGen synthetic data: "
                f"concat score {hitgen_score_dtf['avg_smape_concat']:.4f} vs original "
                f"score {hitgen_score_dtf['avg_smape_original']:.4f}\n\n"
            )

            results.append(
                {
                    "Dataset": DATASET,
                    "Group": DATASET_GROUP,
                    "Method": "HiTGen",
                    "Discriminative Score": hitgen_score_disc,
                    "TSTR (avg_smape_tstr)": hitgen_score_tstr["avg_smape_tstr"],
                    "TRTR (avg_smape_trtr)": hitgen_score_tstr["avg_smape_trtr"],
                    "DTF Concat Score": hitgen_score_dtf["avg_smape_concat"],
                    "DTF Original Score": hitgen_score_dtf["avg_smape_original"],
                }
            )

            # print("\nComputing discriminative score for TimeGAN synthetic data...")
            # score_timegan = compute_discriminative_score(
            #     unique_ids=test_unique_ids,
            #     original_data=test_data_long,
            #     synthetic_data=synthetic_timegan_long,
            #     freq="M",
            #     dataset_name=DATASET,
            #     dataset_group=DATASET_GROUP,
            #     loss=0.0,
            #     samples=5,
            # )

            # print(f"Discriminative score for TimeGAN synthetic data: {score_timegan:.4f}")

            for method in METAFORECAST_METHODS:
                synthetic_metaforecast_long_method = synthetic_metaforecast_long.loc[
                    synthetic_metaforecast_long["method"] == method
                ].copy()

                synthetic_metaforecast_long_method = (
                    synthetic_metaforecast_long_method.merge(
                        create_dataset_vae.original_test_long[["unique_id", "ds"]],
                        on=["unique_id", "ds"],
                        how="right",
                    )
                )
                synthetic_metaforecast_long_method.fillna(0, inplace=True)

                plot_generated_vs_original(
                    synth_data=synthetic_metaforecast_long_method,
                    original_test_data=create_dataset_vae.original_test_long,
                    score=0.0,
                    loss=0.0,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    n_series=4,
                    suffix_name=f"{method}",
                )

                print(
                    f"\nComputing discriminative score for {method} synthetic data generation..."
                )

                score_disc = compute_discriminative_score(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synthetic_metaforecast_long_method,
                    method=method,
                    freq=FREQ,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    loss=0.0,
                    samples=5,
                    split="test",
                )

                print(f"\nComputing TSTR score for {method} synthetic data...")
                score_tstr = tstr(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synthetic_metaforecast_long_method,
                    method=method,
                    freq=FREQ,
                    horizon=H,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    samples=5,
                    split="test",
                )

                print(
                    f"\nComputing downstream task forecasting score for {method} synthetic data..."
                )
                score_dtf = compute_downstream_forecast(
                    unique_ids=test_unique_ids,
                    original_data=create_dataset_vae.original_test_long,
                    synthetic_data=synthetic_metaforecast_long_method,
                    method=method,
                    freq=FREQ,
                    horizon=H,
                    dataset_name=DATASET,
                    dataset_group=DATASET_GROUP,
                    samples=5,
                    split="test",
                )

                print(f"\n\n{DATASET}")
                print(f"{DATASET_GROUP}")

                print(
                    f"Discriminative score for {method} synthetic data: {score_disc:.4f}"
                )

                print(
                    f"TSTR score for {method} synthetic data: "
                    f"TSTR {score_tstr['avg_smape_tstr']:.4f} vs TRTR "
                    f"score {score_tstr['avg_smape_trtr']:.4f}"
                )

                print(
                    f"Downstream task forecasting score for {method} synthetic data: "
                    f"concat score {score_dtf['avg_smape_concat']:.4f} vs original "
                    f"score {score_dtf['avg_smape_original']:.4f}\n\n"
                )

                results.append(
                    {
                        "Dataset": DATASET,
                        "Group": DATASET_GROUP,
                        "Method": method,
                        "Discriminative Score": round(score_disc, 3),
                        "TSTR (avg_smape_tstr)": round(score_tstr["avg_smape_tstr"], 3),
                        "TRTR (avg_smape_trtr)": round(score_tstr["avg_smape_trtr"], 3),
                        "DTF Concat Score": round(score_dtf["avg_smape_concat"], 3),
                        "DTF Original Score": round(score_dtf["avg_smape_original"], 3),
                    }
                )

    results_df = pd.DataFrame(results)
    results_df = results_df.round(3)

    print(results_df)

    results_path = "assets/results/synthetic_data_results.csv"
    results_df.to_csv(results_path, index=False)
