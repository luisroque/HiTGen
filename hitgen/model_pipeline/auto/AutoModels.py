from os import cpu_count
import torch

from ray import tune

from hitgen.model_pipeline.HiTGen import HiTGen
from hitgen.model_pipeline.HiTGenDeep import HiTGenDeep
from hitgen.model_pipeline.HiTGenMixture import HiTGenMixture
from hitgen.model_pipeline.HiTGenDeepMixture import HiTGenDeepMixture
from hitgen.model_pipeline.HiTGenDynamicMixture import HiTGenDynamicMixture
from hitgen.model_pipeline.HiTGenDeepMixtureTempNorm import HiTGenDeepMixtureTempNorm
from hitgen.model_pipeline.HiTGenDeepMixtureTempNormLossNorm import (
    HiTGenDeepMixtureTempNormLossNorm,
)

from ray.tune.search.basic_variant import BasicVariantGenerator
from neuralforecast.losses.pytorch import MAE, MSE
from neuralforecast.auto import BaseAuto


class AutoHiTGen(BaseAuto):

    default_config = {
        "latent_dim": tune.choice([16, 32, 64, 128, 256]),
        "encoder_hidden_dims": tune.choice(
            [[64, 32], [256, 128], [512, 256], [512, 256, 128]]
        ),
        # NHITS-like parameters
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGen, self).__init__(
            cls_model=HiTGen,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenDeep(BaseAuto):

    default_config = {
        "latent_dim": tune.choice([16, 32, 64, 128, 256]),
        "encoder_hidden_dims": tune.choice(
            [[64, 32], [256, 128], [512, 256], [512, 256, 128]]
        ),
        # NHITS-like parameters
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenDeep, self).__init__(
            cls_model=HiTGenDeep,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenMixture(BaseAuto):

    default_config = {
        # mixture params
        "n_beats_nblocks_stack_1": tune.choice([0, 1]),
        "n_beats_nblocks_stack_2": tune.choice([0, 1]),
        "n_beats_nblocks_stack_3": tune.choice([0, 1]),
        # VAE params
        "latent_dim": tune.choice([16, 32, 64, 128, 256]),
        "encoder_hidden_dims": tune.choice(
            [[64, 32], [256, 128], [512, 256], [512, 256, 128]]
        ),
        # NHITS params
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        # model params
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenMixture, self).__init__(
            cls_model=HiTGenMixture,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenDeepMixture(BaseAuto):

    default_config = {
        # mixture params
        "n_beats_nblocks_stack_1": tune.choice([0, 1]),
        "n_beats_nblocks_stack_2": tune.choice([0, 1]),
        "n_beats_nblocks_stack_3": tune.choice([0, 1]),
        # VAE params
        "latent_dim": tune.choice([16, 32, 64, 128, 256]),
        "encoder_hidden_dims": tune.choice(
            [[64, 32], [256, 128], [512, 256], [512, 256, 128]]
        ),
        # NHITS params
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        # model params
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenDeepMixture, self).__init__(
            cls_model=HiTGenDeepMixture,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenDeepMixtureTempNorm(BaseAuto):

    default_config = {
        # mixture params
        "n_beats_nblocks_stack_1": tune.choice([0, 1]),
        "n_beats_nblocks_stack_2": tune.choice([0, 1]),
        "n_beats_nblocks_stack_3": tune.choice([0, 1]),
        # VAE params
        "latent_dim": tune.choice([16, 32, 64, 128, 256]),
        "encoder_hidden_dims": tune.choice(
            [[64, 32], [256, 128], [512, 256], [512, 256, 128]]
        ),
        # NHITS params
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        # model params
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenDeepMixtureTempNorm, self).__init__(
            cls_model=HiTGenDeepMixtureTempNorm,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenDeepMixtureTempNormLossNorm(BaseAuto):

    default_config = {
        # mixture params
        "n_beats_nblocks_stack_1": tune.choice([0, 1]),
        "n_beats_nblocks_stack_2": tune.choice([0, 1]),
        "n_beats_nblocks_stack_3": tune.choice([0, 1]),
        # VAE params
        "latent_dim": tune.choice([16, 32, 64, 128, 256]),
        "encoder_hidden_dims": tune.choice(
            [[64, 32], [256, 128], [512, 256], [512, 256, 128]]
        ),
        # NHITS params
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        # model params
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenDeepMixtureTempNormLossNorm, self).__init__(
            cls_model=HiTGenDeepMixtureTempNormLossNorm,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config


class AutoHiTGenDynamicMixture(BaseAuto):

    default_config = {
        # mixture params
        "n_beats_nblocks_stack_1": tune.choice([0, 1]),
        "n_beats_nblocks_stack_2": tune.choice([0, 1]),
        "n_beats_nblocks_stack_3": tune.choice([0, 1]),
        "deep": tune.choice([True, False]),
        # VAE params
        "latent_dim": tune.choice([16, 32, 64, 128, 256]),
        "encoder_hidden_dims": tune.choice(
            [[64, 32], [256, 128], [512, 256], [512, 256, 128]]
        ),
        # NHITS params
        "input_size_multiplier": [1, 2, 3, 4, 5],
        "h": None,
        "n_pool_kernel_size": tune.choice(
            [[2, 2, 1], 3 * [1], 3 * [2], 3 * [4], [8, 4, 1], [16, 8, 1]]
        ),
        "n_freq_downsample": tune.choice(
            [
                [168, 24, 1],
                [24, 12, 1],
                [180, 60, 1],
                [60, 8, 1],
                [40, 20, 1],
                [1, 1, 1],
            ]
        ),
        # model params
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "scaler_type": tune.choice([None, "robust", "standard"]),
        "max_steps": tune.quniform(lower=500, upper=1500, q=100),
        "batch_size": tune.choice([32, 64, 128, 256]),
        "windows_batch_size": tune.choice([128, 256, 512, 1024]),
        "loss": None,
        "random_seed": tune.randint(lower=1, upper=20),
    }

    def __init__(
        self,
        h,
        loss=MAE(),
        valid_loss=None,
        config=None,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        refit_with_val=False,
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):

        if config is None:
            config = self.get_default_config(h=h, backend=backend)

        super(AutoHiTGenDynamicMixture, self).__init__(
            cls_model=HiTGenDynamicMixture,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=config,
            search_alg=search_alg,
            num_samples=num_samples,
            refit_with_val=refit_with_val,
            cpus=cpus,
            gpus=gpus,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
        )

    @classmethod
    def get_default_config(cls, h, backend, n_series=None):
        config = cls.default_config.copy()
        config["input_size"] = tune.choice(
            [h * x for x in config["input_size_multiplier"]]
        )
        config["step_size"] = tune.choice([1, h])
        del config["input_size_multiplier"]
        if backend == "optuna":
            config = cls._ray_config_to_optuna(config)

        return config
