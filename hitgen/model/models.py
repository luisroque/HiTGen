import logging
import os
from typing import List, Tuple, Union

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from keras import layers
import tensorflow as tf
from tensorflow.keras import utils
from keras import regularizers
from tensorflow import keras

K = keras.backend


class TemporalizeGenerator(utils.Sequence):
    def __init__(
        self,
        data,
        mask,
        dyn_features,
        window_size,
        stride=1,
        batch_size=8,
        shuffle=True,
    ):
        """
        A generator that reshuffles and re-temporalizes the dataset before each epoch.
        """
        self.data = tf.convert_to_tensor(data, dtype=tf.float32)
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        self.dyn_features = tf.convert_to_tensor(dyn_features, dtype=tf.float32)

        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.temporalized_data = self.temporalize(self.data)
        self.temporalized_mask = self.temporalize(self.mask)
        self.temporalized_dyn_features = self.temporalize(self.dyn_features)
        self.indices = tf.range(len(self.temporalized_data))
        self.epoch = 0

    def __len__(self):
        """Number of batches per epoch, including the last incomplete batch."""
        total_batches = len(self.indices) // self.batch_size
        if len(self.indices) % self.batch_size != 0:
            total_batches += 1
        return total_batches

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_data = tf.gather(self.temporalized_data, batch_indices)
        batch_mask = tf.gather(self.temporalized_mask, batch_indices)
        batch_dyn_features = tf.gather(self.temporalized_dyn_features, batch_indices)

        # one-step ahead
        batch_target_indices = tf.minimum(
            batch_indices + 1, len(self.temporalized_data) - 1
        )
        batch_target = tf.gather(self.temporalized_data, batch_target_indices)

        # multi-step ahead
        # future_steps = 24  # number of steps to predict
        # batch_target = tf.map_fn(
        #     lambda i: self.temporalized_data[i + 1 : i + 1 + future_steps],
        #     batch_indices,
        #     fn_output_signature=tf.TensorSpec(
        #         (future_steps, batch_data.shape[2]), tf.float32
        #     ),
        # )

        return (batch_data, batch_mask, batch_dyn_features), batch_target

    def temporalize(self, data):
        """
        Create temporal windows from the input data.
        """
        num_windows = (tf.shape(data)[0] - self.window_size) // self.stride + 1
        indices = tf.range(num_windows) * self.stride
        windows = tf.map_fn(
            lambda i: data[i : i + self.window_size],
            indices,
            fn_output_signature=tf.TensorSpec(
                (self.window_size, data.shape[1]), tf.float32
            ),
        )
        return windows


class Sampling(tf.keras.layers.Layer):
    def __init__(self, noise_scale_init=0.01, **kwargs):
        super(Sampling, self).__init__(**kwargs)
        # define noise_scale as a mutable variable
        self.noise_scale = tf.Variable(
            noise_scale_init, trainable=False, dtype=tf.float32, name="noise_scale"
        )

    def call(self, inputs):
        """
        Performs the reparameterization trick.
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        seq_len = tf.shape(z_mean)[1]
        latent_dim = tf.shape(z_mean)[2]

        epsilon = tf.keras.backend.random_normal(shape=(batch, seq_len, latent_dim))

        # eeparameterization trick with dynamic noise scaling
        return z_mean + tf.exp(0.5 * z_log_var) * self.noise_scale * epsilon


def masked_mse(y_true, y_pred, mask):
    """
    Compute Mean Squared Error only on the unmasked (non-padded) entries.
    """
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    mse = tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))
    return mse


class CVAE(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        kl_weight_initial: float = None,
        forecasting: bool = True,
        **kwargs,
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.kl_weight = tf.Variable(
            kl_weight_initial, trainable=False, name="kl_weight"
        )

        self.forecasting = forecasting

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.prediction_loss_tracker = keras.metrics.Mean(name="prediction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        batch_data, batch_mask, batch_dyn_features = inputs

        z_mean, z_log_var, z = self.encoder(
            [batch_data, batch_mask, batch_dyn_features]
        )

        pred_reconst, pred = self.decoder([z, batch_mask, batch_dyn_features])

        return pred_reconst, z_mean, z_log_var

    @property
    def metrics(self):
        """Ensures that Keras tracks the metrics properly during training."""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.prediction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def compute_loss(
        self, inp_data, pred_reconst, pred, z_mean, z_log_var, mask, batch_target
    ):
        """
        Computes total loss with reconstruction loss, cosine similarity and KL divergence.
        """
        reconstruction_loss = masked_mse(
            y_true=inp_data, y_pred=pred_reconst, mask=mask
        )

        kl_loss = -0.5 * K.mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        if self.forecasting:
            prediction_loss = masked_mse(y_true=batch_target, y_pred=pred, mask=mask)
            total_loss = (
                reconstruction_loss + prediction_loss + self.kl_weight * kl_loss
            )
        else:
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
            prediction_loss = reconstruction_loss

        return total_loss, reconstruction_loss, prediction_loss, kl_loss

    @tf.function
    def train_step(self, data):
        inputs, batch_target = data
        batch_data, batch_mask, batch_dyn_features = inputs

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(
                [batch_data, batch_mask, batch_dyn_features]
            )
            pred_reconst, pred = self.decoder([z, batch_mask, batch_dyn_features])

            total_loss, reconstruction_loss, prediction_loss, kl_loss = (
                self.compute_loss(
                    batch_data,
                    pred_reconst,
                    pred,
                    z_mean,
                    z_log_var,
                    batch_mask,
                    batch_target,
                )
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.prediction_loss_tracker.update_state(prediction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "prediction_loss": self.prediction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        inputs, batch_target = data
        batch_data, batch_mask, batch_dyn_features = inputs

        z_mean, z_log_var, z = self.encoder(
            [batch_data, batch_mask, batch_dyn_features]
        )
        pred_reconst, pred = self.decoder([z, batch_mask, batch_dyn_features])

        total_loss, reconstruction_loss, prediction_loss, kl_loss = self.compute_loss(
            batch_data,
            pred_reconst,
            pred,
            z_mean,
            z_log_var,
            batch_mask,
            batch_target,
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "prediction_loss": prediction_loss,
            "kl_loss": kl_loss,
        }

    def get_config(self):
        config = super(CVAE, self).get_config()
        config.update(
            {
                "encoder": self.encoder.get_config(),
                "decoder": self.decoder.get_config(),
                "kl_weight_initial": self.kl_weight.numpy(),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder")
        decoder_config = config.pop("decoder")

        encoder = keras.Model.from_config(encoder_config)
        decoder = keras.Model.from_config(decoder_config)

        return cls(encoder=encoder, decoder=decoder, **config)


def get_CVAE(
    window_size: int,
    n_series: int,
    latent_dim: int,
    noise_scale_init: float = 0.01,
    n_blocks_encoder: int = 3,
    n_blocks_decoder: int = 3,
    n_hidden: int = 64,
    n_layers: int = 2,
    kernel_size: int = 2,
    forecasting: bool = True,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Constructs and returns the encoder and decoder models for the CVAE.
    """
    input_shape = (window_size, n_series)
    input_shape_dyn_features = (window_size, 6)

    enc = encoder(
        input_shape=input_shape,
        input_shape_dyn_features=input_shape_dyn_features,
        latent_dim=latent_dim,
        noise_scale_init=noise_scale_init,
        n_blocks=n_blocks_encoder,
        n_hidden=n_hidden,
        n_layers=n_layers,
        kernel_size=kernel_size,
    )

    dec = decoder(
        output_shape=input_shape,
        output_shape_dyn_features=input_shape_dyn_features,
        latent_dim=latent_dim,
        n_blocks=n_blocks_decoder,
        n_hidden=n_hidden,
        n_layers=n_layers,
        kernel_size=kernel_size,
        forecasting=forecasting,
    )

    return enc, dec


class MRHIBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        backcast_size,
        n_hidden,
        n_layers,
        kernel_size=3,
        **kwargs,
    ):
        """
        Multi-Rate Hierarchical Interpolation Block.
        """
        super(MRHIBlock, self).__init__(**kwargs)
        self.backcast_size = backcast_size

        self.pooling_layer = layers.MaxPooling1D(
            pool_size=kernel_size, strides=1, padding="same"
        )

        self.mlp_stack = tf.keras.Sequential(
            [layers.Dense(n_hidden, activation="relu") for _ in range(n_layers)]
        )
        self.backcast_layer = layers.TimeDistributed(
            layers.Dense(backcast_size[1], activation="linear")
        )

    def call(self, inputs):
        x = self.pooling_layer(inputs)
        x = self.mlp_stack(x)

        backcast = self.backcast_layer(x)

        return backcast


class UpsampleTimeLayer(tf.keras.layers.Layer):
    def __init__(self, target_len: int, method="bilinear"):
        super().__init__()
        self.target_len = target_len
        self.method = method

    def call(self, x):
        # x shape: [B, current_len, num_features]
        x = tf.expand_dims(x, axis=1)  # => [B, 1, current_len, num_features]
        x_upsampled = tf.image.resize(
            images=x, size=(1, self.target_len), method=self.method
        )  # => [B, 1, target_len, num_features]
        x_upsampled = tf.squeeze(
            x_upsampled, axis=1
        )  # => [B, target_len, num_features]
        return x_upsampled


class MRHIBlock_backcast_forecast(tf.keras.layers.Layer):
    """
    block:
      1) Pools (downsamples) the input time dimension with strides=pool_size.
      2) Flattens and passes through an MLP => outputs a parameter vector θ.
      3) Splits θ into [backcast portion, knots portion].
      4) Upsamples knots to the full sequence length for the forecast.
      5) Reshapes backcast portion to [seq_len, num_features].
    """

    def __init__(
        self,
        backcast_size,
        n_knots: int,
        n_hidden: int,
        n_layers: int,
        kernel_size=3,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seq_len, self.num_features = backcast_size
        self.n_knots = n_knots
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # final MLP output dimension => backcast + knots
        # backcast is (seq_len * num_features), knots is (n_knots * num_features).
        self.n_theta_backcast = self.seq_len * self.num_features
        self.n_theta_forecast = self.n_knots * self.num_features
        self.n_theta = self.n_theta_backcast + self.n_theta_forecast

        self.upsample_layer = UpsampleTimeLayer(
            target_len=self.seq_len, method="bilinear"
        )

        self.pooling_layer = layers.MaxPooling1D(
            pool_size=kernel_size, strides=kernel_size, padding="valid"
        )

        # flatten => MLP => final Dense(n_theta)
        mlp_layers = []
        mlp_layers.append(
            layers.Flatten()
        )  # [B, T//k, num_features] => [B, (T//k)*num_features]
        for _ in range(n_layers):
            mlp_layers.append(layers.Dense(n_hidden, activation=activation))
        mlp_layers.append(layers.Dense(self.n_theta))  # outputs [B, n_theta]
        self.mlp_stack = tf.keras.Sequential(mlp_layers)

    def call(self, x: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """
        x: [B, seq_len, num_features]
        Returns: backcast, forecast => [B, seq_len, num_features]
        """
        pooled = self.pooling_layer(x)  # => [B, seq_len//k, num_features]

        theta = self.mlp_stack(pooled)  # => [B, n_theta]

        backcast_flat = theta[:, : self.n_theta_backcast]  # [B, seq_len * num_features]
        knots_flat = theta[:, self.n_theta_backcast :]  # [B, n_knots * num_features]

        backcast = tf.reshape(backcast_flat, [-1, self.seq_len, self.num_features])

        knots = tf.reshape(knots_flat, [-1, self.n_knots, self.num_features])
        forecast = self.upsample_layer(knots)

        return backcast, forecast


def encoder(
    input_shape: Tuple[int, int],
    input_shape_dyn_features: Tuple[int, int],
    latent_dim: int,
    n_blocks: int = 3,
    n_hidden: int = 64,
    n_layers: int = 2,
    kernel_size: int = 2,
    noise_scale_init: float = 0.01,
):
    main_input = layers.Input(shape=input_shape, name="main_input")
    mask_input = layers.Input(shape=input_shape, name="mask_input")
    dyn_features_input = layers.Input(
        shape=input_shape_dyn_features, name="dyn_features_input"
    )

    masked_input = layers.Multiply(name="masked_input")([main_input, mask_input])
    masked_input = layers.Concatenate()([dyn_features_input, masked_input])

    masked_input = layers.TimeDistributed(
        layers.Dense(input_shape[1], activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(masked_input)

    backcast_total = masked_input
    final_output = 0

    for i in range(n_blocks):
        mrhi_block = MRHIBlock_backcast_forecast(
            backcast_size=input_shape,
            n_knots=4,
            n_hidden=n_hidden,
            n_layers=n_layers,
            kernel_size=kernel_size,
        )

        backcast, _ = mrhi_block(backcast_total)

        backcast_total = backcast_total - backcast

        final_output += backcast

    final_output = layers.TimeDistributed(
        layers.Dense(latent_dim * 2, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(final_output)

    # clip z_log_var and z_mean to prevent KL loss explosion
    z_mean = layers.Lambda(lambda x: tf.clip_by_value(x[:, :, latent_dim:], -5, 5))(
        final_output
    )

    z_log_var = layers.Lambda(lambda x: tf.clip_by_value(x[:, :, :latent_dim], -5, 5))(
        final_output
    )

    z = Sampling(name="sampling", noise_scale_init=noise_scale_init)(
        [z_mean, z_log_var]
    )

    return tf.keras.Model(
        inputs=[main_input, mask_input, dyn_features_input],
        outputs=[z_mean, z_log_var, z],
        name="encoder",
    )


def decoder(
    output_shape: Tuple[int, int],
    output_shape_dyn_features: Tuple[int, int],
    latent_dim: int,
    n_blocks: int = 3,
    n_hidden: int = 64,
    n_layers: int = 2,
    kernel_size: int = 2,
    forecasting: bool = True,
):
    time_steps = output_shape[0]
    num_features = output_shape[1]

    latent_input = layers.Input(shape=(time_steps, latent_dim), name="latent_input")
    dyn_features_input = layers.Input(
        shape=output_shape_dyn_features, name="dyn_features_input"
    )
    mask_input = layers.Input(shape=output_shape, name="mask_input")

    x = layers.TimeDistributed(
        layers.Dense(num_features, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(latent_input)

    x = layers.Concatenate()([dyn_features_input, x])

    x = layers.TimeDistributed(
        layers.Dense(num_features, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(x)

    residual = x
    final_forecast = layers.Lambda(lambda t: tf.zeros_like(t))(x)
    final_backcast = 0  # accumulate partial backcasts

    if forecasting:
        for i in range(n_blocks):
            mrhi_block = MRHIBlock_backcast_forecast(
                backcast_size=output_shape,
                n_knots=4,
                n_hidden=n_hidden,
                n_layers=n_layers,
                kernel_size=kernel_size,
            )

            backcast, forecast = mrhi_block(residual)

            # Update the residual, accumulate partial forecast
            residual = residual - backcast
            final_forecast = final_forecast + forecast
            final_backcast = final_backcast + backcast
    else:
        for i in range(n_blocks):
            mrhi_block = MRHIBlock(
                backcast_size=output_shape,
                n_hidden=n_hidden,
                n_layers=n_layers,
                kernel_size=kernel_size,
            )
            backcast = mrhi_block(residual)
            residual = residual - backcast
            final_backcast = final_backcast + backcast
        final_forecast = final_backcast

    backcast_out = final_backcast
    forecast_out = final_forecast

    return tf.keras.Model(
        inputs=[latent_input, mask_input, dyn_features_input],
        outputs=[backcast_out, forecast_out],
        name="decoder",
    )
