from keras import layers
from keras import backend as K
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from keras.regularizers import l2


def custom_relu_linear_saturation(x):
    """
    Custom activation:
    - 0 for x < 0 (ReLU behavior)
    - Linear (x) for 0 <= x <= 1
    - Saturates at 1 for x > 1
    """
    relu_part = tf.nn.relu(x)

    # linear between 0 and 1 and saturation at 1
    linear_part = tf.minimum(relu_part, 1.0)

    return linear_part


class TemporalizeGenerator(Sequence):
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

        return (batch_data, batch_mask, batch_dyn_features), batch_data

    def on_epoch_end(self):
        """Shuffle data and re-temporalize."""
        if self.shuffle:
            shuffled_indices = tf.random.shuffle(
                tf.range(tf.shape(self.temporalized_data)[0])
            )
            self.temporalized_data = tf.gather(self.temporalized_data, shuffled_indices)
            self.temporalized_mask = tf.gather(self.temporalized_mask, shuffled_indices)
            self.temporalized_dyn_features = tf.gather(
                self.temporalized_dyn_features, shuffled_indices
            )

            tf.print(f"SHUFFLING AND RE-TEMPORALIZING: Epoch {self.epoch}")
        self.epoch += 1

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


class KLAnnealingAndNoiseScalingCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model,
        kl_weight_initial,
        kl_weight_final,
        noise_scale_initial,
        noise_scale_final,
        annealing_epochs,
        annealing,
    ):
        super().__init__()
        self.model = model
        self.kl_weight_initial = kl_weight_initial
        self.kl_weight_final = kl_weight_final
        self.noise_scale_initial = noise_scale_initial
        self.noise_scale_final = noise_scale_final
        self.annealing_epochs = annealing_epochs
        self.annealing = annealing

    def compute_progress(self, epoch):
        return min((epoch + 1) / self.annealing_epochs, 1.0)

    def on_epoch_begin(self, epoch, logs=None):
        if self.annealing:
            # Calculate progress and update variables
            progress = self.compute_progress(epoch)
            new_kl_weight = self.kl_weight_initial + progress * (
                self.kl_weight_final - self.kl_weight_initial
            )
            new_noise_scale = self.noise_scale_initial + progress * (
                self.noise_scale_final - self.noise_scale_initial
            )

            # Update the model's tf.Variable instances
            self.model.kl_weight.assign(new_kl_weight)
            sampling_layer = self.model.encoder.get_layer(name="sampling")
            if sampling_layer:
                sampling_layer.noise_scale.assign(new_noise_scale)

            tf.print(
                f"UPDATING OPT VARIABLES: Epoch {epoch + 1}: KL weight = {new_kl_weight:.4f}, Noise scale = {new_noise_scale:.4f}"
            )

    def on_epoch_end(self, epoch, logs=None):
        """Print KL weight and noise scale at the end of the epoch."""
        sampling_layer = self.model.encoder.get_layer(name="sampling")
        current_noise_scale = (
            sampling_layer.noise_scale.numpy() if sampling_layer else "N/A"
        )
        current_kl_weight = self.model.kl_weight.numpy()
        print(
            f"OPT VARIABLES at Epoch {epoch + 1} end: KL weight = {current_kl_weight:.4f}, Noise scale = {current_noise_scale:.4f}"
        )


def masked_mse(y_true, y_pred, mask):
    """
    Compute Mean Squared Error only on the unmasked (non-padded) entries.
    """
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    mse = tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))
    return mse


def cosine_similarity_loss(y_true, y_pred, mask):
    """
    Computes cosine similarity loss for the unmasked parts of the sequences.
    """
    y_true_masked = y_true * mask
    y_pred_masked = y_pred * mask

    dot_product = tf.reduce_sum(
        y_true_masked * y_pred_masked, axis=-1
    )  # (batch_size, timesteps)
    norm_true = tf.sqrt(tf.reduce_sum(tf.square(y_true_masked), axis=-1) + 1e-8)
    norm_pred = tf.sqrt(tf.reduce_sum(tf.square(y_pred_masked), axis=-1) + 1e-8)

    cosine_similarity = dot_product / (norm_true * norm_pred + 1e-8)

    # turn cosine similarity into a loss (1 - similarity)
    cosine_loss = 1 - tf.reduce_mean(cosine_similarity)

    return cosine_loss


class CVAE(keras.Model):
    def __init__(
        self,
        encoder: keras.Model,
        decoder: keras.Model,
        kl_weight_initial: int = None,
        **kwargs,
    ) -> None:
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        self.kl_weight = tf.Variable(
            kl_weight_initial, trainable=False, name="kl_weight"
        )

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        batch_data, batch_mask, batch_dyn_features = inputs

        z_mean, z_log_var, z = self.encoder(
            [batch_data, batch_mask, batch_dyn_features]
        )

        pred = self.decoder([z, batch_mask, batch_dyn_features])

        return pred, z_mean, z_log_var

    def compute_loss(self, inp_data, pred, z_mean, z_log_var, mask):
        """
        Computes total loss with reconstruction loss, cosine similarity and KL divergence.
        """
        reconstruction_loss = masked_mse(inp_data, pred, mask)

        # cosine_loss = cosine_similarity_loss(inp_data, pred, mask)

        # checking numeric instability
        # tf.print("\nz_mean min:", tf.reduce_min(z_mean), "max:", tf.reduce_max(z_mean))
        # tf.print(
        #     "z_log_var min:", tf.reduce_min(z_log_var), "max:", tf.reduce_max(z_log_var)
        # )
        # tf.print(
        #     "exp(z_log_var) min:",
        #     tf.reduce_min(tf.exp(z_log_var)),
        #     "max:",
        #     tf.reduce_max(tf.exp(z_log_var)),
        # )

        kl_loss = -0.5 * K.mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        # total_loss = reconstruction_loss + 0.1 * cosine_loss + self.kl_weight * kl_loss

        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        inputs, targets = data
        batch_data, batch_mask, batch_dyn_features = inputs

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(
                [batch_data, batch_mask, batch_dyn_features]
            )
            pred = self.decoder([z, batch_mask, batch_dyn_features])

            total_loss, reconstruction_loss, kl_loss = self.compute_loss(
                batch_data, pred, z_mean, z_log_var, batch_mask
            )

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        inputs, targets = data
        batch_data, batch_mask, batch_dyn_features = inputs

        z_mean, z_log_var, z = self.encoder(
            [batch_data, batch_mask, batch_dyn_features]
        )
        pred = self.decoder([z, batch_mask, batch_dyn_features])

        total_loss, reconstruction_loss, kl_loss = self.compute_loss(
            batch_data, pred, z_mean, z_log_var, batch_mask
        )

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
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
    bi_rnn: bool = True,
    noise_scale_init: float = 0.01,
    n_blocks_encoder: int = 3,
    n_blocks_decoder: int = 3,
    n_hidden: int = 64,
    n_layers: int = 2,
    kernel_size: int = 2,
    pooling_mode: str = "max",
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
        bi_rnn=bi_rnn,
        noise_scale_init=noise_scale_init,
        n_blocks=n_blocks_encoder,
        n_hidden=n_hidden,
        n_layers=n_layers,
        kernel_size=kernel_size,
        pooling_mode=pooling_mode,
    )

    dec = decoder(
        output_shape=input_shape,
        output_shape_dyn_features=input_shape_dyn_features,
        latent_dim=latent_dim,
        bi_rnn=bi_rnn,
        n_blocks=n_blocks_decoder,
        n_hidden=n_hidden,
        n_layers=n_layers,
        kernel_size=kernel_size,
        pooling_mode=pooling_mode,
    )

    return enc, dec


class MRHIBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        backcast_size,
        n_hidden,
        n_layers,
        pooling_mode="max",
        kernel_size=3,
        **kwargs,
    ):
        """
        Multi-Rate Hierarchical Interpolation Block for time-series decomposition.
        """
        super(MRHIBlock, self).__init__(**kwargs)
        self.backcast_size = backcast_size

        if pooling_mode == "max":
            self.pooling_layer = layers.MaxPooling1D(
                pool_size=kernel_size, strides=1, padding="same"
            )
        else:
            self.pooling_layer = layers.AveragePooling1D(
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


def encoder(
    input_shape,
    input_shape_dyn_features,
    latent_dim,
    n_blocks=3,
    n_hidden=64,
    n_layers=2,
    kernel_size=2,
    pooling_mode="max",
    bi_rnn=True,
    noise_scale_init=0.01,
):
    main_input = layers.Input(shape=input_shape, name="main_input")
    mask_input = layers.Input(shape=input_shape, name="mask_input")
    dyn_features_input = layers.Input(
        shape=input_shape_dyn_features, name="dyn_features_input"
    )

    masked_input = layers.Multiply(name="masked_input")([main_input, mask_input])
    masked_input = layers.Concatenate()([dyn_features_input, masked_input])

    # Using LeakyReLU instead of ReLU
    masked_input = layers.TimeDistributed(
        layers.Dense(input_shape[1], activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(masked_input)

    backcast_total = masked_input
    final_output = 0

    for i in range(n_blocks):
        mrhi_block = MRHIBlock(
            backcast_size=input_shape,
            n_hidden=n_hidden,
            n_layers=n_layers,
            pooling_mode=pooling_mode,
            kernel_size=kernel_size,
        )

        backcast = mrhi_block(backcast_total)
        backcast_total = backcast_total - backcast
        final_output += backcast

    if bi_rnn:
        backcast = layers.Bidirectional(
            layers.GRU(
                input_shape[1],
                return_sequences=True,
                dropout=0.3,
                kernel_regularizer=l2(0.001),
            )
        )(backcast_total)

        backcast = layers.TimeDistributed(
            layers.Dense(
                input_shape[1], activation=tf.keras.layers.LeakyReLU(alpha=0.01)
            )
        )(backcast)

        final_output += backcast

    final_output = layers.TimeDistributed(
        layers.Dense(latent_dim * 2, activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(final_output)

    # clip z_log_var and z_mean to prevent KL loss explosion
    z_mean = tf.clip_by_value(final_output[:, :, latent_dim:], -5, 5)
    z_log_var = tf.clip_by_value(final_output[:, :, :latent_dim], -5, 5)

    z = Sampling(name="sampling", noise_scale_init=noise_scale_init)(
        [z_mean, z_log_var]
    )

    return tf.keras.Model(
        inputs=[main_input, mask_input, dyn_features_input],
        outputs=[z_mean, z_log_var, z],
        name="encoder",
    )


def decoder(
    output_shape,
    output_shape_dyn_features,
    latent_dim,
    n_blocks=3,
    n_hidden=64,
    n_layers=2,
    kernel_size=2,
    pooling_mode="max",
    bi_rnn=True,
):

    latent_input = layers.Input(
        shape=(output_shape[0], latent_dim),
        name="latent_input",
    )
    dyn_features_input = layers.Input(
        shape=output_shape_dyn_features, name="dyn_features_input"
    )
    mask_input = layers.Input(shape=output_shape, name="mask_input")

    x = layers.TimeDistributed(
        layers.Dense(output_shape[1], activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(latent_input)

    x = layers.Concatenate()([dyn_features_input, x])

    x = layers.TimeDistributed(
        layers.Dense(output_shape[1], activation=tf.keras.layers.LeakyReLU(alpha=0.01))
    )(x)

    backcast_total = x
    final_output = 0

    for i in range(n_blocks):
        mrhi_block = MRHIBlock(
            backcast_size=output_shape,
            n_hidden=n_hidden,
            n_layers=n_layers,
            pooling_mode=pooling_mode,
            kernel_size=kernel_size,
        )

        backcast = mrhi_block(backcast_total)
        backcast_total = backcast_total - backcast
        final_output += backcast

    if bi_rnn:
        backcast = layers.Bidirectional(
            layers.GRU(
                output_shape[1],
                return_sequences=True,
                dropout=0.3,
                kernel_regularizer=l2(0.001),
            )
        )(backcast_total)

        backcast = layers.TimeDistributed(
            layers.Dense(
                output_shape[1], activation=tf.keras.layers.LeakyReLU(alpha=0.01)
            )
        )(backcast)

        final_output += backcast

    out = layers.Flatten(name="flatten_decoder_output_CVAE")(final_output)
    out = layers.Dense(
        output_shape[0] * output_shape[1],
        kernel_regularizer=l2(0.001),
        activation=tf.keras.layers.LeakyReLU(alpha=0.01),
        name="dense_output_CVAE",
    )(out)

    out = layers.Reshape(
        (output_shape[0], output_shape[1]), name="reshape_final_output_CVAE"
    )(out)

    final_output = layers.Multiply(name="masked_output")([out, mask_input])

    final_output = custom_relu_linear_saturation(final_output)

    return tf.keras.Model(
        inputs=[latent_input, mask_input, dyn_features_input],
        outputs=[final_output],
        name="decoder",
    )
