"""
Simple LSTM encoder-decoder network
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from src.util.data_handling import SequenceLoss
from tensorflow.keras import Model

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()


class CustomModel(Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

class LstmVAE:
    def __init__(self, samples_per_sequence=240, vocab_size=128, learning_rate=0.005, compression_factor=2,
                 batch_size=32, eps_std=1.0):
        latent_dim = samples_per_sequence * (vocab_size // 2)
        input_shape = (vocab_size, samples_per_sequence)
        inner_size = int(samples_per_sequence / compression_factor)

        inputs = Input(input_shape)
        h = LSTM(inner_size)(inputs)

        z_mean = Dense(latent_dim)(h)
        z_log_sig = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sig = args
            batch = K.shape(z_log_sig)[0]
            dim = K.shape(z_log_sig)[1]
            eps = K.random_normal(shape=(batch, dim), mean=0.0, stddev=eps_std)
            # eps = K.random_normal(shape=z_log_sig.shape, mean=0.0, stddev=eps_std)
            return z_mean + z_log_sig * eps

        z = Lambda(sampling)((z_mean, z_log_sig))

        decoder_h = LSTM(inner_size, return_sequences=True)
        decoder_mean = LSTM(vocab_size, return_sequences=True)

        h_decoded = RepeatVector(samples_per_sequence)(z)
        h_decoded = decoder_h(h_decoded)

        x_decoded_mean = decoder_mean(h_decoded)
        reordered = tf.keras.layers.Permute((2,1))(x_decoded_mean)

        self.vae = CustomModel(inputs, reordered)

        self.encoder = tf.keras.Model(inputs, z_mean)

        decoder_input = Input(shape=(latent_dim,))

        _h_decoded = RepeatVector(samples_per_sequence)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)

        _x_decoded_mean = decoder_mean(_h_decoded)
        self.generator = tf.keras.Model(decoder_input, _x_decoded_mean)

        # decode1 = RepeatVector(vocab_size)(h)
        # decode2 = LSTM(inner_size, return_sequences=True)(decode1)
        # outputs = TimeDistributed(Dense(samples_per_sequence, name='active_pitches'))(decode2)
        # self.model = tf.keras.Model(inputs, outputs)

        mse = MeanSquaredError()

        def vae_loss(x, x_decoded_mean):
            xent_loss = mse(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sig - K.square(z_mean) - K.exp(z_log_sig))
            loss = xent_loss + kl_loss
            return loss

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.vae.compile(optimizer=optimizer, loss=vae_loss)  # , run_eagerly=True)

        # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=optimizer)
        print('VAE summary:')
        self.vae.summary()
        print('Encoder summary:')
        self.encoder.summary()
        print('Generator summary:')
        self.generator.summary()

    def train(self, train_ds: tf.data.Dataset,
              epochs: int,
              ckpt_path='./training_checkpoints/ckpt_{epoch}',
              batch_size=64):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True),
        ]

        # history = self.model.fit(
        history = self.vae.fit(
            train_ds,
            # batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
        )
        pass

    def evaluate(self, eval_ds: tf.data.Dataset):
        losses = self.vae.evaluate(eval_ds, return_dict=True)
        print(losses)

    def save(self, dir_name='lstm_vae'):
        self.vae.save(dir_name)

    def load(self, dir_name='lstm_vae'):
        self.vae = tf.keras.load_model(dir_name)
